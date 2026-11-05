# 用项目回答面试题

## 30 秒开场

“我把逻辑 host 页、resident block 地址、hot block 租约和 hot row 内容分开。resident 与 hot 共享一个 GPU tensor，GPU resolver 先查驻留，再查 hot tags，只为 miss 搬回选中的 token rows。准入保证 hot region 覆盖单步选择并集，tail 保持驻留，因此历史页被回收后仍能 decode，不需要重新 prefill。”

## 建议的五分钟演示

配套：[图解设计](visual-guide.md)、[代码调用链](code-walkthrough.md)、[可运行例子](examples.md)、[当前 baseline 图表](benchmarks.md)。

1. 运行 `python -m mini_hisparse demo --device cuda:0 --top-k 2 --prompt-tokens 257 --gpu-pages 12`。展示请求依次准入、持续生成，prefill 始终一次。
2. 打开 `hybrid.py`：解释 hot credit、实际 hot blocks、active tail、唯一 slab 和 slot fence。
3. 运行 `python examples/inspect_indices.py`，再打开 `row_index.py`：沿 source/resident table → hot tags/LRU → physical row 讲清寻址；全部 hit 先受保护，miss 才选择 victim。
4. 打开 `row_kernels.py`：展示从 pinned host 仅搬 miss rows，attention 直接读取物理行；resolver 独立 CUDA Graph 测试与整引擎 Graph 是两回事。
5. 展示同轨迹 `compare_layouts.py` 结果和真实权重 `validate_pretrained.py --force-host`。说明固定 hot region 的空间代价及 Quest 与 learned DSA 的区别。页级分窗版本作为额外对照。

## 1. 页面管理：怎样记录位置并完成状态转换？

**面试回答：我把逻辑身份、物理所有权和内容可读性分开。逻辑页被卸载后身份不变；临时加载只恢复选中的 rows，不恢复整页的 resident 身份。**

### 用什么记录位置？

| 状态 / 表 | 当前代码中的内容 | 解决的问题 |
|---|---|---|
| `Page.key` | `(request_id, layer, logical_page)` | 页的稳定身份；与 GPU 位置无关 |
| `Page.host_slot` / `tables.source` | logical page → host block | GPU 回收后仍能找到 backing |
| `Page.slot` / `tables.resident` | logical page → GPU block；无驻留时为 `None` / `-1` | resident bypass |
| `tables.hot` / `tables.tags` | hot block 列表；hot slot → host physical row | 碎片化 hot blocks 中每行具体缓存了什么 |
| `DeviceSlot.owner/generation/fence` | block 当前 owner、复用代次、GPU 完成事件 | 防止旧句柄读到新内容，覆盖前等待旧读者 |
| `content_version / host_committed_version` | GPU 最新内容版本 / 已完成 host 镜像版本 | 判断 host 副本是否真的 clean |

前两类页元数据由 CPU 控制；`source/resident/hot/tags` 是设备表，resolver 用它们翻译逻辑 token。设页面大小为 `P`，逻辑 token 为 `t`：

```text
page, offset = divmod(t, P)
host_row = source[page] * P + offset
resident hit: physical_row = resident[page] * P + offset
hot hit/miss: physical_row = hot[slot // P] * P + slot % P
```

Hot tag 存 host physical row，不是 logical page number，也不是 GPU block owner。多头 selection 先合并 token union，同一 KV row 只搬一次，再用逐头 mask 保持各 head 的语义。[逐项地址例子](visual-guide.md#2-用一组数字走完寻址)。

### 卸载与加载的实际过程

1. **写入：** 新 KV 进入 resident tail；`HostStore.append()` 只镜像新增 rows 到 pinned host。提交 D2H 后记录版本和 `host_ready`，此时尚不能宣称 clean。
2. **成为 clean：** 对应 D2H 完成，`host_committed_version == content_version`。满足 sealed、离开最后两页、无 consumer lease，且 hot region 已可接管时，允许 unpin。
3. **可回收但仍可读：** `clean_unpinned` 进入 reclaimable 队列，保留 resident mapping、owner、generation。若尚未被覆盖，后续访问仍是 resident hit。
4. **实际卸载：** allocator 真正复用该 block 时，回调撤销旧 resident mapping；host/source 不变。新 owner 绑定时 generation 增加；fence 保护尚未结束的 GPU 访问。
5. **临时加载：** resolver 对 host-only token 查 hot tags；miss 才选 slot，产生 host-row → device-row plan。Gather 完成事件后 attention 才能读新内容。页面仍为 host-only，hot 中只存在它的部分 rows；这些 rows 之后可被替换。

因此默认 hybrid 不是简单的 `RESIDENT → HOST → HOT 整页 → RESIDENT` 状态机。共享 `Location.HOT` 枚举用于页级 reference；hybrid 的 hot 内容身份在 tags 中，不能只看 `Page.location` 判断某 token 是否已缓存。Tags 已更新也不等于 gather 已完成，`tables.ready` 才给出可读依赖。

请求完成或取消时，先停止逻辑使用并建立 fence；host backing 和 row tables 一起留到 retirement 事件完成才归还，防止 request ID / host slot 重用造成误读。实现入口：[storage.py](../mini_hisparse/storage.py)、[hybrid.py](../mini_hisparse/hybrid.py)、[allocator.py](../mini_hisparse/allocator.py)。

## 2. 空间分配：同一个池里，驱逐谁、留多少 hot？

**面试回答：先用 hot/tail 最低工作集做准入承诺，再按压力兑现 hot blocks。页面回收与 hot row 替换是两层决策，不能用一个“全局 LRU”概括。**

### 驱逐顺序和保护条件

| 决策 | 当前策略 |
|---|---|
| 先用哪种 block？ | 真正空 block 优先，再用仍可读的 reclaimable block |
| 哪些 resident 页能回收？ | host 镜像完成、sealed、无 consumer lease、逻辑页号不大于当前 tail index 减 2 |
| 哪些页必须保护？ | 最后两个逻辑页、dirty/备份未完成页、正在消费的页 |
| 合格页按什么顺序排队？ | 常规 unpin 按逻辑页号从大到小入队，实际复用按 reclaimable 队列顺序；不是按访问频率排序的全局 page LRU |
| hot 空间不足时怎么办？ | 整组预检后，可用本请求/层合格历史页原子补足；预检失败保留 demand 和映射，稍后重试 |
| hot 内部替换什么？ | 先保护本步所有 hot hits，再在本层剩余 slots 中按 row LRU 选择 victims |

启动 hot 时的本层补足路径按 `_historical_pages()` 的 allocator 顺序选候选，与常规 unpin 的尾部优先顺序不是同一个排序。`RowLease` 当前保守 pin 本层全部 resident pages，避免将 GPU top-k 读回 CPU 求精确页集合；这也会暂时减少可回收容量。当前策略优先保证安全和进展，没有实现全局最优驱逐、按请求收益分配 hot 或动态缩容。

Hot 的低水位或 resident-window 触发会保留一个持久 demand。默认水位为 `max(每层 hot blocks, GPU blocks // 10)`，追加分配前使用 `available-1` 预判；默认 resident window 是 64 页，只是触发阈值。Hot lease 成功后持有到请求结束，row 内容可每步替换。[分配流程图](visual-guide.md#5-hot-空间先做准入承诺再在压力下分配)。

### 预留多少？为什么足以让 miss 继续前进？

令 `U` 为一个请求/层单步选中 token 的最大跨头并集，`H` 为 hot rows，`P` 为 page size，`L` 为层数：

```text
H >= U
每请求 device credit = L * (ceil(H / P) + 2)
所有已准入请求的 device credits 之和 <= GPU KV pool blocks
```

Host 最大长度的 backing 和 GPU index bytes 也必须通过准入。Credit 是承诺，不是立即拿走物理 blocks；resident history 暂时可以借用未兑现 hot 空间。

若本步有 `r` 个 resident hits、`h` 个 hot hits、`m` 个 misses，则：

```text
r + h + m <= U
保护 h 个 hot hits 后，可替换 slots = H - h
H - h >= U - h >= m
```

所以即使选中的旧 KV 全部被卸载，只要 `H>=U`，本步也不需要重新装回完整历史。两页 tail 负责追加路径；跨页时保护范围随 `_writing_key` 向前移动，更老的 sealed 页可让出空间。镜像还在途时，同步 runner 可能等一个有用的 D2H 完成事件，之后重试。

这是**有条件的进展保证**：selection 不超过已承诺 U，host/index/总显存资源足够，consumer lease 正常释放，GPU 操作最终完成，并遵守单层单 active row lease 的接口。它不是“任意小显存都能跑”；更不覆盖完整 prefill workspace。新请求仍可能等待准入。

### 为什么默认 2U，而不是 U？

`U` 是上述容量保证的最低界；`2U` 是给跨步复用留余量的启发式默认值，不是数学必要条件，也不保证最优。

例如 `P=16, U=32, L=2`：`H=32` 需 8 blocks/request；`H=64` 需 12。Hot 加大可能提高时间复用，也会减少 resident history 容量、提高 admission 成本，甚至让整个请求集合无法准入。新增的 [固定总容量 hot sweep](hot-capacity.md) 比较 `1U/1.5U/2U`，并标出 `3U` 不可准入的配置。它用于选配置，不是自动调参策略。

## 3. 效率：怎样协调搬运和 attention？

**面试回答：先减少必须搬的数据，再用 stream/event 表达依赖，避免 CPU 为每个 miss 同步。当前实现保证正确排序并利用 row 复用，但同层 attention 仍等待 gather 完成，不能声称已经隐藏全部加载时间。**

### 实际事件链

| 阶段 | 依赖与执行 |
|---|---|
| Append | compute 写新 KV；transfer 等 `page.ready` 后只 D2H 新增 rows |
| Resolve | compute 等上次 `tables.ready`，解析 resident/hot hits、保护 hits、更新 tags，记录 `resolved` |
| Gather | transfer 等 `resolved` 与上一轮 `tables.last_use`，从 pinned host 加载 misses；此前 D2H 已在同一 transfer stream 排在前面 |
| Attention | compute 等新的 `tables.ready`，kernel 直接读取 resident/hot physical rows |
| Release | 在 compute 记录 `last_use`；下一轮复用或写入仍必须尊重该事件 |

CPU 侧的 lease 保护地址和 owner，GPU event 保护在途字节访问；两者缺一不可。异步使用的 miss-plan tensors 也通过 `record_stream()` 保持生命周期。[时序图](visual-guide.md#6-读写时序地址准备好不代表内容准备好)。

### 已实现的减少等待手段

- Resident hit 和 hot hit 都不需要搬回数据；未选中的 KV 完全不加载。
- Token union 去重后，每个 miss row 只搬一次；搬运 payload 为 `misses * 2 * kv_heads * head_dim * dtype_bytes`。
- Resolver 在设备上生成固定形状的 miss plan，有效前缀之外填 -1，无需把 miss count 读回 CPU 再逐个安排 copy。当前 `Selection` 适配仍有 CPU 操作；原生 `LogicalTopK` 可绕过这层。
- Triton gather 通过 pinned host 的 CUDA 地址由 SM 发起读取，直接写到最终 hot rows；attention 按物理 row 读取。它不等于 copy-engine DMA。
- 常规 clean/retirement 检查只 `event.query()`；生产依赖用 stream wait event，容量耗尽的兜底等待针对有用镜像，不做全设备同步。

### 仍未隐藏的开销与下一步

同一层目前按 `resolve → gather → attention` 前进，attention 没有拆成 resident 部分与 miss 部分并行计算。多个 stream 并不使这种数据依赖自动消失。Gather 也占用 SM，与 attention 并发时可能竞争资源。

现有 [baseline 测评](benchmarks.md) 测的是串行 layer append+attention：分散 token 场景减少 H2D、层级回放更快；Quest 场景反而更慢。它证明某些访问形态受益，**没有测出跨层 overlap 比例或真实 serving 吞吐提升**。

要继续优化，先用 profiler/设备事件拆分 index、resolver、gather、attention 与等待，再考虑 fused resolver、设备原生 indexer、已知共享 index 时的跨层预取。下一 token 的 query 尚未产生时不能假设其 top-k 已知。若拆开 resident/miss attention，必须正确合并 softmax 的最大值、分母和未归一化加权和，并重新验证热槽复用 fence；这些优化当前尚未实现。

## 当前项目能否回答这三问？本轮评估与改进

**结论：作为面试用的单卡最小系统，页面管理和空间分配已经能较完整地回答；效率有可运行的初步解法和部分收益证据，尚不能证明生产级调度或充分隐藏传输等待。**

| 拷问 / 顾虑 | 当前证据 | 判断与剩余边界 |
|---|---|---|
| 页在哪里、会不会读错旧槽？ | source/resident/tags、owner/generation/fence；[地址例子](../examples/inspect_indices.py) | 已实现；暂不覆盖共享 prefix/COW、跨进程、多卡 |
| 卸载后是否必须恢复整请求？ | clean/unpin/reuse 生命周期例子；Engine 测试 prefill 始终一次 | 已回答；需要最低 hot/tail 和 host/index 资源 |
| 为什么热空间够用？ | `H-h>=m` 推导；本轮新增全 miss、最小配额连续追加测试 | 已有构造验证；不把长期不释放 lease 的调用者纳入保证 |
| 驱逐谁，是否会破坏读写？ | 两页 tail、clean completion、consumer lease、延迟复用测试 | 已实现安全规则；page 回收策略是启发式，不宣称最优 |
| 为什么留 2U？ | 本轮新增固定 GPU 配额和触发阈值的 hot sweep | 从默认参数补到可复现实验；没有在线自适应分配 |
| 异步是否正确、是否真的更快？ | CUDA 延迟 D2H/reader 测试；同 selection baseline | 正确排序有证据；减少搬运的收益依赖访问形态，overlap 收益尚未证明 |

本轮改进针对原先“答案分散、容量保证缺推导、2U 缺敏感性证据”的缺口：集中三问回答并链接实现；新增最小工作集的最坏 miss 测试；补上 hot 容量实验与不可准入配置。不把另一个完整调度器或新的预取协议塞进教学 runtime。

可复核的测试入口：

```bash
# 页状态、clean 完成、触发条件、两页 tail、全 miss 最低配额
python -m pytest -q tests/test_residency.py
# hot hit 保护、地址解析、延迟复用、真实 Engine 不重做 prefill
python -m pytest -q tests/test_hybrid.py
```

[test_residency.py](../tests/test_residency.py) 的 `test_minimum_hot_capacity_progresses_when_every_selected_row_misses` 在 CPU/CUDA 上使用 `P=4, U=H=4, GPU blocks=3`：32-token prefill 后连续追加 16 次，交替选择互不相交的四个 host-only tokens，每步均全 miss，跨过四次页边界。它逐步对照完整 K/V 的同 selection oracle，并检查两页 tail、容量与最终 64 次 misses。与已有 Engine 的一次 prefill 测试结合，验证“全 miss 仍可前进”，不把它当成性能测试。

## 容易被追问的边界

**为什么不能把每个 chunk 的 attention 输出平均？** 每块 softmax 的分母不同。必须保留 max、归一化分母和未归一化加权和，重标定后再合并。

**尾页没写满时怎么办？** Hybrid 硬保留最后两页，追加只备份新行。跨页后紧邻的新旧两页都保留，再前一页才滑出保护范围；封口本身不代表立即可逐出。

**Unpin 是否就变成 host-only？** 不会。Host 镜像完成后，旧 clean 页可以成为分配器候选，但仍保留 owner、generation 和 resident table。真正分配给新内容时才通过 reuse callback 撤销旧映射；尚未被复用时继续 resident hit。Consumer lease 会暂时阻止候选被复用。

**D2H 只提交了就能 unpin 吗？** 当前严格要求完成事件可见，`host_committed_version` 才追上最新内容。常规 policy tick 非阻塞查询；实际容量耗尽时同步教学 runner 可以等待单个有用的镜像事件，不等整个设备。上游可把这部分等待交还调度器。

**页已经 free，为什么还要等？** CPU 分配器说的是所有权，GPU stream 说的是访问时序。attention 或 D2H 可能还在读取旧槽。

**为什么不把整个请求 swap 回来？** 这会重新要求与上下文长度成比例的瞬时 HBM，正好违反题目的进展目标。

**top-k 会损害精度吗？** 存储层在相同 selection 下保持结果；选择算法是否损害精度是另一个问题。原生 DSA 的 top-k 是模型定义的一部分，本项目对 Qwen2 使用的 Quest 风格或 mean 选择器是近似改动。实测确实存在 attention mass 损失，不能声称无损。

**trace 有多接近真实模型？** 输入是固定 LongBench 文本，Q/K/V 来自预训练 Qwen2.5 的真实 decode，包含 RoPE、GQA、层差异与时间复用；它是 frozen dense teacher trace，尚非原生 DSA。原生 top-k 逻辑位置可导入，但 MLA 压缩布局仍需专门后端。

**为什么保留旧的公平 LRU？** 它是页级参考路径的策略消融，可展示尾页反复加载的问题。默认 hybrid 使用独立 hot 租约和请求/层内 row LRU；active tail 是硬保证，不再仅靠公平 LRU 的软保护。

**请求取消时如何处理在途数据？** 先撤销逻辑所有权、为 GPU 槽保留 fence，host backing 留在 retirement queue，完成事件可见才回收；slot/request generation 检查防止旧引用命中新内容。

**CPU 存储和索引真的免费吗？** 不是。host pool 必须容量准入；pinned memory 会占用实际 RAM；GPU 页摘要、权重与 workspace 都单独计入显存。

**为什么没有直接复刻上游 fused kernel？** 当前已经把 GPU 页表、row tags、LRU 和 miss plan 实现为固定形状设备运算，并验证 resolver Graph capture。还没有 fuse 成一个内核，也没有捕获整个引擎；这与上游吞吐仍有明显差距。

**怎么证明预取有效？** 同一访问轨迹、相同容量与窗口大小做开关对照，排除 JIT 和 oracle；再看 profiler 中 copy 与 attention 的重叠。开了第二条 stream 本身不是证据。

**有没有最小显存限制？** 有。Hybrid 要兑现 hot region 与 active tail，并另计模型、索引和 workspace。一个槽位可分窗前进属于 `pages` 扩展，不能当成 vLLM HiSparse 的容量保证。

**如果让我继续优化？** 接原生 learned indexer / MLA，消除 Quest 适配中的 CPU 往返；将设备 resolver fuse；增加共享 indexer 的跨层 miss-plan 复用和预取，再验证整步 CUDA Graph 与真实模型质量。完整对齐范围见 [hybrid-alignment.md](hybrid-alignment.md)。
