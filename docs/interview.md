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

## 按题目三个问题回答

**页面管理。** `(request, layer, logical_page)` 保持身份，source table 指向 host block，resident table 指向 GPU block 或 -1。Hot tag 记录的是 host physical row，不是页面 owner。`ready/host_ready/last_use` 表示可读与可复用时间，CPU 解绑不代表字节可立即覆盖。

**空间分配。** 一个 slab 中同时有 resident block 和 hot block 租约。准入预留每层 `ceil(hot_rows/page_size)+2 tails` 的容量 credit。低水位或 resident window 提前触发 hot 申请，整组分配成功后再允许 clean 页 unpin。Hot 内部做 row LRU，host credit、索引字节和 hot/tail footprint 都参与准入。

**效率。** GPU resolver 不读回 miss count，生成紧凑 host/device row 计划；Triton 在 transfer stream 从 pinned host 加载 row，compute 等 ready 后直接按 physical row attention。它是 SM copy，不是 copy engine DMA。当前多算子 resolver 与 Python 调度仍有开销，也未实现跨层共享 index 的预取。

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
