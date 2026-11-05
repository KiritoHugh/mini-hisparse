# Hybrid 路径：对齐 vLLM 的空间租约、row 替换与索引

本轮对照 vLLM `abf98672ce7e822701f7a062bf240efb2c9900d1`。默认 `demo/generate/replay_trace` 使用新的 `HybridKVCache`；原 `PagedKVCache` 仍可通过 `--cache-layout pages` 使用。两者是不同的缓存机制，不能把页级分窗的单槽进展结论套在 hybrid 上。

后续已继续对齐 clean/unpinned 页面状态、hot 水位/window 触发与两页 tail；具体变更及新验证见 [residency-alignment.md](residency-alignment.md)。下方旧性能表保留历史结果，不代表当前策略。

## 已经对齐到哪些机制

| 职责 | vLLM 对照 | mini 当前实现 |
|---|---|---|
| 逻辑 KV 与 resident 地址分离 | source / resident block tables | 每请求、每层分别持有 `source` 和 `resident`，host-only 页面保留 source，resident 为 -1 |
| 共享 GPU 空间，但区分租约类型 | HMA resident/hot groups | 唯一 `[blocks, 2, page, kv_heads, dim]` slab；resident block 与 hot block 有不同 owner |
| hot region 空间租约 | `HiSparseHotManager` | 固定 row 容量，block 向上取整；第一次转 host-backed 时一次取得整组 blocks，支持物理不连续 |
| hot 内容与空间租约分离 | `HiSparseIndexGroup` | GPU `tags[slot] = host_physical_row`、`lru[rank] = slot`；替换 row 不申请/释放 block |
| resident → hot → host 查找 | residency resolver | resident 命中直接得到 physical row，只有非驻留 token 参与 hot 命中与 victim 决策 |
| 本轮 hit 不被本轮 miss 逐出 | resolver union protection | 先识别全部有效选择与 hot hits，再给 misses 分配未命中的 LRU slots |
| compact miss plan | `swap_host/device_physical_rows` | 固定宽度计划的前缀装载 miss，尾部 -1；无需读取 GPU miss count 来决定发射 |
| 只搬运选中 KV rows | compact gather kernel | Triton 直接读 mapped pinned host memory；按 row 复制 K/V，不扩大成完整页 |
| attention 消费 physical indices | sparse attention backend | Triton 直接按 physical row ID 读取同一个 slab，支持 GQA 与每头精确 token 集合 |
| 复用时序 | worker events / block leases | 行消费者事件、block generation、slot fence 和 host retirement queue |

上游入口：[HotManager](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/core/single_type_kv_cache_manager.py)、[runtime](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/hisparse/runtime.py)、[CUDA resolver/gather](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/csrc/hisparse_kernels.cu)。

## 容量：准入 credit 与实际 hot blocks

设页大小为 `P`，一个请求每层 hot 容量为 `H` rows，声明的单步最大选择并集为 `U`：

```text
H >= U
hot_blocks_per_layer = ceil(H / P)
request_block_credits = layers * (hot_blocks_per_layer + 2 active tail pages)
sum(admitted request_block_credits) <= total_gpu_blocks
```

`HotConfig(U)` 默认 `H=2U`，允许额外一份工作集的 LRU 余量。配置 16 个 hot rows 并不表示预留 16 个 pages。最后一个 block 的对齐 padding 不参与 row LRU。

Credit 是准入保证，不是另外分配一块 tensor。Resident history 可以借用尚未兑现的空闲 blocks。低水位或 resident window 触发 hot demand；整组 hot blocks 就绪后，完成 host 镜像且滑出两页 tail 的 sealed pages 可以 unpin。Unpin 保留 resident 映射，仅实际 block 复用才撤销它。Hot block 租约在请求存活期间保持，内部可不断装载不同历史 token。

新请求可能因为 hot/tail footprint 而等待。已经准入的请求不需要恢复完整历史驻留，也不需要重新 prefill。超过声明 `U` 的 selection 会在改变 hot state 前拒绝；不会默默截断，也不会偷偷退回页级分窗。

这是保守的 per-layer credit 实现。水位和 resident window 已成为显式策略，但 window 是配置的页数上限，并非 vLLM 根据 in-flight tokens 算出的调度窗口；HMA group 打包和 scheduler/worker 通信也没有复制，不能直接比较两者 block 数。`reclaim_history()` 默认 unpin；运行中不移动现有 hot block 租约来缩小总 pool。

## 三类索引及完整地址链

1. **选择索引。** Quest min/max 或外部模型产生 logical token positions。它负责决定“读什么”，不拥有缓存空间。
2. **页面地址。** source table 将逻辑页映射到 host backing，resident table 将同一个逻辑页映射到驻留 GPU block 或 -1。
3. **Hot 内容标签。** tags 标识 hot slot 当前保存的 host row；hot block table 决定 slot 在共享 slab 中的物理位置。

```text
t = request-relative logical token
page, offset = t // P, t % P
host_row = source[page] * P + offset

if resident[page] >= 0:
    physical_row = resident[page] * P + offset
else:
    slot = hot lookup(host_row), or an unprotected LRU victim
    physical_row = hot_block_table[slot // P] * P + slot % P
    if miss: copy host_row -> physical_row
```

`LogicalTopK(positions)` 接收设备上的共享 `[K]` 整数 tensor。resolver 在设备上完成去重、无效/未来位置屏蔽、地址翻译、hit 保护、victim 分配、LRU 更新与 miss compaction；无 `.item()`、`.tolist()` 或动态 `nonzero()` 读回。重复位置只在 attention 中出现一次，避免改变 softmax 权重。

Quest 的逐头选择通过 `LogicalTopK.from_selection()` 转换为 union 和 `head_columns`。union 中一个 token 只搬一次，各 head 仍仅读取自己的选择。这个适配仍走 CPU，不能将它描述为端到端无 CPU 往返。Trace 导入也先做 CPU 校验；非法 trace token 拒绝，而直接 GPU 输入的非法/填充位置被屏蔽为 -1。

## GPU 执行和生命周期

`row_index.py` 是固定形状的 PyTorch 设备算子实现，使用 sort/searchsorted/scatter，无平方大小的 token×slot 矩阵。它已通过独立 CUDA Graph capture/replay 测试，但存在多个小 kernel launch；尚未合并成上游的单个 fused resolver。

`row_kernels.py` 查询 PyTorch pinned allocation 的 CUDA device alias，让 Triton gather 直接从 host 加载缺失 rows。它是 **SM 发起的 host memory 访问**，不是 DMA copy engine。没有复制第二份 host pool，也没有接管 PyTorch 内存的注册/注销。

时序为：

```text
compute: append KV -> resolver -> event
transfer: wait append -> D2H new rows -> wait resolver/previous reader -> gather misses -> ready
compute: wait ready -> physical-row attention -> last_use
```

每层同时只允许一个 `RowLease`。为避免把 GPU selection 读回 CPU，租约保守 pin 住该层所有 resident blocks，其他请求/层仍可使用剩余容量。hot tags 可能在实际 gather 前更新，但消费者必须等待 ready；下一次替换必须等待上一读者 last_use。Host 页与索引在请求退休事件完成前不回收，ID 重用不会复用旧 tags。

`close()` / 析构只在 teardown 排空 stream，防止 pinned allocator 在看不到 Triton SM 读者的情况下过早复用 backing。正常 decode、finish/cancel 仍使用事件与延迟回收，不做全设备同步。

## 验证和复现

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m mini_hisparse demo --device cuda:0 \
  --top-k 2 --prompt-tokens 257 --gpu-pages 12
.venv/bin/python scripts/validate_pretrained.py --force-host
.venv/bin/python scripts/compare_layouts.py --selector quest --budget 4 \
  --output results/hybrid-quest.json
.venv/bin/python scripts/compare_layouts.py --selector oracle_token --budget 1 \
  --output results/hybrid-token.json
```

当前 **93 项测试通过**，其中最初 hybrid 版本新增 24 项，此次 residency 对齐再新增 15 项。覆盖 row 粒度搬运、resident bypass、不连续 hot blocks、hit 保护、无效/重复 token、逐头掩码、容量与索引准入、异步旧读者、ID 重用、跨请求/层替换、resolver Graph，以及 readable-unpinned 页面、D2H 完成门槛、低水位/window 触发和两页 tail。CUDA FP16/BF16 与 CPU FP32 均验证。

首版 hybrid 的 Qwen2.5 BF16 强制卸载验证生成 **32 tokens 与 Hugging Face 逐 token 相同**，prefill=1、decode=31，发生 **1,920 row misses / 983,040 B H2D**，保留于 [hybrid-pretrained-parity.json](reports/hybrid-pretrained-parity.json)。当前两页 tail 版本的新验证见 [residency-alignment.md](residency-alignment.md)。这些实验使用完整 selection，检验存储路径，不是稀疏质量结论。

同轨迹、同 GPU block 数的页级/hybrid 对照见 [Quest 结果](reports/hybrid-quest-comparison.json) 与 [token oracle 结果](reports/hybrid-token-comparison.json)。按 token 替换不保证对所有选择分布更优：固定请求 hot region 会占用 resident 容量，Quest 本身选整页，收益与分布和局部性有关。报告保留完整两次运行，计时已预热所有实际 row-kernel 形状，仅为层级 append+attention 回放。

以下为改造 clean page / 两页 tail **之前**的历史数据：A100、BF16、同一 1K LongBench 激活 trace 的 96 个层级事件，重复两次。当前准入下限已经变化，不能用新代码的默认配置原样复现下表：

| 选择 | 相同 GPU blocks | 布局 | H2D bytes | 单层中位数的跨运行中位数 | 最大绝对存储误差 |
|---|---:|---|---:|---:|---:|
| Quest 每头 4 页 | 372 | pages | 1,228,800 | 2.377 ms | 0.001953125 |
| Quest 每头 4 页 | 372 | hybrid | 1,990,656 | 2.724 ms | 0.001953125 |
| token oracle 每头 16 tokens | 192 | pages | 19,095,552 | 8.621 ms | 0.00048828125 |
| token oracle 每头 16 tokens | 192 | hybrid | 1,244,160 | 2.578 ms | 0.0078125 |

第二组 token 分散在很多页面，row 布局的 H2D 减少约 **93.5%**；第一组整页选择反而增加约 **62.0%**，不能宣称统一提升。两组的 selection 不同，不应跨组直接比速度。BF16 输出按容差检验，不承诺张量逐 bit 一致；token oracle 的 0.0078125 也明确保留。精确 greedy token 一致性来自上面的独立生成验证。

额外 resident/hot tables 与 tags/LRU 也占显存：Quest 对照中索引从 774,144 B 增至 850,464 B；token 对照中增至 814,320 B。算子临时 workspace、模型权重与 activation 仍需另计。

## 仍未对齐的部分

- 模型还是 Qwen GQA，Quest 不是 learned DSA。没有 MLA/FP8 后端、原生 DSA 权重或 native trace 的实测质量声明。
- GPU resolver 未 fuse；整个引擎未做 CUDA Graph、batch runner 或多行 speculative decode。`U` 可以描述输入宽度，但不等于实现 speculative 调度。
- 每层独立 hot tags/LRU，没有共享 indexer leader/follower miss-plan 重用与跨层预取。
- Source tables 仍按请求/层组织；没有 vLLM prefix sharing/COW、group 布局、多进程 host pool、TP completion aggregation。
- Resident 页面回收策略、准入信用和生命周期控制面仍是单线程 Python；追求的是可解释的机制对齐，不是上游 ABI 或性能等价。
