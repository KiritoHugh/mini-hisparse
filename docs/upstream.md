# 上游源码阅读：哪些设计真正影响这个题

阅读日期：2026-11-04。以下链接固定到本次实际拉取的 commit，避免把变化中的 main 当作可复现实验依据。源码保存在本地 `.references/`，不纳入 mini 项目的版本控制，也不作为运行依赖。

当前默认 hybrid 路径具有 GPU row resolver、独立 hot block 租约和 row gather；逐项对照见 [hybrid-alignment.md](hybrid-alignment.md)，clean 回收和两页 tail 见 [residency-alignment.md](residency-alignment.md)。页级分窗是另行保留的参考路径，不能用其 CPU resolver 概括默认实现。

| 项目 | Commit | 本地位置 |
|---|---|---|
| vLLM | `abf98672ce7e822701f7a062bf240efb2c9900d1` | `.references/vllm` |
| SGLang | `a73010b648819b4125458888956ce697a0a6274f` | `.references/sglang` |
| nano-vLLM | `bb823b3e06983d71485a8e1f23715ebd87d98ef8` | `.references/nano-vllm` |

## vLLM：最接近题目里的 Hybrid 语义

阅读入口是 [HiSparse 设计](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/docs/design/hisparse.md)，然后沿以下调用链检查实际代码。

1. [layout.py](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/hisparse/layout.py)：`_lay_out_hisparse_groups` 分离 host source、indexer、resident 与 hot 的语义组；`_build_hisparse_kv_cache_tensors` 通过 offset/stride 绑定共享 GPU 分配。共享存储不意味着共享逻辑页所有权。mini 采用一个同构页池来表达这一点。
2. [coordinator.py](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/hisparse/coordinator.py)：`update_residency` 管驻留策略，`_resident_page_blocks` 排除 active tail，`_plan_page_transfer` 建立搬运事务，`update_spills` 接收 worker 通知。逻辑页的生命周期不能交给 attention kernel 猜测。
3. [worker.py](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py)：`_submit_dma_descriptors` 在 stream 依赖后提交拷贝，保留描述符到 completion event；`take_transfer_updates` 区分 enqueued 与 completed。数据搬运已提交并不代表 CPU 可读取目的内存。
4. [runtime.py](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/hisparse/runtime.py)：`PagedCacheView.bind` 把底层 tensor 映射成可消费视图。`_resolve_residency` 调用 GPU resolver，`_swap_rows` 调用 compact gather；`_resolve_and_stage_group` 在 copy stream 为共享索引的后续层提前搬运，每层等待自己的 ready event。
5. [hisparse_kernels.cu](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/csrc/hisparse_kernels.cu)：`hisparse_resolve_residency_kernel` 优先处理 resident 命中，维护 hot 映射及 LRU，并生成物理索引/缺页计划。多验证行需要保护同一请求的选择集合，避免行间互相逐出。`hisparse_gather_compact_kernel` 消费紧凑搬运计划。

**阅读时发现的一处重要细节：** 该 snapshot 的设计文档描述了 enqueued 后按 stream 顺序释放源租约，而 coordinator 的 `_apply_enqueued_spills` 实际还检查 `worker_completions >= expected_worker_completions`，采用更保守的完成门槛。因此面试时应讲“源槽位复用必须有完成保证或明确的 stream 依赖”，不能仅凭函数名声称收到 enqueue ACK 就总能释放。mini 把逻辑解绑与物理复用 fence 显式分开。

另一个区别：该 snapshot 的 runtime 已有 resolver 与 compact gather 的分工，不能把图里的“fused miss-resolution kernel”理解成所有版本都必须只发射一个 CUDA kernel。

## SGLang：固定 hot 工作集、GPU replacement、共享索引预取

阅读 [HiSparse guide](https://github.com/sgl-project/sglang/blob/a73010b648819b4125458888956ce697a0a6274f/docs/docs/advanced_features/hisparse_guide.mdx) 后，再看这些实现：

- [allocator/hisparse.py](https://github.com/sgl-project/sglang/blob/a73010b648819b4125458888956ce697a0a6274f/python/sglang/srt/mem_cache/allocator/hisparse.py)：`logical_attn_allocator` 与 `hisparse_attn_allocator` 是两个不同容量域，`full_to_hisparse_device_index_mapping` 承接身份翻译。mini 对应稳定 `PageKey/host_slot` 与可变化 `slot`。
- [hisparse_coordinator.py](https://github.com/sgl-project/sglang/blob/a73010b648819b4125458888956ce697a0a6274f/python/sglang/srt/managers/hisparse_coordinator.py)：`admit_request_direct`、`alloc_device_buffer` 和 `map_last_loc_to_buffer` 处理准入及新 token 的物理位置；`_eager_backup_previous_token` 在独立 stream 备份新增数据，并跳过已经在 prefill 阶段备份的首步。其 `record_stream` 和 backup event 是对象生命周期的一部分，不是性能装饰。
- 同文件的 `swap_in_selected_pages`：anchor 层记录 miss plan；共享 index 的 follower 层复用选择并提前做 copy-only replay。只有选择确实共享时，这才是精确预取。mini 的各层独立解析选择；下一窗口预取只存在于 `pages` 参考路径。
- [hisparse.cuh](https://github.com/sgl-project/sglang/blob/a73010b648819b4125458888956ce697a0a6274f/python/sglang/kernels/jit/csrc/kvcacheio/hisparse.cuh)：`load_cache_to_device_buffer_kernel` 对短序列直接返回映射；长序列先构造选择集合的 shared-memory hash，再识别全体 hit 与 evictable slot，压缩 miss 并更新 LRU，最后拷贝 host 行。最新 token 使用额外槽位，避开常规 LRU。**先保护本轮所有 hit，再分配 miss** 是 mini `acquire()` 的直接设计依据。

上游通常为固定选择宽度准备足够大的 hot buffer。mini 默认 hybrid 也要求 hot slots 覆盖单步 token union；“selection 大于整个 GPU 池”的分窗 attention 仅属于 `pages` 参考路径。

## nano-vLLM：借鉴组织方式，改变暂停语义

[block_manager.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py) 展示了清晰的物理块分配、引用计数和序列 block table；[scheduler.py](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py) 的 decode 分支在不能追加时调用 `preempt`，释放块、重置 prefill 状态、回到 waiting。

这正是本题要改变的边界：不把局部 GPU 容量不足变成整个请求失去 KV。mini 中旧逻辑页一直有 host backing，decode 前只解决本层当前窗口的缺页；request 不因 KV GPU 配额不足重新进入 prefill。

## 从上游到 mini 的取舍

| 上游机制 | mini 中保留的关键语义 | 为可读性作的简化 |
|---|---|---|
| Host logical block 与 device lease | 身份不随逐出改变 | 同进程 Python 页表 |
| Resident/hot 共享 HMA 存储 | 同一个 GPU tensor | 固定同构页，跨层共同竞争 |
| 新 KV eager mirror | host 数据最终完整，槽复用有依赖 | 每次追加立即备份新增行 |
| GPU hit/miss/LRU | 本轮 hit 全部保护、物理索引正确 | GPU 多个 PyTorch 算子，resolver 单独通过 Graph capture；整引擎未 capture |
| Shared-index layer prefetch | 每层正确等待自己的 miss gather | 没有跨层计划复用；仅页级参考路径有窗口预取 |
| Sparse backend 消费物理行 | attention 直接读 physical row | GQA Triton row 内核，非 MLA/FP8 |
| P/D disaggregation | prefill 后无需再次计算历史 | 单进程初次 prefill，无 RDMA |

本项目没有复制或 vendor 上游实现。上述仓库仅作设计与代码阅读参考；原仓库保留各自的许可证与归属。

## 2026-11-05 补充：Quest 与真实轨迹

另拉取了 `.references/Quest`，commit `01c1623bf9395009520874e989e29f683203b357`。阅读其 `evaluation/quest_attention.py` 的符号变换、页内极值与每头选择后，实现了独立的 min/max 索引和精确 token mask。该上游示例对前两层保留完整 attention；mini 的采集默认从层 2 开始，但实时 `--selector quest` 仍是简化适配，不能据此复制论文质量结论。详细的数据来源、格式和差异见 [traces.md](traces.md)。
