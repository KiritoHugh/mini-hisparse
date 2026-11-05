# Clean page 回收、hot 触发与两页 tail

本轮将默认 `hybrid` 路径的三个管理行为进一步向本地 vLLM 快照 `abf98672ce7e822701f7a062bf240efb2c9900d1` 对齐。核心是：**解除 allocation pin 后仍可读，实际复用才失去 resident 映射；提前申请 hot；保护最后两页。** `pages` 参考路径保持原来的页级机制。

## 改动前后

| 行为 | 上一版 mini | 当前 mini | vLLM 快照 |
|---|---|---|---|
| Clean 历史页回收 | detach 后直接撤销 resident 映射 | 先进入 readable-unpinned 队列；实际复用回调才 detach | unpin 后加入 free queue 尾部；`on_reuse` 才撤销映射 |
| Host 可用性 | 通过搬运事件保护实际复用 | 显式区分镜像已提交与已完成；完成后才允许 unpin | pending spill 完成并发布有效 host copy 后才 unpin |
| Hot 申请 | 主要由空间不足或显式 transition 触发 | 低水位或 resident window 触发，未满足的需求持续保留 | 低水位或 admission window 触发；后续调度分配 hot |
| Tail 保护 | 每层最后 1 页 | 每层最后 2 页 | `ACTIVE_TAIL_PAGES = 2` |

源码入口：mini 的 [allocator.py](../mini_hisparse/allocator.py)、[hybrid.py](../mini_hisparse/hybrid.py)，以及固定版本上游的 [coordinator.py](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/hisparse/coordinator.py)、[block_pool.py](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/core/block_pool.py)、[types.py](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/hisparse/types.py)。

## Clean page：容量可分配，内容仍可读取

```text
dirty resident
    │ 对应版本的 D2H 完成事件已完成
    ▼
clean pinned
    │ hot region 已具备 + sealed + 不属于最后两页 + 无消费租约
    ▼
clean unpinned（resident 映射、owner、generation 和 KV 内容仍有效）
    │ allocator 实际选中此 block，调用 on_reuse
    ▼
host only（撤销旧 resident 映射；新 owner 绑定时增加 generation）
```

`host_enqueued_version` 不能代替 `host_committed_version`。`pending_clean` 记录待发布完成状态的页面，常规 `update_residency()` 只用 `event.query()` 查询；未完成的拷贝不会被提前标记 clean。

mini 将容量分成 `allocator.free` 和有序的 `allocator.reclaimable`。前者没有 owner，后者保留旧 owner 和映射。分配优先使用前者，再从后者回收；解除 pin 的候选按逻辑页号从大到小加入队列，保护较早的前缀。上游使用同一 free queue，将新解除 pin 的 blocks 放在尾部。mini 的两个容器复现了“最后才复用仍有内容的 block”的意图，但不是同一个队列实现。

读取 readable-unpinned 页仍走 resident bypass，不会产生 host→hot miss。临时 `RowLease` 可再次保护这种页面；allocator 排除正在被消费的 blocks，租约结束后的事件继续保护尚未执行完的 GPU 读取。`on_reuse` 在覆盖前撤销 resident table，保留 host/source table；下一次选中旧 token 时才由 row resolver 命中 hot 或加载 host row。

监控分别报告 `empty_device_blocks`、`readable_unpinned_blocks` 和 `allocatable_device_blocks`。没有消费租约时第三项为前两项之和；有租约时扣除不可立即复用的 readable blocks。不能仅用 `owner is None` 判断系统还有多少可分配容量。

`reclaim_history()` 默认执行上述 unpin 行为。`evict=True` 是验证 host 路径的显式诊断开关，会主动撤销 eligible 页的映射，不属于日常回收策略。

## Hot：持久需求、提前触发、完整分配

`LayerTables.hot_required` 记录申请需求，`trigger` 保留首次触发原因。正常策略满足任意条件即提出申请：

- 可分配 blocks 低于水位。默认水位为 `max(本层 hot_blocks, GPU blocks // 10)`；即将分配 append block 时按 `available - 1` 预判。
- 当前层的 resident pages 达到 `resident_window_pages`，默认 64，可通过 `--resident-window-pages` 配置。

分配成功后固定持有整组 hot blocks，再让已完成镜像的历史页解除 pin。若层正在被读取或容量不足，需求保留，下一次 policy tick 重试；不能因申请失败而先清空 resident table。`--hot-watermark` 可覆盖默认水位，设为 0 只关闭常规低水位触发，window 和实际容量压力仍生效。

这里仍有明确的调度差异：

- vLLM 的水位使用所有 hot managers 的 request 成本总和；mini 的分配单位是请求/层，使用本层 hot 成本。
- vLLM 的 resident admission window 来自 `ceil(min(max_in_flight_tokens, max_model_len) / block_size)`，见 [HiSparseResidentSpec](https://github.com/vllm-project/vllm/blob/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/kv_cache_interface.py)。mini 采用显式页数配置，没有 chunked-prefill 调度器。
- vLLM 先标记 demand，在后续 scheduling pass 分配。mini 可在同一次调用完成；空闲容量不足时还允许用本层已完成镜像的历史页组成一次完整的 transition。计划先检查整组容量，之后才修改映射，期间没有并发消费者。
- mini 的同步 prefill runner 无法在层内挂起并交还调度器。如果容量耗尽，且能释放空间的历史页只差 D2H 完成，会等待一个相关页的完成事件再重试，计入 `pressure_mirror_waits`。常规 policy tick 不等待，压力路径也不调用全设备同步；这仍不是生产调度器的完全异步行为。

## 两页 tail 与准入

`ACTIVE_TAIL_PAGES = 2`。当前最后一页为 `t` 时，只有 `page <= t - 2` 的 sealed 历史页才可回收。保留的是最后两个逻辑页：当前 append 页和它前面一页；即使当前页刚好写满，也不会立即把前一页释放。进入新页后，保护范围随之移动；不足两页时全部保留。

每请求的最小容量信用统一由 `HotConfig.admission_blocks()` 计算：

```text
layers × (ceil(hot_rows / page_size) + 2)
```

CLI、trace replay、layout comparison 和真实模型验证脚本均使用这个函数，避免运行脚本仍按一页 tail 计算容量。两页 tail 目前只对应本项目单卡、逐层执行的模型；没有声称复现 vLLM 的 HMA group 打包和全部 admission 逻辑。

## 实测与复现

**93 项测试通过**，本轮新增 15 个 CPU/CUDA 用例，见 [test_residency.py](../tests/test_residency.py)。覆盖 unpin 后仍可读、实际复用与 generation、消费租约、真实异步 D2H 完成门槛、watermark/window 触发、忙碌时保留 demand、两页 tail 边界以及容量耗尽时的针对性事件等待。

[生命周期报告](reports/residency-lifecycle.json) 来自 A100 上运行 [inspect_residency.py](../scripts/inspect_residency.py)。观察请求 a 的第 0 页，GPU pool 共 7 blocks：

| 阶段 | 原 block owner / generation | 真空闲 blocks | 可读且已解除 pin | 累计 resident hits / host row misses |
|---|---|---:|---:|---:|
| Clean pinned | a:page0 / 1 | 4 | 0 | 0 / 0 |
| 分配 hot，解除历史页 pin | a:page0 / 1 | 3 | 1 | 0 / 0 |
| 再读旧 token | a:page0 / 1 | 3 | 1 | 1 / 0 |
| b 消耗剩余空闲 blocks | a:page0 / 1 | 0 | 1 | 1 / 0 |
| b 再分配，实际复用该 block | b:page3 / 2 | 0 | 0 | 1 / 0 |
| 再读 a 的旧 token | b:page3 / 2 | 0 | 0 | 1 / 1 |

复用前后 attention 输出均与 dense oracle 一致；最后一步只搬运一个 64 B 的 KV row。脚本为展示状态在采样点同步，不能用它的运行时间代表 decode 性能。

[真实模型验证](reports/residency-pretrained-parity.json)：Qwen2.5-0.5B-Instruct、BF16、full selection，诊断性撤销 clean 历史页后，32 个生成 token 与 Hugging Face 逐 token 相同，prefill=1、decode=31。共发生 1,536 row misses，786,432 B H2D。该实验验证存储正确性，不证明稀疏选择质量。

[真实 trace 回放](reports/residency-replay.json)：冻结的 LongBench 1K Qwen 激活轨迹，Quest 每头 4 页、96 个层级事件，在 384 GPU blocks 下通过数值校验，最大绝对存储误差 0.001953125；累计 3,696 row misses、1,892,352 B H2D。报告的 2.847 ms 中位数只包含串行层级 append+attention，不是服务吞吐。历史一页 tail 报告仍保留于 [hybrid-alignment.md](hybrid-alignment.md)，配置容量不同，不能据此宣称本轮加速。

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/inspect_residency.py --device cuda:0
.venv/bin/python -m mini_hisparse demo --device cuda:0 \
  --top-k 2 --prompt-tokens 257 --gpu-pages 12
.venv/bin/python scripts/validate_pretrained.py --force-host
# 需要先按 docs/traces.md 采集默认真实 trace
.venv/bin/python scripts/replay_trace.py --output results/residency-replay.json
```

GPU 多算子 row resolver、每层独立 hot tags/LRU、单进程控制面等边界未变；当前没有上游 fused resolver、共享 indexer 的跨层 miss-plan 复用、prefix sharing/COW 或多卡完成聚合。
