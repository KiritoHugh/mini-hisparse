# 第二轮：容量、内容与生命周期（2026-11-05）

第二轮从第一轮的真实轨迹出发。面试官会追问“hot buffer 怎么获得空间、槽位复用后如何防止旧引用读错、请求取消时谁还持有 host 数据”；系统评审则需要检查真实命中率、append 代价和完整显存账本，不能只看 pool 大小。

## 发现与改进

| 发现 | 实际改进 | 验证方式 |
|---|---|---|
| hot-first 逐出会保留冷 resident、丢掉最近使用的 hot 内容 | 独立 `DeviceAllocator`，提供 legacy / LRU / 带软配额的 LRU；运行中的 hot 内容可以跨步保留 | 相同真实 trace、3 种容量、3 种策略对照 |
| 仅按 top-k 数字划 hot 容量，忽略 head 并集及其他消费者 | `WindowPlan` 计算实际 union、已有 lease、一个/两个窗口的最大在途槽数 | 外部 lease、单槽、工作集大于池的数值测试 |
| 请求之间没有容量公平性，硬划分又会浪费空闲槽 | 请求占用计数和可借用的软目标，超额请求优先回收 | 两请求超额回收、单请求借用整个池测试 |
| 未来输出一开始就创建全部 Page 对象 | 准入预留 host credit，首次写入才 materialize 页/映射 | credit 守恒与原子准入失败测试 |
| 槽号/请求 ID 重用缺少显式版本 | slot 与 request generation，lease 验证 owner/generation；写入和 host 镜像分别记版本 | CUDA 延迟读者、ABA、尾页版本测试 |
| 请求结束会同步整个执行流 | 延迟回收队列，GPU 槽带 fence 复用，host backing 等事件完成再归还 | 取消、延迟 D2H/H2D、host 不被提前复用 |
| 索引空间漏出 admission 账本 | 独立 index budget；BF16 min/max 用原 dtype，mean 仍 FP32 | 预算拒绝、真实字节核对、极值精确性测试 |
| append miss 与 attention miss 混在一起 | 分开记录用途、命中数与 H2D bytes | 统计直接暴露尾页抖动 |

这些改动保留一个 GPU KV tensor。hot buffer 没有第二份内存，也没有永久不可回收的区域。只有消费者持有的 lease 是硬约束，其他保留策略都能在压力下退让。

## 本轮内的一次失败与修正

最初的 LRU/软配额版本在 128/256 页时改善，但 32 页时 H2D 从 legacy 的 6,817,792 B 增加到 7,544,832 B。拆分统计发现 append miss 从 11 次变成 84 次：极小缓存容不下循环工作集时，刚使用过的尾页在下一次追加前又被逐出。

这不是“LRU 一定优于旧策略”。加入以下软保护后再测：

- 未封口尾页，避免频繁把下一步追加所需的内容重新载入。
- 最近两轮层级 attention 中实际使用的 sink 页；不用的 sink 会过期。
- 尚未消费的本轮 hit 尽量保留，但不把整个 selection pin 住。

完整失败记录保留在 [round2-policy-first-attempt.json](../reports/round2-policy-first-attempt.json)，没有只留下有利结果。

## 固定 1K 模型轨迹的策略对照

A100，BF16，LongBench 三个样本，4 个采样层 × 8 步，共 96 个层级事件；Quest 每头 4 页，页大小 16。每种配置独立初始化、预热 masked kernel 的全部窗口大小，重复两次。selection 和 oracle 都在计时外。

| GPU 页数 | 策略 | Attention hit | H2D bytes | 单层步骤中位数的跨运行中位数 |
|---:|---|---:|---:|---:|
| 32 | legacy | 15.31% | 6,817,792 | 2.923 ms |
| 32 | 普通 LRU | 9.87% | 7,553,024 | 3.002 ms |
| 32 | 默认 fair LRU + 软保护 | 18.50% | 6,496,256 | 2.847 ms |
| 128 | legacy | 26.93% | 5,885,952 | 3.277 ms |
| 128 | 普通 LRU | 58.48% | 3,423,232 | 2.503 ms |
| 128 | 默认 fair LRU + 软保护 | 64.34% | 2,842,624 | 2.472 ms |
| 256 | legacy | 42.65% | 4,620,288 | 3.647 ms |
| 256 | 普通 LRU | 80.58% | 1,603,584 | 2.269 ms |
| 256 | 默认 fair LRU + 软保护 | 79.55% | 1,630,208 | 2.289 ms |

默认策略相对 legacy 的 H2D 分别减少 **4.7% / 51.7% / 64.7%**。相同 selection 的最大 BF16 存储误差均为 **0.001953125**。256 页场景普通 LRU 略好于默认策略，体现了软公平/anchor 策略的取舍，并非所有预算下唯一最优。

结果：[round2-policy-comparison.json](../reports/round2-policy-comparison.json)。延迟是串行层级回放，不是服务吞吐；两个运行没有统计置信区间。legacy 延迟随容量变大而上升，也受 Python 候选扫描和策略开销影响，不能把延迟变化全归因于 PCIe。

## 更长文本的验证

第二组把 context 上限提高到 4K，同一源样本实际 prompt 长度为 **2,769 / 4,060 / 4,060**。qasper 保留完整 context，其他两类仍截断。每头 16 页（最多 256 tokens）：

| selector | 平均保留 attention mass | 平均相对输出 L2 | 平均页面并集 |
|---|---:|---:|---:|
| mean | 79.28% | 0.2212 | 16.00 |
| Quest 风格 | 79.89% | 0.2015 | 58.07 |
| token oracle | 93.58% | 0.0713 | 184.71 |

Quest 选择即使只有每头 16 页，实际也需要约 58 页的 union。用 **32 槽**重放全部 96 个事件仍成功，BF16 最大存储误差仍为 0.001953125，H2D 为 44,253,184 B。层级中位耗时 16.408 ms，反映了多窗口与 Python metadata 的明显成本。

索引在这次回放占 **2,801,664 B**，大于 262,144 B 的 KV 主池，host pool 也单独存在。索引受预算约束不代表它免费，更不能只用 `gpu_pages` 宣称总 HBM 上界。

原始结果：[round2-trace-4k.json](../reports/round2-trace-4k.json)、[round2-replay-4k.json](../reports/round2-replay-4k.json)。oracle 使用相同最大 token 预算，尾页与强制 sink/tail 使页选择的实际 token 数可能更少；原始报告保留了实际 token fraction。

## 验证结果

- **54 项测试通过**，包含 CPU、CUDA、Triton、精确 token mask、GQA、单槽循环、软配额、生命周期和异步复用。
- 改造后的真实 Qwen2.5-0.5B BF16 完整 attention 生成 32 tokens，仍与 Hugging Face greedy 输出逐 token 相等，prefill=1、decode=31：[round2-pretrained-parity.json](../reports/round2-pretrained-parity.json)。
- Ruff 检查、格式检查、shell 语法与 diff 空白检查通过。

复现：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/compare_policies.py --capacities 32 128 256 --repeats 2
.venv/bin/python scripts/model_trace.py capture --max-context 4096 --steps 8 \
  --trace results/qwen-longbench-4k.pt
.venv/bin/python scripts/model_trace.py analyze --trace results/qwen-longbench-4k.pt --budget 16
.venv/bin/python scripts/replay_trace.py --cache-layout pages --trace results/qwen-longbench-4k.pt \
  --budget 16 --gpu-pages 32
.venv/bin/python scripts/validate_pretrained.py --cache-layout pages
```

## 两轮后仍需明确的不足

1. **尚非原生 DSA/MLA 引擎。** 已有原生 logical top-k 导入口，数值后端仍要求 GQA 布局；没有实际采集 DeepSeek learned indexer，也没有 MLA/FP8 kernel。
2. **质量实验仍是冻结 teacher 轨迹。** 它能分离存储错误和选择错误，但不等于闭环 sparse generation、完整 LongBench 评分或长期误差积累评估。小样本、四层、八步不能代表所有上下文和模型。
3. **控制面仍在 CPU。** 页表、选择回传、mask/metadata 构造与多个归约 kernel 都有成本，尚无 GPU fused resolver、CUDA Graph 或真正 batch runner。没有把预取结构本身当作实际 overlap 的证明。
4. **存储功能仍有边界。** host 按生命周期上界保守预留；没有 prefix sharing/COW、RDMA、共享多进程 host pool 或多 compute stream 的聚合读者事件。

作为面试项目，目前能够用代码、反例和实测回答“部分卸载后如何继续 decode”以及具体策略的代价；还不适合宣称达到 vLLM/SGLang 的生产能力或吞吐水平。
