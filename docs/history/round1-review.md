# 第一轮：真实度与可证伪性（2026-11-05）

面试官视角：原来的数值测试能证明缓存基本正确，但轮转 top-k 与文本无关，无法据此讨论真实 cache hit、选择质量或生产显存收益。

推理系统视角的主要缺口：

| 问题 | 严重性 | 本轮改动 |
|---|---|---|
| 随机 Q/K/V、固定轮转页无法代表模型访问 | 高 | 固定官方 LongBench revision，采集预训练 Qwen2.5 的真实 Q/K/V 和 decode 历史 |
| 把页面搬运粒度当成 attention 语义 | 高 | 精确 token selection、每个 query head 独立 mask、`-1` padding 与去重校验 |
| 页均值会抵消页内关键 token | 高 | Quest 风格 min/max 上界索引；保留 mean 作为对照 |
| 只报缓存误差，没有稀疏化误差 | 高 | attention mass、oracle token recall、输出相对 L2、时间 Jaccard |
| 单个 top-k 数字不能描述 GQA 工作集 | 高 | 分别报告每头预算和页面并集放大率 |
| 轨迹没有来源、版本与截断记录 | 中 | dataset/model revision、样本行、context/prompt hash、截断策略 |

本次采集 Qwen2.5-0.5B-Instruct 在 qasper、gov_report、repobench-p 各一条样本上的密集 teacher trajectory，取层 2/7/15/23，每条 8 步，共 96 个事件。context 截断到约 1K tokens。输入是实际数据集文本；这是小样本、截断、冻结轨迹的系统实验，不是完整 LongBench 任务评测，也不是 DeepSeek DSA 的原生轨迹。

页大小 16，每头 4 页预算，统计均值：

| selector | attention mass | 输出相对 L2 | 页面并集 | token 时间 Jaccard |
|---|---:|---:|---:|---:|
| 旧 shared mean | 0.6763 | 0.4001 | 4.00 | 0.7716 |
| Quest 风格，每头选择 | 0.6720 | 0.3924 | 10.14 | 0.7160 |
| 精确 QK token oracle | 0.8877 | 0.1154 | 51.35 | 0.4956 |

不能从表中得出 Quest 一定更好。预算很小、token/head 分布和 GQA 页面并集都影响结果。token oracle 需要读取完整 K，仅作为离线对照；不能拿其 top-k 当成免费 indexer。完整结果见 [round1-trace-analysis.json](../reports/round1-trace-analysis.json)。

真实 Quest 选择在 32 页 GPU 池中的回放：96 个事件全部通过相同 selection 的数值对照，BF16 最大存储误差 0.001953125；旧策略发生 835 次 miss，H2D 6,817,792 bytes，单层步骤中位数约 2.786 ms。见 [round1-replay.json](../reports/round1-replay.json)。时延是层级回放，不是模型吞吐。

第一轮验证：41 项 CPU/CUDA 测试通过，包含不同 head 在一个窗口没有选中 token 的 softmax 边界，精确 token mask、真实模型适配、轨迹校验及回放。

复现：

```bash
.venv/bin/python scripts/prepare_longbench.py
.venv/bin/python scripts/model_trace.py capture --max-context 1024 --steps 8
.venv/bin/python scripts/model_trace.py analyze --output results/trace-analysis.json
.venv/bin/python scripts/replay_trace.py --cache-layout pages --gpu-pages 32
```

参考：[Quest 原论文](https://proceedings.mlr.press/v235/tang24l.html)、[Quest 实现，固定 commit](https://github.com/mit-han-lab/Quest/blob/01c1623bf9395009520874e989e29f683203b357/evaluation/quest_attention.py)、[官方 LongBench 固定 revision](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672)。本实现保留 sink/tail 页并适配 GQA，不宣称是完整 Quest 运行时复刻。

第二轮需要解决的实际问题：旧策略把冷 resident 长期保留下来，却优先逐出刚使用的 hot；未使用的未来 host 页也建立了对象；请求结束会同步全部 stream；索引内存没有准入上限；本轮页面并集没有参与显式容量规划。
