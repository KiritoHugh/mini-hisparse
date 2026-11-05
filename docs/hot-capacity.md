# Hot 应该预留多少：固定总容量的敏感性实验

[面试三问与评估](interview.md) · [整体 baseline](benchmarks.md) · [图解分配](visual-guide.md#5-hot-空间先做准入承诺再在压力下分配)

**`H>=U` 解决单步能否容纳 selection；`H=2U` 不是必需，也不是性能最优保证。** 本实验补上默认值的取舍证据：当总 GPU KV pool 不变时，扩大 hot 会挤占 resident history，不能只看 hot hit 数。

## 实验控制

复用 [baseline](benchmarks.md) 的同一份 LongBench / Qwen2.5 frozen trace，SHA256 为 `a18b1c1df245293fbd50a048af659515f7f021a292d179e5a0c6ed06a875b527`。单张 A100-SXM4-40GB，BF16，runtime `17ef5a723e52b6aebfae9f2614d9af4309b824e7`；每配置 96 个 layer events，五次独立 replay，轮换配置顺序。

每组冻结相同 selection/oracle，固定总 KV blocks、resident window 64 和水位，只改变每请求/层的 `H=ceil(multiplier*U)`。`U` 是 trace 的已知最大跨头 token union；这是离线配额实验，线上必须有可证明的上界或受控的选择预算。

| 场景 | 每头选择 | U | 固定 GPU blocks | 固定水位 |
|---|---|---:|---:|---:|
| Quest | 4 页 | 239 | 384 | 38 |
| 分散 token | 离线 oracle 16 tokens | 116 | 204 | 20 |

三个请求 × 四个采样层，共 12 个 request/layer groups。所有 groups 的 admission 为 `12*(ceil(H/16)+2)`。水位固定为该组 `2U` 配置的默认值，避免把触发时机改变与 hot 大小改变混在一起。

## 结果 A：Quest 整页选择

| Hot 比例 | 每层 rows / blocks | 最低 admission blocks | 信用扣除后余量 | Resident / hot hits | Misses | H2D MiB | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1U | 239 / 15 | 204 | 180 | 6,528 / 5,599 | 2,721 | 1.329 | 2.844 |
| 1.5U | 359 / 23 | 300 | 84 | 4,496 / 7,280 | 3,072 | 1.500 | 2.806 |
| 2U | 478 / 30 | 384 | 0 | 2,320 / 8,832 | 3,696 | 1.805 | 2.781 |

**观察：** 1U 相比 2U 少搬约 26.4% 的 H2D payload，但层级 median 并未更快。Hot 变大后 hot hits 增多，resident hits 同时减少，总 misses 反而上升。因此“hot 命中率更高”不能单独证明分配更好。

五次 layer median 的范围分别为 `2.835–2.880 / 2.801–2.810 / 2.775–2.784 ms`；最大存储 abs error 均为 `0.001953125`。[完整逐次报告](reports/hot-sweep-quest.json)。

## 结果 B：离线 oracle 分散 token

| Hot 比例 | 每层 rows / blocks | 最低 admission blocks | 信用扣除后余量 | Resident / hot hits | Misses | H2D MiB | Median ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1U | 116 / 8 | 120 | 84 | 2,337 / 2,414 | 1,995 | 0.974 | 2.715 |
| 1.5U | 174 / 11 | 156 | 48 | 2,099 / 2,659 | 1,988 | 0.971 | 2.682 |
| 2U | 232 / 15 | 204 | 0 | 1,514 / 3,007 | 2,225 | 1.086 | 2.664 |

**观察：** 1.5U 的 payload 最小，2U 的本机 layer median 最低。二者目标不同，没有一个比率同时支配所有指标；也不能将这个小样本用于自动改变全局默认值。

五次 layer median 的范围分别为 `2.708–2.723 / 2.676–2.689 / 2.657–2.671 ms`；最大存储 abs error 均为 `0.0078125`，在 replay 的 BF16 `atol=rtol=0.008` 内。[完整逐次报告](reports/hot-sweep-token.json)。

## 放不下的配置也要报告

| 场景 | 3U hot rows/layer | 所需 admission blocks | 实际 KV pool | 结果 |
|---|---:|---:|---:|---|
| Quest | 717 | 564 | 384 | 配额预检不通过，未执行 replay |
| 分散 token | 348 | 288 | 204 | 配额预检不通过，未执行 replay |

这些行是容量公式的预检结果，没有伪造“失败运行”的 timing。表中“信用扣除后余量”是 `GPU blocks - admission blocks`，不是此时实际 free blocks，也不是 resident page 数。Hot 尚未兑现时历史页可借用信用；兑现后是否存在额外驻留取决于整个分配过程。

## 怎样据此选配置？

1. 根据模型 indexer/selector 确定可支持的最大 union U；不能把 per-head K 当作 union。
2. 先检查 `H>=U` 和所有请求的 hot/tail、host/index 配额。对不能准入的配置先调整并发数、选择预算或资源。
3. 固定总容量，在代表性轨迹上比较 `1U/1.5U/2U` 的 **resident hits + hot hits、miss bytes、时延与准入能力**，保留不同目标之间的取舍。
4. 若再做在线自适应，需要定义变更租约时机、消费者 fence、metadata 重建、收益/成本和防振荡规则；当前项目没有自动扩缩 hot lease。

本实验只能说明空间分配确实有代价；无法从总耗时反推出 resolver、Python 驻留 bookkeeping、SM gather 各自的比例，也没有测到跨层 overlap。相比减少少量 H2D，固定算子与调度成本也可能主导短层级回放，原因需 profiler 拆分验证。

## 复现

先按 [baseline 采集步骤](benchmarks.md#如何复现) 准备 `results/qwen-longbench.pt`，再运行：

```bash
python scripts/sweep_hot.py --selector quest --budget 4 --gpu-pages 384 \
  --multipliers 1 1.5 2 3 --repeats 5 --output results/hot-sweep-quest.json
python scripts/sweep_hot.py --selector oracle_token --budget 1 --gpu-pages 204 \
  --multipliers 1 1.5 2 3 --repeats 5 --output results/hot-sweep-token.json
```

[脚本](../scripts/sweep_hot.py) 复用既有 `prepare/replay`，没有另写缓存或 attention 实现。计时/预热口径与 layout comparison 一致：prefill、选择器、oracle 和 JIT 在计时外；计时内含串行 append、解析/搬运和 attention，并在事件间同步。

报告保存 runtime commit、脚本/trace hash、命令、五次 run 的完整 counters 和数值误差。两组实验的计数在本次五次运行中相同，时间是 run medians 的 median；统计窗口是 96 events。它们是 frozen dense-teacher Qwen 激活，不是 learned DSA、稀疏闭环生成质量或服务吞吐评测。
