# 实测与验证

本页是 2026-11-04 初版的历史结果。2026-11-05 改进后的真实轨迹、策略对照与当前测试结果见 [第一轮](round1-review.md) 与 [第二轮](round2-review.md)，不要混用两版的索引大小、默认 selector 和逐出策略。

后续默认路径已改为 hybrid block leases / token-row replacement，当时 **93 项测试通过**。Row 管理的历史实测见 [HiSparse 对齐报告](../hybrid-alignment.md)；clean page、hot 触发和两页 tail 的最新验证见 [residency 对齐报告](../residency-alignment.md)。

测试日期：2026-11-04。单卡 **NVIDIA A100-SXM4-40GB**，Python 3.11，PyTorch `2.15.0+cu132`，Triton `3.9.0`，Transformers `5.22.1`。完整 Python 依赖见 [requirements-tested.txt](../../requirements-tested.txt)。

## 正确性

```bash
.venv/bin/python -m pytest -q
# 33 passed
.venv/bin/ruff check mini_hisparse tests scripts
.venv/bin/ruff format --check mini_hisparse tests scripts
```

覆盖：CPU / CUDA、串行 / 双窗口、1/2/5 槽、不断追加和跨页、重复选择去重、尾页屏蔽、hot 命中、动态逻辑预算、host 准入失败回滚、请求结束与 ID 复用、多请求多层随机逐出、FP16/BF16、非 2 次幂页大小、GQA，以及延迟消费者的槽位覆盖竞态。

端到端验证包括：

- 随机小模型多请求生成：2 槽和 64 槽在相同 selection 下 token 一致。
- 完整 attention：逐 token 解码与从完整历史独立计算的 logits/greedy tokens 对照。
- Hugging Face 随机初始化 Qwen2：严格加载同一 state dict，prefill 和三个 decode 步的 logits 在 FP32 容差内一致。
- 保存/重新加载本地 Hugging Face 模型，实际经过 `from_pretrained` 适配器。

有一条 Triton 默认 FP fusion 策略将来变化的上游 warning；当前测试结果没有忽略数值失败。

## 真实预训练权重

```bash
.venv/bin/python scripts/validate_pretrained.py --cache-layout pages
```

`Qwen/Qwen2.5-0.5B-Instruct`，BF16，完整 attention，8 个 GPU 槽：

| 指标 | 实测 |
|---|---:|
| 生成 token 数 | 32 |
| 与 Hugging Face greedy generation 逐 token 相等 | 是 |
| Prefill 次数 | 1 |
| Decode 次数 | 31 |
| GPU KV 主池 | 65,536 bytes（64 KiB） |
| 缺页加载次数 | 3,456 |
| H2D 数据量 | 25,522,176 bytes |

原始结果：[pretrained-parity.json](../reports/pretrained-parity.json)。这是一个提示词上的运行与一致性验证，不是模型质量基准，也不能据此保证所有 BF16 提示词都会有完全相同的 greedy token；浮点归约顺序在临界 logits 上可能改变采样结果。

另外已运行 CLI 的 48-token 生成，`prefill_count=1`、`decode_count=47`，期间 4,680 次缺页，运行记录在本机 `results/qwen-demo.json`。

## KV 微基准

```bash
.venv/bin/python -m mini_hisparse bench \
  --device cuda:0 --tokens 1024 --steps 20 --requests 3 \
  --gpu-pages 8 --selected-pages 8 --chunk-pages 2 \
  --output results/benchmark.json
```

单层、3 个请求，每请求初始 1,024 tokens，追加 20 步，页大小 16，每步选 8 页，2 页一窗口，2 个 KV heads、4 个 query heads、head dim 32、FP32。数据与 selection 对所有策略固定。

| 模式 | GPU KV 主池 | 中位 step | P95 step | 对 CPU oracle 最大绝对误差 |
|---|---:|---:|---:|---:|
| 全部驻留 | 1,622,016 B（198 页） | 3.863 ms | 3.989 ms | 4.17e-7 |
| 8 页，串行缺页加载 | 65,536 B | 5.488 ms | 6.767 ms | 4.17e-7 |
| 8 页，双窗口预取 | 65,536 B | 5.422 ms | 5.510 ms | 4.17e-7 |

原始结果：[benchmark-a100.json](../reports/benchmark-a100.json)。

这组实验说明 KV 主池缩小 **24.75 倍**仍然可以继续执行，结果在浮点容差内一致。它不是总显存缩小 24.75 倍：host pool 为 32 MiB，GPU 页摘要另占 50,688 B，权重、临时张量和 CUDA 运行时均在该账本外。

这里的 step 是三个请求各执行一次“新增 K/V 写入 + 固定选择 attention”。不包含模型投影、MLP、tokenizer 或网络层，所以 JSON 中 `request_steps_per_second` 不能写成 LLM serving tokens/s。JIT warmup 和 oracle 在计时外，测量步边界同步；跨步流水不在本实验范围内。

双窗口相对串行的中位耗时差异约 1.2%，样本小且没有置信区间，**不据此宣称性能提升**。hits 统计也包含追加尾页时的 acquire，不应直接作为纯 attention 命中率。

## Trace 揭示的实际限制

```bash
.venv/bin/python -m mini_hisparse bench --device cuda:0 --steps 5 \
  --gpu-pages 8 --trace results/overlap.json --output results/profiled.json
```

最初把 `[K/V, tokens]` 的非连续尾页切片整体 `copy_`，trace 出现 `Device -> Pageable`：虽然 host 基础 tensor 已 pinned，PyTorch 仍因布局而用了临时 staging。已修复为每个 K/V 平面单独连续复制，完整页仍一次复制。

修复后采集到：

- 30 次 `Device -> Pinned`，对应 15 个 request-step 的 K、V 新行备份。
- 225 次 `Pinned -> Device`，包含 KV 页和 attention metadata。
- 30 次 `Pageable -> Device`，是微基准在计时前准备的新 K/V 输入；不属于 cache 的缺页或写回路径。
- 60 次 Triton `_attention_window`。

**这次小页工作负载没有观测到 pinned H2D 与 attention kernel 的时间区间重叠。** 双 stream/event 的依赖允许重叠，但 CPU 调度和微小 DMA/kernel 发射开销使该样本不能证明硬件 IO 被隐藏。profiled run 仅用于观察拷贝类型与时序，profiler 会扰动计时。

后续性能工作的明确入口是 CPU 页表扫描、metadata 构造、GPU top-k 回传与窗口状态合并的 kernel 发射。把 resolver/metadata 留在 GPU、批量提交搬运和融合归约之后，再重新测量实际 overlap。当前实现交付的是可执行的部分驻留解码与其正确性证据，不把结构上的可并发性包装成已经获得的加速。
