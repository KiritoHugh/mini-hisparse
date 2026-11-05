# 从机制演示到真实文档问答

[文档首页](README.md) · 前置：[图解设计](visual-guide.md) · 下一步：[baseline 测评](benchmarks.md)

以下命令均从仓库根目录运行，假设已激活虚拟环境并 `pip install -e '.[dev]'`。CPU 不需要下载模型；真实问答另需 `pip install -e '.[models]'` 和兼容的 CUDA PyTorch。

## 例子一：观察一个 token 怎样找到 KV

```bash
python examples/inspect_indices.py
# 有 CUDA 时同样的断言在设备上执行
python examples/inspect_indices.py --device cuda:0
```

这是最适合先读的 70 行左右例子：构造四个逻辑页、两个分散 hot blocks，调用真实 GPU/CPU resolver。输出中重点看：

```json
{
  "logical_tokens": [9, 2, 6, 9, 14, -1],
  "physical_rows": [24, 5, 30, -1, 14, -1],
  "counts": {"resident_hits": 2, "hot_hits": 1, "misses": 1}
}
```

对照 [逐项寻址表](visual-guide.md#2-用一组数字走完寻址)。这不是模型生成的 top-k，也不搬运实际 K/V；它把索引含义固定下来，便于调试一个真实 indexer 的输出。

继续跑实际数据搬运与 attention：

```bash
python scripts/inspect_residency.py --device cpu
# 或 --device cuda:0，验证 CUDA 事件和 kernel 路径
```

观察 `clean_pinned → unpinned_but_readable → resident_hit_after_unpin → unused_blocks_consumed_first → block_reused_mapping_revoked → host_row_loaded_after_real_reuse`。Unpin 后 resident mapping 保留；真正复用后才出现 64-byte H2D。这组 snapshot 是诊断工具，会同步设备，不用于性能计时。

## 例子二：三个请求竞争一个小 KV 池

```bash
mini-hisparse demo --device cpu --requests 3 --top-k 2 \
  --prompt-tokens 257 --max-new-tokens 12 --gpu-pages 12 \
  --output results/demo-12.json

mini-hisparse demo --device cpu --requests 3 --top-k 2 \
  --prompt-tokens 257 --max-new-tokens 12 --gpu-pages 36 \
  --output results/demo-36.json
```

这个 tiny model 有两层，page size 16；每层 top-k 2 保留共有的 sink/tail 页，所以 token union 上界为 32，默认 hot rows 为 64。每请求 credit 为 `2 * (64/16 + 2) = 12 blocks`。CLI 为三个 prompt 分别生成 257、264、271 个输入 tokens，全部逻辑 KV 都远大于池容量。

| 本机实际运行 | 12 blocks | 36 blocks |
|---|---:|---:|
| 最大同时 running 请求数 | 1 | 3 |
| 每请求 prefill_count | 1 | 1 |
| 每请求 decode_count | 11 | 11 |
| 每请求生成 tokens | 12 | 12 |
| 持久 GPU KV tensor 容量 | 96 KiB | 288 KiB |

第一枚生成 token 来自 prefill，因此是 1 次 prefill + 11 次 decode。12 blocks 时新请求等待当前请求归还 credit；已准入请求持续生成，不重做 prefill。36 blocks 时多个请求轮流前进，但仍是串行教学调度器，不代表 batch 并行。

结果中的 `timeline` 展示准入过程；末尾 `locations` 可能为空，因为请求完成已清理。`stats/row_stats` 是累计计数，仍能观察 miss 和搬运。换成 `--device cuda:0` 可演示同一存储机制；随机权重输出的 token IDs 没有语言质量含义。

## 例子三：在有限 KV 配额下询问本地文档

[long_context_chat.py](../examples/long_context_chat.py) 展示如何在自己的应用里组合 `MiniModel + HybridKVCache + Engine`。读取本地 UTF-8 文档，套用 Qwen chat template，自动计算 host credit 和 hot/tail 最小配额：

```bash
python -m pip install -e '.[models]'
python examples/long_context_chat.py \
  --context docs/design.md \
  --question 'Why keep two tail pages resident? Answer briefly.' \
  --context-tokens 1024 --max-new-tokens 32 --top-k 4 \
  --device cuda:0 --output results/document-chat.json
```

首次运行下载 `Qwen/Qwen2.5-0.5B-Instruct`；也可用 `--model /path/to/local/checkpoint`。CUDA 使用 BF16，CPU 使用 FP32。上下文超限时保留开头和结尾，报告原始 token 数、保留数与最终 prompt 长度；问题和 chat 模板不会作为 context 被裁掉。它是单文档问答示例，没有检索器、HTTP 服务或多轮会话历史管理。

默认 GPU blocks 取**可证明的选择并集上界**对应的准入下限。Qwen 的 query heads 可能各选不同页，不能直接套用前例 tiny model 的 12 blocks。可显式增大 `--gpu-pages` 观察 resident hit 变化；设得低于 hot/tail 最低承诺会拒绝请求。

输出关注 `answer`、`prefill_count`、`decode_count`、`minimum_device_blocks` 和 `cache_after_finish.row_stats`。[本次运行报告](reports/document-chat.json) 验证真实模型走通存储路径；短回答和机械计数不构成问答质量评测。换一份文档、增加 `--context-tokens` 或调整 `--top-k` 均可能改变答案。

`--top-k 4` 是对 dense Qwen 的近似 Quest 稀疏化。可用 `--top-k all` 观察完整选择，但它需要更大的 hot 配额；本例不会自动承诺稀疏输出与 dense 输出相同。要检查 KV 存储的全选择生成一致性，运行仓库已有的：

```bash
python scripts/validate_pretrained.py --force-host
```

它对固定 prompt 比较 Hugging Face greedy tokens，并主动回收合格历史页。当前记录是 32-token match、一次 prefill，见 [数值验证](validation.md#pretrained-generation)。

## 如何根据输出定位问题？

| 现象 | 先看什么 | 解释 |
|---|---|---|
| 请求一直 waiting | host/index/admission credits | 先区分未准入与已准入 decode 停顿 |
| H2D 不为零但没有重新 prefill | row misses、prefill_count | 这是部分驻留下正常的按需加载 |
| unpin 后没有 H2D | resident mapping / block owner | block 尚未复用，原内容仍可直接读 |
| hot_pending_groups 增加 | 可用 blocks、consumer lease、D2H completion | demand 会保留并重试，不应先撤销有效映射 |
| 减少 gpu-pages 后 OOM | 权重、索引、prefill workspace、host pinning | KV pool 配额不是全部 GPU/RAM 占用 |
| 答案与 Hugging Face 不同 | 是否用了 top-k 稀疏选择 | 先用 full selection 区分选择误差和存储错误 |

要比较速度与搬运量，请转到 [固定 trace 测评](benchmarks.md)。不要给这些逐步打印 snapshot 的例子计时后当作 serving 性能。
