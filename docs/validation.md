# Current validation

These results describe the default hybrid implementation with readable-unpinned clean pages and **two active tail pages**. Earlier page-based and one-tail experiments are archived separately; their capacity and timing numbers are not interchangeable.

For current all-resident/page/hybrid comparisons, plots, timing methodology and reproduction commands, see [baseline benchmarks](benchmarks.md). This page focuses on numerical correctness and lifetime checks.

## Reproducible checks

```bash
python -m pip install -e '.[dev,models]'
python -m pytest -q
python -m ruff check mini_hisparse reference experiments scripts examples tests
python -m ruff format --check mini_hisparse reference experiments scripts examples tests
python -m build
python scripts/inspect_residency.py --device cpu
python examples/inspect_indices.py --device cpu
```

The recorded GPU suite has **96 passing tests**. Coverage includes CPU/CUDA output agreement, GQA and per-head selection, duplicate/invalid token handling, hot LRU protection, fragmented block leases, clean-page completion and reuse, two-tail boundaries, admission failure, request cancellation/ID reuse, delayed readers and standalone resolver CUDA Graph capture.

The interview review adds CPU/CUDA cases for `H=U=4` with only three GPU blocks: one hot block and two tails. After a 32-token prefill, 16 consecutive appends alternate disjoint selections of four host-only rows. All 64 selected rows miss, every attention output matches its same-selection oracle, and the two tails remain resident across page boundaries. See `test_minimum_hot_capacity_progresses_when_every_selected_row_misses` in [test_residency.py](../tests/test_residency.py). This is a progress/correctness test, not a latency measurement.

The earlier refactor validation with CUDA hidden passed **53 CPU tests**, with **41 CUDA cases skipped**, before the two new cases above. Wheel and source distributions built successfully; the installed wheel also ran the three-request CPU demo from outside the source tree.

CPU-only environments skip CUDA cases. The optional Hugging Face test uses a tiny randomly initialized Qwen2 model and a temporary local checkpoint; it does not download pretrained weights. CI installs the `models` extra and runs with the Hugging Face hub offline. CPU CI is not evidence of CUDA kernel correctness.

Recorded GPU environment: NVIDIA A100-SXM4-40GB, Python 3.11, PyTorch `2.15.0+cu132`, Triton `3.9.0`, Transformers `5.22.1`. [requirements-tested.txt](../requirements-tested.txt) preserves that environment's dependency snapshot; it is not a portable install requirement. One upstream Triton FP-fusion warning was present without numerical test failures.

An additional subprocess test blocks imports of `reference` and `experiments` and still runs the default CLI through prefill and decode. The original numerical/lifetime cases remain in the suite. [Refactor details](refactor.md).

## Clean-page lifecycle

```bash
python scripts/inspect_residency.py --device cuda:0
```

[Raw report](reports/residency-lifecycle.json). A seven-block pool demonstrates that unpinning preserves resident hits, empty blocks are used first, and the original resident mapping is revoked only when another request actually reuses the block. The old token then loads one 64-byte KV row from host. Both attention reads match the dense oracle.

The inspection script synchronizes at diagnostic snapshots; its execution time is not a decode benchmark.

## Pretrained generation

```bash
python scripts/validate_pretrained.py --force-host
```

[Raw report](reports/refactor-pretrained-parity.json). Qwen2.5-0.5B-Instruct, BF16, full selection, diagnostic eviction of eligible clean history:

| Measurement | Recorded result |
|---|---:|
| Generated tokens | 32 |
| Exact greedy token match with Hugging Face | Yes |
| Prefill / decode count | 1 / 31 |
| Device blocks | 312 |
| Row misses | 1,536 |
| H2D bytes | 786,432 |

This checks the storage path on one prompt. It is not a sparse-quality benchmark or a guarantee of identical greedy tokens for every BF16 prompt and reduction order.

## Frozen real-model trace

```bash
python scripts/prepare_longbench.py
python scripts/model_trace.py capture --max-context 1024 --steps 8
python scripts/model_trace.py analyze
python scripts/replay_trace.py --output results/residency-replay.json
```

[Raw report](reports/refactor-replay.json). Three LongBench samples, four Qwen layers, eight steps per sample; Quest-style selection with four pages per head:

| Measurement | Recorded result |
|---|---:|
| Layer events | 96 |
| Device blocks | 384 |
| Maximum absolute storage error, BF16 | 0.001953125 |
| Row misses | 3,696 |
| H2D bytes | 1,892,352 |
| Median serialized layer append + attention | 2.786 ms |

Selection and oracle preparation are outside the timing scope. This is not model serving throughput. Inputs are frozen dense-teacher activations; the experiment does not measure sparse generation feedback, learned DSA selection or LongBench task scores. See [trace provenance and limitations](traces.md).

## Historical experiments

The [initial validation](history/initial-validation.md), [round 1 review](history/round1-review.md), [round 2 review](history/round2-review.md) and [early hybrid comparison](hybrid-alignment.md) retain earlier measurements, including unfavorable policy results. The early hybrid comparison used one tail page and different admission minima. No speedup claim is made by comparing those reports with the current two-tail defaults.
