# Contributing

Install Python 3.10+, a PyTorch build appropriate for your device, and the development dependencies:

```bash
python -m pip install -e '.[dev,models]'
python -m pytest -q
python -m ruff check mini_hisparse reference experiments scripts tests
python -m ruff format --check mini_hisparse reference experiments scripts tests
python -m build
```

Tests skip CUDA cases when a GPU is unavailable. Installing the `models` extra enables the small randomly initialized Hugging Face compatibility test without downloading pretrained weights. GitHub Actions runs CPU checks; changes to stream dependencies, pinned memory access, row kernels or lifetime handling also need CUDA validation.

Useful smoke checks:

```bash
mini-hisparse demo --device cpu --top-k 2 --prompt-tokens 257 --gpu-pages 12
python scripts/inspect_residency.py --device cpu
# With a CUDA GPU:
python scripts/inspect_residency.py --device cuda:0
python scripts/validate_pretrained.py --force-host
```

Runtime code belongs in `mini_hisparse/`; optional page-based comparisons belong in `reference/` and offline measurement code in `experiments/`. Keep the default import/decode path independent of the latter two packages. The boundary test enforces this through a real subprocess CLI run.

Keep patches small and explain the concrete behavior they change. For cache changes, identify the ownership invariant and the condition under which a block can be reused. Test the relevant failure case against a dense or same-selection oracle; performance measurements should preserve identical selections and exclude warmup/oracle work.

Record hardware, dtype, page size, hot capacity, selection source and timing scope with experimental results. Do not describe layer replay as serving throughput or frozen dense-teacher traces as native DSA quality. The existing reports are historical evidence; write new runs to `results/` and add a clearly identified report when needed.

Do not commit model weights, dataset contents, tensor traces, credentials or `.references/` checkouts. Reference upstream work using pinned revisions and preserve attribution for any future code reuse. The project uses the MIT license in [LICENSE](LICENSE).
