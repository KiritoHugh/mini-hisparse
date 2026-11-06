# Contributing

Install Python 3.10+, a PyTorch build appropriate for your device, and the development dependencies:

```bash
python -m pip install -e '.[dev,models]'
python -m pytest -q
python -m ruff check mini_hisparse reference experiments scripts examples tests
python -m ruff format --check mini_hisparse reference experiments scripts examples tests
python -m build
```

Tests skip CUDA cases when a GPU is unavailable. Installing the `models` extra enables the small randomly initialized Hugging Face compatibility test without downloading pretrained weights. GitHub Actions runs CPU checks; changes to stream dependencies, pinned memory access, row kernels or lifetime handling also need CUDA validation.

Useful smoke checks:

```bash
mini-hisparse demo --device cpu --top-k 2 --prompt-tokens 257 --gpu-pages 12
python scripts/inspect_residency.py --device cpu
python examples/inspect_indices.py --device cpu
# With a CUDA GPU:
python scripts/inspect_residency.py --device cuda:0
python scripts/validate_pretrained.py --force-host
```

Runtime code belongs in `mini_hisparse/`; optional page-based comparisons belong in `reference/` and offline measurement code in `experiments/`. Keep the default import/decode path independent of the latter two packages. The boundary test enforces this through a real subprocess CLI run.

Keep patches small and explain the concrete behavior they change. For cache changes, identify the ownership invariant and the condition under which a block can be reused. Test the relevant failure case against a dense or same-selection oracle; performance measurements should preserve identical selections and exclude warmup/oracle work.

Record hardware, dtype, page size, hot capacity, selection source and timing scope with experimental results. Do not describe layer replay as serving throughput or frozen dense-teacher traces as native DSA quality. The existing reports are historical evidence; write new runs to `results/` and add a clearly identified report when needed.

Documentation starts at [the reading guide](docs/README.md). Keep mechanism diagrams in Mermaid near the explanation and link them to actual code or a runnable example. Generate benchmark figures from JSON with `python scripts/plot_comparison.py` after installing `.[plots]`; do not hand-edit measured values in the SVG. Keep current two-tail results separate from archived one-tail experiments, and explain any unfavorable result alongside the favorable cases.

The standalone [design answer](docs/partial-residency-design.md) and its [PDF](docs/partial-residency-design.pdf) share one Markdown source. Its seven static diagrams are generated from explicit layouts in `scripts/design_figures.py` into `docs/assets/design/`, so both GitHub and PDF show the same figures without automatic edge routing. To rebuild, install Pango and Noto Sans/Serif CJK SC fonts, then run:

```bash
python -m pip install -e '.[docs]'
python scripts/build_design.py
```

Commit the Markdown, diagram sources/generated SVGs, print CSS and regenerated PDF together. The optional `docs` extra is only needed for publishing; it is not part of inference dependencies. Review PDF page breaks, diagram labels and searchable Chinese text after content changes.

Do not commit model weights, dataset contents, tensor traces, credentials or `.references/` checkouts. Reference upstream work using pinned revisions and preserve attribution for any future code reuse. The project uses the MIT license in [LICENSE](LICENSE).
