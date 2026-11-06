# Changelog

## Unreleased

- Publish a self-contained Chinese design answer with six vector diagrams, pseudocode, a PDF edition and a reproducible Graphviz/WeasyPrint build script.

- Expand the three interview answers with exact page/row state transitions, victim policy, conditional progress reasoning and asynchronous execution limits.
- Add a fixed-capacity hot-size sweep and recorded A100 results, including infeasible admission configurations; add CPU/CUDA worst-case all-miss progress coverage at the minimum hot/tail footprint.

- Organize documentation into illustrated design, code walkthrough, runnable examples and current baseline comparisons.
- Add executable address translation and local-document Qwen examples; include the CPU address example in CI.
- Extend frozen-trace comparisons with an all-resident control, rotating repeat order and provenance; publish current two-tail A100 reports and a reproducible SVG plot.
- Keep plotting dependencies optional and include examples/diagram assets in source distributions.

## 0.2.0

- Replace `HybridKVCache(PagedKVCache)` inheritance with explicit `HostStore` and `DeviceAllocator` composition.
- Move page replacement, window planning and page attention to `reference/`.
- Move benchmark, model-trace tooling and replay to `experiments/`.
- Route model decoding directly through the cache's attention interface; the default runtime does not import optional reference/experiment packages.
- Keep caller-owned metadata and host backing alive through the same retirement event and index-budget lifetime.
- Preserve CLI commands and comparison modes, and document the changed Python module/state APIs in [the migration guide](docs/refactor.md).
- Retain numerical/lifetime coverage, add an isolated runtime-boundary check, and repeat real-model and real-trace validation.

## 0.1.0

Initial public implementation with a shared GPU KV pool, host backing, row-resolution kernels, clean-page reuse, proactive hot allocation and two resident tail pages.
