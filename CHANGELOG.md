# Changelog

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
