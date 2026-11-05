"""Deterministic KV microbenchmark. This does not measure LLM serving throughput."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from math import ceil
from statistics import median
from time import perf_counter

import torch

from .attention import dense_attention, paged_attention
from .cache import CacheConfig, PagedKVCache


def benchmark(
    config: CacheConfig,
    tokens=1024,
    steps=20,
    requests=3,
    selected_pages=8,
    backend="auto",
    trace=None,
):
    if min(tokens, steps, requests, selected_pages) < 1:
        raise ValueError("benchmark dimensions must be positive")
    torch.manual_seed(42)
    # Identical CPU-generated data and page selections for every policy.
    histories = {
        str(i): [
            torch.randn(tokens + steps, config.kv_heads, config.head_dim, dtype=config.dtype)
            for _ in range(2)
        ]
        for i in range(requests)
    }
    queries = torch.randn(steps, requests, config.kv_heads * 2, config.head_dim, dtype=config.dtype)
    total_pages = ceil((tokens + steps) / config.page_size) * requests
    rows = []
    for mode, slots, overlap in [
        ("resident", total_pages, False),
        ("pressured_serial", config.gpu_pages, False),
        ("pressured_overlap", config.gpu_pages, True),
    ]:
        cfg = replace(
            config, gpu_pages=slots, host_pages=max(config.host_pages, total_pages), overlap=overlap
        )
        cache = PagedKVCache(cfg)
        for name, (k, v) in histories.items():
            cache.reserve(name, tokens + steps, 1)
            cache.write(name, 0, k[:tokens].to(cfg.device), v[:tokens].to(cfg.device))
        q_device = queries.to(cfg.device)
        # Compile/warm the exact backend and chunk sizes outside the timer.
        page_count = ceil(tokens / cfg.page_size)
        for name in histories:
            paged_attention(
                q_device[0, int(name)],
                cache,
                name,
                0,
                list(range(min(selected_pages, page_count))),
                backend,
            )
        cache.synchronize()
        cache.stats.clear()
        observations, durations = [], []
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cache.cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = (
            torch.profiler.profile(activities=activities)
            if trace and mode == "pressured_overlap"
            else nullcontext()
        )
        with profiler as profile:
            for step in range(steps):
                # One-token projection inputs are prepared before timing.
                incoming = {
                    name: (
                        k[tokens + step : tokens + step + 1].to(cfg.device),
                        v[tokens + step : tokens + step + 1].to(cfg.device),
                    )
                    for name, (k, v) in histories.items()
                }
                cache.synchronize()
                started = perf_counter()
                for name, (k, v) in incoming.items():
                    cache.write(name, 0, k, v)
                    count = ceil((tokens + step + 1) / cfg.page_size)
                    # Rotate the set; always include the live tail. This is a
                    # reproducible access trace, not the approximate indexer.
                    pages = list(
                        dict.fromkeys(
                            [count - 1]
                            + [(step * 3 + j) % count for j in range(selected_pages - 1)]
                        )
                    )
                    result = paged_attention(
                        q_device[step, int(name)], cache, name, 0, pages, backend
                    )
                    observations.append((step, name, pages, result))
                cache.synchronize()
                durations.append((perf_counter() - started) * 1000)
        if profile is not None:
            profile.export_chrome_trace(str(trace))
        # Correctness is checked AFTER timing, entirely against CPU KV history.
        max_error = 0.0
        for step, name, pages, observed in observations:
            indices = [
                t
                for p in pages
                for t in range(p * cfg.page_size, min((p + 1) * cfg.page_size, tokens + step + 1))
            ]
            k, v = histories[name]
            expected = dense_attention(queries[step, int(name)], k[indices], v[indices])
            error = (observed.cpu().float() - expected.float()).abs().max().item()
            max_error = max(max_error, error)
            torch.testing.assert_close(
                observed.cpu(),
                expected,
                atol=3e-5 if cfg.dtype == torch.float32 else 0.006,
                rtol=3e-4 if cfg.dtype == torch.float32 else 0.03,
            )
        cache.assert_invariants()
        rows.append(
            {
                "mode": mode,
                "median_step_ms": median(durations),
                "p95_step_ms": sorted(durations)[
                    min(len(durations) - 1, ceil(0.95 * len(durations)) - 1)
                ],
                "request_steps_per_second": steps * requests / (sum(durations) / 1000),
                "max_abs_error": max_error,
                **cache.snapshot(),
            }
        )
        for name in histories:
            cache.release_request(name)
    return {
        "workload": {
            "tokens": tokens,
            "steps": steps,
            "requests": requests,
            "selected_pages": selected_pages,
            "page_size": config.page_size,
            "chunk_pages": config.chunk_pages,
            "backend": backend,
            "dtype": str(config.dtype),
            "device": str(config.device),
            "gpu": torch.cuda.get_device_name(config.device)
            if str(config.device).startswith("cuda")
            else None,
            "torch": torch.__version__,
            "seed": 42,
            "timing": "append + fixed-selection attention; oracle and input preparation excluded",
            "profiled_overlap": bool(trace),
        },
        "results": rows,
    }
