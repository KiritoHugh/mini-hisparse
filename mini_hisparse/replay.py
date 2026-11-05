"""Replay frozen, model-derived Q/K/V and exact selections across cache policies."""

from collections import Counter
from dataclasses import replace
from math import ceil
from statistics import median
from time import perf_counter

import torch

from .attention import paged_attention
from .cache import CacheConfig, PagedKVCache
from .hybrid import HotConfig, HybridKVCache
from .selection import Selection, selection_oracle
from .traces import choose, validate_trace


@torch.inference_mode()
def prepare(trace, page_size, selection_budget, selector, device):
    validate_trace(trace)
    histories = {
        (r["request"], r["layer"]): (r["k"].to(device), r["v"].to(device))
        for r in trace["prefills"]
    }
    starts = {key: len(k) for key, (k, _) in histories.items()}
    events = []
    for row in trace["events"]:
        key = row["request"], row["layer"]
        q, new_k, new_v = [row[name].to(device) for name in ("q", "k", "v")]
        k, v = [torch.cat((old, new)) for old, new in zip(histories[key], (new_k, new_v))]
        histories[key] = k, v
        selection = choose(q, k, page_size, selection_budget, selector, row.get("native_topk"))
        expected, _ = selection_oracle(q, k, v, selection)
        events.append(
            {
                "request": key[0],
                "layer": key[1],
                "step": row["position"] - starts[key],
                "q": q,
                "k": new_k,
                "v": new_v,
                "selection": selection,
                "expected": expected.cpu(),
            }
        )
    return sorted(events, key=lambda e: (e["step"], e["request"], e["layer"]))


@torch.inference_mode()
def replay(trace, events, config: CacheConfig, hot: HotConfig | None = None):
    # Sparse captured layers are compacted into this cache's layer dimension.
    layer_ids = sorted({r["layer"] for r in trace["prefills"]})
    layer_map = {layer: i for i, layer in enumerate(layer_ids)}
    request_names = list(dict.fromkeys(r["request"] for r in trace["prefills"]))
    lengths = {r["request"]: len(r["k"]) for r in trace["prefills"]}
    outputs = Counter(e["request"] for e in events)
    maximum = {name: length + outputs[name] // len(layer_ids) for name, length in lengths.items()}
    host_pages = sum(ceil(n / config.page_size) * len(layer_ids) for n in maximum.values())
    config = replace(config, host_pages=host_pages)
    cache = HybridKVCache(config, hot) if hot else PagedKVCache(config)
    for name in request_names:
        cache.reserve(name, maximum[name], len(layer_ids))
    for row in trace["prefills"]:
        cache.write(
            row["request"],
            layer_map[row["layer"]],
            row["k"].to(config.device),
            row["v"].to(config.device),
        )
    # Warm mask/window specialization without changing logical KV history.
    first = trace["prefills"][0]
    query = events[0]["q"]
    warm_pages = min(cache.window_pages, ceil(len(first["k"]) / config.page_size))
    if hot:
        warm_pages = min(warm_pages, hot.max_union_rows // config.page_size)
    for width in range(1, warm_pages + 1):
        warm = Selection.from_pages(
            list(range(width)), len(first["k"]), config.page_size, query.shape[0], "warmup"
        )
        paged_attention(query, cache, first["request"], layer_map[first["layer"]], selection=warm)
    if hot and cache.cuda:
        # Row kernels specialize on selected widths, including partial tails.
        # Compile every measured shape without changing any cache contents.
        from .row_index import RowPlan
        from .row_kernels import attend_rows, gather_rows

        warmed = set()
        for event in events:
            selection = event["selection"]
            width = max(map(len, selection.positions))
            union = len({t for row in selection.positions for t in row})
            shape = (len(selection.positions), width, union)
            if shape in warmed:
                continue
            warmed.add(shape)
            rows = torch.full(shape[:2], -1, dtype=torch.int64, device=cache.device)
            empty = torch.full((union,), -1, dtype=torch.int64, device=cache.device)
            plan = RowPlan(rows, empty, empty, cache.row_stats)
            gather_rows(cache._host_pointer, cache.pool, plan, config)
            attend_rows(event["q"], cache.pool, rows, config)
    cache.synchronize()
    if hot:
        cache.reset_stats()
    else:
        cache.stats.clear()
    observed, timings = [], []
    per_request = {name: Counter() for name in request_names}
    for event in events:
        name, layer = event["request"], layer_map[event["layer"]]
        cache.synchronize()
        before = cache.stats.copy()
        before_rows = cache.row_stats.clone() if hot else None
        started = perf_counter()
        cache.write(name, layer, event["k"], event["v"])
        output = paged_attention(event["q"], cache, name, layer, selection=event["selection"])
        cache.synchronize()
        timings.append((perf_counter() - started) * 1000)
        observed.append(output.cpu())
        per_request[name].update(cache.stats - before)
        if hot:
            counts = (cache.row_stats - before_rows).cpu().tolist()
            per_request[name].update(
                dict(zip(("row_resident_hits", "row_hot_hits", "row_misses"), counts))
            )
    maximum_error = 0.0
    for actual, event in zip(observed, events):
        error = (actual.float() - event["expected"].float()).abs().max().item()
        maximum_error = max(maximum_error, error)
        tolerance = 0.008 if config.dtype != torch.float32 else 3e-5
        torch.testing.assert_close(actual, event["expected"], atol=tolerance, rtol=tolerance)
    cache.assert_invariants()
    result = {
        "events": len(events),
        "selection_source": events[0]["selection"].source,
        "median_layer_step_ms": median(timings),
        "max_storage_abs_error": maximum_error,
        "per_request": {name: dict(stats) for name, stats in per_request.items()},
        "timing_scope": "serialized layer append+attention; selection/oracle excluded; not serving throughput",
        **cache.snapshot(),
    }
    for name in request_names:
        cache.release_request(name)
    return result
