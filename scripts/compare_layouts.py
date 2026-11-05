"""Compare frozen selections with bounded page/row caches and a resident control."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from statistics import median

import torch

from experiments.replay import prepare, replay
from experiments.traces import load_trace
from mini_hisparse.config import CacheConfig
from mini_hisparse.hybrid import HotConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("results/qwen-longbench.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/layout-comparison.json"))
    parser.add_argument(
        "--selector", choices=["quest", "mean", "native", "oracle_token"], default="quest"
    )
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--hot-rows", type=int)
    parser.add_argument("--gpu-pages", type=int)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-resident", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.set_num_threads(1)
    trace = load_trace(args.trace)
    events = prepare(trace, 16, args.budget, args.selector, args.device)
    unions = [len({t for row in e["selection"].positions for t in row}) for e in events]
    hot = HotConfig(max(unions), args.hot_rows)
    minimum = hot.admission_blocks(16, len(trace["prefills"]))
    capacity = args.gpu_pages if args.gpu_pages is not None else minimum
    k = trace["prefills"][0]["k"]
    config = CacheConfig(
        gpu_pages=capacity,
        page_size=16,
        kv_heads=k.shape[1],
        head_dim=k.shape[2],
        dtype=k.dtype,
        device=args.device,
    )
    appended = Counter((e["request"], e["layer"]) for e in events)
    full_pages = sum(
        ceil((len(p["k"]) + appended[p["request"], p["layer"]]) / config.page_size)
        for p in trace["prefills"]
    )
    layouts = ["resident", "pages", "hybrid"] if args.include_resident else ["pages", "hybrid"]
    collected = {layout: [] for layout in layouts}
    orders = []
    for repeat in range(args.repeats):
        order = layouts[repeat % len(layouts) :] + layouts[: repeat % len(layouts)]
        orders.append(order)
        for layout in order:
            run_config = replace(config, gpu_pages=full_pages) if layout == "resident" else config
            result = replay(trace, events, run_config, hot if layout == "hybrid" else None)
            if layout == "resident":
                assert result["stats"].get("h2d_bytes", 0) == 0, "resident control must not reload"
            collected[layout].append(result)
            print(
                f"repeat={repeat + 1} layout={layout} median_ms={result['median_layer_step_ms']:.3f}",
                flush=True,
            )
    cases = []
    for layout, runs in collected.items():
        times = [r["median_layer_step_ms"] for r in runs]
        summary = {
            "layout": layout,
            "gpu_pages": full_pages if layout == "resident" else capacity,
            "kv_pool_bytes": (full_pages if layout == "resident" else capacity)
            * config.page_size
            * 2
            * k.shape[1]
            * k.shape[2]
            * k.element_size(),
            "median_of_run_medians_ms": median(times),
            "run_median_range_ms": [min(times), max(times)],
            "h2d_bytes": (
                runs[-1]["row_stats"]["h2d_bytes"]
                if layout == "hybrid"
                else runs[-1]["stats"].get("h2d_bytes", 0)
            ),
            "index_bytes": runs[-1]["index_bytes"],
            "max_storage_abs_error": max(r["max_storage_abs_error"] for r in runs),
        }
        print(json.dumps(summary), flush=True)
        cases.append({**summary, "runs": runs})
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_commit": subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "mini_hisparse", "reference", "experiments"],
            text=True,
        ).strip(),
        "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": ["python", "scripts/compare_layouts.py", *sys.argv[1:]],
        "trace": str(args.trace),
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "provenance": {
            **trace["provenance"],
            "samples": [
                {k: v for k, v in sample.items() if k not in ("prompt_ids", "generated_ids")}
                for sample in trace["provenance"].get("samples", [])
            ],
        },
        "selector": args.selector,
        "selection_budget": args.budget,
        "selection_note": "frozen Qwen activation selection; oracle_token is an offline quality oracle, not a learned DSA indexer",
        "gpu_pages": capacity,
        "minimum_hybrid_admission_pages": minimum,
        "max_union_rows": hot.max_union_rows,
        "hot_rows_per_layer": hot.hot_rows,
        "mean_union_rows": sum(unions) / len(unions),
        "repeats": args.repeats,
        "run_orders": orders,
        "events_per_run": len(events),
        "page_size": config.page_size,
        "kv_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "dtype": str(config.dtype),
        "resident_control": "same selected attention via page reference with all history resident; extra KV memory, not dense attention or an optimized lower bound",
        "timing_scope": "serialized synchronized layer append + cache resolution/load + selected attention; prefill, JIT warmup, selection and oracle excluded; not end-to-end serving",
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
