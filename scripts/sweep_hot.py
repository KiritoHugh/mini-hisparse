"""Measure hot/resident capacity tradeoffs at a fixed GPU pool and frozen selection."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from math import ceil, isfinite
from pathlib import Path
from statistics import median

import torch

from experiments.replay import prepare, replay
from experiments.traces import load_trace
from mini_hisparse import CacheConfig, HotConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("results/qwen-longbench.pt"))
    parser.add_argument("--selector", choices=["quest", "oracle_token", "native"], default="quest")
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--gpu-pages", type=int, required=True)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[1, 1.5, 2, 3])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("results/hot-sweep.json"))
    args = parser.parse_args()
    if args.repeats < 1 or any(not isfinite(m) or m < 1 for m in args.multipliers):
        parser.error("repeats must be positive; hot multipliers must be finite and >= 1")
    torch.set_num_threads(1)
    trace = load_trace(args.trace)
    events = prepare(trace, 16, args.budget, args.selector, args.device)
    union = max(len({t for row in e["selection"].positions for t in row}) for e in events)
    groups = len(trace["prefills"])
    k = trace["prefills"][0]["k"]
    config = CacheConfig(
        gpu_pages=args.gpu_pages,
        page_size=16,
        kv_heads=k.shape[1],
        head_dim=k.shape[2],
        dtype=k.dtype,
        device=args.device,
    )
    # Hold trigger settings fixed as hot capacity changes: isolate the allocation tradeoff.
    watermark = max(ceil(2 * union / config.page_size), args.gpu_pages // 10)
    cases, policies = [], {}
    for multiplier in dict.fromkeys(args.multipliers):
        hot = HotConfig(union, ceil(multiplier * union), transition_watermark=watermark)
        minimum = hot.admission_blocks(config.page_size, groups)
        fits = minimum <= args.gpu_pages
        cases.append(
            {
                "multiplier": multiplier,
                "hot_rows_per_layer": hot.hot_rows,
                "hot_blocks_per_layer": ceil(hot.hot_rows / config.page_size),
                "minimum_admission_blocks": minimum,
                "blocks_beyond_hot_and_two_tails": args.gpu_pages - minimum,
                "status": "measured" if fits else "infeasible_by_admission_formula",
                "runs": [],
            }
        )
        if fits:
            policies[len(cases) - 1] = hot
    orders = []
    for repeat in range(args.repeats):
        indices = list(policies)
        if not indices:
            break
        offset = repeat % len(indices)
        order = indices[offset:] + indices[:offset]
        orders.append([cases[i]["multiplier"] for i in order])
        for i in order:
            result = replay(trace, events, config, policies[i])
            cases[i]["runs"].append(result)
            print(
                f"repeat={repeat + 1} hot={cases[i]['multiplier']}U "
                f"median_ms={result['median_layer_step_ms']:.3f}",
                flush=True,
            )
    for case in cases:
        runs = case["runs"]
        if not runs:
            continue
        times = [r["median_layer_step_ms"] for r in runs]
        h2d = [r["row_stats"]["h2d_bytes"] for r in runs]
        case.update(
            median_of_run_medians_ms=median(times),
            run_median_range_ms=[min(times), max(times)],
            median_h2d_bytes=median(h2d),
            h2d_range_bytes=[min(h2d), max(h2d)],
            median_row_stats={
                key: median(r["row_stats"][key] for r in runs) for key in runs[0]["row_stats"]
            },
            index_bytes=runs[0]["index_bytes"],
            max_storage_abs_error=max(r["max_storage_abs_error"] for r in runs),
        )
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_commit": subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "mini_hisparse", "reference", "experiments"],
            text=True,
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": ["python", "scripts/sweep_hot.py", *sys.argv[1:]],
        "trace": str(args.trace),
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "selection_source": events[0]["selection"].source,
        "selection_budget": args.budget,
        "max_union_rows": union,
        "request_layer_groups": groups,
        "gpu_pages": args.gpu_pages,
        "fixed_watermark": watermark,
        "resident_window_pages": 64,
        "page_size": config.page_size,
        "dtype": str(k.dtype),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "events_per_run": len(events),
        "repeats": args.repeats,
        "run_orders": orders,
        "timing_scope": "serialized layer append+attention; same replay warmup as layout comparison; selection/oracle excluded; not serving throughput",
        "note": "offline measured-union sizing, fixed capacity and triggers; infeasible cases are preflight calculations, not executed runs",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in c.items() if k != "runs"} for c in cases], indent=2))


if __name__ == "__main__":
    main()
