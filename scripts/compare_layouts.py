"""Compare page vs token-row storage at identical capacity and frozen selections."""

import argparse
import json
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
    cases = []
    for layout in ("pages", "hybrid"):
        runs = [
            replay(trace, events, config, hot if layout == "hybrid" else None)
            for _ in range(args.repeats)
        ]
        summary = {
            "layout": layout,
            "median_of_run_medians_ms": median(r["median_layer_step_ms"] for r in runs),
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
        "trace": str(args.trace),
        "provenance": trace["provenance"],
        "selector": args.selector,
        "selection_budget": args.budget,
        "selection_note": "frozen Qwen activation selection; oracle_token is an offline quality oracle, not a learned DSA indexer",
        "gpu_pages": capacity,
        "minimum_hybrid_admission_pages": minimum,
        "max_union_rows": hot.max_union_rows,
        "hot_rows_per_layer": hot.hot_rows,
        "mean_union_rows": sum(unions) / len(unions),
        "repeats": args.repeats,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
