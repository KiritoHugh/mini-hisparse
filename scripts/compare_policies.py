"""Ablate policies on the identical frozen trajectory; never claim model tps."""

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from mini_hisparse.cache import CacheConfig
from mini_hisparse.replay import prepare, replay
from mini_hisparse.traces import load_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("results/qwen-longbench.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/policy-comparison.json"))
    parser.add_argument("--capacities", type=int, nargs="+", default=[32, 128, 256])
    parser.add_argument("--policies", nargs="+", default=["legacy", "lru", "fair_lru"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument(
        "--selector", default="quest", choices=["quest", "oracle_token", "native", "mean"]
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.set_num_threads(1)
    trace = load_trace(args.trace)
    events = prepare(trace, 16, args.budget, args.selector, args.device)
    shape = trace["prefills"][0]["k"]
    rows = []
    for capacity in args.capacities:
        for policy in args.policies:
            config = CacheConfig(
                gpu_pages=capacity,
                page_size=16,
                kv_heads=shape.shape[1],
                head_dim=shape.shape[2],
                dtype=shape.dtype,
                device=args.device,
                eviction_policy=policy,
            )
            runs = [replay(trace, events, config) for _ in range(args.repeats)]
            stats = runs[-1]["stats"]
            hits = stats.get("attention_resident_hits", 0) + stats.get("attention_hot_hits", 0)
            misses = stats.get("attention_misses", 0)
            rows.append(
                {
                    "capacity_pages": capacity,
                    "policy": policy,
                    "median_of_run_medians_ms": median(r["median_layer_step_ms"] for r in runs),
                    "attention_hit_rate": hits / (hits + misses),
                    "h2d_bytes": stats.get("h2d_bytes", 0),
                    "max_storage_abs_error": max(r["max_storage_abs_error"] for r in runs),
                    "runs": runs,
                }
            )
            print(json.dumps({k: v for k, v in rows[-1].items() if k != "runs"}), flush=True)
    result = {
        "trace": str(args.trace),
        "provenance": trace["provenance"],
        "selector": args.selector,
        "budget_pages_per_head": args.budget,
        "repeats": args.repeats,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
