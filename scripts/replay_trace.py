import argparse
import json
from pathlib import Path

import torch

from mini_hisparse.cache import CacheConfig
from mini_hisparse.hybrid import HotConfig
from mini_hisparse.replay import prepare, replay
from mini_hisparse.traces import load_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=Path("results/qwen-longbench.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/replay.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-pages", type=int)
    parser.add_argument("--cache-layout", choices=["hybrid", "pages"], default="hybrid")
    parser.add_argument("--hot-rows", type=int)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--policy", choices=["legacy", "lru", "fair_lru"], default="fair_lru")
    parser.add_argument(
        "--selector", choices=["quest", "mean", "oracle_token", "native"], default="quest"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    trace = load_trace(args.trace)
    events = prepare(trace, args.page_size, args.budget, args.selector, args.device)
    shape = trace["prefills"][0]["k"]
    hot = None
    if args.cache_layout == "hybrid":
        union = max(len({t for row in e["selection"].positions for t in row}) for e in events)
        hot = HotConfig(union, args.hot_rows)
    groups = len(trace["prefills"])
    minimum_pages = hot.admission_blocks(args.page_size, groups) if hot else 32
    config = CacheConfig(
        gpu_pages=args.gpu_pages if args.gpu_pages is not None else minimum_pages,
        page_size=args.page_size,
        kv_heads=shape.shape[1],
        head_dim=shape.shape[2],
        dtype=shape.dtype,
        device=args.device,
        eviction_policy=args.policy,
    )
    result = replay(trace, events, config, hot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
