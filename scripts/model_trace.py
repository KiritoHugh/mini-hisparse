"""Capture/inspect real model trajectories with pinned dataset provenance."""

import argparse
import json
from pathlib import Path

import torch

from experiments.traces import analyze, capture, load_trace
from mini_hisparse.model import MiniModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["capture", "analyze"])
    parser.add_argument("--trace", type=Path, default=Path("results/qwen-longbench.pt"))
    parser.add_argument("--data", type=Path, default=Path("results/longbench.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 7, 15, 23])
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/trace-analysis.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "capture":
        from transformers import AutoTokenizer

        model = MiniModel.from_pretrained(args.model, args.device, torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        samples = [json.loads(line) for line in args.data.read_text().splitlines() if line]
        trace = capture(model, tokenizer, samples, args.max_context, args.steps, args.layers)
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(args.model)
        trace["provenance"].update(
            model=args.model, model_revision=getattr(cfg, "_commit_hash", None)
        )
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trace, args.trace)
        print(
            json.dumps(
                {
                    "trace": str(args.trace),
                    "events": len(trace["events"]),
                    "samples": len(samples),
                    "layers": args.layers,
                }
            )
        )
    else:
        result = analyze(load_trace(args.trace), args.page_size, args.budget, args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
