from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark import benchmark
from .cache import CacheConfig, PagedKVCache
from .engine import Engine
from .hybrid import HotConfig, HybridKVCache
from .model import MiniModel


def _top_k(value):
    if value == "all":
        return None
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("top-k must be 'all' or a positive page count")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Mini-HiSparse: decode through partial KV residency"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "generate", "bench"):
        command = sub.add_parser(name)
        command.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
        command.add_argument("--gpu-pages", type=int)
        command.add_argument("--host-pages", type=int, default=4096)
        command.add_argument("--page-size", type=int, default=16)
        command.add_argument("--chunk-pages", type=int, default=2, help="pages layout only")
        command.add_argument("--backend", choices=["auto", "torch", "triton"], default="auto")
        command.add_argument(
            "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
        )
        command.add_argument("--output", type=Path)
        command.add_argument(
            "--eviction-policy",
            choices=["legacy", "lru", "fair_lru"],
            default="fair_lru",
            help="pages layout only; hybrid uses request/layer-local row LRU",
        )
        command.add_argument("--index-budget-mib", type=int, default=64)
        if name != "bench":
            command.add_argument("--cache-layout", choices=["hybrid", "pages"], default="hybrid")
            command.add_argument("--max-union-rows", type=int)
            command.add_argument("--hot-rows", type=int)
            command.add_argument("--resident-window-pages", type=int, default=64)
            command.add_argument("--hot-watermark", type=int)
            command.add_argument(
                "--top-k", type=_top_k, default=None, help="page count, or all (default)"
            )
            command.add_argument("--max-new-tokens", type=int, default=12)
            command.add_argument("--no-overlap", action="store_true", help="pages layout only")
            command.add_argument("--selector", choices=["quest", "mean"], default="quest")
        if name == "generate":
            command.add_argument(
                "--model", required=True, help="local Qwen2/2.5 directory or HF repo"
            )
            command.add_argument("--prompt", default="Explain how a KV cache works.")
            command.add_argument(
                "--chat", action="store_true", help="apply the tokenizer's chat template"
            )
        elif name == "demo":
            command.add_argument("--requests", type=int, default=3)
            command.add_argument("--prompt-tokens", type=int, default=65)
        else:
            command.add_argument("--tokens", type=int, default=1024)
            command.add_argument("--steps", type=int, default=20)
            command.add_argument("--requests", type=int, default=3)
            command.add_argument("--selected-pages", type=int, default=8)
            command.add_argument(
                "--trace", type=Path, help="optional Chrome trace (adds profiler overhead)"
            )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    common = {
        "gpu_pages": args.gpu_pages if args.gpu_pages is not None else 8,
        "host_pages": args.host_pages,
        "page_size": args.page_size,
        "chunk_pages": args.chunk_pages,
        "device": args.device,
        "dtype": dtype,
        "eviction_policy": args.eviction_policy,
        "index_budget_bytes": args.index_budget_mib * 1024 * 1024,
    }
    if args.command == "bench":
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
        result = benchmark(
            CacheConfig(**common),
            args.tokens,
            args.steps,
            args.requests,
            args.selected_pages,
            args.backend,
            args.trace,
        )
    else:
        tokenizer = None
        if args.command == "generate":
            from transformers import AutoTokenizer

            model = MiniModel.from_pretrained(args.model, args.device, dtype)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            prompt = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": args.prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                if args.chat
                else tokenizer.encode(args.prompt)
            )
            prompts = [prompt]
        else:
            if args.requests < 1 or args.prompt_tokens < 1:
                parser.error("requests and prompt-tokens must be positive")
            model = MiniModel().eval().to(device=args.device, dtype=dtype)
            prompts = [
                torch.randint(0, model.config.vocab_size, (args.prompt_tokens + i * 7,)).tolist()
                for i in range(args.requests)
            ]
        cfg = model.config
        hot = None
        if args.cache_layout == "hybrid":
            maximum = max(map(len, prompts)) + args.max_new_tokens
            if args.top_k is not None:
                # Quest forces sink/tail pages shared by heads; remaining
                # pages can differ. This is a bound, not a trace-based guess.
                forced = min(2, args.top_k)
                heads = cfg.num_attention_heads if args.selector == "quest" else 1
                maximum = min(maximum, (forced + (args.top_k - forced) * heads) * args.page_size)
            hot = HotConfig(
                maximum if args.max_union_rows is None else args.max_union_rows,
                args.hot_rows,
                resident_window_pages=args.resident_window_pages,
                transition_watermark=args.hot_watermark,
            )
            if args.gpu_pages is None:
                common["gpu_pages"] = hot.admission_blocks(args.page_size, cfg.num_hidden_layers)
        cache_config = CacheConfig(
            **common,
            kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            overlap=not args.no_overlap,
        )
        cache = HybridKVCache(cache_config, hot) if hot else PagedKVCache(cache_config)
        engine = Engine(
            model,
            cache,
            args.top_k,
            args.backend,
            tokenizer.eos_token_id if tokenizer else None,
            selector=args.selector,
        )
        for i, prompt in enumerate(prompts):
            engine.add_request(str(i), prompt, args.max_new_tokens)
        timeline = []
        while engine.waiting or engine.running:
            engine.step()
            timeline.append(
                {
                    "active": len(engine.running),
                    "waiting": len(engine.waiting),
                    "locations": cache.snapshot()["locations"],
                    "generated": {name: len(r.generated) for name, r in engine.requests.items()},
                }
            )
        outputs = {name: r.generated for name, r in engine.requests.items()}
        cache.collect(wait=True)
        result = {
            "model": args.model if tokenizer else "tiny random decoder (mechanics demo)",
            "selection": "full attention"
            if args.top_k is None
            else f"{args.selector}: approximate top-{args.top_k} pages per selection head",
            "outputs": {
                name: tokenizer.decode(tokens, skip_special_tokens=True) if tokenizer else tokens
                for name, tokens in outputs.items()
            },
            "requests": {
                name: {
                    "prefill_count": r.prefill_count,
                    "decode_count": r.decode_count,
                    "generated_tokens": len(r.generated),
                }
                for name, r in engine.requests.items()
            },
            "timeline": timeline,
            **cache.snapshot(),
        }
        cache.assert_invariants()
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
