"""Ask Qwen about a local document while bounding the persistent GPU KV pool."""

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mini_hisparse import CacheConfig, HotConfig, HybridKVCache
from mini_hisparse.engine import Engine
from mini_hisparse.model import MiniModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--question", default="Why keep two tail pages resident? Answer briefly.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", default="4", help="Quest pages per head, or all")
    parser.add_argument("--gpu-pages", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.context_tokens, args.max_new_tokens) < 1:
        parser.error("context-tokens and max-new-tokens must be positive")
    try:
        top_k = None if args.top_k == "all" else int(args.top_k)
        if top_k is not None and top_k < 1:
            raise ValueError
    except ValueError:
        parser.error("top-k must be all or a positive page count")
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    document = args.context.read_bytes()
    context = tokenizer.encode(document.decode("utf-8"), add_special_tokens=False)
    original_tokens = len(context)
    if len(context) > args.context_tokens:
        front = (args.context_tokens + 1) // 2
        back = args.context_tokens // 2
        context = context[:front] + (context[-back:] if back else [])
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": f"Document:\n{tokenizer.decode(context)}\n\nQuestion: {args.question}",
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = MiniModel.from_pretrained(args.model, args.device, dtype)
    cfg, page_size = model.config, 16
    max_tokens = len(prompt) + args.max_new_tokens
    union = max_tokens
    if top_k is not None:
        forced = min(2, top_k)  # sink and tail are shared across query heads
        union = min(union, (forced + (top_k - forced) * cfg.num_attention_heads) * page_size)
    hot = HotConfig(union)
    minimum = hot.admission_blocks(page_size, cfg.num_hidden_layers)
    cache = HybridKVCache(
        CacheConfig(
            gpu_pages=minimum if args.gpu_pages is None else args.gpu_pages,
            host_pages=cfg.num_hidden_layers * ceil(max_tokens / page_size),
            page_size=page_size,
            kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            device=args.device,
            dtype=dtype,
        ),
        hot,
    )
    engine = Engine(model, cache, top_k=top_k, eos_token_id=tokenizer.eos_token_id)
    request = engine.add_request("document", prompt, args.max_new_tokens)
    output = engine.run()["document"]
    report = {
        "model": args.model,
        "context_file": str(args.context),
        "context_sha256": hashlib.sha256(document).hexdigest(),
        "question": args.question,
        "original_context_tokens": original_tokens,
        "kept_context_tokens": len(context),
        "truncation": "head+tail" if len(context) < original_tokens else "none",
        "prompt_tokens": len(prompt),
        "selection": "full" if top_k is None else f"quest {top_k} pages/head",
        "minimum_device_blocks": minimum,
        "prefill_count": request.prefill_count,
        "decode_count": request.decode_count,
        "generated_tokens": len(output),
        "answer": tokenizer.decode(output, skip_special_tokens=True),
        "cache_after_finish": cache.snapshot(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
