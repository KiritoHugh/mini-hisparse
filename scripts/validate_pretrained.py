"""Optional real-weight validation; downloads the model on first use.

Run after installing .[models]: python scripts/validate_pretrained.py
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_hisparse.engine import Engine
from mini_hisparse.hybrid import HotConfig, HybridKVCache
from mini_hisparse.model import MiniModel
from reference.cache import PagedConfig as CacheConfig
from reference.cache import PagedKVCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-layout", choices=["hybrid", "pages"], default="hybrid")
    parser.add_argument("--force-host", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "大语言模型逐 token 解码时，为什么要缓存 attention 的 Key 和 Value？请简短回答。",
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    hf = (
        AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        .eval()
        .to(args.device)
    )
    inputs = torch.tensor([prompt], device=args.device)
    with torch.inference_mode():
        expected = hf.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )[0, len(prompt) :].tolist()
    del hf
    model = MiniModel.from_pretrained(args.model, args.device, torch.bfloat16)
    cfg = model.config
    hot = HotConfig(len(prompt) + args.max_new_tokens)
    config = CacheConfig(
        gpu_pages=hot.admission_blocks(16, cfg.num_hidden_layers)
        if args.cache_layout == "hybrid"
        else 8,
        host_pages=1024,
        page_size=16,
        kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        device=args.device,
        dtype=torch.bfloat16,
    )
    cache = HybridKVCache(config, hot) if args.cache_layout == "hybrid" else PagedKVCache(config)
    engine = Engine(model, cache, eos_token_id=tokenizer.eos_token_id)
    request = engine.add_request("a", prompt, args.max_new_tokens)
    if args.force_host:
        if args.cache_layout != "hybrid":
            parser.error("--force-host requires --cache-layout hybrid")
        engine.step()
        if not request.finished:
            cache.storage.synchronize()  # explicit pressure experiment, outside normal decode
            for layer in range(cfg.num_hidden_layers):
                cache.reclaim_history("a", layer, evict=True)
    actual = engine.run()["a"]
    result = {
        "model": args.model,
        "selection": "full attention",
        "dtype": "bfloat16",
        "generated_tokens": len(actual),
        "exact_token_match": actual == expected,
        "prefill_count": request.prefill_count,
        "decode_count": request.decode_count,
        "forced_host_history": args.force_host,
        "text": tokenizer.decode(actual, skip_special_tokens=True),
        "reference_text": tokenizer.decode(expected, skip_special_tokens=True),
        **cache.snapshot(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if actual != expected:
        raise AssertionError("greedy tokens diverged; inspect numerical/model differences")


if __name__ == "__main__":
    main()
