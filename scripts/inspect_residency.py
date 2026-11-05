"""Show clean -> unpinned/readable -> actually reused -> host-row miss."""

import argparse
import json
from pathlib import Path

import torch

from mini_hisparse.attention import dense_attention
from mini_hisparse.cache import CacheConfig
from mini_hisparse.hybrid import HotConfig, HybridKVCache, row_attention
from mini_hisparse.row_index import LogicalTopK


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/residency-lifecycle.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cache = HybridKVCache(
        CacheConfig(
            gpu_pages=7, host_pages=64, page_size=4, kv_heads=1, head_dim=8, device=args.device
        ),
        HotConfig(4, 4, transition_watermark=0),  # isolate explicit transition for this demo
    )
    for request in ("a", "b"):
        cache.reserve(request, 32, 1)

    def write(name, tokens):
        kv = [torch.randn(tokens, 1, 8, device=cache.device) for _ in range(2)]
        cache.write(name, 0, *kv)
        return kv

    k, v = write("a", 12)
    watched = ("a", 0, 0)
    original_slot = cache.pages[watched].slot
    records = []

    def record(label):
        cache.synchronize()  # diagnostic snapshots, not part of timed decode
        records.append(
            {
                "stage": label,
                "page": cache.page_state(watched),
                "original_slot": original_slot,
                "current_slot_owner": cache.owners[original_slot],
                "current_slot_generation": cache.allocator.slots[original_slot].generation,
                **cache.snapshot(),
            }
        )

    record("clean_pinned")
    cache.reclaim_history("a", 0)
    record("unpinned_but_readable")
    q = torch.randn(2, 8, device=cache.device)
    topk = LogicalTopK(torch.tensor([0], dtype=torch.int32, device=cache.device))
    expected = dense_attention(q, k[:1], v[:1])
    torch.testing.assert_close(row_attention(q, cache, "a", 0, topk), expected)
    record("resident_hit_after_unpin")
    write("b", 12)
    record("unused_blocks_consumed_first")
    write("b", 1)
    record("block_reused_mapping_revoked")
    torch.testing.assert_close(row_attention(q, cache, "a", 0, topk), expected)
    record("host_row_loaded_after_real_reuse")
    cache.assert_invariants()
    report = {
        "device": args.device,
        "source_commit": "abf98672ce7e822701f7a062bf240efb2c9900d1",
        "all_attention_outputs_match": True,
        "stages": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "stages": [r["stage"] for r in records]}))
    for request in ("a", "b"):
        cache.release_request(request)
    cache.assert_invariants()


if __name__ == "__main__":
    main()
