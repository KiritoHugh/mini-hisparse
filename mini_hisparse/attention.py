"""Bounded-window attention with exact online-softmax reduction."""

from __future__ import annotations

from math import ceil, sqrt

import torch

from .cache import Lease, PagedKVCache
from .selection import Selection


def select_pages(
    cache: PagedKVCache, request_id: str, layer: int, query: torch.Tensor, top_k: int | None
) -> list[int]:
    """A small page-mean indexer for demonstrations, NOT the learned DSA indexer.

    None selects the full causal history. Sparse mode keeps the first and last
    page (within its budget), then uses query/mean-key scores for the remainder.
    """
    request = cache.requests[request_id]
    count = ceil(request.lengths[layer] / cache.config.page_size)
    if count == 0:
        raise ValueError("cannot attend to empty KV")
    if top_k is None or top_k >= count:
        return list(range(count))
    if top_k < 1:
        raise ValueError("top_k must be positive or None")
    required = [count - 1] if top_k == 1 else [0, count - 1]
    if top_k == len(required):
        return sorted(required)
    groups = query.shape[0] // cache.config.kv_heads
    grouped_query = query.float().reshape(cache.config.kv_heads, groups, -1).mean(1)
    scores = torch.einsum("hd,phd->p", grouped_query, request.summaries[layer, :count])
    scores[required] = -torch.inf
    # Deliberate reference-control-plane synchronization; a production fused
    # resolver keeps these indices and the residency table entirely on GPU.
    chosen = scores.topk(top_k - len(required)).indices.tolist()
    return sorted(required + chosen)


def dense_attention(query: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Independent full-KV oracle, also useful for correctness tests."""
    groups = query.shape[0] // k.shape[1]
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    scores = torch.einsum("hd,thd->ht", query.float(), k.float()) / sqrt(query.shape[-1])
    return torch.einsum("ht,thd->hd", scores.softmax(-1), v.float()).to(query.dtype)


def _merge(left, right):
    if left is None:
        return right
    m1, l1, a1 = left
    m2, l2, a2 = right
    m = torch.maximum(m1, m2)
    # A head may select nothing in one window, while another head uses it.
    w1 = torch.where(torch.isfinite(m1), (m1 - m).exp(), 0.0)
    w2 = torch.where(torch.isfinite(m2), (m2 - m).exp(), 0.0)
    return m, w1 * l1 + w2 * l2, w1[:, None] * a1 + w2[:, None] * a2


def _torch_chunk(query: torch.Tensor, lease: Lease, mask=None):
    state = None
    for index, page in enumerate(lease.pages):
        kv = lease.cache.pool[page.slot, :, : page.valid_tokens]
        groups = query.shape[0] // kv.shape[2]
        k, v = kv[0].repeat_interleave(groups, 1), kv[1].repeat_interleave(groups, 1)
        scores = torch.einsum("hd,thd->ht", query.float(), k.float()) / sqrt(query.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask[:, index, : page.valid_tokens], -torch.inf)
        maximum = scores.max(-1).values
        safe_max = torch.where(torch.isfinite(maximum), maximum, 0.0)
        weights = (scores - safe_max[:, None]).exp()
        state = _merge(
            state, (maximum, weights.sum(-1), torch.einsum("ht,thd->hd", weights, v.float()))
        )
    return state


@torch.inference_mode()
def paged_attention(
    query: torch.Tensor,
    cache: PagedKVCache,
    request_id: str,
    layer: int,
    pages: list[int] | None = None,
    backend: str = "auto",
    *,
    selection: Selection | None = None,
) -> torch.Tensor:
    """Read any number of selected pages with at most two small windows live.

    H2D(next window) may overlap attention(current). A lease is released only
    AFTER its last kernel has been enqueued. Its event protects subsequent reuse.
    """
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be auto, torch or triton")
    if (
        query.ndim != 2
        or query.shape[1] != cache.config.head_dim
        or query.shape[0] % cache.config.kv_heads
    ):
        raise ValueError("expected [query_heads, head_dim], query_heads divisible by KV heads")
    if query.device != cache.pool.device:
        raise ValueError("query must be on the cache device")
    if selection is not None:
        if (
            pages is not None
            or selection.length != cache.requests[request_id].lengths[layer]
            or selection.page_size != cache.config.page_size
            or len(selection.positions) != query.shape[0]
        ):
            raise ValueError("selection has stale length, wrong shape, or conflicts with pages")
        pages = selection.pages
    if pages is None:
        pages = select_pages(cache, request_id, layer, query, None)
    pages = list(dict.fromkeys(pages))
    count = ceil(cache.requests[request_id].lengths[layer] / cache.config.page_size)
    if not pages or any(p < 0 or p >= count for p in pages):
        raise ValueError("selection must contain valid logical pages")
    # The hybrid path resolves token rows, never acquires whole selected pages.
    from .hybrid import HybridKVCache, row_attention
    from .row_index import LogicalTopK

    if isinstance(cache, HybridKVCache):
        if selection is None:
            selection = Selection.from_pages(
                pages,
                cache.requests[request_id].lengths[layer],
                cache.config.page_size,
                query.shape[0],
            )
        return row_attention(
            query,
            cache,
            request_id,
            layer,
            LogicalTopK.from_selection(selection, query.device),
            backend,
        )
    use_triton = backend == "triton" or (backend == "auto" and cache.cuda)
    if use_triton:
        if not cache.cuda:
            raise ValueError("Triton requires CUDA")
        from .kernels import triton_chunk

        chunk_fn = triton_chunk
    else:
        chunk_fn = _torch_chunk
    all_keys = [(request_id, layer, p) for p in pages]
    plan = cache.plan(all_keys)
    width = plan.width
    previous_hint = cache.lookahead
    if cache.config.eviction_policy != "legacy":
        cache.lookahead = set(all_keys)
    cache.stats["planned_union_pages"] += plan.union_pages
    cache.stats["planned_hot_slots"] += plan.hot_slot_target
    chunks = [
        [(request_id, layer, p) for p in pages[i : i + width]] for i in range(0, len(pages), width)
    ]
    current, future, state = None, None, None
    try:
        current = cache.acquire(chunks[0])
        for i in range(len(chunks)):
            if plan.pipelined and i + 1 < len(chunks):
                future = cache.acquire(chunks[i + 1])
                cache.stats["prefetch_windows"] += 1
            current.wait()
            mask = (
                selection.mask([p.key[2] for p in current.pages], query.device)
                if selection
                else None
            )
            state = _merge(state, chunk_fn(query, current, mask))
            current.release()
            current = None
            cache.stats["attention_windows"] += 1
            if i + 1 < len(chunks):
                current = future if future is not None else cache.acquire(chunks[i + 1])
                future = None
    finally:
        for lease in (current, future):
            if lease is not None and not lease.released:
                lease.release()
        cache.lookahead = previous_hint
    _, denominator, numerator = state
    return (numerator / denominator[:, None]).to(query.dtype)
