"""One Triton program per query head; reads physical slots without KV gathering.

This kernel fuses QK, online softmax and PV for one window. It deliberately does
not implement the upstream fused host-memory miss resolver.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_window(
    Q,
    Pool,
    Slots,
    Lengths,
    Mask,
    M,
    L,
    A,
    Q_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MASKED: tl.constexpr,
):
    head = tl.program_id(0)
    kv_head = head // GROUPS
    d = tl.arange(0, BLOCK_D)
    t = tl.arange(0, BLOCK_P)
    query = tl.load(Q + head * Q_STRIDE + d, d < DIM, other=0).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    accumulator = tl.full((BLOCK_D,), 0, tl.float32)
    for i in range(N):
        slot = tl.load(Slots + i)
        length = tl.load(Lengths + i)
        offset = slot * 2 * PAGE * KV_HEADS * DIM + t[:, None] * KV_HEADS * DIM
        offset = offset + kv_head * DIM + d[None, :]
        mask = (t[:, None] < length) & (d[None, :] < DIM)
        keys = tl.load(Pool + offset, mask, other=0).to(tl.float32)
        values = tl.load(Pool + offset + PAGE * KV_HEADS * DIM, mask, other=0).to(tl.float32)
        score = tl.sum(keys * query[None, :], 1) * (DIM**-0.5)
        valid = t < length
        if MASKED:
            valid = valid & tl.load(Mask + (head * N + i) * PAGE + t, t < PAGE, other=False)
        score = tl.where(valid, score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 0))
        alpha = tl.where(maximum == -float("inf"), 0.0, tl.exp(maximum - next_max))
        weights = tl.exp(score - tl.where(next_max == -float("inf"), 0.0, next_max))
        accumulator = accumulator * alpha + tl.sum(weights[:, None] * values, 0)
        denominator = denominator * alpha + tl.sum(weights, 0)
        maximum = next_max
    tl.store(M + head, maximum)
    tl.store(L + head, denominator)
    tl.store(A + head * DIM + d, accumulator, d < DIM)


def triton_chunk(query, lease, mask=None):
    cfg = lease.store.config
    if query.stride(-1) != 1:
        query = query.contiguous()
    slots = torch.tensor([p.slot for p in lease.pages], dtype=torch.int32, pin_memory=True).to(
        query.device, non_blocking=True
    )
    lengths = torch.tensor(
        [p.valid_tokens for p in lease.pages], dtype=torch.int32, pin_memory=True
    ).to(query.device, non_blocking=True)
    maximum = torch.empty(query.shape[0], device=query.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    accumulator = torch.empty(query.shape, device=query.device, dtype=torch.float32)
    _attention_window[(query.shape[0],)](
        query,
        lease.store.pool,
        slots,
        lengths,
        mask,
        maximum,
        denominator,
        accumulator,
        query.stride(0),
        cfg.page_size,
        cfg.kv_heads,
        query.shape[0] // cfg.kv_heads,
        cfg.head_dim,
        len(lease.pages),
        triton.next_power_of_2(cfg.page_size),
        triton.next_power_of_2(cfg.head_dim),
        mask is not None,
        num_warps=4,
    )
    return maximum, denominator, accumulator
