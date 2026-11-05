from dataclasses import replace

import pytest
import torch

from mini_hisparse.attention import dense_attention, paged_attention
from mini_hisparse.cache import CacheConfig, CapacityError, PagedKVCache
from mini_hisparse.engine import Engine
from mini_hisparse.model import MiniModel, ModelConfig


def cfg(**kwargs):
    return replace(
        CacheConfig(gpu_pages=3, host_pages=32, page_size=4, kv_heads=1, head_dim=4, device="cpu"),
        **kwargs,
    )


def write(cache, name, count):
    k, v = [torch.randn(count, 1, 4, device=cache.device) for _ in range(2)]
    cache.write(name, 0, k, v)
    return k, v


@pytest.mark.parametrize("policy,keeps_hot", [("legacy", False), ("lru", True), ("fair_lru", True)])
def test_recent_hot_contents_survive_cold_resident(policy, keeps_hot):
    cache = PagedKVCache(cfg(gpu_pages=2, eviction_policy=policy))
    cache.reserve("a", 12, 1)
    write(cache, "a", 12)
    with cache.acquire([("a", 0, 0)]):
        pass
    with cache.acquire([("a", 0, 1)]):
        pass
    assert (cache.pages[("a", 0, 0)].slot is not None) == keeps_hot
    cache.assert_invariants()


def test_soft_fair_share_and_borrowing_idle_capacity():
    cache = PagedKVCache(cfg(gpu_pages=4))
    cache.reserve("a", 24, 1)
    cache.reserve("b", 8, 1)
    write(cache, "b", 4)  # oldest page belongs to the under-share request
    write(cache, "a", 12)
    write(cache, "b", 4)
    assert cache.allocator.occupancy == {"a": 2, "b": 2}
    assert cache.pages[("b", 0, 0)].slot is not None
    cache.release_request("b")
    write(cache, "a", 12)
    assert cache.allocator.occupancy["a"] == 4
    cache.assert_invariants()


def test_sink_soft_protection_expires_when_access_pattern_changes():
    cache = PagedKVCache(cfg(gpu_pages=2))
    cache.reserve("a", 12, 1)
    write(cache, "a", 12)
    q = torch.randn(2, 4)
    paged_attention(q, cache, "a", 0, [0])
    for page in [1, 2, 1, 2]:
        paged_attention(q, cache, "a", 0, [page])
    assert cache.pages[("a", 0, 0)].slot is None
    cache.assert_invariants()


def test_lookahead_preserves_future_hit_and_window_respects_other_leases():
    cache = PagedKVCache(cfg())
    cache.reserve("a", 16, 1)
    k, v = write(cache, "a", 16)
    q = torch.randn(2, 4)
    before = cache.stats["attention_misses"]
    actual = paged_attention(q, cache, "a", 0, [0, 1, 3])
    assert cache.stats["attention_misses"] - before == 1
    torch.testing.assert_close(
        actual, dense_attention(q, k[[*range(8), *range(12, 16)]], v[[*range(8), *range(12, 16)]])
    )
    cache.reserve("b", 4, 1)
    write(cache, "b", 4)
    with cache.acquire([("b", 0, 0)]):
        plan = cache.plan([("a", 0, i) for i in range(4)])
        assert plan.pinned_outside == 1 and plan.live_slot_limit <= 2
        torch.testing.assert_close(paged_attention(q, cache, "a", 0), dense_attention(q, k, v))
    cache.assert_invariants()


def test_lazy_materialization_and_index_admission_are_atomic():
    cache = PagedKVCache(cfg(host_pages=20, index_budget_bytes=960))
    cache.reserve("a", 40, 2)
    assert len(cache.pages) == 0 and len(cache.free_host) == 20
    assert cache.available_host_credits == 0
    write(cache, "a", 1)
    assert len(cache.pages) == 1 and cache.requests["a"].host_credits == 19
    cache.assert_invariants()
    cache.release_request("a")
    assert len(cache.free_host) == 20
    cache = PagedKVCache(cfg(host_pages=20, index_budget_bytes=959))
    with pytest.raises(CapacityError, match="index budget"):
        cache.reserve("a", 40, 2)
    assert not cache.requests and len(cache.free_host) == 20


def test_bfloat16_bounds_are_exact_and_index_budget_counts_actual_bytes():
    cache = PagedKVCache(cfg(dtype=torch.bfloat16))
    cache.reserve("a", 12, 1)
    k = torch.randn(7, 1, 4).bfloat16()
    cache.write("a", 0, k, k)
    request = cache.requests["a"]
    assert request.minimum.dtype == torch.bfloat16
    torch.testing.assert_close(request.minimum[0, 0], k[:4].amin(0))
    assert cache.index_bytes == cache.index_reservation_bytes(12, 1) == 3 * 1 * 4 * 8


def test_shrink_rejects_active_slot_atomically_and_versions_track_tail():
    cache = PagedKVCache(cfg(gpu_pages=3))
    cache.reserve("a", 12, 1)
    write(cache, "a", 9)
    owner = cache.owners[2]
    with cache.acquire([owner]):
        with pytest.raises(CapacityError):
            cache.set_budget(1)
        assert cache.budget == 3
    cache.set_budget(1)
    write(cache, "a", 3)
    state = cache.page_state(("a", 0, 2))
    assert state["sealed"] and state["content_version"] == 2
    assert state["host_committed_version"] == 2
    cache.assert_invariants()


def test_attention_can_share_fully_leased_resident_pages():
    cache = PagedKVCache(cfg(gpu_pages=2))
    cache.reserve("a", 8, 1)
    k, v = write(cache, "a", 8)
    q = torch.randn(2, 4)
    with cache.acquire([("a", 0, 0), ("a", 0, 1)]):
        torch.testing.assert_close(paged_attention(q, cache, "a", 0), dense_attention(q, k, v))
    cache.assert_invariants()


@pytest.mark.cuda
def test_deferred_reclaim_protects_host_bytes_and_slot_generation():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache = PagedKVCache(cfg(device="cuda:0", gpu_pages=1, host_pages=1))
    cache.reserve("a", 4, 1)
    old_request_generation = cache.requests["a"].generation
    old, _ = write(cache, "a", 4)
    with cache.acquire([("a", 0, 0)]) as lease:
        handle = lease.handles[0]
        torch.cuda._sleep(100_000_000)
        observed = cache.pool[0, 0].clone()
    cache.release_request("a", wait=False)
    # No global synchronization; host backing is still owned by the event.
    assert cache.retired and not cache.free_host
    with pytest.raises(CapacityError):
        cache.reserve("a", 4, 1)
    cache.collect(wait=True)
    cache.reserve("a", 4, 1)
    assert cache.requests["a"].generation > old_request_generation
    new, _ = write(cache, "a", 4)
    with pytest.raises(RuntimeError, match="generation"):
        cache.allocator.validate(*handle)
    torch.testing.assert_close(observed, old)
    torch.testing.assert_close(cache.host_kv("a", 0)[0], new.cpu())
    cache.assert_invariants()


def test_cancellation_reclaims_only_cancelled_request():
    model = MiniModel(
        ModelConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
        )
    ).eval()
    cache = PagedKVCache(cfg(gpu_pages=1))
    engine = Engine(model, cache)
    engine.add_request("a", [1] * 5, 5)
    engine.add_request("b", [2] * 7, 4)
    engine.add_request("c", [3] * 9, 4)
    engine.cancel("b")
    engine.step()
    engine.cancel("a")
    engine.run()
    assert engine.requests["a"].cancelled and engine.requests["b"].cancelled
    assert len(engine.requests["c"].generated) == 4
    assert not cache.requests and not cache.retired
    cache.assert_invariants()
