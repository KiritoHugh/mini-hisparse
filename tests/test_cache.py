from dataclasses import replace

import pytest
import torch

from mini_hisparse.attention import dense_attention, paged_attention, select_pages
from mini_hisparse.cache import CacheConfig, CapacityError, Location, PagedKVCache


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def config(device, **changes):
    return replace(
        CacheConfig(
            gpu_pages=3,
            host_pages=48,
            page_size=4,
            kv_heads=2,
            head_dim=8,
            chunk_pages=2,
            device=device,
        ),
        **changes,
    )


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 5])
def test_partial_residency_append_and_sparse_oracle(device, overlap, budget):
    torch.manual_seed(17)
    cache = PagedKVCache(config(device, gpu_pages=budget, overlap=overlap))
    cache.reserve("a", 35, 1)
    k, v = [torch.randn(19, 2, 8, device=device) for _ in range(2)]
    cache.write("a", 0, k, v)
    pointer = cache.pool.data_ptr()
    for _ in range(10):
        new_k, new_v = [torch.randn(1, 2, 8, device=device) for _ in range(2)]
        cache.write("a", 0, new_k, new_v)
        k, v = torch.cat((k, new_k)), torch.cat((v, new_v))
        q = torch.randn(4, 8, device=device)
        # Reordered duplicates must NOT change the softmax's token multiplicity.
        pages = [0, (len(k) - 1) // 4, 1, 0]
        tokens = [
            t for page in dict.fromkeys(pages) for t in range(page * 4, min(page * 4 + 4, len(k)))
        ]
        for backend in ["torch", "triton"] if cache.cuda else ["torch"]:
            actual = paged_attention(q, cache, "a", 0, pages, backend)
            torch.testing.assert_close(
                actual, dense_attention(q, k[tokens], v[tokens]), atol=3e-6, rtol=3e-5
            )
        cache.assert_invariants()
    host_k, host_v = cache.host_kv("a", 0)
    torch.testing.assert_close(host_k, k.cpu())
    torch.testing.assert_close(host_v, v.cpu())
    assert cache.pool.data_ptr() == pointer
    assert cache.stats["peak_device_pages"] <= budget
    assert cache.stats["evictions"] > 0
    if budget < 3:
        assert cache.stats["misses"] > 0


def test_acquire_protects_later_hits_and_rejects_overcommit(device):
    cache = PagedKVCache(config(device, gpu_pages=2))
    cache.reserve("a", 12, 1)
    k, v = [torch.randn(12, 2, 8, device=device) for _ in range(2)]
    cache.write("a", 0, k, v)
    # Page 0 is missing; pages 1 and 2 are resident. Protect page 1 BEFORE
    # allocating page 0, even though page 1 is older than page 2.
    before = cache.pages[("a", 0, 1)].slot
    lease = cache.acquire([("a", 0, 0), ("a", 0, 1)])
    assert cache.pages[("a", 0, 1)].slot == before
    snapshot = list(cache.owners)
    with pytest.raises(CapacityError):
        cache.acquire([("a", 0, 2)])
    assert cache.owners == snapshot
    with pytest.raises(RuntimeError):
        cache.release_request("a")
    lease.wait()
    lease.release()
    cache.release_request("a")
    cache.assert_invariants()
    assert len(cache.free_host) == cache.config.host_pages


def test_hot_reuse_no_reload_then_dynamic_pressure(device):
    cache = PagedKVCache(config(device, gpu_pages=5))
    cache.reserve("a", 28, 1)
    k, v = [torch.randn(28, 2, 8, device=device) for _ in range(2)]
    q = torch.randn(4, 8, device=device)
    cache.write("a", 0, k, v)
    paged_attention(q, cache, "a", 0, [0])
    assert cache.pages[("a", 0, 0)].location == Location.HOT
    misses = cache.stats["misses"]
    paged_attention(q, cache, "a", 0, [0])
    assert cache.stats["misses"] == misses
    cache.set_budget(1)
    actual = paged_attention(q, cache, "a", 0)
    torch.testing.assert_close(actual, dense_attention(q, k, v), atol=3e-6, rtol=3e-5)
    cache.set_budget(5)
    cache.assert_invariants()


def test_resident_hits_never_fetch(device):
    cache = PagedKVCache(config(device, gpu_pages=5))
    cache.reserve("a", 16, 1)
    k, v = [torch.randn(13, 2, 8, device=device) for _ in range(2)]
    cache.write("a", 0, k, v)
    paged_attention(torch.randn(4, 8, device=device), cache, "a", 0)
    assert cache.stats["misses"] == cache.stats["h2d_bytes"] == 0
    assert cache.stats["resident_hits"] == 4


def test_host_admission_is_atomic_and_ids_are_reusable(device):
    cache = PagedKVCache(config(device, host_pages=4))
    cache.reserve("a", 8, 2)
    with pytest.raises(CapacityError):
        cache.reserve("b", 1, 1)
    assert "b" not in cache.requests
    cache.release_request("a")
    cache.reserve("a", 16, 1)
    cache.assert_invariants()


def test_indexer_tail_and_invalid_selections(device):
    cache = PagedKVCache(config(device))
    cache.reserve("a", 20, 1)
    k, v = [torch.randn(20, 2, 8, device=device) for _ in range(2)]
    cache.write("a", 0, k, v)
    q = torch.randn(4, 8, device=device)
    assert select_pages(cache, "a", 0, q, 1) == [4]
    chosen = select_pages(cache, "a", 0, q, 3)
    assert 0 in chosen and 4 in chosen and len(chosen) == 3
    for invalid in [[], [-1], [5]]:
        with pytest.raises(ValueError):
            paged_attention(q, cache, "a", 0, invalid)
    cache.assert_invariants()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_triton_and_non_power_of_two_page(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache = PagedKVCache(config("cuda:0", page_size=7, head_dim=24, dtype=dtype))
    cache.reserve("a", 33, 1)
    k, v = [torch.randn(33, 2, 24, device="cuda", dtype=dtype) for _ in range(2)]
    q = torch.randn(6, 24, device="cuda", dtype=dtype)
    cache.write("a", 0, k, v)
    actual = paged_attention(q, cache, "a", 0, backend="triton")
    torch.testing.assert_close(actual, dense_attention(q, k, v), atol=0.003, rtol=0.015)


@pytest.mark.cuda
def test_delayed_compute_cannot_be_overwritten_by_next_copy():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache = PagedKVCache(config("cuda:0", gpu_pages=1))
    cache.reserve("a", 8, 1)
    k, v = [torch.randn(8, 2, 8, device="cuda") for _ in range(2)]
    cache.write("a", 0, k, v)
    for _ in range(16):
        with cache.acquire([("a", 0, 0)]) as lease:
            torch.cuda._sleep(100000)
            observed = cache.pool[lease.pages[0].slot, 0].clone()
        with cache.acquire([("a", 0, 1)]):
            pass
        torch.testing.assert_close(observed, k[:4])
    cache.assert_invariants()


def test_randomized_multi_request_eviction_and_reclamation(device):
    rng = torch.Generator().manual_seed(19)
    cache = PagedKVCache(config(device, gpu_pages=3, host_pages=128))
    histories = {}
    for request_id in ["a", "b", "c"]:
        cache.reserve(request_id, 40, 2)
        for layer in range(2):
            k, v = [torch.randn(13, 2, 8, generator=rng).to(device) for _ in range(2)]
            cache.write(request_id, layer, k, v)
            histories[(request_id, layer)] = (k, v)
    for step in range(30):
        request_id = ["a", "b", "c"][step % 3]
        layer = (step // 3) % 2
        k, v = histories[(request_id, layer)]
        new_k, new_v = [torch.randn(1, 2, 8, generator=rng).to(device) for _ in range(2)]
        cache.write(request_id, layer, new_k, new_v)
        k, v = torch.cat((k, new_k)), torch.cat((v, new_v))
        histories[(request_id, layer)] = k, v
        q = torch.randn(4, 8, generator=rng).to(device)
        cache.set_budget(1 + step % 3)
        torch.testing.assert_close(
            paged_attention(q, cache, request_id, layer),
            dense_attention(q, k, v),
            atol=3e-6,
            rtol=3e-5,
        )
        cache.assert_invariants()
    for request_id in ["b", "a", "c"]:
        cache.release_request(request_id)
        cache.assert_invariants()
