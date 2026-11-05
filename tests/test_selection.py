import pytest
import torch

from mini_hisparse.attention import paged_attention
from mini_hisparse.cache import CacheConfig, PagedKVCache
from mini_hisparse.selection import Selection, quest_selection, selection_oracle


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("budget", [1, 3])
def test_exact_tokens_head_masks_and_empty_windows(device, budget):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(91)
    cache = PagedKVCache(
        CacheConfig(
            gpu_pages=budget, host_pages=16, page_size=4, kv_heads=2, head_dim=8, device=device
        )
    )
    cache.reserve("r", 19, 1)
    k, v = [torch.randn(19, 2, 8, device=device) for _ in range(2)]
    q = torch.randn(4, 8, device=device)
    cache.write("r", 0, k, v)
    selection = Selection.from_tokens([[0, 0, -1], [18], [5, 6], [12, 18]], 19, 4, 4)
    expected, _ = selection_oracle(q, k, v, selection)
    for backend in ["torch", "triton"] if cache.cuda else ["torch"]:
        actual = paged_attention(q, cache, "r", 0, backend=backend, selection=selection)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    cache.assert_invariants()


def test_quest_bound_handles_mean_cancellation_and_incremental_tail():
    cache = PagedKVCache(
        CacheConfig(gpu_pages=1, host_pages=4, page_size=2, kv_heads=1, head_dim=2, device="cpu")
    )
    cache.reserve("r", 8, 1)
    k = torch.tensor([[[10.0, 0.0]], [[-10.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]]])
    cache.write("r", 0, k[:4], k[:4])
    cache.write("r", 0, k[4:], k[4:])
    r = cache.requests["r"]
    selection = quest_selection(
        torch.tensor([[1.0, 0.0]]), r.minimum[0], r.maximum[0], 5, 2, 2, keep_sink=False
    )
    assert selection.pages == [0, 2]  # mean scoring would choose page 1
    torch.testing.assert_close(r.minimum[0, 0], k[:2].amin(0))
    torch.testing.assert_close(r.maximum[0, 0], k[:2].amax(0))


def test_selection_rejects_invalid_and_shared_native_indexer_rows():
    s = Selection.from_tokens([3, 1, 1, -1], 4, 2, 6, source="native_fixture")
    assert all(row == (1, 3) for row in s.positions)
    for rows in [[4], [-2], [-1], [1.2], []]:
        with pytest.raises((ValueError, TypeError)):
            Selection.from_tokens(rows, 4, 2, 2)
