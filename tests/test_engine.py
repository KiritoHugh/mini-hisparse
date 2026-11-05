import pytest
import torch

from mini_hisparse.engine import Engine
from mini_hisparse.model import MiniModel, ModelConfig
from reference.cache import PagedConfig as CacheConfig
from reference.cache import PagedKVCache


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("top_k", [None, 2])
def test_real_autoregressive_decode_matches_resident_engine(device, top_k):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(31)
    torch.set_num_threads(1)
    model = (
        MiniModel(
            ModelConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=64,
            )
        )
        .eval()
        .to(device)
    )
    outputs = []
    for budget in [64, 2]:
        cache = PagedKVCache(
            CacheConfig(
                gpu_pages=budget, host_pages=64, page_size=4, kv_heads=2, head_dim=8, device=device
            )
        )
        engine = Engine(model, cache, top_k=top_k)
        engine.add_request("a", list(range(1, 18)), 8)
        engine.add_request("b", list(range(8, 30)), 6)
        outputs.append(engine.run())
        assert all(r.prefill_count == 1 for r in engine.requests.values())
        assert all(r.decode_count == r.max_new_tokens - 1 for r in engine.requests.values())
        assert not cache.storage.requests and not any(cache.owners)
        cache.assert_invariants()
        if budget == 2:
            assert cache.storage.stats["misses"] > 0
    assert outputs[0] == outputs[1]
    if top_k is None:
        for name, prompt in [("a", list(range(1, 18))), ("b", list(range(8, 30)))]:
            history = prompt[:]
            for token in outputs[0][name]:
                assert model.prefill(history).argmax().item() == token
                history.append(token)


def test_host_backpressure_eventually_admits_waiter():
    torch.manual_seed(4)
    model = MiniModel(
        ModelConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    ).eval()
    cache = PagedKVCache(
        CacheConfig(gpu_pages=1, host_pages=4, page_size=4, kv_heads=1, head_dim=8, device="cpu")
    )
    engine = Engine(model, cache)
    for i in range(3):
        engine.add_request(str(i), [1] * 10, 5)
    engine.step()
    assert len(engine.waiting) == 2
    engine.run()
    assert all(len(r.generated) == 5 and r.prefill_count == 1 for r in engine.requests.values())


def test_qwen2_huggingface_parity(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(8)
    cfg = ModelConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
    )
    hf_cfg = transformers.Qwen2Config(**vars(cfg), attention_dropout=0, tie_word_embeddings=False)
    hf = transformers.Qwen2ForCausalLM(hf_cfg).eval()
    model = MiniModel(cfg).eval()
    model.load_state_dict(hf.state_dict(), strict=True)
    # Also exercise the actual pretrained adapter, including config migrations.
    hf.save_pretrained(tmp_path)
    loaded = MiniModel.from_pretrained(str(tmp_path), device="cpu", dtype=torch.float32)
    tokens = [1, 7, 3, 4, 9, 11, 17]
    with torch.inference_mode():
        expected = hf(torch.tensor([tokens])).logits[0, -1]
    torch.testing.assert_close(model.prefill(tokens), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(loaded.prefill(tokens), expected, atol=2e-6, rtol=2e-5)
    cache = PagedKVCache(
        CacheConfig(gpu_pages=1, host_pages=16, page_size=3, kv_heads=2, head_dim=8, device="cpu")
    )
    cache.reserve("a", 12, 2)
    model.prefill(tokens, cache, "a")
    for token in [12, 19, 5]:
        tokens.append(token)
        actual = model.decode(token, "a", cache)
        with torch.inference_mode():
            expected = hf(torch.tensor([tokens])).logits[0, -1]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
