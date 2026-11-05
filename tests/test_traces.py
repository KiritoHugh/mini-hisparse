import copy

import pytest
import torch

from experiments.replay import prepare, replay
from experiments.traces import analyze, load_trace, validate_trace
from mini_hisparse.config import CacheConfig
from mini_hisparse.hybrid import HotConfig


def fixture_trace():
    torch.manual_seed(12)
    return {
        "schema_version": 1,
        "provenance": {"family": "unit_fixture", "native_sparse": False},
        "prefills": [
            {"request": "a", "layer": 3, "k": torch.randn(11, 1, 8), "v": torch.randn(11, 1, 8)}
        ],
        "events": [
            {
                "request": "a",
                "layer": 3,
                "position": pos,
                "q": torch.randn(2, 8),
                "k": torch.randn(1, 1, 8),
                "v": torch.randn(1, 1, 8),
            }
            for pos in range(11, 14)
        ],
    }


def test_trace_validation_safe_round_trip_and_replay(tmp_path):
    trace = fixture_trace()
    path = tmp_path / "trace.pt"
    torch.save(trace, path)
    loaded = load_trace(path)
    summary = analyze(loaded, page_size=4, budget=1, device="cpu")
    assert summary["events"] == 3
    for mode in ["quest", "mean", "oracle_token"]:
        metrics = summary["summary"][mode]
        assert 0 < metrics["retained_attention_mass"] <= 1
        events = prepare(loaded, 4, 1, mode, "cpu")
        union = max(len({t for row in e["selection"].positions for t in row}) for e in events)
        for hot in (None, HotConfig(union)):
            result = replay(
                loaded,
                events,
                CacheConfig(
                    gpu_pages=hot.admission_blocks(4) if hot else 1,
                    page_size=4,
                    kv_heads=1,
                    head_dim=8,
                    device="cpu",
                ),
                hot,
            )
            assert result["max_storage_abs_error"] < 1e-5
    broken = copy.deepcopy(trace)
    broken["events"][1]["position"] += 1
    with pytest.raises(ValueError, match="monotonic"):
        validate_trace(broken)


def test_native_trace_uses_exact_logged_positions():
    trace = fixture_trace()
    for event in trace["events"]:
        event["native_topk"] = [0, 3, -1, 3]
    events = prepare(trace, 4, 2, "native", "cpu")
    assert all(event["selection"].positions == ((0, 3), (0, 3)) for event in events)


def test_trace_rejects_unsupported_heterogeneous_layout_and_empty_trace():
    trace = fixture_trace()
    trace["prefills"].append(
        {"request": "b", "layer": 3, "k": torch.randn(11, 2, 8), "v": torch.randn(11, 2, 8)}
    )
    with pytest.raises(ValueError, match="homogeneous"):
        validate_trace(trace)
    trace = fixture_trace()
    trace["events"] = []
    with pytest.raises(ValueError, match="nonempty"):
        validate_trace(trace)
