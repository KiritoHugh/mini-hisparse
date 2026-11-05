"""Offline model traces and selection diagnostics, kept outside runtime timing."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from math import ceil
from statistics import mean

import torch

from mini_hisparse.selection import Selection, quest_selection, selection_oracle
from reference.cache import PagedConfig as CacheConfig
from reference.cache import PagedKVCache


class Recorder:
    def __init__(self, layers):
        self.layers = set(layers)
        self.prefills, self.events, self.lengths = [], [], {}

    def prefill(self, request, layer, k, v):
        if layer in self.layers:
            self.prefills.append({"request": request, "layer": layer, "k": k.cpu(), "v": v.cpu()})
            self.lengths[(request, layer)] = len(k)

    def decode(self, request, layer, query, k, v):
        if layer in self.layers:
            key = (request, layer)
            self.events.append(
                {
                    "request": request,
                    "layer": layer,
                    "position": self.lengths[key],
                    "q": query.cpu(),
                    "k": k.cpu(),
                    "v": v.cpu(),
                }
            )
            self.lengths[key] += len(k)


@torch.inference_mode()
def capture(model, tokenizer, samples, max_context=1024, steps=8, layers=None):
    """Dense teacher trajectory: frozen Q/K/V lets every storage policy see the
    same accesses. It does NOT measure sparse autoregressive quality or DSA.
    """
    cfg = model.config
    layers = list(range(cfg.num_hidden_layers)) if layers is None else list(layers)
    if not layers or any(i < 0 or i >= cfg.num_hidden_layers for i in layers):
        raise ValueError("invalid capture layers")
    if max_context < 64 or steps < 1:
        raise ValueError("need context >=64 and at least one decode step")
    recorder, sources = Recorder(layers), []
    for sample in samples:
        context = tokenizer.encode(sample["context"], add_special_tokens=False)
        instruction = f"\n\nTask: {sample['question']}"
        overhead = len(tokenizer.encode(instruction, add_special_tokens=False)) + 64
        available = max_context - overhead
        if available < 16:
            raise ValueError("question/template leaves no context budget")
        original_length = len(context)
        if len(context) > available:
            context = context[: available // 2] + context[-(available - available // 2) :]
        text = tokenizer.decode(context) + instruction
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        count = ceil((len(prompt) + steps) / 16) * cfg.num_hidden_layers
        cache = PagedKVCache(
            CacheConfig(
                gpu_pages=count,
                host_pages=count,
                page_size=16,
                kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                device=str(model.device),
                dtype=model.dtype,
            )
        )
        request = sample["id"]
        cache.reserve(request, len(prompt) + steps, cfg.num_hidden_layers)
        logits = model.prefill(prompt, cache, request, observer=recorder)
        generated = []
        for _ in range(steps):
            token = int(logits.argmax())
            generated.append(token)
            logits = model.decode(token, request, cache, observer=recorder)
        sources.append(
            {
                "request": request,
                "source": sample["source"],
                "task": sample["task"],
                "prompt_tokens": len(prompt),
                "original_context_tokens": original_length,
                "truncated": original_length > available,
                "truncation": "context head+tail",
                "prompt_ids": prompt,
                "prompt_sha256": sha256(str(prompt).encode()).hexdigest(),
                "generated_ids": generated,
                "steps": steps,
            }
        )
        cache.release_request(request)
    return {
        "schema_version": 1,
        "provenance": {
            "family": "dense_qwen_teacher",
            "native_sparse": False,
            "torch": str(torch.__version__),
            "model_config": asdict(cfg),
            "layers": layers,
            "samples": sources,
        },
        "prefills": recorder.prefills,
        "events": recorder.events,
    }


def load_trace(path):
    trace = torch.load(path, map_location="cpu", weights_only=True)
    validate_trace(trace)
    return trace


def validate_trace(trace):
    if trace.get("schema_version") != 1 or not trace.get("provenance"):
        raise ValueError("trace requires schema_version=1 and provenance")
    if not trace.get("prefills") or not trace.get("events"):
        raise ValueError("trace needs nonempty prefill and decode records")
    lengths, shapes, dtypes, starts = {}, {}, {}, {}
    request_layers = {}
    for record in trace["prefills"]:
        key = record["request"], record["layer"]
        k, v = record["k"], record["v"]
        if (
            key in lengths
            or k.ndim != 3
            or k.shape != v.shape
            or min(k.shape) < 1
            or k.dtype != v.dtype
            or k.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        ):
            raise ValueError("invalid or duplicate prefill")
        lengths[key], shapes[key] = len(k), k.shape[1:]
        dtypes[key], starts[key] = k.dtype, len(k)
        request_layers.setdefault(key[0], set()).add(key[1])
    if (
        len({tuple(shape) for shape in shapes.values()}) != 1
        or len(set(dtypes.values())) != 1
        or len({tuple(sorted(layers)) for layers in request_layers.values()}) != 1
    ):
        raise ValueError(
            "replay requires homogeneous GQA dimensions, dtype and captured layer sets"
        )
    for request in request_layers:
        if len({n for (r, _), n in starts.items() if r == request}) != 1:
            raise ValueError("layers of a request must have the same prefix length")
    for record in trace["events"]:
        key = record["request"], record["layer"]
        if key not in lengths or record["position"] != lengths[key]:
            raise ValueError("non-monotonic, missing, or stale trace event")
        q, k, v = record["q"], record["k"], record["v"]
        if (
            k.shape != v.shape
            or k.shape != (1, *shapes[key])
            or q.ndim != 2
            or q.shape[-1] != k.shape[-1]
            or q.shape[0] % k.shape[1]
            or q.shape[0] < 1
            or k.dtype != dtypes[key]
            or v.dtype != dtypes[key]
        ):
            raise ValueError("incompatible trace Q/K/V shapes")
        if "native_topk" in record:
            Selection.from_tokens(record["native_topk"], lengths[key] + 1, 16, q.shape[0])
        lengths[key] += 1
    for request in request_layers:
        if len({n for (r, _), n in lengths.items() if r == request}) != 1:
            raise ValueError("incomplete decode step: captured layers have different lengths")
    return trace


def choose(query, k, page_size, budget, mode, native_topk=None):
    if budget < 1 or page_size < 1:
        raise ValueError("selection budget and page size must be positive")
    length, heads = len(k), query.shape[0]
    count = ceil(length / page_size)
    if mode == "native":
        if native_topk is None:
            raise ValueError("native selection requires captured model indexer positions")
        return Selection.from_tokens(native_topk, length, page_size, heads, "native")
    if mode == "oracle_token":
        groups = heads // k.shape[1]
        scores = torch.einsum("hd,thd->ht", query.float(), k.repeat_interleave(groups, 1).float())
        return Selection.from_tokens(
            scores.topk(min(budget * page_size, length), -1).indices, length, page_size, heads, mode
        )
    chunks = [k[i : i + page_size].float() for i in range(0, length, page_size)]
    if mode == "quest":
        return quest_selection(
            query,
            torch.stack([x.amin(0) for x in chunks]),
            torch.stack([x.amax(0) for x in chunks]),
            length,
            page_size,
            budget,
        )
    if mode == "mean":
        grouped = query.float().reshape(k.shape[1], heads // k.shape[1], -1).mean(1)
        scores = torch.einsum("hd,phd->p", grouped, torch.stack([x.mean(0) for x in chunks]))
        required = [count - 1] if budget == 1 else [0, count - 1]
        scores[required] = torch.inf
        return Selection.from_pages(
            scores.topk(min(budget, count)).indices.tolist(), length, page_size, heads, mode
        )
    raise ValueError("unknown selector")


@torch.inference_mode()
def analyze(trace, page_size=16, budget=4, device="cuda:0"):
    validate_trace(trace)
    histories = {
        (r["request"], r["layer"]): (r["k"].to(device), r["v"].to(device))
        for r in trace["prefills"]
    }
    rows, previous = [], {}
    for event in trace["events"]:
        key = event["request"], event["layer"]
        k, v = [
            torch.cat((old, event[name].to(device)))
            for old, name in zip(histories[key], ("k", "v"))
        ]
        histories[key] = k, v
        q = event["q"].to(device)
        modes = ["native"] if "native_topk" in event else ["mean", "quest", "oracle_token"]
        for mode in modes:
            selection = choose(q, k, page_size, budget, mode, event.get("native_topk"))
            _, metrics = selection_oracle(q, k, v, selection)
            old = previous.get((*key, mode))
            sets = [set(row) for row in selection.positions]
            jaccard = mean(len(a & b) / len(a | b) for a, b in zip(old, sets)) if old else None
            previous[(*key, mode)] = sets
            rows.append(
                {
                    "request": key[0],
                    "layer": key[1],
                    "position": event["position"],
                    "selector": mode,
                    "temporal_jaccard": jaccard,
                    "page_union_amplification": len(selection.pages)
                    / min(budget, ceil(len(k) / page_size)),
                    **metrics,
                }
            )
    summary = {}
    for mode in sorted({row["selector"] for row in rows}):
        subset = [r for r in rows if r["selector"] == mode]
        summary[mode] = {
            field: mean(r[field] for r in subset if r[field] is not None)
            for field in [
                "retained_attention_mass",
                "oracle_token_recall",
                "relative_output_l2",
                "selected_token_fraction",
                "union_pages",
                "temporal_jaccard",
                "page_union_amplification",
            ]
            if any(r[field] is not None for r in subset)
        }
    return {
        "provenance": trace["provenance"],
        "page_size": page_size,
        "budget_pages_per_head": budget,
        "events": len(trace["events"]),
        "summary": summary,
        "rows": rows,
    }
