"""Logical identity survives eviction; physical slots are short-lived leases.

All device KV lives in ONE tensor. Resident/hot are policies, not allocations.
The Python control plane is deliberately single-threaded and uses one compute
stream. Every slot reuse joins its producer, last reader and host mirror events.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from math import ceil

import torch

from .allocator import DeviceAllocator, WindowPlan

PageKey = tuple[str, int, int]  # request, layer, logical page


class CapacityError(RuntimeError):
    pass


class Location(str, Enum):
    HOST = "host"
    RESIDENT = "resident"
    HOT = "hot"


@dataclass(frozen=True)
class CacheConfig:
    gpu_pages: int = 16
    host_pages: int = 1024
    page_size: int = 16
    kv_heads: int = 2
    head_dim: int = 32
    chunk_pages: int = 4
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    overlap: bool = True
    eviction_policy: str = "fair_lru"
    index_budget_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for name in ("gpu_pages", "host_pages", "page_size", "kv_heads", "head_dim", "chunk_pages"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("KV dtype must be float32, float16 or bfloat16")
        if self.eviction_policy not in ("legacy", "lru", "fair_lru"):
            raise ValueError("eviction_policy must be legacy, lru or fair_lru")
        if self.index_budget_bytes < 1:
            raise ValueError("index budget must be positive")


@dataclass
class Page:
    key: PageKey
    host_slot: int
    slot: int | None = None
    location: Location = Location.HOST
    valid_tokens: int = 0
    pins: int = 0
    touched: int = 0
    ready: object = None
    last_use: object = None
    host_ready: object = None
    request_generation: int = 0
    content_version: int = 0
    host_enqueued_version: int = 0
    host_committed_version: int = 0
    sealed: bool = False
    last_attention_epoch: int = -1000000000


@dataclass
class RequestPages:
    max_tokens: int
    layers: int
    page_count: int
    lengths: list[int]
    summaries: torch.Tensor  # per-page mean K, always on device (the small index)
    minimum: torch.Tensor
    maximum: torch.Tensor
    host_credits: int
    generation: int
    keys: list[PageKey] = field(default_factory=list)


class Lease:
    """Pins protect mappings on CPU; events protect bytes on the accelerator."""

    def __init__(self, cache: PagedKVCache, pages: list[Page]):
        self.cache, self.pages, self.released = cache, pages, False
        self.handles = [(p.slot, cache.allocator.slots[p.slot].generation, p.key) for p in pages]

    def _validate(self):
        self.cache._check_stream()
        if self.released:
            raise RuntimeError("lease already released")
        for handle in self.handles:
            self.cache.allocator.validate(*handle)

    def wait(self):
        self._validate()
        for page in self.pages:
            self.cache._wait(self.cache.compute, page.ready)

    def release(self):
        self._validate()
        done = self.cache._event(self.cache.compute)
        for page in self.pages:
            page.last_use = done
            page.pins -= 1
            if page.pins == 0:
                self.cache.pinned.discard(page.key)
        self.released = True

    def __enter__(self):
        self.wait()
        return self

    def __exit__(self, *_):
        self.release()


class PagedKVCache:
    def __init__(self, config: CacheConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.cuda = self.device.type == "cuda"
        if self.cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use device='cpu' for the reference path")
        shape = (2, config.page_size, config.kv_heads, config.head_dim)
        self.allocator = DeviceAllocator(config)
        self.pool = self.allocator.storage
        self.host = torch.empty(
            (config.host_pages, *shape), dtype=config.dtype, pin_memory=self.cuda
        )
        self.compute = torch.cuda.current_stream(self.device) if self.cuda else None
        self.transfer = torch.cuda.Stream(device=self.device) if self.cuda else None
        self.pages: dict[PageKey, Page] = {}
        self.requests: dict[str, RequestPages] = {}
        self.free_host = list(reversed(range(config.host_pages)))
        self.stats: Counter = Counter()
        self.clock = 0
        self.pinned: set[PageKey] = set()
        self.lookahead: set[PageKey] = set()
        self.retired: list[tuple[object, RequestPages, list[Page]]] = []
        self.request_generation = 0
        self.attention_epoch = 0

    @property
    def budget(self):
        return self.allocator.budget

    @property
    def owners(self):
        return [slot.owner for slot in self.allocator.slots]

    @property
    def index_bytes(self):
        requests = [*self.requests.values(), *(r for _, r, _ in self.retired)]
        return sum(
            t.numel() * t.element_size()
            for r in requests
            for t in (r.summaries, r.minimum, r.maximum)
        )

    def index_reservation_bytes(self, max_tokens, layers):
        return (
            ceil(max_tokens / self.config.page_size)
            * layers
            * self.config.kv_heads
            * self.config.head_dim
            * (4 + 2 * self.pool.element_size())
        )

    @property
    def available_host_credits(self):
        return len(self.free_host) - sum(r.host_credits for r in self.requests.values())

    def _context(self, stream):
        return torch.cuda.stream(stream) if self.cuda else nullcontext()

    def _event(self, stream):
        if not self.cuda:
            return None
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def _wait(self, stream, event):
        if event is not None:
            stream.wait_event(event)

    def _check_stream(self):
        if self.cuda and torch.cuda.current_stream(self.device) != self.compute:
            raise RuntimeError("cache operations must use the construction compute stream")

    @property
    def page_bytes(self):
        return self.pool[0].numel() * self.pool.element_size()

    @property
    def window_pages(self):
        # A soft reserve: two windows can coexist. Residents may borrow EVERY
        # unused slot; clean pages can be reclaimed before starting a window.
        windows = 2 if self.config.overlap and self.budget > 1 else 1
        return min(self.config.chunk_pages, max(1, self.budget // windows))

    def reserve(self, request_id: str, max_tokens: int, layers: int):
        """Reserve host backing for the whole admitted lifetime, never GPU quota."""
        self._check_stream()
        self.collect()
        if request_id in self.requests:
            raise ValueError(f"duplicate request {request_id}")
        if max_tokens < 1 or layers < 1:
            raise ValueError("max_tokens and layers must be positive")
        n = ceil(max_tokens / self.config.page_size)
        if n * layers > self.available_host_credits:
            raise CapacityError(
                f"need {n * layers} host credits; have {self.available_host_credits}"
            )
        index_bytes = self.index_reservation_bytes(max_tokens, layers)
        if self.index_bytes + index_bytes > self.config.index_budget_bytes:
            raise CapacityError("GPU index budget exhausted; defer request admission")
        summaries = torch.zeros(
            (layers, n, self.config.kv_heads, self.config.head_dim),
            device=self.device,
            dtype=torch.float32,
        )
        request = RequestPages(
            max_tokens,
            layers,
            n,
            [0] * layers,
            summaries,
            # Bounds are actual K extrema and exactly representable in KV dtype.
            torch.full_like(summaries, torch.inf, dtype=self.config.dtype),
            torch.full_like(summaries, -torch.inf, dtype=self.config.dtype),
            n * layers,
            self.request_generation,
        )
        self.request_generation += 1
        self.requests[request_id] = request

    def _materialize(self, request_id, layer, index):
        key = (request_id, layer, index)
        if key not in self.pages:
            request = self.requests[request_id]
            assert request.host_credits > 0 and self.free_host
            request.host_credits -= 1
            request.keys.append(key)
            self.pages[key] = Page(key, self.free_host.pop(), request_generation=request.generation)
            self.stats["materialized_pages"] += 1
        return self.pages[key]

    def _touch(self, page: Page):
        self.clock += 1
        page.touched = self.clock
        if page.slot is not None:
            self.allocator.touch(page.key)

    def _detach(self, page: Page):
        assert page.slot is not None and page.pins == 0
        slot = page.slot
        assert page.host_enqueued_version == page.content_version
        # GPU slot becomes logically free immediately, but reuse is fenced.
        # In particular, a queued D2H copy still READS this slot!
        for event in (page.ready, page.last_use, page.host_ready):
            self._wait(self.transfer, event)
        self.allocator.release(slot, self._event(self.transfer))
        page.slot, page.location = None, Location.HOST
        self.stats["evictions"] += 1

    def _slot(self):
        # Append locality matters when the whole selection cannot stay resident.
        # Sink retention is earned by actual use and expires after two rounds.
        horizon = 2 * sum(request.layers for request in self.requests.values())
        anchors = {
            key
            for key in self.allocator.lru
            if (
                not self.pages[key].sealed
                or (
                    key[2] == 0
                    and self.attention_epoch - self.pages[key].last_attention_epoch <= horizon
                )
            )
        }
        if self.config.eviction_policy != "fair_lru":
            anchors = set()
        slot = self.allocator.choose(
            self.pages, self.config.eviction_policy, len(self.requests), self.lookahead, anchors
        )
        if slot is None:
            raise CapacityError("all device slots leased; split the attention working set")
        owner = self.allocator.slots[slot].owner
        if owner is not None:
            self._detach(self.pages[owner])
        return slot

    def _bind(self, page: Page, slot: int, location: Location):
        page.slot, page.location = slot, location
        page.last_use = None
        self.allocator.bind(slot, page.key)
        self._touch(page)
        self.stats["peak_device_pages"] = max(
            self.stats["peak_device_pages"], len(self.allocator.lru)
        )

    def acquire(self, keys: list[PageKey], purpose="attention") -> Lease:
        self._check_stream()
        keys = list(dict.fromkeys(keys))
        if not keys:
            raise ValueError("empty working set")
        pages = [self.pages[key] for key in keys]
        if any(p.valid_tokens == 0 for p in pages):
            raise ValueError("cannot read an unwritten page")
        if len(self.pinned | set(keys)) > self.budget:
            raise CapacityError("working set exceeds available leases; no state changed")
        # Pin ALL hits before allocating any miss. Otherwise a later hit could
        # be evicted while resolving an earlier miss in this same selection.
        for page in pages:
            page.pins += 1
            self.pinned.add(page.key)
        for page in pages:
            if purpose == "attention":
                page.last_attention_epoch = self.attention_epoch
            if page.slot is not None:
                self.stats[f"{page.location.value}_hits"] += 1
                self.stats[f"{purpose}_{page.location.value}_hits"] += 1
            else:
                slot = self._slot()
                self._wait(self.transfer, self.allocator.slots[slot].fence)
                self._wait(self.transfer, page.host_ready)
                with self._context(self.transfer):
                    if page.valid_tokens == self.config.page_size:
                        self.pool[slot].copy_(self.host[page.host_slot], non_blocking=self.cuda)
                    else:
                        # [:, :n] is strided across K/V planes. Copy each plane
                        # contiguously to avoid pageable temporary staging.
                        for plane in (0, 1):
                            self.pool[slot, plane, : page.valid_tokens].copy_(
                                self.host[page.host_slot, plane, : page.valid_tokens],
                                non_blocking=self.cuda,
                            )
                page.ready = self._event(self.transfer)
                self._bind(page, slot, Location.HOT)
                self.stats["misses"] += 1
                self.stats[f"{purpose}_misses"] += 1
                self.stats[f"{purpose}_h2d_bytes"] += (
                    self.page_bytes * page.valid_tokens // self.config.page_size
                )
                self.stats["h2d_bytes"] += (
                    self.page_bytes * page.valid_tokens // self.config.page_size
                )
            self._touch(page)
        return Lease(self, pages)

    def plan(self, keys):
        """Bound live leases by the real union, subtracting other consumers."""
        keys = set(keys)
        self.attention_epoch += 1
        outside = len(self.pinned - keys)
        # Existing leases cannot be released by this attention invocation, even
        # when they overlap its selection. Count all of them conservatively.
        available = self.budget - len(self.pinned)
        if available < 1:
            if keys <= self.pinned:
                return WindowPlan(
                    min(self.config.chunk_pages, len(keys)),
                    False,
                    len(keys),
                    0,
                    outside,
                    len(keys),
                    0,
                )
            raise CapacityError("no available window; all capacity leased by other consumers")
        pipeline = self.config.overlap and available > 1
        width = min(self.config.chunk_pages, max(1, available // (2 if pipeline else 1)))
        misses = sum(self.pages[key].slot is None for key in keys)
        live = min(len(keys), width * (2 if pipeline else 1))
        return WindowPlan(width, pipeline, len(keys), misses, outside, live, min(misses, live))

    @torch.inference_mode()
    def write(self, request_id: str, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Append projected, RoPE-applied K/V; eagerly mirror ONLY the new rows."""
        self._check_stream()
        request = self.requests[request_id]
        cfg = self.config
        if not 0 <= layer < request.layers:
            raise ValueError("invalid layer")
        expected = (cfg.kv_heads, cfg.head_dim)
        if k.ndim != 3 or k.shape != v.shape or tuple(k.shape[1:]) != expected:
            raise ValueError(f"expected matching [tokens, {expected[0]}, {expected[1]}] K/V")
        if k.device != self.pool.device or v.device != self.pool.device:
            raise ValueError("K/V must be on the cache device")
        if k.dtype != cfg.dtype or v.dtype != cfg.dtype:
            raise ValueError("K/V dtype must match the cache")
        start = request.lengths[layer]
        if start + len(k) > request.max_tokens:
            raise CapacityError("request exceeds its reserved host capacity")
        if any(key[0] == request_id and key[1] == layer for key in self.pinned):
            raise RuntimeError("cannot append while this layer has active attention leases")
        if len(self.pinned) >= self.budget:
            raise CapacityError("no slot available for append")
        offset = 0
        while offset < len(k):
            position = start + offset
            index, within = divmod(position, cfg.page_size)
            page = self._materialize(request_id, layer, index)
            count = min(cfg.page_size - within, len(k) - offset)
            lease = None
            if page.valid_tokens:
                lease = self.acquire([page.key], purpose="append")
                lease.wait()
            else:
                slot = self._slot()
                self._wait(self.compute, self.allocator.slots[slot].fence)
                self._bind(page, slot, Location.RESIDENT)
            # Join previous reads and host mirrors before modifying the page.
            for event in (page.ready, page.last_use, page.host_ready):
                self._wait(self.compute, event)
            target = self.pool[page.slot]
            target[0, within : within + count].copy_(k[offset : offset + count])
            target[1, within : within + count].copy_(v[offset : offset + count])
            page.valid_tokens = within + count
            page.content_version += 1
            page.sealed = page.valid_tokens == cfg.page_size
            request.summaries[layer, index].copy_(target[0, : page.valid_tokens].float().mean(0))
            new_keys = target[0, within : within + count].float()
            request.minimum[layer, index].copy_(
                torch.minimum(request.minimum[layer, index], new_keys.amin(0))
            )
            request.maximum[layer, index].copy_(
                torch.maximum(request.maximum[layer, index], new_keys.amax(0))
            )
            page.ready = self._event(self.compute)
            page.location = Location.RESIDENT
            self._wait(self.transfer, page.ready)
            with self._context(self.transfer):
                if within == 0 and count == cfg.page_size:
                    self.host[page.host_slot].copy_(target, non_blocking=self.cuda)
                else:
                    for plane in (0, 1):
                        self.host[page.host_slot, plane, within : within + count].copy_(
                            target[plane, within : within + count], non_blocking=self.cuda
                        )
            page.host_ready = self._event(self.transfer)
            page.host_enqueued_version = page.content_version
            if not self.cuda:
                page.host_committed_version = page.content_version
            self.stats["d2h_bytes"] += self.page_bytes * count // cfg.page_size
            if lease is not None:
                lease.release()
            self._after_write(page)
            offset += count
        request.lengths[layer] += len(k)

    def _after_write(self, page):
        """Optional per-page policy hook after the mirror has been submitted."""

    def set_budget(self, pages: int):
        """Inject logical pressure inside a fixed allocation (does not cudaFree)."""
        self._check_stream()
        if not 1 <= pages <= self.config.gpu_pages:
            raise ValueError("budget must be within the allocated pool")
        victims = [self.pages[key] for key in self.owners[pages:] if key is not None]
        if any(p.pins for p in victims):
            raise CapacityError("cannot shrink over active leases")
        for page in victims:
            self._detach(page)
        self.allocator.budget = pages

    def release_request(self, request_id: str, wait=True):
        self._check_stream()
        request = self.requests[request_id]
        if any(self.pages[key].pins for key in request.keys):
            raise RuntimeError("cannot free a request with active leases")
        retired = []
        for key in request.keys:
            page = self.pages[key]
            if page.slot is not None:
                self._detach(page)
            for event in (page.ready, page.last_use, page.host_ready):
                self._wait(self.transfer, event)
            retired.append(self.pages.pop(key))
        del self.requests[request_id]
        self.retired.append((self._event(self.transfer), request, retired))
        self.stats["retired_requests"] += 1
        self.collect(wait=wait)

    def collect(self, wait=False):
        """Publish reusable host slots only AFTER all in-flight users finish.

        Normal scheduler polls never synchronize the CPU. When idle or shutting
        down, wait drains retirement events, rather than synchronizing a device.
        """
        pending = []
        for event, request, pages in self.retired:
            if wait and event is not None:
                event.synchronize()
            if event is None or event.query():
                self.free_host.extend(page.host_slot for page in pages)
                self.stats["reclaimed_requests"] += 1
            else:
                pending.append((event, request, pages))
        self.retired = pending

    def page_state(self, key):
        page = self.pages[key]
        backup_pending = page.host_ready is not None and not page.host_ready.query()
        if not backup_pending:
            page.host_committed_version = page.host_enqueued_version
        loading = page.ready is not None and not page.ready.query()
        return {
            "location": page.location.value,
            "loading": loading,
            "backup_pending": backup_pending,
            "content_version": page.content_version,
            "host_committed_version": page.host_committed_version,
            "sealed": page.sealed,
            "request_generation": page.request_generation,
            "slot_generation": self.allocator.slots[page.slot].generation
            if page.slot is not None
            else None,
        }

    def synchronize(self):
        if self.cuda:
            self.compute.synchronize()
            self.transfer.synchronize()

    def host_kv(self, request_id: str, layer: int):
        """Blocking diagnostic/oracle read, never called by the decode path."""
        self.synchronize()
        request = self.requests[request_id]
        chunks = [
            self.host[
                self.pages[(request_id, layer, i)].host_slot,
                :,
                : self.pages[(request_id, layer, i)].valid_tokens,
            ]
            for i in range(ceil(request.lengths[layer] / self.config.page_size))
        ]
        if not chunks:
            raise ValueError("layer has no KV")
        kv = torch.cat(chunks, dim=1)
        return kv[0], kv[1]

    def snapshot(self):
        counts = Counter(p.location.value for p in self.pages.values() if p.valid_tokens)
        return {
            "budget_pages": self.budget,
            "allocated_device_bytes": self.pool.numel() * self.pool.element_size(),
            "allocated_host_bytes": self.host.numel() * self.host.element_size(),
            "index_bytes": self.index_bytes,
            "index_budget_bytes": self.config.index_budget_bytes,
            "host_materialized_pages": len(self.pages),
            "host_unused_credits": sum(r.host_credits for r in self.requests.values()),
            "host_available_credits": self.available_host_credits,
            "retiring_requests": len(self.retired),
            "eviction_policy": self.config.eviction_policy,
            "request_device_pages": dict(self.allocator.occupancy),
            "fair_share_target": max(1, self.budget // max(1, len(self.requests))),
            "locations": dict(counts),
            "stats": dict(self.stats),
        }

    def assert_invariants(self):
        occupied = [key for key in self.owners if key is not None]
        assert len(occupied) == len(set(occupied))
        assert all(key is None for key in self.owners[self.budget :])
        retired_pages = [p for _, _, pages in self.retired for p in pages]
        assert len(self.free_host) + len(self.pages) + len(retired_pages) == self.config.host_pages
        host_slots = [p.host_slot for p in [*self.pages.values(), *retired_pages]]
        assert len(set(host_slots + self.free_host)) == self.config.host_pages
        assert self.available_host_credits >= 0
        assert self.pinned == {p.key for p in self.pages.values() if p.pins}
        assert set(self.allocator.lru) == set(occupied)
        assert self.allocator.free == {i for i, key in enumerate(self.owners) if key is None}
        assert +self.allocator.occupancy == Counter(key[0] for key in occupied)
        assert self.index_bytes <= self.config.index_budget_bytes
        for request in self.requests.values():
            assert len(request.keys) + request.host_credits == request.layers * request.page_count
        for key, page in self.pages.items():
            assert page.pins >= 0
            assert 0 <= page.valid_tokens <= self.config.page_size
            if page.slot is None:
                assert page.location == Location.HOST
            else:
                assert self.owners[page.slot] == key
                assert page.location in (Location.RESIDENT, Location.HOT)
            assert page.pins == 0 or page.slot is not None
            assert page.request_generation == self.requests[key[0]].generation
            assert page.host_committed_version <= page.host_enqueued_version == page.content_version
