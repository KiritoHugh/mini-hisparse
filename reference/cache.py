"""Optional whole-page replacement and bounded-window reference."""

from dataclasses import dataclass

from mini_hisparse.allocator import DeviceAllocator
from mini_hisparse.config import CacheConfig, CapacityError
from mini_hisparse.storage import HostStore, Location, Page, PageKey, PageLease


@dataclass(frozen=True)
class PagedConfig(CacheConfig):
    chunk_pages: int = 4
    overlap: bool = True
    eviction_policy: str = "fair_lru"

    def __post_init__(self):
        super().__post_init__()
        if self.chunk_pages < 1:
            raise ValueError("chunk_pages must be positive")
        if self.eviction_policy not in ("legacy", "lru", "fair_lru"):
            raise ValueError("invalid page eviction policy")


@dataclass(frozen=True)
class WindowPlan:
    width: int
    pipelined: bool
    union_pages: int
    misses: int
    pinned_outside: int
    live_slot_limit: int
    hot_slot_target: int


class PagedKVCache:
    def __init__(self, config: PagedConfig):
        self.config = config
        self.allocator = DeviceAllocator(config)
        self.storage = HostStore(config, self.allocator)
        self.pool, self.device = self.allocator.storage, self.storage.device
        self.lookahead = set()
        self.clock = self.attention_epoch = 0
        self.last_attention = {}
        self.touched = {}

    @property
    def budget(self):
        return self.allocator.budget

    @property
    def owners(self):
        return self.storage.owners

    @property
    def index_bytes(self):
        return self.storage.index_bytes

    def index_reservation_bytes(self, max_tokens, layers):
        return self.storage.index_reservation_bytes(max_tokens, layers)

    def reserve(self, request_id, max_tokens, layers):
        self.storage.reserve(request_id, max_tokens, layers)

    def write(self, request_id, layer, k, v):
        self.storage.append(request_id, layer, k, v, self._prepare_append)

    def _prepare_append(self, page):
        if page.valid_tokens:
            return self.acquire([page.key], purpose="append")
        slot = self._slot()
        self.storage.wait(self.storage.compute, self.allocator.slots[slot].fence)
        self._bind(page, slot, Location.RESIDENT)

    def _bind(self, page, slot, location):
        self.storage.bind(page, slot, location)
        self._touch(page)

    def _detach(self, page):
        self.storage.detach(page)

    def collect(self, wait=False):
        self.storage.collect(wait)

    def release_request(self, request_id, wait=True):
        request = self.storage.requests[request_id]
        if any(self.storage.pages[key].pins for key in request.keys):
            raise RuntimeError("cannot free a request with active leases")
        for key in request.keys:
            page = self.storage.pages[key]
            if page.slot is not None:
                self._detach(page)
            self.last_attention.pop(key, None)
            self.touched.pop(key, None)
        self.storage.retire(request_id)
        self.collect(wait)

    def snapshot(self):
        return {
            **self.storage.snapshot(),
            "eviction_policy": self.config.eviction_policy,
            "fair_share_target": max(1, self.budget // max(1, len(self.storage.requests))),
        }

    def assert_invariants(self):
        self.storage.assert_invariants()

    def attention(self, query, request_id, layer, selection, backend="auto"):
        from .attention import paged_attention

        return paged_attention(query, self, request_id, layer, backend=backend, selection=selection)

    @property
    def window_pages(self):
        # A soft reserve: two windows can coexist. Residents may borrow EVERY
        # unused slot; clean pages can be reclaimed before starting a window.
        windows = 2 if self.config.overlap and self.budget > 1 else 1
        return min(self.config.chunk_pages, max(1, self.budget // windows))

    def _touch(self, page: Page):
        self.clock += 1
        self.touched[page.key] = self.clock
        if page.slot is not None:
            self.allocator.touch(page.key)

    def _slot(self):
        # Append locality matters when the whole selection cannot stay resident.
        # Sink retention is earned by actual use and expires after two rounds.
        horizon = 2 * sum(request.layers for request in self.storage.requests.values())
        anchors = {
            key
            for key in self.allocator.lru
            if (
                not self.storage.pages[key].sealed
                or (
                    key[2] == 0
                    and self.attention_epoch - self.last_attention.get(key, -1000000000) <= horizon
                )
            )
        }
        if self.config.eviction_policy != "fair_lru":
            anchors = set()
        slot = self._choose(
            self.storage.pages,
            self.config.eviction_policy,
            len(self.storage.requests),
            self.lookahead,
            anchors,
        )
        if slot is None:
            raise CapacityError("all device slots leased; split the attention working set")
        owner = self.allocator.slots[slot].owner
        if owner is not None:
            self._detach(self.storage.pages[owner])
        return slot

    def acquire(self, keys: list[PageKey], purpose="attention") -> PageLease:
        self.storage.check_stream()
        keys = list(dict.fromkeys(keys))
        if not keys:
            raise ValueError("empty working set")
        pages = [self.storage.pages[key] for key in keys]
        if any(p.valid_tokens == 0 for p in pages):
            raise ValueError("cannot read an unwritten page")
        if len(self.storage.pinned | set(keys)) > self.budget:
            raise CapacityError("working set exceeds available leases; no state changed")
        # Pin ALL hits before allocating any miss. Otherwise a later hit could
        # be evicted while resolving an earlier miss in this same selection.
        for page in pages:
            page.pins += 1
            self.storage.pinned.add(page.key)
        for page in pages:
            if purpose == "attention":
                self.last_attention[page.key] = self.attention_epoch
            if page.slot is not None:
                self.storage.stats[f"{page.location.value}_hits"] += 1
                self.storage.stats[f"{purpose}_{page.location.value}_hits"] += 1
            else:
                slot = self._slot()
                self.storage.wait(self.storage.transfer, self.allocator.slots[slot].fence)
                self.storage.wait(self.storage.transfer, page.host_ready)
                with self.storage.context(self.storage.transfer):
                    if page.valid_tokens == self.config.page_size:
                        self.pool[slot].copy_(
                            self.storage.host[page.host_slot], non_blocking=self.storage.cuda
                        )
                    else:
                        # [:, :n] is strided across K/V planes. Copy each plane
                        # contiguously to avoid pageable temporary staging.
                        for plane in (0, 1):
                            self.pool[slot, plane, : page.valid_tokens].copy_(
                                self.storage.host[page.host_slot, plane, : page.valid_tokens],
                                non_blocking=self.storage.cuda,
                            )
                page.ready = self.storage.event(self.storage.transfer)
                self._bind(page, slot, Location.HOT)
                self.storage.stats["misses"] += 1
                self.storage.stats[f"{purpose}_misses"] += 1
                self.storage.stats[f"{purpose}_h2d_bytes"] += (
                    self.storage.page_bytes * page.valid_tokens // self.config.page_size
                )
                self.storage.stats["h2d_bytes"] += (
                    self.storage.page_bytes * page.valid_tokens // self.config.page_size
                )
            self._touch(page)
        return PageLease(self.storage, pages)

    def plan(self, keys):
        """Bound live leases by the real union, subtracting other consumers."""
        keys = set(keys)
        self.attention_epoch += 1
        outside = len(self.storage.pinned - keys)
        # Existing leases cannot be released by this attention invocation, even
        # when they overlap its selection. Count all of them conservatively.
        available = self.budget - len(self.storage.pinned)
        if available < 1:
            if keys <= self.storage.pinned:
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
        misses = sum(self.storage.pages[key].slot is None for key in keys)
        live = min(len(keys), width * (2 if pipeline else 1))
        return WindowPlan(width, pipeline, len(keys), misses, outside, live, min(misses, live))

    def set_budget(self, pages: int):
        """Inject logical pressure inside a fixed allocation (does not cudaFree)."""
        self.storage.check_stream()
        if not 1 <= pages <= self.config.gpu_pages:
            raise ValueError("budget must be within the allocated pool")
        victims = [self.storage.pages[key] for key in self.owners[pages:] if key is not None]
        if any(p.pins for p in victims):
            raise CapacityError("cannot shrink over active leases")
        for page in victims:
            self._detach(page)
        self.allocator.budget = pages

    def _choose(self, pages, policy, active_requests, lookahead, anchors=()):
        available = self.allocator.free & set(range(self.budget))
        if available:
            return min(available)
        candidates = [pages[key] for key in self.allocator.lru if pages[key].pins == 0]
        if not candidates:
            return None
        if policy == "legacy":
            return min(
                candidates, key=lambda p: (p.location.value != "hot", self.touched[p.key])
            ).slot
        # Soft fair share, not a hard quota: a request can borrow the entire
        # idle pool. Current leases always override fairness and lookahead.
        share = max(1, self.budget // max(1, active_requests))
        victim = min(
            candidates,
            key=lambda p: (
                p.key in anchors,
                p.key in lookahead,
                self.allocator.occupancy[p.key[0]] <= share if policy == "fair_lru" else False,
                self.touched[p.key],
            ),
        )
        return victim.slot
