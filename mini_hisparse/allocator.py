"""Physical capacity and leases, independent of host KV contents."""

from collections import Counter, OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class DeviceSlot:
    owner: tuple | None = None
    generation: int = 0
    fence: object = None


@dataclass(frozen=True)
class WindowPlan:
    width: int
    pipelined: bool
    union_pages: int
    misses: int
    pinned_outside: int
    live_slot_limit: int
    hot_slot_target: int


class DeviceAllocator:
    def __init__(self, config):
        self.storage = torch.empty(
            (config.gpu_pages, 2, config.page_size, config.kv_heads, config.head_dim),
            device=config.device,
            dtype=config.dtype,
        )
        self.slots = [DeviceSlot() for _ in range(config.gpu_pages)]
        self.free = set(range(config.gpu_pages))
        # Empty blocks and readable-but-unpinned blocks are distinct. Both
        # provide capacity; reuse of the latter must revoke the old mapping.
        self.reclaimable = OrderedDict()
        self.lru = OrderedDict()
        self.occupancy = Counter()
        self.budget = config.gpu_pages

    def touch(self, key):
        self.lru.move_to_end(key)

    def bind(self, slot, key):
        assert slot in self.free and slot < self.budget
        self.free.remove(slot)
        frame = self.slots[slot]
        frame.generation += 1
        frame.owner = key
        self.lru[key] = slot
        self.occupancy[key[0]] += 1

    def release(self, slot, fence):
        frame = self.slots[slot]
        assert frame.owner is not None
        key = frame.owner
        self.reclaimable.pop(slot, None)
        del self.lru[key]
        self.occupancy[key[0]] -= 1
        frame.owner, frame.fence = None, fence
        self.free.add(slot)

    def unpin(self, slot, on_reuse):
        assert self.slots[slot].owner is not None and slot not in self.free
        self.reclaimable.setdefault(slot, on_reuse)

    def reclaim(self, slot):
        """Run the ownership callback only when the block is actually taken."""
        self.reclaimable[slot](slot)
        assert slot in self.free and slot not in self.reclaimable

    def choose(self, pages, policy, active_requests, lookahead, anchors=()):
        available = self.free & set(range(self.budget))
        if available:
            return min(available)
        candidates = [pages[key] for key in self.lru if pages[key].pins == 0]
        if not candidates:
            return None
        if policy == "legacy":
            return min(candidates, key=lambda p: (p.location.value != "hot", p.touched)).slot
        # Soft fair share, not a hard quota: a request can borrow the entire
        # idle pool. Current leases always override fairness and lookahead.
        share = max(1, self.budget // max(1, active_requests))
        victim = min(
            candidates,
            key=lambda p: (
                p.key in anchors,
                p.key in lookahead,
                self.occupancy[p.key[0]] <= share if policy == "fair_lru" else False,
                p.touched,
            ),
        )
        return victim.slot

    def validate(self, slot, generation, owner):
        frame = self.slots[slot]
        if frame.owner != owner or frame.generation != generation:
            raise RuntimeError("stale device-slot generation (ABA)")
