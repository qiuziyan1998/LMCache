# SPDX-License-Identifier: Apache-2.0
"""Two disjoint CPU page pools shared by cache and checkpoint allocations."""

# Standard
from dataclasses import dataclass
from typing import Mapping
import threading


@dataclass(frozen=True)
class PageSlotSpec:
    """Runtime layout and one group quota shared by all allocation purposes."""

    layer_token_bytes: tuple[int, ...]
    chunk_tokens: int
    slots: int
    alignment: int = 4096

    def __post_init__(self) -> None:
        if (
            not self.layer_token_bytes
            or any(type(n) is not int or n <= 0 for n in self.layer_token_bytes)
            or any(
                type(n) is not int
                for n in (
                    self.chunk_tokens,
                    self.slots,
                    self.alignment,
                )
            )
            or self.chunk_tokens <= 0
            or self.slots <= 0
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
        ):
            raise ValueError("Invalid fixed page-slot layout or quota")

    @property
    def slot_bytes(self) -> int:
        """Return the full-chunk physical size, rounded to buffer alignment."""
        size = sum(self.layer_token_bytes) * self.chunk_tokens
        return (size + self.alignment - 1) // self.alignment * self.alignment


class PageSlotPool:
    """Manage offsets in two immutable regions without splitting free slots.

    All admission, address ownership and accounting share one lock. A failed
    batch changes no state. Offsets remain owned until their MemoryObj releases
    them through the existing reference/pin/DMA lifecycle.
    """

    def __init__(self, slab_bytes: int, specs: tuple[PageSlotSpec, ...]) -> None:
        if len(specs) != 2:
            raise ValueError("Fixed page slots require exactly two KV groups")
        self.specs = specs
        self.capacity = sum(spec.slot_bytes * spec.slots for spec in specs)
        if self.capacity > slab_bytes:
            raise ValueError("Fixed page-slot quotas exceed the CPU slab")
        self.unassigned_bytes = slab_bytes - self.capacity
        self.lock = threading.Lock()
        self.free_slots: list[list[int]] = []
        self.live: dict[int, tuple[int, int, str]] = {}
        self.payload_bytes = [0, 0]
        self.partial_slots = [0, 0]
        self.purpose_counts = [dict(cache=0, checkpoint=0, workspace=0) for _ in specs]
        offset = 0
        for spec in specs:
            self.free_slots.append(
                [offset + i * spec.slot_bytes for i in reversed(range(spec.slots))]
            )
            offset += spec.slots * spec.slot_bytes

    @property
    def total_allocated_size(self) -> int:
        """Return bytes charged to live owners, including partial-page padding."""
        with self.lock:
            return sum(
                (spec.slots - len(free)) * spec.slot_bytes
                for spec, free in zip(self.specs, self.free_slots, strict=True)
            )

    def get_capacity_bytes(self) -> tuple[int, int]:
        """Return free and usable pool bytes in one synchronized snapshot."""
        with self.lock:
            return sum(
                len(free) * spec.slot_bytes
                for spec, free in zip(self.specs, self.free_slots, strict=True)
            ), self.capacity

    def get_free_size(self) -> int:
        """Return physical slot bytes available to every allocation purpose."""
        return self.get_capacity_bytes()[0]

    def get_heap_size(self) -> int:
        """Return the usable quota, excluding the unassigned slab remainder."""
        return self.capacity

    def snapshot(self) -> dict[int, dict[str, int]]:
        """Return per-group capacity, payload and allocation-purpose counters."""
        with self.lock:
            return {
                group: dict(
                    slot_bytes=spec.slot_bytes,
                    total_slots=spec.slots,
                    free_slots=len(self.free_slots[group]),
                    allocated_bytes=(spec.slots - len(self.free_slots[group]))
                    * spec.slot_bytes,
                    logical_payload_bytes=self.payload_bytes[group],
                    padding_bytes=(spec.slots - len(self.free_slots[group]))
                    * spec.slot_bytes
                    - self.payload_bytes[group],
                    partial_slots=self.partial_slots[group],
                    **{
                        f"{purpose}_in_use": count
                        for purpose, count in self.purpose_counts[group].items()
                    },
                )
                for group, spec in enumerate(self.specs)
            }

    def shortages(self, counts: Mapping[int, int], purpose: str) -> dict[int, int]:
        """Return missing slots by group; this is a snapshot, not a reservation."""
        self._validate_purpose(purpose)
        with self.lock:
            return {
                group: max(0, count - len(self.free_slots[group]))
                for group, count in counts.items()
                if self._validate_count(group, count)
            }

    def allocate_pages(
        self, group: int, payload_bytes: int, count: int, purpose: str
    ) -> list[tuple[int, int]]:
        """Atomically take full slots; raise MemoryError without partial success."""
        self._validate_count(group, count)
        spec = self.specs[group]
        if not 0 < payload_bytes <= sum(spec.layer_token_bytes) * spec.chunk_tokens:
            raise ValueError("Page payload exceeds its runtime fixed-slot layout")
        self._validate_purpose(purpose)
        with self.lock:
            free = self.free_slots[group]
            if len(free) < count:
                raise MemoryError(f"Group {group} has insufficient {purpose} slots")
            addresses = [free.pop() for _ in range(count)]
            for address in addresses:
                self.live[address] = (group, payload_bytes, purpose)
            self.payload_bytes[group] += count * payload_bytes
            self.partial_slots[group] += count * (
                payload_bytes < sum(spec.layer_token_bytes) * spec.chunk_tokens
            )
            self.purpose_counts[group][purpose] += count
            return [(address, spec.slot_bytes) for address in addresses]

    def free(self, address: int, size: int) -> None:
        """Return exactly one owned slot; reject double frees and wrong extents."""
        with self.lock:
            group, payload, purpose = self.live[address]
            spec = self.specs[group]
            if size != spec.slot_bytes:
                raise ValueError("Fixed page slots must be released individually")
            del self.live[address]
            self.free_slots[group].append(address)
            self.payload_bytes[group] -= payload
            self.partial_slots[group] -= (
                payload < sum(spec.layer_token_bytes) * spec.chunk_tokens
            )
            self.purpose_counts[group][purpose] -= 1

    def allocate(self, size: int) -> tuple[int, int]:
        """Reject legacy objects instead of silently cutting a fixed page slot."""
        raise ValueError("Fixed-slot mode only accepts all-layer page allocations")

    def batched_allocate(self, size: int, batch_size: int) -> list[tuple[int, int]]:
        """Reject legacy batches; callers must use the page allocation API."""
        raise ValueError("Fixed-slot mode only accepts all-layer page allocations")

    def compute_aligned_size(self, size: int) -> int:
        """Reject byte-only sizing: fixed-slot capacity requires group identity."""
        raise ValueError("Fixed-slot sizing requires an explicit KV group")

    def check_consistency(self) -> bool:
        """Check address ownership and accounting; intended for diagnostics."""
        with self.lock:
            offset = 0
            for group, spec in enumerate(self.specs):
                free = set(self.free_slots[group])
                live = {a: v for a, v in self.live.items() if v[0] == group}
                expected = set(
                    range(
                        offset, offset + spec.slots * spec.slot_bytes, spec.slot_bytes
                    )
                )
                if (
                    len(free) != len(self.free_slots[group])
                    or free & live.keys()
                    or free | live.keys() != expected
                    or sum(v[1] for v in live.values()) != self.payload_bytes[group]
                    or sum(self.purpose_counts[group].values()) != len(live)
                ):
                    return False
                offset += spec.slots * spec.slot_bytes
            return True

    def _validate_count(self, group: int, count: int) -> bool:
        if (
            type(group) is not int
            or group not in (0, 1)
            or type(count) is not int
            or count < 0
        ):
            raise ValueError("Invalid KV group or page-slot count")
        return True

    @staticmethod
    def _validate_purpose(purpose: str) -> None:
        # Purpose labels are accounting only; they never reserve free slots.
        if purpose not in ("cache", "checkpoint", "workspace"):
            raise ValueError(f"Unknown page allocation purpose: {purpose}")
