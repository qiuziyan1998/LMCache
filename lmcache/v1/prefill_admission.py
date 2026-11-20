# SPDX-License-Identifier: Apache-2.0
"""CPU-only admission accounting for layerwise prefill requests."""

# Standard
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PrefillAdmissionBudget:
    """Reserve a full prompt before lookup pins or worker allocations start.

    Free plus evictable bytes are reported by the actual worker allocator.
    Reservations are deliberately not discounted by estimated materialization:
    some existing allocations can therefore be counted twice. This conservative
    policy also covers outstanding saves whose ownership has left the request.
    """

    world_size: int
    reports: dict[int, dict] = field(default_factory=dict)
    reservations: dict[str, int] = field(default_factory=dict)
    retiring: set[str] = field(default_factory=set)
    retired_ranks: dict[str, set[int]] = field(default_factory=dict)
    waiting: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("Prefill admission requires at least one worker")

    @property
    def ready(self) -> bool:
        return set(self.reports) == set(range(self.world_size))

    def update(self, reports: tuple[dict, ...]) -> None:
        """Accept allocator snapshots and all-rank retirement acknowledgements."""
        for report in reports:
            rank = int(report["rank"])
            if not 0 <= rank < self.world_size:
                raise ValueError("Prefill admission received an invalid worker rank")
            pool = report["pool"]
            if pool is not None and not (
                0 <= pool["available"] <= pool["capacity"]
                and pool["capacity"] > 0
                and report["page_bytes"] > 0
                and report["chunk_tokens"] > 0
                and report["max_batch_tokens"] > 0
            ):
                raise ValueError("Invalid prefill CPU allocator capacity report")
            self.reports[rank] = report
            for req_id in report["retired"]:
                if req_id in self.reservations:
                    self.retired_ranks.setdefault(req_id, set()).add(rank)
        for req_id, ranks in list(self.retired_ranks.items()):
            if req_id in self.retiring and len(ranks) == self.world_size:
                self.reservations.pop(req_id, None)
                self.retiring.discard(req_id)
                del self.retired_ranks[req_id]

    def finish(self, req_id: str) -> None:
        """Keep credit charged until worker cleanup, including aborted lookups."""
        self.waiting.discard(req_id)
        if req_id in self.reservations:
            self.retiring.add(req_id)
        self.update(())

    def admit(self, req_id: str, tokens: int) -> tuple[bool, Optional[str], dict]:
        """Return (admitted, permanent error, diagnostic byte counts)."""
        if req_id in self.reservations:
            return True, None, {}
        self.waiting.add(req_id)
        if not self.ready:
            return False, None, {}
        pools = [r for r in self.reports.values() if r["pool"] is not None]
        if not pools:
            raise RuntimeError(
                "Layerwise prefill has no CPU allocator admission report"
            )
        if len({(r["chunk_tokens"], r["max_batch_tokens"]) for r in pools}) != 1:
            raise ValueError("Prefill CPU pool layouts disagree across workers")
        chunk = int(pools[0]["chunk_tokens"])
        # Full prompt + concurrently live replacement/save chunk + tail pages.
        pages = (tokens + chunk - 1) // chunk
        store_pages = (
            min(tokens, int(pools[0]["max_batch_tokens"])) + chunk - 1
        ) // chunk
        need = (pages + store_pages + 2) * max(int(r["page_bytes"]) for r in pools)
        capacity = min(int(r["pool"]["capacity"]) for r in pools)
        available = min(int(r["pool"]["available"]) for r in pools)
        reserved = sum(self.reservations.values())
        details = dict(
            need_bytes=need,
            reserved_bytes=reserved,
            available_bytes=available,
            capacity_bytes=capacity,
        )
        if need > capacity:
            self.waiting.discard(req_id)
            return (
                False,
                (
                    f"LMCache CPU capacity cannot fit this prefill request: "
                    f"tokens={tokens}, required_bytes={need}, capacity_bytes={capacity}"
                ),
                details,
            )
        if need + reserved > available:
            return False, None, details
        self.reservations[req_id] = need
        self.waiting.discard(req_id)
        return True, None, details
