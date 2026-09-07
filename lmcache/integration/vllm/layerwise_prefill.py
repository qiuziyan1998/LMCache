# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned request bindings for synchronous shared-bank P execution."""

# Standard
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class LayerwisePrefillRequest:
    """One prefill step's absolute token ranges and banked allocation.

    ``block_ids_by_bank[bank][group]`` addresses the complete logical prefix.
    ``restore_end`` can include the recomputed final token of an external hit;
    its original extent must be retained for partial-chunk cache keys.
    """

    request_id: str
    allocation_generation: int
    token_ids: tuple[int, ...]
    compute_start: int
    compute_end: int
    restore_end: int
    block_ids_by_bank: tuple[tuple[tuple[int, ...], ...], ...]
    block_size: int
    request_configs: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if (
            not self.request_id
            or self.allocation_generation <= 0
            or self.block_size <= 0
            or not 0 <= self.compute_start < self.compute_end <= len(self.token_ids)
            or not self.compute_start <= self.restore_end <= len(self.token_ids)
        ):
            raise ValueError(
                "Invalid layerwise-prefill request identity or token range"
            )
        if len(self.block_ids_by_bank) != 2 or any(
            len(bank) != 2 for bank in self.block_ids_by_bank
        ):
            raise ValueError("Layerwise prefill requires two banks of two KV groups")
        required_tokens = max(self.compute_end, self.restore_end)
        for bank in self.block_ids_by_bank:
            for blocks in bank:
                if len(blocks) * self.block_size < required_tokens or any(
                    block <= 0 for block in blocks
                ):
                    raise ValueError(
                        "Layerwise-prefill bank has incomplete or null blocks"
                    )
