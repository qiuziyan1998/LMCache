# SPDX-License-Identifier: Apache-2.0
"""Value-only mapping of canonical token intervals to a decoder KV shard."""

# Standard
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class DcpWriteLayout:
    """Describe Group-0 addressing without changing canonical CPU page positions.

    ``live_origin`` is the number of physical block-table columns preceding
    ordinary local positions (the compact scratch reservation), not tokens.
    Group 1 must continue to use its ordinary replicated mapping.
    """

    world_size: int
    block_size: int
    block_ids: tuple[int, ...]
    live_origin: int = 0

    def __post_init__(self) -> None:
        for value in (self.world_size, self.block_size, self.live_origin):
            if type(value) is not int:
                raise ValueError("DCP layout dimensions must be integers")
        if (
            self.world_size not in (2, 4, 8)
            or self.block_size <= 0
            or self.live_origin < 0
        ):
            raise ValueError("Invalid DCP write layout")

    def runs(self, start: int, end: int, rank: int) -> Iterator[tuple[int, int, int]]:
        """Yield (global token, physical slot, count) for exactly this rank.

        Reject missing/null destination columns rather than remapping them to
        scratch. Each run stays within one physical block and canonical range.
        """
        if any(type(v) is not int for v in (start, end, rank)) or not (
            0 <= start <= end and 0 <= rank < self.world_size
        ):
            raise ValueError("Invalid DCP capture interval or rank")
        block = self.block_size
        first = start // block
        stripe = first + (rank - first % self.world_size) % self.world_size
        while stripe * block < end:
            left, right = max(start, stripe * block), min(end, (stripe + 1) * block)
            column = self.live_origin + stripe // self.world_size
            if column >= len(self.block_ids):
                raise ValueError("DCP capture exceeds its allocated block table")
            physical = self.block_ids[column]
            if type(physical) is not int or physical <= 0:
                raise ValueError("DCP capture includes a null or invalid block")
            slot = physical * block + left % block
            if slot + right - left > (1 << 63) - 1:
                raise OverflowError("DCP capture slot exceeds int64")
            yield left, slot, right - left
            stripe += self.world_size
