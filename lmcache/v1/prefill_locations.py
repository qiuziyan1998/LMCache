# SPDX-License-Identifier: Apache-2.0
"""Append-only location/key rows for a request-owned shared prefill prefix."""

# Standard
from collections import Counter
from collections.abc import Sequence
from typing import Any

# First Party
from lmcache.v1.prefill_metadata import PrefillSequenceView


class PrefillLocationPlan:
    """Keep both row orders without transposing historical layer keys.

    The owner must validate the retained source prefix before ``truncate`` and
    finish consuming an old retrieval before updating this plan. Storage is
    owned by the same request lease as the corresponding source objects.
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.keys: list[Sequence[Any]] = []
        self.locations: list[Sequence[str]] = []
        self.layer_keys: list[list[Any]] = [[] for _ in range(num_layers)]
        self.layer_locations: list[list[str]] = [[] for _ in range(num_layers)]
        self.location_counts: Counter[str] = Counter()
        self.page_chunks = 0
        self.valid_chunks = 0

    def truncate(self, chunks: int) -> None:
        """Discard a changed suffix while leaving every historical row intact."""
        if not 0 <= chunks <= len(self.starts):
            raise ValueError("Invalid retained prefill location prefix")
        for locations in self.locations[chunks:]:
            self.location_counts.subtract(locations)
        self.location_counts += Counter()
        for row in (
            self.starts,
            self.ends,
            self.keys,
            self.locations,
            *self.layer_keys,
            *self.layer_locations,
        ):
            del row[chunks:]
        self.page_chunks = min(self.page_chunks, chunks)
        self.valid_chunks = chunks

    def append(
        self,
        start: int,
        end: int,
        keys: Sequence[Any],
        locations: Sequence[str],
        *,
        page: bool,
    ) -> None:
        """Append one newly resolved chunk in both indexing orders."""
        if len(keys) != self.num_layers or len(locations) != self.num_layers:
            raise ValueError("Prefill chunk layer count differs from its plan")
        if page and self.page_chunks != len(self.starts):
            raise ValueError("Shared layer pages must form a prefix")
        self.starts.append(start)
        self.ends.append(end)
        self.keys.append(keys)
        self.locations.append(locations)
        self.location_counts.update(locations)
        for layer, key, location in zip(
            range(self.num_layers), keys, locations, strict=True
        ):
            self.layer_keys[layer].append(key)
            self.layer_locations[layer].append(location)
        self.page_chunks += int(page)
        self.valid_chunks = len(self.starts)

    @property
    def location(self) -> str | None:
        """Return the common location, mixed, or None for an empty plan."""
        if len(self.location_counts) == 1:
            return next(iter(self.location_counts))
        return "mixed" if self.location_counts else None

    def key_rows(self) -> tuple[Sequence[Any], ...]:
        """Return bounded views of the prepared layer-major key rows."""
        return tuple(PrefillSequenceView(row) for row in self.layer_keys)

    def location_rows(self) -> tuple[Sequence[str], ...]:
        """Return bounded views of the prepared layer-major location rows."""
        return tuple(PrefillSequenceView(row) for row in self.layer_locations)
