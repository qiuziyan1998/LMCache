# SPDX-License-Identifier: Apache-2.0
"""Request-owned CPU plans for scheduler-validated append-only P prefill."""

# Standard
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import chain, islice
from typing import Any, Generic, TypeVar, overload

# First Party
from lmcache.utils import CacheEngineKey

T = TypeVar("T")
_ORIG_CACHE_KEY_SPLIT_LAYERS = CacheEngineKey.split_layers
_ORIG_CACHE_KEY_GET_LAYER = CacheEngineKey.get_layer


@lru_cache(maxsize=32768)
def _split_cache_engine_key_layers(
    key: CacheEngineKey, num_layers: int
) -> tuple[Any, ...]:
    return tuple(_ORIG_CACHE_KEY_SPLIT_LAYERS(key, num_layers))


def split_layers_cached(key: Any, num_layers: int) -> Sequence[Any]:
    """Return immutable per-layer keys, caching standard CacheEngineKey splits."""
    if (
        type(key) is CacheEngineKey
        and CacheEngineKey.split_layers is _ORIG_CACHE_KEY_SPLIT_LAYERS
        and CacheEngineKey.get_layer is _ORIG_CACHE_KEY_GET_LAYER
        and "split_layers" not in getattr(key, "__dict__", ())
        and (
            not key.request_configs
            or len(key.request_configs) == len(key.tags or ())
        )
    ):
        return _split_cache_engine_key_layers(key, num_layers)
    return key.split_layers(num_layers)


class PrefillSequenceView(Sequence[T], Generic[T]):
    """Fixed-length, read-only view of append-only storage and an optional tail.

    ``start``/``stop`` bound the backing sequence at construction. Later appends
    are invisible, including through slices. Existing entries must not mutate.
    """

    values: Sequence[T]
    start: int
    stop: int
    tail: tuple[T, ...]
    _base_len: int
    _len: int

    def __init__(
        self,
        values: Sequence[T],
        start: int = 0,
        stop: int | None = None,
        *,
        tail: tuple[T, ...] = (),
    ) -> None:
        val_len = len(values)
        stop = val_len if stop is None else stop
        if not 0 <= start <= stop <= val_len:
            raise ValueError("Invalid prefill sequence bounds")
        if isinstance(values, PrefillSequenceView):
            inner_base = values._base_len
            if stop <= inner_base:
                stop = values.start + stop
                start = values.start + start
                values = values.values
            elif start <= inner_base:
                tail = values.tail[: stop - inner_base] + tail
                stop = values.stop
                start = values.start + start
                values = values.values
            else:
                stop = stop - inner_base
                start = start - inner_base
                values = values.tail
        self.values = values
        self.start = start
        self.stop = stop
        self.tail = tail
        self._base_len = stop - start
        self._len = self._base_len + len(tail)

    def __len__(self) -> int:
        return self._len

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        total_len = self._len
        if isinstance(index, slice):
            start, stop, step = index.indices(total_len)
            if step == 1:
                return PrefillSequenceView(self, start, max(start, stop))
            return tuple(self[i] for i in range(start, stop, step))
        if index < 0:
            index += total_len
        if not 0 <= index < total_len:
            raise IndexError(index)
        size = self._base_len
        return (
            self.values[self.start + index] if index < size else self.tail[index - size]
        )

    def __iter__(self) -> Iterator[T]:
        head = islice(self.values, self.start, self.stop)
        return chain(head, self.tail) if self.tail else head

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, PrefillSequenceView):
            return NotImplemented
        return len(self) == len(other) and all(
            a == b for a, b in zip(self, other, strict=True)
        )


@dataclass(frozen=True)
class PrefillMetadataPlan:
    """Fixed prefix views; ``keys_new`` counts new per-layer key objects."""

    starts: Sequence[int]
    ends: Sequence[int]
    candidates: Sequence[tuple[int, int, Any]]
    base_keys: Sequence[Any]
    keys_chunk_major: Sequence[Sequence[Any]]
    keys_layer_major: Sequence[Sequence[Any]]
    page_keys_layer_major: Sequence[Sequence[Any]]
    hashes_new: int
    keys_new: int
    keys_kept: int


@dataclass
class _GroupKeys:
    base: list[Any] = field(default_factory=list)
    candidates: list[tuple[int, int, Any]] = field(default_factory=list)
    chunks: list[Sequence[Any]] = field(default_factory=list)
    layers: list[list[Any]] = field(default_factory=list)
    partials: dict[int, tuple[Any, Sequence[Any] | None]] = field(
        default_factory=dict
    )


class PrefillMetadataCache:
    """Incremental hashes/keys for one immutable prompt and request scope.

    The adapter must replace this cache on request/generation changes,
    preemption, true prompt rollback, or multimodal identity changes. Tokens
    supplied to ``prepare`` must be prefixes of that same append-only stream;
    this contract avoids comparing or hashing old tokens on each chunk. Short
    retrieve queries after a longer store query do not roll the cache back.
    Returned views remain valid while other groups/store append to this cache.
    This object owns CPU metadata only, and is used by the worker owner thread.
    """

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self.database: Any = None
        self.signature: Any = None
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.hashes: list[Any] = []
        self.partial_hashes: dict[int, Any] = {}
        self.groups: dict[tuple[int, int], _GroupKeys] = {}

    def seed_precomputed_hashes(
        self,
        token_database: Any,
        hashes: Sequence[Any],
        offsets: Sequence[int],
        *,
        request_configs: dict | None = None,
    ) -> None:
        """Seed validated scheduler chunk hashes so prepare() skips rehashing."""
        if not hashes or len(hashes) != len(offsets):
            return
        chunk_size = int(getattr(token_database, "chunk_size", 0) or 0)
        if chunk_size <= 0:
            return
        if any(type(off) is not int or off != chunk_size for off in offsets[:-1]):
            return
        last_off = offsets[-1]
        if type(last_off) is not int or not (0 < last_off <= chunk_size):
            return
        save_tail = bool(getattr(token_database.config, "save_unfull_chunk", True))
        signature = (chunk_size, save_tail, request_configs or {})
        if self.database is not token_database or self.signature != signature:
            self._reset()
            self.database = token_database
            self.signature = deepcopy(signature)

        complete_seed = len(hashes) if last_off == chunk_size else len(hashes) - 1
        common = min(len(self.hashes), complete_seed)
        if any(self.hashes[i] != hashes[i] for i in range(common)):
            self._reset()
            self.database = token_database
            self.signature = deepcopy(signature)
        for idx in range(len(self.hashes), complete_seed):
            start = idx * chunk_size
            self.starts.append(start)
            self.ends.append(start + chunk_size)
            self.hashes.append(hashes[idx])
        if last_off < chunk_size and save_tail:
            total_end = complete_seed * chunk_size + last_off
            self.partial_hashes.setdefault(total_end, hashes[-1])

    def prepare(
        self,
        token_database: Any,
        tokens: Any,
        *,
        request_configs: dict | None = None,
        kv_group: int = 0,
        num_layers: int,
        skip_tokens: int = 0,
        split_layers: bool = True,
    ) -> PrefillMetadataPlan:
        """Return a prefix plan, hashing/splitting only new chunks or its tail.

        Args:
            token_database: Chunked database exposing incremental processing.
            tokens: CPU token list/tensor for the requested prefix.
            request_configs: Small request key configuration; changes invalidate.
            kv_group: Group whose dtype/schema keys should be returned.
            num_layers: Number of layer keys per chunk.
            skip_tokens: Chunk-aligned masked prefix to omit from the views.
            split_layers: Materialize per-layer ``LayerCacheEngineKey`` rows when
                true; passive compact consumers pass false to reuse ``base``.

        Raises:
            ValueError: For invalid layer counts or unaligned/outside masks.
        """
        chunk_size = int(token_database.chunk_size)
        total = len(tokens)
        if num_layers <= 0 or not 0 <= skip_tokens <= total:
            raise ValueError("Invalid prefill layer count or masked prefix")
        if skip_tokens % chunk_size:
            raise ValueError("Prefill masked prefix must be chunk aligned")
        save_tail = bool(getattr(token_database.config, "save_unfull_chunk", True))
        signature = (chunk_size, save_tail, request_configs or {})
        if self.database is not token_database or self.signature != signature:
            # Replace containers: outstanding fixed views retain their old data.
            self._reset()
            self.database = token_database
            self.signature = deepcopy(signature)

        complete = total // chunk_size
        hashed = 0
        if complete > len(self.hashes):
            prefix_count = len(self.hashes) * chunk_size
            generated = (
                token_database.process_tokens_from_prefix(
                    tokens,
                    prefix_token_count=prefix_count,
                    prefix_hash=self.hashes[-1],
                    make_key=False,
                )
                if prefix_count
                else token_database.process_tokens(tokens=tokens, make_key=False)
            )
            for start, end, value in generated:
                hashed += 1
                if end - start == chunk_size:
                    self.starts.append(start)
                    self.ends.append(end)
                    self.hashes.append(value)
                else:
                    self.partial_hashes[end] = value

        has_tail = save_tail and total % chunk_size != 0
        if has_tail and total not in self.partial_hashes:
            prefix_count = complete * chunk_size
            generated = (
                token_database.process_tokens_from_prefix(
                    tokens,
                    prefix_token_count=prefix_count,
                    prefix_hash=self.hashes[complete - 1],
                    make_key=False,
                )
                if prefix_count
                else token_database.process_tokens(tokens=tokens, make_key=False)
            )
            for _, end, value in generated:
                hashed += 1
                self.partial_hashes[end] = value

        group_id = (kv_group, num_layers)
        group = self.groups.get(group_id)
        if group is None:
            group = _GroupKeys(layers=[[] for _ in range(num_layers)])
            self.groups[group_id] = group
        old_base = len(group.base)
        new_base = max(0, complete - old_base)
        if new_base:
            generated = token_database.process_tokens(
                hashes=self.hashes[old_base:complete],
                offsets=[chunk_size] * new_base,
                request_configs=request_configs,
                kv_group=kv_group,
            )
            for index, (_, _, key) in enumerate(generated, old_base):
                group.base.append(key)
                group.candidates.append((self.starts[index], self.ends[index], key))

        new_split = 0
        if split_layers:
            old_split = len(group.chunks)
            new_split = max(0, complete - old_split)
            for index in range(old_split, complete):
                split = split_layers_cached(group.base[index], num_layers)
                group.chunks.append(split)
                for layer, layer_key in zip(group.layers, split, strict=True):
                    layer.append(layer_key)

        tail_base: tuple[Any, ...] = ()
        tail_chunks: tuple[Sequence[Any], ...] = ()
        tail_candidates: tuple[tuple[int, int, Any], ...] = ()
        tail_start: tuple[int, ...] = ()
        tail_end: tuple[int, ...] = ()
        tail_base_new = 0
        tail_split_new = 0
        if has_tail:
            tail_split: Sequence[Any] | None
            if total not in group.partials:
                _, _, key = next(
                    iter(
                        token_database.process_tokens(
                            hashes=[self.partial_hashes[total]],
                            offsets=[total % chunk_size],
                            request_configs=request_configs,
                            kv_group=kv_group,
                        )
                    )
                )
                tail_split = (
                    split_layers_cached(key, num_layers)
                    if split_layers
                    else None
                )
                group.partials[total] = (key, tail_split)
                tail_base_new = 1
                tail_split_new = int(split_layers)
            else:
                key, tail_split = group.partials[total]
                if split_layers and tail_split is None:
                    tail_split = split_layers_cached(key, num_layers)
                    group.partials[total] = (key, tail_split)
                    tail_split_new = 1
            tail_base = (key,)
            if split_layers:
                assert tail_split is not None
                tail_chunks = (tail_split,)
            tail_start, tail_end = (complete * chunk_size,), (total,)
            tail_candidates = ((complete * chunk_size, total, key),)

        first = min(skip_tokens // chunk_size, complete)
        base = PrefillSequenceView(group.base, first, complete, tail=tail_base)
        page_layers = (base,) * num_layers
        if split_layers:
            layers = tuple(
                PrefillSequenceView(
                    row,
                    first,
                    complete,
                    tail=(tail_chunks[0][layer],) if has_tail else (),
                )
                for layer, row in enumerate(group.layers)
            )
            chunks_view: Sequence[Sequence[Any]] = PrefillSequenceView(
                group.chunks, first, complete, tail=tail_chunks
            )
            keys_new = (new_split + tail_split_new) * num_layers
            keys_kept = max(0, len(base) * num_layers - keys_new)
        else:
            layers = page_layers
            chunks_view = ()
            keys_new = new_base + tail_base_new
            keys_kept = max(0, len(base) - keys_new)
        return PrefillMetadataPlan(
            PrefillSequenceView(self.starts, first, complete, tail=tail_start),
            PrefillSequenceView(self.ends, first, complete, tail=tail_end),
            PrefillSequenceView(
                group.candidates, first, complete, tail=tail_candidates
            ),
            base,
            chunks_view,
            layers,
            page_layers,
            hashed,
            keys_new,
            keys_kept,
        )
