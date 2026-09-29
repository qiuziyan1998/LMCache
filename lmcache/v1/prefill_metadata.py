# SPDX-License-Identifier: Apache-2.0
"""Request-owned CPU plans for scheduler-validated append-only P prefill."""

# Standard
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, overload

T = TypeVar("T")


class PrefillSequenceView(Sequence[T], Generic[T]):
    """Fixed-length, read-only view of append-only storage and an optional tail.

    ``start``/``stop`` bound the backing sequence at construction. Later appends
    are invisible, including through slices. Existing entries must not mutate.
    """

    def __init__(
        self,
        values: Sequence[T],
        start: int = 0,
        stop: int | None = None,
        *,
        tail: tuple[T, ...] = (),
    ) -> None:
        stop = len(values) if stop is None else stop
        if not 0 <= start <= stop <= len(values):
            raise ValueError("Invalid prefill sequence bounds")
        self.values = values
        self.start = start
        self.stop = stop
        self.tail = tail

    def __len__(self) -> int:
        return self.stop - self.start + len(self.tail)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return PrefillSequenceView(self, start, max(start, stop))
            return tuple(self[i] for i in range(start, stop, step))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        size = self.stop - self.start
        return (
            self.values[self.start + index] if index < size else self.tail[index - size]
        )

    def __iter__(self) -> Iterator[T]:
        for index in range(self.start, self.stop):
            yield self.values[index]
        yield from self.tail


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
    partials: dict[int, tuple[Any, Sequence[Any]]] = field(default_factory=dict)


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
        self.database: Any = None
        self.signature: Any = None
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.hashes: list[Any] = []
        self.partial_hashes: dict[int, Any] = {}
        self.groups: dict[tuple[int, int], _GroupKeys] = {}

    def prepare(
        self,
        token_database: Any,
        tokens: Any,
        *,
        request_configs: dict | None = None,
        kv_group: int = 0,
        num_layers: int,
        skip_tokens: int = 0,
    ) -> PrefillMetadataPlan:
        """Return a prefix plan, hashing/splitting only new chunks or its tail.

        Args:
            token_database: Chunked database exposing incremental processing.
            tokens: CPU token list/tensor for the requested prefix.
            request_configs: Small request key configuration; changes invalidate.
            kv_group: Group whose dtype/schema keys should be returned.
            num_layers: Number of layer keys per chunk.
            skip_tokens: Chunk-aligned masked prefix to omit from the views.

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
            self.__init__()
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
        old_count = len(group.base)
        new_count = max(0, complete - old_count)
        if new_count:
            generated = token_database.process_tokens(
                hashes=self.hashes[old_count:complete],
                offsets=[chunk_size] * new_count,
                request_configs=request_configs,
                kv_group=kv_group,
            )
            for index, (_, _, key) in enumerate(generated, old_count):
                split = key.split_layers(num_layers)
                group.base.append(key)
                group.candidates.append((self.starts[index], self.ends[index], key))
                group.chunks.append(split)
                for layer, layer_key in zip(group.layers, split, strict=True):
                    layer.append(layer_key)

        tail_base: tuple[Any, ...] = ()
        tail_chunks: tuple[Sequence[Any], ...] = ()
        tail_candidates: tuple[tuple[int, int, Any], ...] = ()
        tail_start: tuple[int, ...] = ()
        tail_end: tuple[int, ...] = ()
        tail_new = 0
        if has_tail:
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
                group.partials[total] = (key, key.split_layers(num_layers))
                tail_new = 1
            key, split = group.partials[total]
            tail_base, tail_chunks = (key,), (split,)
            tail_start, tail_end = (complete * chunk_size,), (total,)
            tail_candidates = ((complete * chunk_size, total, key),)

        first = min(skip_tokens // chunk_size, complete)
        base = PrefillSequenceView(group.base, first, complete, tail=tail_base)
        layers = tuple(
            PrefillSequenceView(
                row,
                first,
                complete,
                tail=(tail_chunks[0][layer],) if has_tail else (),
            )
            for layer, row in enumerate(group.layers)
        )
        keys_new = (new_count + tail_new) * num_layers
        return PrefillMetadataPlan(
            PrefillSequenceView(self.starts, first, complete, tail=tail_start),
            PrefillSequenceView(self.ends, first, complete, tail=tail_end),
            PrefillSequenceView(
                group.candidates, first, complete, tail=tail_candidates
            ),
            base,
            PrefillSequenceView(group.chunks, first, complete, tail=tail_chunks),
            layers,
            (base,) * num_layers,
            hashed,
            keys_new,
            max(0, len(base) * num_layers - keys_new),
        )
