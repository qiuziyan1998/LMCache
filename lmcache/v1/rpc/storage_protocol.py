# SPDX-License-Identifier: Apache-2.0
"""Value-only decoder storage contracts; no engine or device imports."""

# Standard
from hashlib import sha256
from typing import Any, Literal
import struct

# Third Party
import msgspec


PROTOCOL_VERSION = 1
_FRAME_EXT = 76
_FRAME_THRESHOLD = 256
_DTYPE_BYTES = {
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "float32": 4,
    "int32": 4,
    "float64": 8,
    "int64": 8,
}


def checked_uint(value: int, bits: int = 64) -> int:
    """Reject negative values, bools and arithmetic overflow."""
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise ValueError(f"Expected uint{bits}, got {value!r}")
    return value


def pack_uints(values: Any, bits: int = 64) -> bytes:
    """Pack a checked immutable little-endian numeric frame."""
    values = tuple(checked_uint(value, bits) for value in values)
    code = {32: "I", 64: "Q"}[bits]
    return struct.pack(f"<{len(values)}{code}", *values)


def unpack_uints(frame: bytes, count: int, bits: int = 64) -> tuple[int, ...]:
    """Validate cardinality before interpreting a numeric frame."""
    checked_uint(count)
    if not isinstance(frame, bytes) or len(frame) != count * (bits // 8):
        raise ValueError("Numeric frame has the wrong byte length")
    code = {32: "I", 64: "Q"}[bits]
    return struct.unpack(f"<{count}{code}", frame)


def wire_hash(value: int | bytes, width: int) -> int | bytes:
    """Preserve digest width; wide integer hashes travel as exact bytes."""
    if type(width) is not int or width <= 0 or width % 8:
        raise ValueError("Hash width must be a positive number of whole bytes")
    if isinstance(value, bytes):
        if len(value) != width // 8:
            raise ValueError("Hash digest has the wrong width")
        return value
    if type(value) is int and width <= 64 and -(1 << (width - 1)) <= value < 0:
        # Python's builtin prefix hash is signed; preserve its integer key.
        return value
    checked_uint(value, width)
    return value if width <= 64 else value.to_bytes(width // 8, "big")


def restore_hash(value: int | bytes, width: int, kind: str) -> int | bytes:
    """Restore the token database's declared integer or bytes representation."""
    value = wire_hash(value, width)
    if kind == "int":
        return int.from_bytes(value, "big") if isinstance(value, bytes) else value
    if kind == "bytes":
        return value if isinstance(value, bytes) else value.to_bytes(width // 8, "big")
    raise ValueError("Unknown hash representation")


def content_digest(value: Any) -> bytes:
    """Hash a canonical value-only record, including exact binary frame bytes."""

    def canonical(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: canonical(item[key]) for key in sorted(item)}
        if isinstance(item, (tuple, list)):
            return [canonical(child) for child in item]
        return item

    builtins = msgspec.to_builtins(value, builtin_types=(bytes,))
    return sha256(msgspec.msgpack.encode(canonical(builtins))).digest()


class OperationIdentity(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Identity shared by work and controls targeting that original work."""

    cache_instance_id: str
    service_epoch: str
    client_session: str
    operation_id: int
    request_id: str
    request_attempt: str
    load_generation: int
    purpose: str

    def validate(self) -> None:
        """Reject incomplete identities before registry or allocator access."""
        if not all(
            (
                self.cache_instance_id,
                self.service_epoch,
                self.client_session,
                self.request_id,
                self.request_attempt,
                self.purpose,
            )
        ):
            raise ValueError("Incomplete storage operation identity")
        checked_uint(self.operation_id)
        checked_uint(self.load_generation)

    @property
    def key(self) -> tuple[str, str, int]:
        """Registry key; request identity remains part of the checked payload."""
        return self.service_epoch, self.client_session, self.operation_id


class GroupLayout(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Registered physical owners and per-token plane geometry for one group."""

    group_id: int
    layer_names: tuple[str, ...]
    layer_dtypes: tuple[str, ...]
    plane_token_bytes: tuple[tuple[int, ...], ...]
    memory_format: int
    shared_owner_map: tuple[int, ...] = ()


class LayoutSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Frozen source/key identity, distinct from worker HBM destinations."""

    model_identity: str
    cache_namespace: str
    hash_algorithm: str
    hash_kind: Literal["int", "bytes"]
    hash_width: int
    source_tp_size: int
    chunk_size: int
    groups: tuple[GroupLayout, ...]
    precision_descriptor: bytes = b""
    allocator_alignment: int = 64
    layout_version: int = 1

    def validate(self) -> None:
        """Validate registered cardinalities without consulting a connector."""
        if (
            not self.model_identity
            or not self.cache_namespace
            or not self.hash_algorithm
        ):
            raise ValueError("Layout requires model, cache and hash identities")
        wire_hash(0, self.hash_width)
        if self.hash_kind not in ("int", "bytes"):
            raise ValueError("Invalid hash representation")
        for value in (
            self.source_tp_size,
            self.chunk_size,
            self.allocator_alignment,
            self.layout_version,
        ):
            if checked_uint(value, 32) == 0:
                raise ValueError("Layout dimensions must be positive")
        if self.allocator_alignment & (self.allocator_alignment - 1):
            raise ValueError("Allocator alignment must be a power of two")
        if tuple(group.group_id for group in self.groups) != tuple(
            range(len(self.groups))
        ):
            raise ValueError("Group IDs must be complete and ordered")
        for group in self.groups:
            count = len(group.layer_names)
            if (
                not count
                or len(group.layer_dtypes) != count
                or len(group.plane_token_bytes) != count
            ):
                raise ValueError("Incomplete physical group layer layout")
            if len(set(group.layer_names)) != count:
                raise ValueError("Physical owner names must be unique")
            for dtype, planes in zip(
                group.layer_dtypes, group.plane_token_bytes, strict=True
            ):
                if dtype.removeprefix("torch.") not in _DTYPE_BYTES or not planes:
                    raise ValueError("Unknown source dtype or plane geometry")
                if any(checked_uint(width, 32) == 0 for width in planes):
                    raise ValueError("Source planes must have positive widths")
            if any(not 0 <= owner < count for owner in group.shared_owner_map):
                raise ValueError("Shared indexer owner map exceeds physical layers")

    @property
    def digest(self) -> bytes:
        """Stable layout identity for ticket and source import validation."""
        return content_digest(self)


class SourceGeometry(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Actual byte boundaries in a merged allocation, not inferred from valid tokens."""

    token_capacity: int
    layer_starts: tuple[int, ...]
    layer_sizes: tuple[int, ...]
    layer_shapes: tuple[tuple[int, ...], ...]
    layer_dtypes: tuple[str, ...]
    plane_ranges: tuple[tuple[tuple[int, int], ...], ...]

    def validate(self, layers: int, extent: int) -> None:
        """Check every layer and plane against its physical allocation extent."""
        if checked_uint(self.token_capacity, 32) == 0:
            raise ValueError("Source capacity must be positive")
        if any(
            len(field) != layers
            for field in (
                self.layer_starts,
                self.layer_sizes,
                self.layer_shapes,
                self.layer_dtypes,
                self.plane_ranges,
            )
        ):
            raise ValueError("Geometry physical layer count disagrees with layout")
        previous = 0
        for start, size, shape, dtype, planes in zip(
            self.layer_starts,
            self.layer_sizes,
            self.layer_shapes,
            self.layer_dtypes,
            self.plane_ranges,
            strict=True,
        ):
            end = checked_uint(checked_uint(start) + checked_uint(size))
            if not size or start < previous or end > extent:
                raise ValueError("Source layer overlaps or exceeds physical extent")
            element_size = _DTYPE_BYTES.get(dtype.removeprefix("torch."))
            if element_size is None:
                raise ValueError("Unknown source dtype")
            elements = 1
            for dimension in shape:
                elements = checked_uint(elements * checked_uint(dimension))
            if not shape or elements * element_size != size:
                raise ValueError(
                    "Source shape does not match exact logical layer bytes"
                )
            plane_end = 0
            for offset, length in planes:
                checked_uint(offset)
                checked_uint(length)
                if not length or offset < plane_end or offset + length > size:
                    raise ValueError("Source plane overlaps or exceeds its layer")
                plane_end = offset + length
            if not planes or plane_end != size:
                raise ValueError("Source planes must cover the complete logical layer")
            previous = end


class GroupManifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Merged pages and ragged flat chunks in canonical order; offsets are local."""

    group_id: int
    source_generation: int
    layout_digest: bytes
    key_plan_digest: bytes
    lease_id: str
    chunk_starts: tuple[int, ...]
    chunk_ends: tuple[int, ...]
    page_offsets: bytes
    page_extents: bytes
    geometry_ids: bytes
    geometries: tuple[SourceGeometry, ...]
    flat_offsets: bytes = b""
    flat_extents: bytes = b""
    flat_logical_sizes: bytes = b""
    flat_token_capacities: bytes = b""
    source_start: int = 0
    page_chunk_indices: bytes = b""

    def page_positions(self) -> tuple[int, ...]:
        """Return canonical chunk positions, retaining the legacy prefix encoding."""
        pages = len(self.page_offsets) // 8
        return (
            unpack_uints(self.page_chunk_indices, pages, 32)
            if self.page_chunk_indices
            else tuple(range(pages))
        )

    def flat_positions(self) -> tuple[int, ...]:
        """Return canonical positions represented by the layer-major flat frames."""
        merged = set(self.page_positions())
        return tuple(
            index for index in range(len(self.chunk_starts)) if index not in merged
        )

    def validate(
        self,
        layout: LayoutSpec,
        slab_size: int,
        source_end: int,
        *,
        key_plan_digest: bytes | None = None,
        allow_checkpoint_tail: bool = False,
    ) -> None:
        """Validate import geometry and both physical and requested token coverage."""
        if self.layout_digest != layout.digest or not self.lease_id:
            raise ValueError("Stale source layout or absent physical lease")
        if key_plan_digest is not None and self.key_plan_digest != key_plan_digest:
            raise ValueError(
                "Source manifest disagrees with the authoritative key plan"
            )
        checked_uint(self.source_generation)
        checked_uint(slab_size)
        if not 0 <= self.group_id < len(layout.groups):
            raise ValueError("Unknown source group")
        chunks = len(self.chunk_starts)
        if not chunks or len(self.chunk_ends) != chunks or source_end <= 0:
            raise ValueError("Incomplete source chunk ranges")
        checked_uint(self.source_start)
        if self.source_start and not (
            self.group_id == 1
            and allow_checkpoint_tail
            and self.source_start < source_end
        ):
            raise ValueError(
                "Only a checkpoint Group-1 tail may omit the source prefix"
            )
        end = self.source_start
        for index, (start, stop) in enumerate(
            zip(self.chunk_starts, self.chunk_ends, strict=True)
        ):
            checked_uint(start)
            checked_uint(stop)
            if start != end or not start < stop or stop - start > layout.chunk_size:
                raise ValueError(
                    "Source chunk ranges have a hole, overlap or invalid length"
                )
            if index < chunks - 1 and stop - start != layout.chunk_size:
                raise ValueError("Only the final source chunk may be partial")
            end = stop
        if end < source_end or self.chunk_starts[-1] >= source_end:
            raise ValueError("Source consumption range disagrees with object coverage")
        if len(self.page_offsets) % 8:
            raise ValueError("Invalid page offset frame")
        pages = len(self.page_offsets) // 8
        if pages > chunks:
            raise ValueError("Too many merged source pages")
        offsets = unpack_uints(self.page_offsets, pages)
        extents = unpack_uints(self.page_extents, pages)
        indices = unpack_uints(self.geometry_ids, pages, 32)
        page_positions = self.page_positions()
        if tuple(sorted(set(page_positions))) != page_positions or any(
            index >= chunks for index in page_positions
        ):
            raise ValueError("Merged source chunk positions are invalid")
        flat_positions = self.flat_positions()
        group = layout.groups[self.group_id]
        layers = len(group.layer_names)
        validated = set()
        for index, (offset, extent, geometry_id) in enumerate(
            zip(offsets, extents, indices, strict=True)
        ):
            if (
                not extent
                or checked_uint(offset + extent) > slab_size
                or offset % layout.allocator_alignment
                or geometry_id >= len(self.geometries)
            ):
                raise ValueError("Merged source page exceeds slab or geometry table")
            geometry = self.geometries[geometry_id]
            if (
                geometry.token_capacity
                < self.chunk_ends[page_positions[index]]
                - self.chunk_starts[page_positions[index]]
                or geometry.token_capacity > layout.chunk_size
            ):
                raise ValueError("Valid source tokens exceed physical capacity")
            if (geometry_id, extent) not in validated:
                geometry.validate(layers, extent)
                for layer in range(layers):
                    widths = group.plane_token_bytes[layer]
                    lengths = tuple(width * geometry.token_capacity for width in widths)
                    position = 0
                    planes = []
                    for length in lengths:
                        planes.append((position, length))
                        position += length
                    if (
                        geometry.layer_dtypes[layer].removeprefix("torch.")
                        != group.layer_dtypes[layer].removeprefix("torch.")
                        or geometry.layer_sizes[layer] != position
                        or geometry.plane_ranges[layer] != tuple(planes)
                    ):
                        raise ValueError(
                            "Source geometry disagrees with registered precision/planes"
                        )
                validated.add((geometry_id, extent))
        tail = chunks - pages
        flat_offsets = unpack_uints(self.flat_offsets, layers * tail)
        flat_extents = unpack_uints(self.flat_extents, layers * tail)
        logical_sizes = unpack_uints(self.flat_logical_sizes, layers * tail)
        capacities = unpack_uints(self.flat_token_capacities, layers * tail, 32)
        for index, (offset, extent, logical_size, capacity) in enumerate(
            zip(flat_offsets, flat_extents, logical_sizes, capacities, strict=True)
        ):
            layer, suffix_index = divmod(index, tail)
            chunk_index = flat_positions[suffix_index]
            valid = self.chunk_ends[chunk_index] - self.chunk_starts[chunk_index]
            if (
                not extent
                or checked_uint(offset + extent) > slab_size
                or offset % layout.allocator_alignment
                or capacity < valid
                or capacity > layout.chunk_size
                or logical_size != capacity * sum(group.plane_token_bytes[layer])
                or logical_size > extent
            ):
                raise ValueError("Flat source allocation exceeds slab extent")

    def pointer_rows(
        self, local_device_base: int, layers: int
    ) -> tuple[tuple[int, ...], ...]:
        """Rebase a validated manifest without constructing per-chunk MemoryObjs."""
        checked_uint(local_device_base)
        pages = len(self.page_offsets) // 8
        offsets = unpack_uints(self.page_offsets, pages)
        indices = unpack_uints(self.geometry_ids, pages, 32)
        tail = len(self.chunk_starts) - pages
        suffix = unpack_uints(self.flat_offsets, layers * tail)
        if self.page_chunk_indices:
            rows = [[0] * len(self.chunk_starts) for _ in range(layers)]
            for layer, row in enumerate(rows):
                for index, position in enumerate(self.page_positions()):
                    row[position] = checked_uint(
                        local_device_base
                        + offsets[index]
                        + self.geometries[indices[index]].layer_starts[layer]
                    )
                for index, position in enumerate(self.flat_positions()):
                    row[position] = checked_uint(
                        local_device_base + suffix[layer * tail + index]
                    )
            return tuple(tuple(row) for row in rows)
        return tuple(
            tuple(
                checked_uint(
                    local_device_base + offset + self.geometries[g].layer_starts[layer]
                )
                for offset, g in zip(offsets, indices, strict=True)
            )
            + tuple(
                checked_uint(local_device_base + value)
                for value in suffix[layer * tail : (layer + 1) * tail]
            )
            for layer in range(layers)
        )


class LookupReply(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A source ticket and the canonical group key plan, not an execution-ready hit."""

    matched_tokens: int
    ticket_id: str
    lookup_generation: int
    key_plan_digest: bytes
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    hashes: tuple[int | bytes, ...]
    request_configs: dict[str, Any]


class StorageLookupTicket(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Scheduler handoff retaining the exact lookup operation and canonical plan."""

    reply: LookupReply
    identity: OperationIdentity
    payload_digest: bytes


class PrepareSpec(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Value-only CPU source preparation request; no worker tensors or futures."""

    ticket_id: str
    key_plan_digest: bytes
    source_end: int
    layout_digest: bytes
    participants: tuple[str, ...]
    mode: Literal["compact", "resident", "checkpoint_restore", "dense"] = "compact"
    checkpoint_generation: int | None = None


class PreparedReply(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Service-owned slab sources, separately leased for each KV group."""

    identity: OperationIdentity
    source_end: int
    slab_name: str
    slab_size: int
    slab_generation: int
    groups: tuple[GroupManifest, ...]
    timings: dict[str, float] = msgspec.field(default_factory=dict)
    actions: tuple[str, ...] = ()
    restore_lease_id: str | None = None


class StorageRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Work or lifecycle envelope bound to its original work digest."""

    identity: OperationIdentity
    method: str
    payload: Any
    payload_digest: bytes
    session_nonce: bytes
    version: int = PROTOCOL_VERSION
    correlation_id: int = 0
    participant_session: str = ""


class StorageResult(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Structured asynchronous reply; pending/unknown never mean safe to free."""

    identity: OperationIdentity
    status: str
    value: Any = None
    error: str | None = None
    version: int = PROTOCOL_VERSION
    payload_digest: bytes = b""
    correlation_id: int = 0
    session_nonce: bytes = b""


def encode_message(record: StorageRequest | StorageResult) -> list[bytes]:
    """Encode a header plus immutable large byte frames without pickling objects."""
    frames: list[bytes] = []

    def extract(value: Any) -> Any:
        if isinstance(value, bytes) and len(value) >= _FRAME_THRESHOLD:
            index = len(frames)
            frames.append(value)
            return msgspec.msgpack.Ext(
                _FRAME_EXT, struct.pack("<IQ", index, len(value))
            )
        if isinstance(value, dict):
            return {key: extract(child) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            return [extract(child) for child in value]
        return value

    header = msgspec.msgpack.encode(
        extract(msgspec.to_builtins(record, builtin_types=(bytes,)))
    )
    return [header, *frames]


def decode_message(frames: list[bytes], record_type: type) -> Any:
    """Reject missing, mis-sized or unused frames before typed envelope decoding."""
    if not frames:
        raise ValueError("Missing storage message header")
    used = set()

    def restore(code: int, value: memoryview) -> bytes:
        if code != _FRAME_EXT or len(value) != 12:
            raise ValueError("Invalid storage frame reference")
        index, size = struct.unpack("<IQ", value)
        if index + 1 >= len(frames) or len(frames[index + 1]) != size:
            raise ValueError("Missing or truncated storage frame")
        used.add(index + 1)
        return frames[index + 1]

    value = msgspec.msgpack.decode(frames[0], ext_hook=restore)
    if used != set(range(1, len(frames))):
        raise ValueError("Unexpected unused storage frames")
    record = msgspec.convert(value, type=record_type)
    if record.version != PROTOCOL_VERSION:
        raise ValueError("Unsupported decoder storage protocol version")
    record.identity.validate()
    return record
