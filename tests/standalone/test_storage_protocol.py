# SPDX-License-Identifier: Apache-2.0
"""CPU tests for exact process wire values and hostile source geometry."""

import subprocess
import sys

import msgspec
import pytest

from lmcache.v1.rpc.storage_protocol import (
    GroupLayout,
    GroupManifest,
    LayoutSpec,
    OperationIdentity,
    SourceGeometry,
    StorageRequest,
    content_digest,
    decode_message,
    encode_message,
    pack_uints,
    restore_hash,
    wire_hash,
)


def identity():
    return OperationIdentity(
        "cache-dp0", "epoch", "tp0", 1, "request", "attempt", 0, "load"
    )


def layout():
    return LayoutSpec(
        "model",
        "namespace",
        "sha256",
        "bytes",
        256,
        4,
        4,
        (
            GroupLayout(
                0, ("latent0", "latent1"), ("bfloat16", "bfloat16"), ((4,), (4,)), 1
            ),
            GroupLayout(1, ("index0",), ("uint8",), ((8, 4),), 1),
        ),
    )


def manifest(**changes):
    values = dict(
        group_id=0,
        source_generation=1,
        layout_digest=layout().digest,
        key_plan_digest=b"proof",
        lease_id="lease",
        chunk_starts=(0, 4),
        chunk_ends=(4, 5),
        page_offsets=pack_uints([64]),
        page_extents=pack_uints([64]),
        geometry_ids=pack_uints([0], 32),
        geometries=(
            SourceGeometry(
                4,
                (0, 32),
                (16, 16),
                ((4, 2), (4, 2)),
                ("bfloat16", "bfloat16"),
                (((0, 16),), ((0, 16),)),
            ),
        ),
        flat_offsets=pack_uints([192, 128]),
        flat_extents=pack_uints([4, 4]),
        flat_logical_sizes=pack_uints([4, 4]),
        flat_token_capacities=pack_uints([1, 1], 32),
    )
    values.update(changes)
    return GroupManifest(**values)


def test_protocol_import_does_not_load_device_or_serving_modules():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lmcache.v1.rpc.storage_protocol; "
            "assert not any(m in sys.modules for m in ('torch', 'torch_npu', 'vllm'))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kind,value", [("int", 2**250 + 1), ("bytes", b"\0" * 31 + b"\1")]
)
def test_wide_hash_round_trip_keeps_width_and_leading_zero(kind, value):
    encoded = wire_hash(value, 256)
    request = StorageRequest(
        identity(), "lookup", {"hash": encoded}, b"digest", b"nonce"
    )
    restored = decode_message(encode_message(request), StorageRequest)
    assert restore_hash(restored.payload["hash"], 256, kind) == value


@pytest.mark.parametrize("value", [-1, 2**256, True, b"short"])
def test_invalid_hash_rejected(value):
    with pytest.raises(ValueError):
        wire_hash(value, 256)


def test_builtin_hash_preserves_negative_signed_integer_identity():
    value = -(2**63) + 123
    request = StorageRequest(
        identity(), "lookup", {"hash": wire_hash(value, 64)}, b"digest", b"nonce"
    )
    restored = decode_message(encode_message(request), StorageRequest)
    assert restore_hash(restored.payload["hash"], 64, "int") == value


def test_large_numeric_frames_are_exact_and_strictly_checked():
    payload = {"offsets": pack_uints(range(1024)), "nested": [b"x" * 512]}
    request = StorageRequest(identity(), "prepare", payload, b"digest", b"nonce")
    frames = encode_message(request)
    assert len(frames) == 3
    assert decode_message(frames, StorageRequest) == request
    for damaged in (
        frames[:-1],
        [*frames, b"extra"],
        [frames[0], frames[1][:-1], frames[2]],
    ):
        with pytest.raises(ValueError):
            decode_message(damaged, StorageRequest)


def test_unknown_version_fields_and_incomplete_identity_rejected():
    request = StorageRequest(identity(), "prepare", {}, b"digest", b"nonce")
    raw = msgspec.to_builtins(request)
    raw["unexpected"] = 1
    with pytest.raises(msgspec.ValidationError):
        decode_message([msgspec.msgpack.encode(raw)], StorageRequest)
    raw.pop("unexpected")
    raw["version"] = 2
    with pytest.raises(ValueError):
        decode_message([msgspec.msgpack.encode(raw)], StorageRequest)
    raw["version"] = 1
    raw["identity"]["request_attempt"] = ""
    with pytest.raises(ValueError):
        decode_message([msgspec.msgpack.encode(raw)], StorageRequest)


def test_canonical_digest_is_mapping_order_independent_and_binary_exact():
    assert content_digest({"a": [1, b"\0"], "b": 2}) == content_digest(
        {"b": 2, "a": (1, b"\0")}
    )
    assert content_digest(b"\0") != content_digest(b"")


def test_physical_capacity_and_alignment_are_distinct_from_valid_token_length():
    source = manifest()
    source.validate(layout(), 256, 5)
    # Layer 1 starts at the producer's aligned byte boundary, not 4 * 4.
    assert source.pointer_rows(4096, 2) == ((4160, 4288), (4192, 4224))


@pytest.mark.parametrize(
    "changes",
    [
        {"chunk_starts": (0, 3)},
        {"chunk_ends": (3, 5)},
        {"page_offsets": pack_uints([240])},
        {"page_extents": pack_uints([0])},
        {"geometry_ids": pack_uints([1], 32)},
        {"flat_extents": pack_uints([4])},
        {"layout_digest": b"stale"},
        {"lease_id": ""},
        {"flat_offsets": pack_uints([255, 224])},
    ],
)
def test_invalid_manifest_never_yields_worker_pointers(changes):
    with pytest.raises(ValueError):
        manifest(**changes).validate(layout(), 256, 5)


def test_source_geometry_checks_capacity_shapes_and_planes():
    source = manifest()
    geo = source.geometries[0]
    for changes in (
        {"token_capacity": 3},
        {"layer_starts": (0, 8)},
        {"layer_shapes": ((4, 1), (4, 2))},
        {"plane_ranges": (((0, 8),), ((0, 16),))},
    ):
        updated = msgspec.structs.replace(geo, **changes)
        with pytest.raises(ValueError):
            manifest(geometries=(updated,)).validate(layout(), 256, 5)


def test_layout_preserves_unequal_group_layer_counts_and_mixed_planes():
    spec = layout()
    spec.validate()
    assert tuple(len(group.layer_names) for group in spec.groups) == (2, 1)
    assert spec.groups[1].plane_token_bytes == ((8, 4),)
    with pytest.raises(ValueError):
        msgspec.structs.replace(spec, allocator_alignment=3).validate()
