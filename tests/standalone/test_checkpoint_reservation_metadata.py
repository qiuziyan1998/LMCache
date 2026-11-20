# SPDX-License-Identifier: Apache-2.0
"""CPU reservation metadata reuse without changing page ownership or geometry."""

from collections import Counter

import msgspec
import pytest

from test_ascend_cpu_storage_runtime import setup  # noqa: F401
from test_ascend_cpu_storage_runtime import capture, retire


@pytest.mark.parametrize("group", [0, 1])
def test_allocation_reuses_only_same_token_count_and_layout(
    request, monkeypatch, group
):
    _, runtime, _ = request.getfixturevalue("setup")
    runtime.checkpoint_plane_widths(group)
    original = runtime._expected_shared_cpu_chunk_metadata
    calls = Counter()

    def measured(**kwargs):
        calls[kwargs["num_tokens"]] += 1
        return original(**kwargs)

    monkeypatch.setattr(runtime, "_expected_shared_cpu_chunk_metadata", measured)
    for tokens in (4, 4, 4, 1, 1, 4):
        page, _ = runtime.allocate_checkpoint_fragment(group, tokens)
        try:
            assert page.valid_tokens == tokens
            assert page.get_shapes() == [
                original(kv_group=group, num_tokens=tokens, layer_id=i)[0]
                for i in range(runtime.num_layers_for_group(group))
            ]
        finally:
            page.ref_count_down()
    layers = runtime.num_layers_for_group(group)
    assert calls == {4: layers * 2, 1: layers}
    # An equivalent but newly bound immutable layout invalidates the cached proof.
    runtime.layout = msgspec.structs.replace(runtime.layout)
    runtime.checkpoint_plane_widths(group)
    calls.clear()
    page, _ = runtime.allocate_checkpoint_fragment(group, 4)
    page.ref_count_down()
    assert calls == {4: layers}


@pytest.mark.parametrize("invalid_second", [False, True])
def test_equal_export_geometries_are_validated_once_per_run(
    request, monkeypatch, invalid_second
):
    from lmcache.v1.rpc.storage_protocol import SourceGeometry

    _, runtime, work = request.getfixturevalue("setup")
    validate = SourceGeometry.validate
    validated = []

    def measured(self, layers, extent):
        validated.append((self, layers, extent))
        return validate(self, layers, extent)

    monkeypatch.setattr(SourceGeometry, "validate", measured)
    source_geometry = runtime.source_geometry
    calls = 0

    def source(group, page):
        nonlocal calls
        calls += 1
        geometry = source_geometry(group, page)
        return (
            msgspec.structs.replace(geometry, token_capacity=0)
            if invalid_second and calls == 2
            else geometry
        )

    monkeypatch.setattr(runtime, "source_geometry", source)
    result, operation = capture(work, end=13)
    if invalid_second:
        assert result.status == "FAILED_SAFE"
        assert len(validated) == 2
        assert runtime.registry.snapshot()["leases"] == 0
        return
    assert result.status == "COMPLETED", result.error
    assert [len(group["starts"]) for group in result.value["groups"]] == [4, 4]
    work(
        "worker-0",
        "cancel_write",
        {
            "reservation_id": result.value["reservation_id"],
            "proof": "never_imported",
        },
    )
    for group in result.value["groups"]:
        retire(runtime, operation, group["lease_id"])
    assert len(validated) == 4  # Full and partial pages, independently per group.
