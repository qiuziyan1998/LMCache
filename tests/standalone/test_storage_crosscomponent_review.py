# SPDX-License-Identifier: Apache-2.0
"""Independent M-to-W ownership review with real CPU storage and registry refs.

Only device pointer upload and its completion fence are injected. This is not an
NPU transport or graph qualification.
"""

from concurrent.futures import Future
from types import SimpleNamespace
import gc

import pytest

from lmcache.v1.rpc.storage_import import CompactSourceImporter

from test_cpu_cache_runtime import (
    control,
    lookup,
    prepare,
    setup as runtime_setup,
)


@pytest.fixture
def setup():
    yield from runtime_setup.__wrapped__()


@pytest.mark.parametrize("completion_unknown", [False, True])
def test_partial_pointer_upload_holds_both_actual_group_acquisitions_until_fence(
    setup, completion_unknown
):
    owner, runtime, work, bootstrap, populate = setup
    populate()
    bootstrap()
    hit, _ = lookup(work)
    prepared, original = prepare(work, runtime, hit.value)
    assert prepared.status == "COMPLETED", prepared.error
    assert [
        len(group.geometries[0].layer_starts) for group in prepared.value.groups
    ] == [2, 3]
    owner.local.clear()
    held_capacity = owner.allocator.get_capacity_bytes()[0]
    assert held_capacity < owner.allocator.buffer.numel()
    events, queued_operands = [], []

    def retire(identity, digest, lease_id, proof):
        assert identity == original.identity and digest == original.payload_digest
        events.append(("retire", lease_id, proof))
        future = Future()
        future.set_result(
            control(runtime, original, "retire", {"lease_id": lease_id, "proof": proof})
        )
        return future

    def upload(table):
        queued_operands.append(table)
        events.append(("upload", len(queued_operands)))
        # Failure after a submission attempt, including the second group's
        # pointer table, cannot turn either acquisition into never-imported.
        if len(queued_operands) == 2:
            raise MemoryError("second group pointer upload failed")
        return table

    def fence():
        events.append(("fence",))
        assert runtime.registry.snapshot()["leases"] == 2
        assert owner.allocator.get_capacity_bytes()[0] == held_capacity
        if completion_unknown:
            raise RuntimeError("pointer upload completion unknown")

    importer = CompactSourceImporter(
        layout=runtime.layout,
        mapping=SimpleNamespace(
            shm_name=owner.allocator.shm_name,
            size=owner.allocator.buffer.numel(),
            generation=1,
            tensor=owner.allocator.buffer,
        ),
        local_device_base=4096,
        client=SimpleNamespace(retire=retire),
        upload=upload,
        fence_upload=fence,
    )
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        expected = RuntimeError if completion_unknown else MemoryError
        with pytest.raises(expected):
            importer.import_reply(
                prepared.value,
                identity=original.identity,
                payload_digest=original.payload_digest,
                key_plan_digest=hit.value.key_plan_digest,
                source_end=5,
            )
        assert events[:3] == [("upload", 1), ("upload", 2), ("fence",)]
        assert runtime.registry.snapshot()["leases"] == 2
        assert owner.allocator.get_capacity_bytes()[0] == held_capacity
        if completion_unknown:
            assert not any(event[0] == "retire" for event in events)
            assert not importer.sources_drained()
            assert len(importer._quarantined_imports) == 1
            # Later peer proof cannot certify this worker's queued upload.
            for manifest in prepared.value.groups:
                control(
                    runtime,
                    original,
                    "retire",
                    {"lease_id": manifest.lease_id, "proof": "never_imported"},
                    "worker-1",
                )
            assert runtime.registry.snapshot()["leases"] == 2
            assert owner.allocator.get_capacity_bytes()[0] == held_capacity
        else:
            assert [event[2] for event in events if event[0] == "retire"] == [
                "fenced",
                "fenced",
            ]
            assert importer.sources_drained()
            for manifest in prepared.value.groups:
                control(
                    runtime,
                    original,
                    "retire",
                    {"lease_id": manifest.lease_id, "proof": "never_imported"},
                    "worker-1",
                )
            assert runtime.registry.snapshot()["leases"] == 0
            # The first clear correctly refused pinned cache entries. Once
            # leases unpin them, clear may drop those independent cache refs.
            owner.local.clear()
            assert (
                owner.allocator.get_capacity_bytes()[0]
                == owner.allocator.buffer.numel()
            )
    finally:
        if was_enabled:
            gc.enable()
