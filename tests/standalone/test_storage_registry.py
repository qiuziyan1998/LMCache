# SPDX-License-Identifier: Apache-2.0
"""Adversarial ordering tests for real storage registry and physical ownership."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
import gc

import pytest

from lmcache.v1.cpu_cache import StorageBusy, StorageOperationRegistry
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    content_digest,
)


def request(operation_id=1, method="prepare", payload=None):
    payload = {} if payload is None else payload
    identity = OperationIdentity(
        "cache", "epoch", "tp0", operation_id, "req", "attempt", 1, "load"
    )
    return StorageRequest(
        identity, method, payload, content_digest((method, payload)), b"zero"
    )


def control(work, method, payload=None, participant=""):
    return StorageRequest(
        work.identity,
        method,
        payload or {},
        work.payload_digest,
        b"one" if participant == "tp1" else b"zero",
        correlation_id=1,
        participant_session=participant,
    )


def registry(**kwargs):
    return StorageOperationRegistry(
        "cache", "epoch", {"tp0": b"zero", "tp1": b"one"}, **kwargs
    )


def ack(reg, work, result):
    return reg.handle(
        control(
            work,
            "acknowledge_terminal",
            {
                "terminal_result_digest": content_digest(
                    (result.status, result.value, result.error)
                ),
            },
        ),
        lambda _: pytest.fail("Controls must not enter the work executor"),
    )


def test_cancel_before_work_installs_tombstone_and_never_allocates():
    reg, work = registry(), request()
    cancelled = reg.handle(
        control(work, "cancel"), lambda _: pytest.fail("unexpected work")
    )
    assert cancelled.status == "CANCELLED_NOT_STARTED"
    replay = reg.handle(
        work, lambda _: pytest.fail("delayed work allocated after cancel")
    )
    assert replay.status == "CANCELLED_NOT_STARTED"
    assert ack(reg, work, replay).status == "ACKNOWLEDGED"
    with pytest.raises(ValueError, match="watermark"):
        reg.handle(work, lambda _: None)


def test_lost_terminal_ack_and_retire_retries_are_safe_after_compaction():
    reg, work = registry(), request()
    freed = []
    result = reg.handle(
        work,
        lambda _: reg.add_lease(
            work.identity,
            (object(),),
            participants=("tp0",),
            release=lambda owners: freed.extend(owners),
        ),
    )
    retire = control(work, "retire", {"lease_id": result.value, "proof": "fenced"})
    assert reg.handle(retire, lambda _: None).status == "RETIRED"
    assert ack(reg, work, result).status == "ACKNOWLEDGED"
    assert reg.snapshot()["operations"] == 0
    # Replies may be lost after the service has already compacted the record.
    assert ack(reg, work, result).status == "ACKNOWLEDGED"
    assert reg.handle(retire, lambda _: None).status == "RETIRED"
    assert len(freed) == 1
    with pytest.raises(ValueError, match="watermark"):
        reg.handle(work, lambda _: pytest.fail("Old work must never allocate again"))


def test_duplicate_while_running_is_nonblocking_and_result_is_replayed():
    reg, work = registry(), request()
    started, finish = Event(), Event()
    calls = []

    def dispatch(_):
        calls.append(1)
        started.set()
        assert finish.wait(5)
        return {"lease": "value"}

    with ThreadPoolExecutor(1) as pool:
        original = pool.submit(reg.handle, work, dispatch)
        assert started.wait(5)
        assert reg.handle(work, dispatch).status == "PENDING"
        assert reg.handle(control(work, "cancel"), dispatch).value[
            "cancellation_requested"
        ]
        finish.set()
        result = original.result(5)
    assert reg.handle(work, dispatch).value == result.value
    assert len(calls) == 1


def test_changed_payload_or_request_attempt_cannot_reuse_operation_id():
    reg, work = registry(), request()
    reg.handle(work, lambda _: None)
    with pytest.raises(ValueError, match="different inputs"):
        reg.handle(request(payload={"different": True}), lambda _: None)
    import msgspec

    changed = msgspec.structs.replace(
        work, identity=msgspec.structs.replace(work.identity, request_attempt="other")
    )
    with pytest.raises(ValueError, match="different inputs"):
        reg.handle(changed, lambda _: None)


def test_all_participants_and_opposite_bank_producer_must_finish_before_release():
    reg, work, freed = registry(), request(), []
    enabled = gc.isenabled()
    gc.disable()
    try:
        result = reg.handle(
            work,
            lambda _: reg.add_lease(
                work.identity,
                (object(),),
                participants=("tp0", "tp1"),
                release=lambda objects: freed.extend(objects),
                producer_holds=2,
            ),
        )
        lease_id = result.value
        for participant in ("tp0", "tp1"):
            reg.handle(
                control(
                    work,
                    "retire",
                    {"lease_id": lease_id, "proof": "fenced"},
                    participant,
                ),
                lambda _: None,
            )
        assert not freed
        reg.producer_done(lease_id, known_complete=True)
        assert not freed
        assert ack(reg, work, result).status == "PENDING"
        reg.producer_done(lease_id, known_complete=True)
        assert len(freed) == 1
        reg.handle(
            control(work, "retire", {"lease_id": lease_id, "proof": "fenced"}),
            lambda _: None,
        )
        assert len(freed) == 1
        assert ack(reg, work, result).status == "ACKNOWLEDGED"
    finally:
        if enabled:
            gc.enable()


def test_unknown_native_completion_retains_actual_owners_and_history():
    reg, work, freed = registry(max_operations=1), request(), []
    result = reg.handle(
        work,
        lambda _: reg.add_lease(
            work.identity,
            (object(),),
            participants=("tp0",),
            release=lambda objects: freed.extend(objects),
            producer_holds=1,
        ),
    )
    reg.producer_done(result.value, known_complete=False)
    reg.handle(
        control(work, "retire", {"lease_id": result.value, "proof": "fenced"}),
        lambda _: None,
    )
    assert not freed
    assert ack(reg, work, result).status == "PENDING"
    with pytest.raises(StorageBusy):
        reg.handle(request(2), lambda _: None)


def test_acknowledged_out_of_order_history_compacts_only_contiguous_ids():
    reg = registry(max_operations=3)
    works = [request(index) for index in (1, 2, 3)]
    results = [reg.handle(work, lambda _: None) for work in works]
    assert ack(reg, works[2], results[2]).status == "ACKNOWLEDGED"
    assert reg.snapshot()["operations"] == 2
    assert reg.snapshot()["watermarks"]["tp0"] == 0
    assert ack(reg, works[0], results[0]).status == "ACKNOWLEDGED"
    assert reg.snapshot()["operations"] == 1
    assert reg.snapshot()["watermarks"]["tp0"] == 1
    assert ack(reg, works[1], results[1]).status == "ACKNOWLEDGED"
    assert reg.snapshot()["operations"] == 0


def test_long_live_source_does_not_retain_completed_request_manifests():
    reg = registry(max_operations=4)
    long = request(1)
    result = reg.handle(
        long,
        lambda _: reg.add_lease(
            long.identity,
            (object(),),
            participants=("tp0",),
            release=lambda _: None,
        ),
    )
    for index in range(2, 22):
        work = request(index)
        short = reg.handle(work, lambda _: {"manifest": b"done"})
        assert ack(reg, work, short).status == "ACKNOWLEDGED"
    assert reg.snapshot()["leases"] == 1
    assert reg.snapshot()["operations"] == 1
    assert reg.snapshot()["watermarks"]["tp0"] == 0
    with pytest.raises(ValueError, match="watermark"):
        reg.handle(request(8), lambda _: pytest.fail("Acknowledged work replayed"))
    reg.handle(
        control(long, "retire", {"lease_id": result.value, "proof": "fenced"}),
        lambda _: None,
    )
    assert ack(reg, long, result).status == "ACKNOWLEDGED"
    assert reg.snapshot()["watermarks"]["tp0"] == 21


def test_wrong_epoch_or_nonce_cannot_create_cancellation_tombstone():
    import msgspec

    reg, work = registry(), request()
    cancel = control(work, "cancel")
    for bad in (
        msgspec.structs.replace(cancel, session_nonce=b"wrong"),
        msgspec.structs.replace(
            cancel, identity=msgspec.structs.replace(work.identity, service_epoch="old")
        ),
    ):
        with pytest.raises(ValueError, match="authentication"):
            reg.handle(bad, lambda _: None)
    assert reg.snapshot()["operations"] == 0


def test_peer_cannot_cancel_origin_or_retire_without_safety_proof():
    reg, work = registry(), request()
    result = reg.handle(
        work,
        lambda _: reg.add_lease(
            work.identity,
            (object(),),
            participants=("tp0", "tp1"),
            release=lambda _: None,
        ),
    )
    with pytest.raises(ValueError, match="original session"):
        reg.handle(control(work, "cancel", participant="tp1"), lambda _: None)
    with pytest.raises(ValueError, match="safety proof"):
        reg.handle(
            control(work, "retire", {"lease_id": result.value}, "tp1"), lambda _: None
        )
