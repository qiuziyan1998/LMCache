# SPDX-License-Identifier: Apache-2.0
"""Service-local body proofs preserve direct-call validation and ownership."""

# Standard
from threading import Event
import gc

# Third Party
import msgspec
import pytest

# First Party
import lmcache.v1.cpu_cache as cache_module
from lmcache.v1.cpu_cache import validate_storage_work

# Local
from test_storage_registry import control, registry, request
import test_cpu_cache_runtime as runtime_tests


@pytest.fixture
def runtime_setup():
    yield from runtime_tests.setup.__wrapped__()


def test_validated_entry_hashes_body_once_and_replays_without_allocation(monkeypatch):
    reg, work = registry(), request(payload={"frame": b"x" * 4096})
    digest, bodies, allocations = cache_module.content_digest, [], []

    def counted(value):
        if isinstance(value, tuple) and value[0] == "prepare":
            bodies.append(value)
        return digest(value)

    monkeypatch.setattr(cache_module, "content_digest", counted)
    proof = validate_storage_work(work)
    result = reg.handle_validated(work, lambda _: allocations.append(1), proof)
    replay = reg.handle_validated(work, lambda _: pytest.fail("duplicate work"), proof)
    assert result.status == replay.status == "COMPLETED"
    assert allocations == [1] and len(bodies) == 1


def test_proof_is_local_to_exact_request_and_cannot_replace_authentication():
    reg, work = registry(), request()
    proof = validate_storage_work(work)
    duplicate = msgspec.structs.replace(work)
    with pytest.raises(ValueError, match="proof"):
        reg.handle_validated(duplicate, lambda _: None, proof)
    with pytest.raises(ValueError, match="proof"):
        reg.handle_validated(work, lambda _: None, {"validated": True})
    forged = msgspec.structs.replace(work, session_nonce=b"wrong")
    with pytest.raises(ValueError, match="authentication"):
        reg.handle_validated(forged, lambda _: None, validate_storage_work(forged))
    assert reg.snapshot()["operations"] == 0


@pytest.mark.parametrize("entry", ["handle", "validate_storage_work"])
def test_bad_direct_work_is_rejected_before_admission(entry):
    reg = registry()
    malformed = msgspec.structs.replace(request(), payload_digest=b"0" * 32)
    with pytest.raises(ValueError, match="digest"):
        if entry == "handle":
            reg.handle(malformed, lambda _: pytest.fail("bad work dispatched"))
        else:
            validate_storage_work(malformed)
    assert reg.snapshot()["operations"] == 0


def test_lifecycle_keeps_original_digest_instead_of_hashing_control_payload():
    reg, work = registry(), request()
    cancellation = control(work, "cancel")
    with pytest.raises(ValueError, match="Lifecycle"):
        validate_storage_work(cancellation)
    result = reg.handle(cancellation, lambda _: pytest.fail("control dispatched"))
    assert result.status == "CANCELLED_NOT_STARTED"


@pytest.mark.parametrize("unknown", [False, True])
def test_validated_runtime_retains_dispatch_failure_classification(
    runtime_setup, unknown
):
    _, runtime, _, _, _ = runtime_setup
    from lmcache.v1.cpu_cache import StorageFailedSafe

    def failed(_):
        if unknown:
            raise ValueError("injected internal dispatch failure")
        raise StorageFailedSafe("injected safe failure")

    runtime._dispatch = failed
    # Use the fixture's exact authenticated request identity rather than a wire
    # shortcut: all authentication still runs on the validated entry.
    work = runtime_tests.StorageRequest(
        runtime_tests.OperationIdentity(
            "cache", "epoch", "worker-0", 99, "req", "attempt", 1, "load"
        ),
        "prepare",
        {},
        cache_module.content_digest(("prepare", {})),
        b"a",
    )
    result = runtime.handle_validated(work, validate_storage_work(work))
    assert result.status == ("FAILED_UNKNOWN" if unknown else "FAILED_SAFE")
    assert runtime.registry.is_healthy() == (not unknown)


@pytest.mark.parametrize("failure", ["explicit", "dispatch", "release"])
def test_every_fatal_transition_signals_idle_transport(failure):
    reg, work = registry(), request()
    wake = Event()
    assert reg.set_transport_notifier(wake.set)
    if failure == "explicit":
        reg.mark_fatal("injected fatal")
    elif failure == "dispatch":

        def bad(_):
            raise RuntimeError("injected dispatch failure")

        assert reg.handle(work, bad).status == "FAILED_UNKNOWN"
    else:

        def release(_):
            raise RuntimeError("injected release failure")

        saved = reg.handle(
            work,
            lambda _: reg.add_lease(
                work.identity,
                (object(),),
                participants=("tp0",),
                release=release,
            ),
        )
        with pytest.raises(RuntimeError, match="release failure"):
            reg.handle(
                control(work, "retire", {"lease_id": saved.value, "proof": "fenced"}),
                lambda _: None,
            )
    assert wake.is_set() and not reg.is_healthy()
    wake.clear()
    assert reg.set_transport_notifier(wake.set)  # pending failure on attachment
    assert wake.is_set()
    reg.set_transport_notifier(None)


def test_terminal_notification_signals_after_last_physical_hold_with_gc_off():
    reg, work = registry(), request()
    wake, freed = Event(), []
    reg.set_transport_notifier(wake.set)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        saved = reg.handle(
            work,
            lambda _: reg.add_lease(
                work.identity,
                (object(),),
                participants=("tp0",),
                release=lambda owners: freed.extend(owners),
                producer_holds=1,
            ),
        )
        assert not wake.is_set()
        reg.handle(
            control(work, "retire", {"lease_id": saved.value, "proof": "fenced"}),
            lambda _: None,
        )
        assert not wake.is_set() and not freed
        reg.producer_done(saved.value, known_complete=True)
        assert wake.is_set() and len(freed) == 1
        assert len(reg.drain_terminal_notifications()) == 1
    finally:
        reg.set_transport_notifier(None)
        if was_enabled:
            gc.enable()
