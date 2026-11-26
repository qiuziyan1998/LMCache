# SPDX-License-Identifier: Apache-2.0
"""Actual registry backpressure and CPU owners at checkpoint retirement."""

# Standard
from threading import Event
from types import SimpleNamespace as NS

# First Party
from lmcache.v1.storage_service import StorageService

# Local
from test_process_checkpoint_worker import setup, spec, wait_drained, worker  # noqa: F401


def test_lifecycle_cancel_busy_retries_original_after_actual_registry_recovers(
    request, monkeypatch
):
    instance, runtime, client, _ = request.getfixturevalue("worker")
    # RuntimeClient excludes socket dispatch. Apply the actual service handler,
    # which turns the real registry's StorageBusy exception into its wire result.
    service = NS(runtime=NS(handle=runtime.handle), _reply=StorageService._reply)
    cancel_busy = Event()
    cancel_ids = []

    def dispatch(request):
        result = StorageService._handle(service, request)
        if request.method == "cancel" and result.status == "BUSY":
            cancel_ids.append(request.identity)
            cancel_busy.set()
        return result

    monkeypatch.setattr(runtime, "handle", dispatch)
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    real, extras = client.request, []

    def fill_during_cancel(method, payload, **kwargs):
        if method == "cancel_write" and not extras:
            # A terminal result awaiting its legitimate ACK occupies the other
            # registry slot. Its pages are unrelated to the fenced checkpoint.
            extra = real(
                "admin",
                {"method": "snapshot"},
                request_id="other",
                request_attempt="other-attempt",
                load_generation=0,
                purpose="admin",
            )
            assert extra.result().status == "COMPLETED"
            extras.append(extra)
            runtime.registry.max_operations = 2
            assert runtime.registry.snapshot()["operations"] == 2
        return real(method, payload, **kwargs)

    monkeypatch.setattr(client, "request", fill_during_cancel)
    instance.cancel("request", 1)
    assert cancel_busy.wait(2)
    state = instance.reservations["request", 1]
    original = state.cancel_future.identity
    assert state.cancel_future.result().status == "BUSY"
    assert not state.released and not state.job.quarantined
    assert runtime.registry.snapshot()["leases"] == 2
    client.acknowledge_terminal(extras[0].identity).result()
    # No decode poll drives this completion. Retained views and the reservation
    # survive the refused lifecycle control, then drain after capacity recovers.
    wait_drained(instance)
    assert cancel_ids and all(identity == original for identity in cancel_ids)
    assert runtime.registry.snapshot()["leases"] == 0
    assert not state.job.quarantined
    assert [method for method, _ in client.controls].count("cancel_write") == 2
