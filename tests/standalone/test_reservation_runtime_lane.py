# SPDX-License-Identifier: Apache-2.0
"""Real socket routing and CPU allocator; device operations are not exercised."""

# Standard
from contextlib import contextmanager
from threading import Event
from typing import Any, Iterator
import gc

# Third Party
import pytest

# First Party
from lmcache.v1.cpu_cache import StorageFailedSafe
from lmcache.v1.rpc.storage_client import StorageClient, StorageFuture
from lmcache.v1.rpc.storage_protocol import PrepareSpec, StorageRequest, StorageResult
from lmcache.v1.storage_service import StorageService

# Local
import test_ascend_cpu_storage_runtime
from tests.v1.test_storage_service_process import _endpoint

runtime_setup = test_ascend_cpu_storage_runtime.setup


class FixtureRuntime:
    """Leave fixture-managed backend shutdown to the actual runtime fixture."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def handle(self, request: StorageRequest) -> StorageResult:
        """Run the real CPU runtime on the service's selected executor."""
        return self.runtime.handle(request)

    def is_healthy(self) -> bool:
        """Use the actual runtime's cached health contract."""
        return self.runtime.is_healthy()

    def close(self) -> None:
        """Check all CPU ownership before the enclosing fixture closes storage."""
        self.runtime.check_close_ready()


@contextmanager
def serving(runtime: Any) -> Iterator[StorageClient]:
    """Serve one source-loaded runtime through real sockets and client threads."""
    work, lifecycle = _endpoint(), _endpoint()
    service = StorageService(
        FixtureRuntime(runtime),
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions=runtime.registry.sessions,
        max_work=4,
    )
    service.start()
    client = StorageClient(
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="worker-0",
        session_nonce=b"a",
    )
    try:
        yield client
    finally:
        service.close()
        client.close()


def request(client: StorageClient, method: str, payload: Any) -> StorageFuture:
    """Keep the same request attempt across allocation and cancellation controls."""
    return client.request(
        method,
        payload,
        request_id="lane-request",
        request_attempt="attempt",
        load_generation=1,
        purpose="reservation-lane-test",
    )


def reserve(client: StorageClient) -> StorageFuture:
    """Reserve both actual group layouts, including mixed index keys/scales."""
    return request(
        client,
        "reserve_write",
        {
            "purpose": "checkpoint",
            "generation": 1,
            "base": 0,
            "end": 1,
            "resident_start": 0,
            "prefix_end": 0,
            "request_configs": {},
        },
    )


def cancel_reservation(client: StorageClient, original: StorageFuture) -> None:
    """Release private producer holds, then retire exported worker ownership."""
    value = original.result(2).value
    cancelled = request(
        client,
        "cancel_write",
        {"reservation_id": value["reservation_id"], "proof": "never_imported"},
    )
    assert cancelled.result(2).value["status"] == "CANCELLED"
    for group in value["groups"]:
        assert (
            client.retire(
                original.identity,
                original.payload_digest,
                group["lease_id"],
                "never_imported",
            )
            .result(2)
            .status
            == "RETIRED"
        )
    for work in (original, cancelled):
        assert (
            client.acknowledge_terminal(work.identity).result(2).status
            == "ACKNOWLEDGED"
        )


def test_real_cpu_reservation_runs_beside_held_preparation(
    runtime_setup: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, runtime, _ = runtime_setup
    entered, release = Event(), Event()
    free_before = owner.allocator.address_manager.get_free_size()

    def prepare(*args: Any, **kwargs: Any) -> Any:
        # Hold a real CPU allocation while the separate reservation lane allocates.
        page, _ = runtime.allocate_checkpoint_fragment(0, 1)
        try:
            entered.set()
            assert release.wait(5)
            raise StorageFailedSafe("controlled prepare miss")
        finally:
            page.ref_count_down()

    monkeypatch.setattr(runtime, "prepare", prepare)
    with serving(runtime) as client:
        preparation = request(
            client, "prepare", PrepareSpec("held", b"plan", 1, b"layout", ("worker-0",))
        )
        try:
            assert entered.wait(2)
            destination = reserve(client)
            assert destination.result(2).status == "COMPLETED"
            assert {
                entry["group_id"] for entry in destination.result().value["groups"]
            } == {0, 1}
            assert not preparation.done()
        finally:
            release.set()
        assert preparation.result(2).status == "FAILED_SAFE"
        cancel_reservation(client, destination)
        assert (
            client.acknowledge_terminal(preparation.identity).result(2).status
            == "ACKNOWLEDGED"
        )
        assert owner.allocator.address_manager.get_free_size() == free_before


def test_real_reservation_limit_and_reclaim_survive_gc_disabled(
    runtime_setup: Any,
) -> None:
    owner, runtime, _ = runtime_setup
    free_before = owner.allocator.address_manager.get_free_size()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with serving(runtime) as client:
            first, second = reserve(client), reserve(client)
            assert first.result(2).status == second.result(2).status == "COMPLETED"
            refused = reserve(client)
            assert refused.result(2).status == "FAILED_SAFE"
            assert "staging is busy" in refused.result().error
            assert (
                client.acknowledge_terminal(refused.identity).result(2).status
                == "ACKNOWLEDGED"
            )
            held_free = owner.allocator.address_manager.get_free_size()
            assert held_free < free_before
            cancel_reservation(client, first)
            assert owner.allocator.address_manager.get_free_size() > held_free
            replacement = reserve(client)
            assert replacement.result(2).status == "COMPLETED"
            cancel_reservation(client, second)
            cancel_reservation(client, replacement)
            assert runtime.registry.snapshot()["leases"] == 0
            assert owner.allocator.address_manager.get_free_size() == free_before
    finally:
        if was_enabled:
            gc.enable()
