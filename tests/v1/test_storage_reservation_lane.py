# SPDX-License-Identifier: Apache-2.0
"""Real sockets and registry with controlled CPU work; no device transfers."""

# Standard
from collections import Counter
from contextlib import contextmanager
from threading import Event, Lock, current_thread
from typing import Iterator

# Third Party
import pytest

# First Party
from lmcache.v1.cpu_cache import StorageFailedSafe, StorageOperationRegistry
from lmcache.v1.rpc.storage_client import StorageClient
from lmcache.v1.rpc.storage_protocol import StorageRequest, StorageResult
from lmcache.v1.storage_service import StorageService

# Local
from tests.v1.test_storage_service_process import NONCE, _client, _endpoint, _request


class ReservationRuntime:
    """Use the actual operation registry around event-controlled CPU callbacks."""

    def __init__(self) -> None:
        self.registry = StorageOperationRegistry("cache", "epoch", {"worker-0": NONCE})
        self.gates: dict[str, Event] = {}
        self.entered: dict[str, Event] = {}
        self.threads: dict[str, list[str]] = {}
        self.allocations = 0
        self.calls: Counter[str] = Counter()
        self.active = 0
        self.closed = Event()
        self.lock = Lock()

    def block(self, method: str) -> tuple[Event, Event]:
        """Hold one work method until its test explicitly releases the gate."""
        self.gates[method] = Event()
        self.entered[method] = Event()
        return self.entered[method], self.gates[method]

    def handle(self, request: StorageRequest) -> StorageResult:
        """Keep authentication, replay and cancellation on the real registry."""
        return self.registry.handle(request, self._dispatch)

    def is_healthy(self) -> bool:
        """Return the registry's cached health without blocking on held work."""
        return self.registry.is_healthy()

    def close(self) -> None:
        """Prove close occurs after every admitted callback finishes."""
        with self.lock:
            assert not self.active
            self.closed.set()

    def _dispatch(self, request: StorageRequest) -> dict:
        method = request.method
        with self.lock:
            self.active += 1
            self.calls[method] += 1
            self.threads.setdefault(method, []).append(current_thread().name)
        try:
            if method in self.gates:
                self.entered[method].set()
                assert self.gates[method].wait(10), "test did not release held work"
            if self.registry.cancellation_requested(request.identity):
                raise StorageFailedSafe("cancelled before export")
            with self.lock:
                if method == "reserve_write":
                    self.allocations += 1
                return {"method": method, "allocations": self.allocations}
        finally:
            with self.lock:
                self.active -= 1


@contextmanager
def running(
    runtime: ReservationRuntime, *, max_work: int = 2
) -> Iterator[tuple[StorageService, StorageClient]]:
    """Start real service/client socket owners and release test gates on failure."""
    work, lifecycle = _endpoint(), _endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"worker-0": NONCE},
        max_work=max_work,
    )
    service.start()
    client = _client(work, lifecycle)
    try:
        yield service, client
    finally:
        for gate in runtime.gates.values():
            gate.set()
        service.close()
        client.close()


def test_reservation_finishes_while_bulk_prepare_is_held() -> None:
    runtime = ReservationRuntime()
    entered, release = runtime.block("prepare")
    with running(runtime) as (_, client):
        preparation = _request(client, "prepare")
        assert entered.wait(2)
        reserved = _request(client, "reserve_write").result(2)
        assert reserved.status == "COMPLETED"
        assert reserved.value["allocations"] == 1
        assert not preparation.done()
        assert runtime.threads["reserve_write"][0].startswith("storage-reserve")
        assert runtime.threads["prepare"][0].startswith("storage-prepare")
        release.set()
        assert preparation.result(2).status == "COMPLETED"


def test_reservation_and_prepare_share_the_existing_bulk_admission_budget() -> None:
    runtime = ReservationRuntime()
    preparing, release_prepare = runtime.block("prepare")
    reserving, release_reserve = runtime.block("reserve_write")
    with running(runtime, max_work=2) as (_, client):
        preparation = _request(client, "prepare")
        assert preparing.wait(2)
        reservation = _request(client, "reserve_write")
        assert reserving.wait(2)
        assert _request(client, "reserve_write").result(2).status == "BUSY"
        assert _request(client, "prepare").result(2).status == "BUSY"
        assert _request(client, "lookup").result(2).status == "COMPLETED"
        assert _request(client, "persist").result(2).status == "COMPLETED"
        assert runtime.calls == {
            "prepare": 1,
            "reserve_write": 1,
            "lookup": 1,
            "persist": 1,
        }
        release_prepare.set()
        release_reserve.set()
        assert preparation.result(2).status == "COMPLETED"
        assert reservation.result(2).status == "COMPLETED"


def test_reservation_replay_and_cancellation_do_not_allocate_twice() -> None:
    runtime = ReservationRuntime()
    entered, release = runtime.block("reserve_write")
    with running(runtime) as (_, client):
        cancelled = _request(client, "reserve_write")
        assert entered.wait(2)
        assert client.replay(cancelled.identity) is cancelled
        assert client.cancel(cancelled.identity).result(2).status == "RUNNING"
        release.set()
        assert cancelled.result(2).status == "FAILED_SAFE"
        assert runtime.allocations == 0
        assert (
            client.acknowledge_terminal(cancelled.identity).result(2).status
            == "ACKNOWLEDGED"
        )

        allocated = _request(client, "reserve_write")
        assert allocated.result(2).value["allocations"] == 1
        assert client.replay(allocated.identity) is allocated
        status = client.operation_status(allocated.identity).result(2)
        assert status.value["result"]["value"]["allocations"] == 1
        assert runtime.allocations == 1
        assert runtime.calls["reserve_write"] == 2
        assert (
            client.acknowledge_terminal(allocated.identity).result(2).status
            == "ACKNOWLEDGED"
        )


def test_close_waits_for_the_reservation_lane_without_releasing_runtime() -> None:
    runtime = ReservationRuntime()
    entered, release = runtime.block("reserve_write")
    with running(runtime) as (service, client):
        reservation = _request(client, "reserve_write")
        assert entered.wait(2)
        with pytest.raises(TimeoutError, match="unresolved"):
            service.close(0.05)
        assert not runtime.closed.is_set()
        assert _request(client, "lookup").result(2).status == "BUSY"
        assert (
            client.operation_status(reservation.identity).result(2).status == "RUNNING"
        )
        release.set()
        assert reservation.result(2).status == "COMPLETED"
        service.close()
        assert runtime.closed.is_set()
