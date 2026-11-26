# SPDX-License-Identifier: Apache-2.0
"""Independent real-socket registry-capacity and uncertainty classification."""

from lmcache.v1.cpu_cache import StorageOperationRegistry
from lmcache.v1.storage_service import StorageService

from tests.v1.test_storage_service_process import NONCE, _client, _endpoint, _request


class Runtime:
    def __init__(self, *, fail=False):
        self.registry = StorageOperationRegistry(
            "cache", "epoch", {"worker-0": NONCE}, max_operations=1
        )
        self.allocations, self.freed, self.fail = 0, 0, fail

    def handle(self, request):
        def dispatch(work):
            if self.fail:
                raise RuntimeError("producer completion is unknown")
            self.allocations += 1
            return self.registry.add_lease(
                work.identity,
                (object(),),
                participants=("worker-0",),
                release=lambda objects: self.release(),
            )

        return self.registry.handle(request, dispatch)

    def release(self):
        self.freed += 1

    def is_healthy(self):
        return self.registry.is_healthy()

    def close(self):
        assert self.registry.snapshot()["leases"] == 0


def start(runtime):
    work, lifecycle = _endpoint(), _endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"worker-0": NONCE},
    )
    service.start()
    return service, _client(work, lifecycle)


def test_full_registry_defers_work_and_cancel_without_claiming_unknown_completion():
    runtime = Runtime()
    service, client = start(runtime)
    first = result = None
    try:
        first = _request(client, "prepare", {})
        result = first.result(2)
        assert result.status == "COMPLETED"
        second = _request(client, "prepare", {})
        assert second.result(2).status == "BUSY"
        assert client.operation_status(second.identity).result(2).status == "UNKNOWN"
        assert client.cancel(second.identity).result(2).status == "BUSY"
        assert runtime.allocations == 1 and runtime.freed == 0
        assert runtime.registry.snapshot()["operations"] == 1
        assert runtime.is_healthy()

        # Draining the genuine live operation frees the bounded cancellation
        # history slot. Retry the same original identity, never replacement work.
        client.retire(
            first.identity, first.payload_digest, result.value, "fenced"
        ).result(2)
        assert (
            client.acknowledge_terminal(first.identity).result(2).status
            == "ACKNOWLEDGED"
        )
        cancelled = client.cancel(second.identity).result(2)
        assert cancelled.status == "CANCELLED_NOT_STARTED"
        assert (
            client.acknowledge_terminal(
                second.identity,
                {"terminal_result_digest": cancelled.value["terminal_result_digest"]},
            )
            .result(2)
            .status
            == "ACKNOWLEDGED"
        )
        assert runtime.allocations == runtime.freed == 1
        assert runtime.registry.snapshot()["operations"] == 0
    finally:
        if first is not None and result is not None:
            client.retire(
                first.identity, first.payload_digest, result.value, "fenced"
            ).result(2)
            client.acknowledge_terminal(first.identity).result(2)
        client.close()
        service.close()


def test_noncapacity_runtime_failure_remains_unknown_and_fatal():
    runtime = Runtime(fail=True)
    service, client = start(runtime)
    try:
        failed = _request(client, "prepare", {}).result(2)
        assert failed.status == "FAILED_UNKNOWN"
        assert "producer completion is unknown" in failed.error
        assert not runtime.is_healthy()
        assert runtime.registry.snapshot()["quarantined"] == 1
    finally:
        client.close()
        service.close()
