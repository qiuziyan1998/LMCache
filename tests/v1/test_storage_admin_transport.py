# SPDX-License-Identifier: Apache-2.0
"""Real transport admission pressure against the explicit management facade."""

import threading
from types import SimpleNamespace as NS

import pytest

from lmcache.v1.rpc.storage_admin import StorageAdminClient
from lmcache.v1.storage_service import StorageService

from tests.v1.test_storage_service_process import NONCE, _client, _endpoint, _request
from tests.v1.test_storage_terminal_notifications import RegistryRuntime, wait_for


def test_busy_admin_reconciles_original_id_without_exhausting_client_capacity():
    started, unblock = threading.Event(), threading.Event()

    class PausedRuntime(RegistryRuntime):
        def handle(self, request):
            if request.method == "pause_work":
                started.set()
                assert unblock.wait(5)
                return self.registry.handle(request, lambda _: True)
            return super().handle(request)

    runtime = PausedRuntime()
    work, lifecycle = _endpoint(), _endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"worker-0": NONCE},
        max_work=1,
    )
    service.start()
    client = _client(work, lifecycle, max_pending=4)
    admin = StorageAdminClient(client, NS(blocking_timeout_secs=2, local_cpu=True))
    try:
        first = _request(client, "pause_work")
        assert started.wait(2)
        for _ in range(12):
            # Explicit API error semantics remain BUSY, but a rejected operation
            # cannot retain a client slot forever after its CPU cancellation.
            with pytest.raises(RuntimeError, match="BUSY"):
                admin.list_backends()
            wait_for(lambda: not admin._pending)
            assert set(client._records) == {first.identity.key}
        unblock.set()
        assert first.result(2).status == "COMPLETED"
        client.acknowledge_terminal(first.identity).result(2)
        assert runtime.registry.snapshot()["operations"] == 0
    finally:
        unblock.set()
        first.result(2)
        client.acknowledge_terminal(first.identity).result(2)
        client.close()
        service.close()


def test_admin_reconciles_busy_cancel_after_registry_capacity_recovers():
    class FullRuntime(RegistryRuntime):
        def __init__(self):
            super().__init__()
            self.registry.max_operations = 2

        def handle(self, request):
            if request.method == "admin":
                return self.registry.handle(
                    request, lambda _: {"LocalCPUBackend": "online"}
                )
            return super().handle(request)

    runtime = FullRuntime()
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
    client = _client(work, lifecycle, max_pending=4)
    admin = StorageAdminClient(client, NS(blocking_timeout_secs=2, local_cpu=True))
    try:
        first = _request(client, "prepare")
        lease = first.result(2).value["lease_id"]
        second = _request(client, "prepare")
        second_lease = second.result(2).value["lease_id"]
        with pytest.raises(RuntimeError, match="history is full"):
            admin.list_backends()
        wait_for(
            lambda: admin._pending
            and all(f.done() for f in admin._pending.values())
        )
        assert all(f.result().status == "BUSY" for f in admin._pending.values())

        client.retire(
            first.identity, first.payload_digest, lease, "fenced"
        ).result(2)
        client.acknowledge_terminal(first.identity).result(2)
        client.retire(
            second.identity, second.payload_digest, second_lease, "fenced"
        ).result(2)
        client.acknowledge_terminal(second.identity).result(2)
        # A later explicit API call must retry the original cancellation; it
        # cannot leave the refused call consuming a client credit permanently.
        assert admin.list_backends() == {"LocalCPUBackend": "online"}
        wait_for(lambda: not admin._pending)
        assert runtime.registry.snapshot()["operations"] == 0
    finally:
        client.retire(
            first.identity, first.payload_digest, lease, "fenced"
        ).result(2)
        client.acknowledge_terminal(first.identity).result(2)
        client.retire(
            second.identity, second.payload_digest, second_lease, "fenced"
        ).result(2)
        client.acknowledge_terminal(second.identity).result(2)
        client.close()
        service.close()
