# SPDX-License-Identifier: Apache-2.0
"""Actual CPU runtime behind sockets; body validation stays off the I/O loop."""

# Standard
from threading import Event, current_thread
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import zmq

# First Party
import lmcache.v1.cpu_cache as cpu_cache
import lmcache.v1.storage_service as storage_service
from lmcache.v1.rpc.storage_client import StorageClient
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    StorageResult,
    decode_message,
    encode_message,
)
from lmcache.v1.storage_service import StorageService

# Local
import test_cpu_cache_runtime
from tests.v1.test_storage_service_process import _endpoint, _request

runtime_setup = test_cpu_cache_runtime.setup


def test_actual_cpu_runtime_hashes_work_once_outside_io(
    runtime_setup: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner, runtime, _, bootstrap, populate = runtime_setup
    bootstrap()
    populate()
    free_before = owner.allocator.address_manager.get_free_size()
    entered, release = Event(), Event()
    hashes = []
    original = cpu_cache.content_digest

    def digest(value: Any) -> bytes:
        if isinstance(value, tuple) and len(value) == 2 and value[0] == "lookup":
            hashes.append(current_thread().name)
            entered.set()
            assert release.wait(5)
        return original(value)

    monkeypatch.setattr(cpu_cache, "content_digest", digest)
    # The baseline also imports this function into the socket-service module.
    # Observe both former call sites, so this regression detects a body walk on
    # the socket thread rather than only counting the registry's own validation.
    monkeypatch.setattr(storage_service, "content_digest", digest, raising=False)
    work, lifecycle = _endpoint(), _endpoint()
    # Leave backend closure to the fixture while using the actual runtime methods.
    service = StorageService(
        SimpleNamespace(
            handle=runtime.handle,
            handle_validated=runtime.handle_validated,
            set_transport_notifier=runtime.set_transport_notifier,
            drain_terminal_notifications=runtime.drain_terminal_notifications,
            is_healthy=runtime.is_healthy,
            close=runtime.check_close_ready,
        ),
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions=runtime.registry.sessions,
    )
    service.start()
    client = StorageClient(
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="scheduler",
        session_nonce=b"s",
    )
    context = zmq.Context()
    raw = context.socket(zmq.DEALER)
    raw.setsockopt(zmq.RCVTIMEO, 2000)
    raw.connect(work)
    try:
        source = _request(
            client,
            "lookup",
            {
                "starts": (0, 4),
                "ends": (4, 5),
                "hashes": (7, 8),
                "request_configs": {},
            },
        )
        assert entered.wait(2)
        # Hashing is deliberately held. Lifecycle control still reaches the real
        # registry, proving that a large body cannot occupy the socket owner.
        assert client.operation_status(source.identity).result(1).status == "UNKNOWN"
        release.set()
        reply = source.result(2)
        assert reply.status == "COMPLETED"
        assert reply.value["matched_tokens"] == 5
        assert len(hashes) == 1 and hashes[0].startswith("storage-lookup")
        client.cancel(source.identity).result(2)
        assert (
            client.acknowledge_terminal(source.identity).result(2).status
            == "ACKNOWLEDGED"
        )
        before = runtime.registry.snapshot()["operations"]
        request = StorageRequest(
            OperationIdentity(
                "cache", "epoch", "scheduler", 100, "bad", "attempt", 1, "prepare"
            ),
            "prepare",
            {"invalid": True},
            b"0" * 32,
            b"s",
        )
        raw.send_multipart(encode_message(request))
        rejected = decode_message(raw.recv_multipart(), StorageResult)
        assert rejected.status == "REJECTED"
        assert runtime.registry.snapshot()["operations"] == before
        assert runtime.registry.snapshot()["leases"] == 0
        assert owner.allocator.address_manager.get_free_size() == free_before
    finally:
        release.set()
        # A failed assertion while the digest is held must still drop any lookup
        # ticket produced after release, before the fixture unmaps its real slab.
        source.result(2)
        try:
            client.cancel(source.identity).result(2)
            client.acknowledge_terminal(source.identity).result(2)
        except ValueError:
            pass  # The successful path already acknowledged this exact source.
        client.close()
        raw.close(linger=0)
        context.term()
        service.close()
