# SPDX-License-Identifier: Apache-2.0
"""Independent real socket/registry review of source terminal reclamation."""

from concurrent.futures import TimeoutError
import time
import threading

import pytest
import zmq

from lmcache.v1.cpu_cache import StorageOperationRegistry
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    StorageResult,
    content_digest,
    decode_message,
    encode_message,
)
from lmcache.v1.storage_service import StorageService

from tests.v1.test_storage_service_process import NONCE, _client, _endpoint, _request


def wait_for(condition):
    deadline = time.monotonic() + 3
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.002)


class RegistryRuntime:
    def __init__(self):
        self.registry = StorageOperationRegistry("cache", "epoch", {"worker-0": NONCE})
        self.freed = 0

    def handle(self, request):
        def prepare(work):
            if work.method == "health":
                return True
            lease = self.registry.add_lease(
                work.identity,
                (object(),),
                participants=("worker-0",),
                release=self.release,
            )
            return {"lease_id": lease}

        return self.registry.handle(request, prepare)

    def release(self, objects):
        self.freed += 1

    def drain_terminal_notifications(self):
        return self.registry.drain_terminal_notifications()

    def is_healthy(self):
        return self.registry.is_healthy()

    def close(self):
        assert self.registry.snapshot()["leases"] == 0


def test_more_than_client_capacity_sources_auto_ack_without_step_polling():
    runtime = RegistryRuntime()
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
    try:
        # Establish its lifecycle route once; source final-access retirement
        # then triggers M's authenticated terminal notice, without status RPCs.
        health = _request(client, "health")
        assert health.result(2).status == "COMPLETED"
        client.acknowledge_terminal(health.identity).result(2)
        for _ in range(24):
            prepared = _request(client, "prepare")
            lease = prepared.result(2).value["lease_id"]
            assert (
                client.retire(
                    prepared.identity, prepared.payload_digest, lease, "fenced"
                )
                .result(2)
                .status
                == "RETIRED"
            )
            wait_for(lambda key=prepared.identity.key: key not in client._records)
            assert (
                client.acknowledge_terminal(prepared.identity).result(2).status
                == "ACKNOWLEDGED"
            )
        assert runtime.freed == 24
        assert runtime.registry.snapshot()["operations"] == 0
    finally:
        client.close()
        service.close()


def test_busy_before_registry_admission_requires_original_cancel_tombstone():
    """A saturated real service returns BUSY while registry status stays UNKNOWN."""
    started, unblock = threading.Event(), threading.Event()

    class PausedRuntime(RegistryRuntime):
        def handle(self, request):
            if request.method == "prepare" and request.payload.get("pause"):
                started.set()
                assert unblock.wait(3)
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
    client = _client(work, lifecycle)
    try:
        first = _request(client, "prepare", {"pause": True})
        assert started.wait(2)
        busy = _request(client, "prepare", {})
        assert busy.result(2).status == "BUSY"
        unknown = client.operation_status(busy.identity).result(2)
        assert unknown.status == "UNKNOWN" and unknown.value is None
        cancelled = client.cancel(busy.identity).result(2)
        assert cancelled.status == "CANCELLED_NOT_STARTED"
        assert (
            client.acknowledge_terminal(
                busy.identity,
                {
                    "terminal_result_digest": cancelled.value["terminal_result_digest"],
                },
            )
            .result(2)
            .status
            == "ACKNOWLEDGED"
        )
        unblock.set()
        result = first.result(2)
        client.retire(
            first.identity, first.payload_digest, result.value["lease_id"], "fenced"
        ).result(2)
        wait_for(lambda: first.identity.key not in client._records)
        assert runtime.registry.snapshot()["operations"] == 0
    finally:
        unblock.set()
        client.close()
        service.close()


@pytest.mark.parametrize("invalid", ["nonce", "digest", "correlation"])
def test_forged_terminal_nonce_before_work_reply_cannot_resolve_operation(invalid):
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.RCVTIMEO, 2000)
    endpoint = _endpoint()
    router.bind(endpoint)
    client = _client(endpoint, _endpoint())
    try:
        future = _request(client, "prepare")
        route, *frames = router.recv_multipart()
        request = decode_message(frames, StorageRequest)
        forged = StorageResult(
            request.identity,
            "TERMINAL",
            {"terminal_result_digest": b"x" * 32},
            payload_digest=b"wrong" if invalid == "digest" else request.payload_digest,
            session_nonce=b"wrong" if invalid == "nonce" else NONCE,
            correlation_id=2 if invalid == "correlation" else 0,
        )
        router.send_multipart([route, *encode_message(forged)])
        with pytest.raises(TimeoutError):
            future.result(0.05)
        assert client.ignored_responses == 1
        valid = StorageResult(
            request.identity,
            "COMPLETED",
            {"lease_id": "x"},
            payload_digest=request.payload_digest,
        )
        router.send_multipart([route, *encode_message(valid)])
        assert future.result(2) == valid
    finally:
        client.close()
        router.close(linger=0)
        context.term()


def test_terminal_notice_before_original_reply_waits_for_exact_result_digest():
    context = zmq.Context()
    routers = [context.socket(zmq.ROUTER), context.socket(zmq.ROUTER)]
    endpoints = [_endpoint(), _endpoint()]
    for router, endpoint in zip(routers, endpoints, strict=False):
        router.setsockopt(zmq.RCVTIMEO, 2000)
        router.bind(endpoint)
    client = _client(*endpoints)
    try:
        future = _request(client, "prepare")
        route, *frames = routers[0].recv_multipart()
        request = decode_message(frames, StorageRequest)
        result = StorageResult(
            request.identity,
            "COMPLETED",
            {"lease_id": "x"},
            payload_digest=request.payload_digest,
        )
        notice = StorageResult(
            request.identity,
            "TERMINAL",
            {
                "terminal_result_digest": content_digest(
                    (result.status, result.value, result.error)
                )
            },
            payload_digest=request.payload_digest,
            session_nonce=NONCE,
        )
        routers[0].send_multipart([route, *encode_message(notice)])
        with pytest.raises(TimeoutError):
            future.result(0.05)
        routers[0].send_multipart([route, *encode_message(result)])
        assert future.result(2) == result
        ack_route, *frames = routers[1].recv_multipart()
        ack = decode_message(frames, StorageRequest)
        assert ack.method == "acknowledge_terminal"
        assert ack.identity == request.identity
        routers[1].send_multipart(
            [
                ack_route,
                *encode_message(
                    StorageResult(
                        ack.identity,
                        "ACKNOWLEDGED",
                        payload_digest=ack.payload_digest,
                        correlation_id=ack.correlation_id,
                    )
                ),
            ]
        )
        wait_for(lambda: request.identity.key not in client._records)
    finally:
        client.close()
        for router in routers:
            router.close(linger=0)
        context.term()


def test_terminal_ack_retries_when_foreign_retirements_release_lifecycle_credit():
    """Pause lifecycle replies on the wire, preserving the real client IO loop."""
    context = zmq.Context()
    routers = [context.socket(zmq.ROUTER), context.socket(zmq.ROUTER)]
    endpoints = [_endpoint(), _endpoint()]
    for router, endpoint in zip(routers, endpoints, strict=False):
        router.setsockopt(zmq.RCVTIMEO, 1000)
        router.bind(endpoint)
    client = _client(*endpoints, max_pending=1)
    try:
        future = _request(client, "prepare")
        route, *frames = routers[0].recv_multipart()
        request = decode_message(frames, StorageRequest)
        result = StorageResult(
            request.identity,
            "COMPLETED",
            {"lease_id": "own"},
            payload_digest=request.payload_digest,
        )
        routers[0].send_multipart([route, *encode_message(result)])
        future.result(2)
        controls = []
        for operation in range(1, 9):
            peer = OperationIdentity(
                "cache", "epoch", "worker-1", operation, "peer", "attempt", 1, "prepare"
            )
            client.retire(peer, b"p" * 32, f"peer-{operation}", "fenced")
            peer_route, *frames = routers[1].recv_multipart()
            controls.append((peer_route, decode_message(frames, StorageRequest)))
        notice = StorageResult(
            request.identity,
            "TERMINAL",
            {
                "terminal_result_digest": content_digest(
                    (result.status, result.value, result.error)
                )
            },
            payload_digest=request.payload_digest,
            session_nonce=NONCE,
        )
        routers[0].send_multipart([route, *encode_message(notice)])
        wait_for(
            lambda: client._records[request.identity.key].terminal_digest is not None
        )
        for peer_route, control in controls:
            routers[1].send_multipart(
                [
                    peer_route,
                    *encode_message(
                        StorageResult(
                            control.identity,
                            "RETIRED",
                            payload_digest=control.payload_digest,
                            correlation_id=control.correlation_id,
                        )
                    ),
                ]
            )
        # Credits are now free. No unrelated work/model step should be required
        # to wake reclamation of the already-terminal original source.
        wait_for(lambda: len(client._pending) <= 2)
        ack_route, *frames = routers[1].recv_multipart()
        ack = decode_message(frames, StorageRequest)
        assert ack.method == "acknowledge_terminal"
        assert ack.identity == request.identity
        routers[1].send_multipart(
            [
                ack_route,
                *encode_message(
                    StorageResult(
                        ack.identity,
                        "ACKNOWLEDGED",
                        payload_digest=ack.payload_digest,
                        correlation_id=ack.correlation_id,
                    )
                ),
            ]
        )
        wait_for(lambda: request.identity.key not in client._records)
    finally:
        client.close()
        for router in routers:
            router.close(linger=0)
        context.term()
