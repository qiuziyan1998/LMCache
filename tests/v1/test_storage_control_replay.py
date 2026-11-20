# SPDX-License-Identifier: Apache-2.0
"""Real socket retry after a deliberately lost, already-applied terminal ACK."""

import zmq

from lmcache.v1.cpu_cache import StorageOperationRegistry
from lmcache.v1.rpc import storage_client
from lmcache.v1.rpc.storage_protocol import (
    StorageRequest,
    decode_message,
    encode_message,
)
from tests.v1.test_storage_service_process import NONCE, _client, _endpoint, _request


def test_lost_terminal_reply_reuses_the_same_control_identity_and_credit():
    context = zmq.Context()
    routers = [context.socket(zmq.ROUTER) for _ in range(2)]
    endpoints = [_endpoint(), _endpoint()]
    for router, endpoint in zip(routers, endpoints, strict=True):
        router.setsockopt(zmq.RCVTIMEO, 500)
        router.bind(endpoint)
    client = _client(*endpoints, max_pending=1)
    registry = StorageOperationRegistry("cache", "epoch", {"worker-0": NONCE})
    try:
        future = _request(client, "prepare", {})
        route, *frames = routers[0].recv_multipart()
        work = decode_message(frames, StorageRequest)
        result = registry.handle(work, lambda _: None)
        routers[0].send_multipart([route, *encode_message(result)])
        future.result(2)
        ack = client.acknowledge_terminal(future.identity)
        route, *frames = routers[1].recv_multipart()
        first = decode_message(frames, StorageRequest)
        assert registry.handle(first, lambda _: None).status == "ACKNOWLEDGED"
        assert registry.snapshot()["operations"] == 0
        # Drop that reply. A caller retry must replay the exact admitted control,
        # rather than returning an inert Future or allocating a replacement ID.
        assert client.acknowledge_terminal(future.identity) is ack
        repeated_route, *repeated = routers[1].recv_multipart()
        assert repeated_route == route
        assert repeated == frames
        retry = decode_message(repeated, StorageRequest)
        routers[1].send_multipart(
            [route, *encode_message(registry.handle(retry, lambda _: None))]
        )
        assert ack.result(2).status == "ACKNOWLEDGED"
    finally:
        client.close()
        for router in routers:
            router.close(linger=0)
        context.term()


def test_control_retries_hash_only_new_payload(monkeypatch):
    client = _client(_endpoint(), _endpoint())
    try:
        future = _request(client, "prepare", {})
        payload = {"terminal_result_digest": b"original"}
        ack = client.acknowledge_terminal(future.identity, payload)
        # The admitted control owns its frozen bytes, even if the caller reuses
        # the dictionary. Its digest can be reused only for that frozen record.
        payload["terminal_result_digest"] = b"changed"
        digest = storage_client.content_digest
        calls = []

        def counted(value):
            calls.append(value)
            return digest(value)

        monkeypatch.setattr(storage_client, "content_digest", counted)
        for _ in range(10):
            assert client.acknowledge_terminal(
                future.identity, {"terminal_result_digest": b"original"}
            ) is ack
        assert len(calls) == 10
        assert client.acknowledge_terminal(future.identity, payload) is not ack
    finally:
        client.close()
