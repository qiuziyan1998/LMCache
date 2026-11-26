# SPDX-License-Identifier: Apache-2.0
"""Packed seals keep exact tokens, checkpoint ownership and legacy fallback."""

# Standard
from typing import Any
import gc

# Third Party
import msgspec
import pytest
import torch
import zmq

# First Party
from lmcache.integration.vllm.preemption_checkpoint import SealSpec
from lmcache.v1.cpu_cache import StorageFailedSafe, StorageOperationRegistry
from lmcache.v1.rpc.storage_client import StorageClient
from lmcache.v1.rpc.storage_protocol import (
    OperationIdentity,
    StorageRequest,
    content_digest,
    decode_message,
    encode_message,
    pack_checkpoint_tokens_u32,
    unpack_uints,
)

# Local
from test_ascend_cpu_storage_runtime import (
    capture,
    fill_capture,
    retire,
    setup as cpu_setup,
)
from test_process_checkpoint_worker import (
    finish,
    spec,
    wait_drained,
    worker as checkpoint_worker,
)

setup = cpu_setup
worker = checkpoint_worker


@pytest.mark.parametrize("tokens", [(), (0,), (2**32 - 1,), (0, 151329, 2**32 - 1)])
def test_compiled_uint32_tokens_round_trip(tokens: tuple[int, ...]) -> None:
    frame = pack_checkpoint_tokens_u32(tokens)
    assert type(frame) is bytes
    assert unpack_uints(frame, len(tokens), 32) == tokens


@pytest.mark.parametrize("bad", [True, False, -1, 1.0, "1", None, 2**32, 2**64])
def test_uint32_codec_rejects_invalid_types_and_overflow(bad: Any) -> None:
    with pytest.raises(msgspec.ValidationError):
        pack_checkpoint_tokens_u32((0, bad))


@pytest.mark.parametrize("count", [100_000, 1_000_000])
def test_full_token_prefix_freezes_into_immutable_replay_frames(count: int) -> None:
    tokens = [index % 160_000 for index in range(count)]
    packed = pack_checkpoint_tokens_u32(tokens)
    payload = {"phase": "seal", "token_count": count, "tokens_u32_le": packed}
    identity = OperationIdentity("cache", "epoch", "worker-0", 1, "r", "a", 1, "seal")
    request = StorageRequest(
        identity,
        "commit_write",
        payload,
        content_digest(("commit_write", payload)),
        b"n",
    )
    frames = encode_message(request)
    frozen = decode_message(frames, StorageRequest)
    tokens[0] = 99
    assert len(frames) == 2
    assert frames[1] is packed
    assert encode_message(frozen) == frames
    assert frozen.payload_digest == content_digest((frozen.method, frozen.payload))
    restored = unpack_uints(frozen.payload["tokens_u32_le"], count, 32)
    assert restored[0] == 0
    assert restored == tuple(index % 160_000 for index in range(count))


@pytest.mark.parametrize("capability", [None, False, True])
def test_actual_worker_negotiates_packed_seal_and_preserves_legacy(
    worker: Any, capability: bool | None
) -> None:
    instance, runtime, client, _ = worker
    instance.context.descriptor = (
        {} if capability is None else {"checkpoint_tokens_u32_le_v1": capability}
    )
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    assert instance.poll()[0].status == "captured"
    instance.seal(SealSpec("request", 1, tuple(range(7))))
    assert [(result.status, result.end) for result in finish(instance)] == [
        ("ready", 7)
    ]
    wait_drained(instance)
    sent = next(
        payload for method, payload in client.controls if method == "commit_write"
    )
    if capability is True:
        assert "tokens" not in sent
        assert unpack_uints(sent["tokens_u32_le"], sent["token_count"], 32) == tuple(
            range(7)
        )
    else:
        assert sent["tokens"] == tuple(range(7))
        assert "tokens_u32_le" not in sent
    assert runtime.checkpoint_store.manifest("request", 1).tokens == tuple(range(7))


def test_advertised_sender_keeps_supported_wide_integers_exact(worker: Any) -> None:
    instance, runtime, client, _ = worker
    instance.context.descriptor = runtime.bootstrap_descriptor()
    assert instance.context.descriptor["checkpoint_tokens_u32_le_v1"] is True
    tokens = (0, 2**32, 2**63, 3, 4, 5, 6)
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    instance.seal(SealSpec("request", 1, tokens))
    assert finish(instance)[0].status == "ready"
    wait_drained(instance)
    sent = next(
        payload for method, payload in client.controls if method == "commit_write"
    )
    assert sent["tokens"] == tokens
    assert "tokens_u32_le" not in sent
    assert runtime.checkpoint_store.manifest("request", 1).tokens == tokens


@pytest.mark.parametrize("bad", [True, -1, 1.0, "1", None])
def test_advertised_worker_rejects_invalid_token_types_without_publication(
    worker: Any, bad: Any
) -> None:
    instance, runtime, client, _ = worker
    instance.context.descriptor = runtime.bootstrap_descriptor()
    instance.capture(spec(), {0: [None] * 2, 1: [None] * 3}, 4)
    instance.poll()
    instance.seal(SealSpec("request", 1, (0, bad, 2, 3, 4, 5, 6)))
    assert finish(instance)[0].status == "failed"
    wait_drained(instance)
    assert not any(method == "commit_write" for method, _ in client.controls)
    assert runtime.checkpoint_store.manifest("request", 1) is None
    assert runtime.registry.snapshot()["leases"] == 0


def test_actual_client_replay_keeps_frozen_packed_bytes_after_input_mutation() -> None:
    context = zmq.Context()
    routers = [context.socket(zmq.ROUTER) for _ in range(2)]
    endpoints = []
    for router in routers:
        router.setsockopt(zmq.RCVTIMEO, 2000)
        port = router.bind_to_random_port("tcp://127.0.0.1")
        endpoints.append(f"tcp://127.0.0.1:{port}")
    client = StorageClient(
        *endpoints,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="worker-0",
        session_nonce=b"n",
    )
    registry = StorageOperationRegistry("cache", "epoch", {"worker-0": b"n"})
    try:
        tokens = list(range(100_000))
        payload = {
            "token_count": len(tokens),
            "tokens_u32_le": pack_checkpoint_tokens_u32(tokens),
        }
        future = client.request(
            "commit_write",
            payload,
            request_id="r",
            request_attempt="a",
            load_generation=1,
            purpose="seal",
        )
        route, *first = routers[0].recv_multipart()
        payload["tokens_u32_le"] = b"mutated"
        payload["token_count"] = 0
        tokens[0] = 99
        assert client.replay(future.identity) is future
        repeated_route, *repeated = routers[0].recv_multipart()
        assert repeated_route == route
        assert repeated == first
        request = decode_message(repeated, StorageRequest)
        assert unpack_uints(
            request.payload["tokens_u32_le"], request.payload["token_count"], 32
        ) == tuple(range(100_000))
        result = registry.handle(request, lambda _: None)
        routers[0].send_multipart([route, *encode_message(result)])
        assert future.result(2).status == "COMPLETED"
    finally:
        client.close()
        for router in routers:
            router.close(linger=0)
        context.term()


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"token_count": 7},
        {"tokens_u32_le": b"\0" * 28},
        {"token_count": True, "tokens_u32_le": b"\0" * 4},
        {"token_count": 7.0, "tokens_u32_le": b"\0" * 28},
        {"token_count": "7", "tokens_u32_le": b"\0" * 28},
        {"token_count": -1, "tokens_u32_le": b""},
        {"token_count": 2**64, "tokens_u32_le": b""},
        {"token_count": 7, "tokens_u32_le": b"\0" * 27},
        {"token_count": 7, "tokens_u32_le": b"\0" * 29},
        {"token_count": 7, "tokens_u32_le": bytearray(28)},
        {"token_count": 0, "tokens_u32_le": b""},
        {"token_count": 10, "tokens_u32_le": b"\0" * 40},
        {"token_count": 7, "tokens_u32_le": b"\0" * 28, "tokens": tuple(range(7))},
    ],
)
def test_malformed_packed_seal_preserves_private_holds_and_never_publishes(
    setup: Any, fields: dict[str, Any]
) -> None:
    _, runtime, work = setup
    reserved, original = capture(work)
    reply = reserved.value
    reservation = runtime._write_reservations[reply["reservation_id"]]
    before = runtime.handle(
        msgspec.structs.replace(original, method="operation_status", payload={})
    ).value["leases"]
    # Direct call avoids msgpack's own bool/oversized-int rejection, ensuring the
    # receiver independently validates adversarial in-process requests as well.
    with pytest.raises(StorageFailedSafe):
        runtime.commit_write(
            {
                "reservation_id": reply["reservation_id"],
                "phase": "seal",
                "proof": "fenced",
                **fields,
            },
            original.identity,
        )
    after = runtime.handle(
        msgspec.structs.replace(original, method="operation_status", payload={})
    ).value["leases"]
    assert after == before
    assert not reservation.captured
    assert not reservation.committing
    assert runtime.checkpoint_store.manifest("request", 1) is None
    runtime.cancel_write(
        {"reservation_id": reply["reservation_id"], "proof": "never_imported"},
        original.identity,
    )
    for group in reply["groups"]:
        retire(runtime, original, group["lease_id"])
    assert runtime.registry.snapshot()["leases"] == 0


def test_packed_accepted_tail_keeps_mixed_c8_scales_and_restore_prefix_gc_off(
    setup: Any,
) -> None:
    _, runtime, work = setup
    enabled = gc.isenabled()
    gc.disable()
    try:
        reserved, original = capture(work)
        reply = reserved.value
        fill_capture(runtime, reply["reservation_id"])
        sealed, _ = work(
            "worker-0",
            "commit_write",
            {
                "reservation_id": reply["reservation_id"],
                "phase": "seal",
                "proof": "fenced",
                "token_count": 7,
                "tokens_u32_le": pack_checkpoint_tokens_u32(tuple(range(7))),
            },
        )
        assert sealed.value["status"] == "COMMITTED", sealed.error
        for group in reply["groups"]:
            retire(runtime, original, group["lease_id"])
        assert runtime.registry.snapshot()["leases"] == 0
        assert runtime.checkpoint_store.manifest("request", 1).tokens == tuple(range(7))
        base, owners = runtime.checkpoint_store.normalize(
            "request", 1, list(range(7)), {}
        )
        try:
            assert base == 0
            index_keys = list(
                runtime.token_database.process_tokens(tokens=list(range(7)), kv_group=1)
            )
            pages, count = runtime.checkpoint_backend().batched_get_layer_page_prefix(
                [index_keys[-1][2]]
            )
            assert count == 1
            page = pages[0]
            try:
                assert page.valid_tokens == 3
                for layer, planes in enumerate(runtime.checkpoint_plane_widths(1)):
                    begin = 0
                    for plane, width in enumerate(planes):
                        end = begin + width * 3
                        assert torch.all(
                            page.layer_tensor(layer)[begin:end]
                            == 20 + layer * 3 + plane + 1
                        )
                        begin = end
            finally:
                page.ref_count_down()
        finally:
            runtime._release_shared_retrieve_objs(owners, unpin=False)
        with pytest.raises(ValueError, match="history is unavailable"):
            runtime.checkpoint_store.normalize("request", 1, [99, 1, 2, 3, 4, 5, 6], {})
    finally:
        if enabled:
            gc.enable()
