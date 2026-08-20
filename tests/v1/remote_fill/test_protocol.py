# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for the bounded remote-fill wire protocol."""

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.remote_fill import (
    ProtocolLimits,
    ProtocolValidationError,
    RemoteFillService,
    ResultCode,
    decode_request,
    decode_response,
    destination_descriptor_digest,
    encode_request,
    seal_request,
    verify_descriptor,
)


def test_control_manifest_is_pointer_free(harness) -> None:
    """Only page identities and sizes cross the request boundary."""

    request = seal_request(harness.requests.reserve())
    encoded = encode_request(request, ProtocolLimits())

    assert b"source_ptr" not in encoded
    assert b"destination_ptr" not in encoded
    decoded = decode_request(encoded, ProtocolLimits())
    assert decoded.control_pages == request.control_pages


def test_unknown_pointer_field_is_rejected(harness) -> None:
    """Unknown fields cannot smuggle source-layout state through RPC."""

    request = seal_request(harness.requests.reserve())
    builtins = msgspec.to_builtins(request)
    builtins["source_ptr"] = 0xBAD

    with pytest.raises(ProtocolValidationError, match="invalid remote-fill message"):
        decode_request(msgspec.msgpack.encode(builtins), ProtocolLimits())


def test_unknown_response_field_is_rejected(harness) -> None:
    """A response cannot append unvalidated address-bearing fields."""

    harness.client.execute(harness.requests.open())
    response = harness.client.execute(harness.requests.reserve())
    builtins = msgspec.to_builtins(response)
    builtins["source_ptr"] = 0xBAD

    with pytest.raises(ProtocolValidationError, match="invalid remote-fill response"):
        decode_response(msgspec.msgpack.encode(builtins), ProtocolLimits())


def test_page_count_is_bounded(harness) -> None:
    """A window cannot serialize more control pages than configured."""

    request = seal_request(harness.requests.reserve(harness.requests.pages(5)))

    with pytest.raises(ProtocolValidationError, match="page count exceeds"):
        encode_request(request, ProtocolLimits(max_control_pages_per_window=8))


def test_message_size_is_checked_before_decode(harness) -> None:
    """Oversized untrusted input produces a bounded pointer-free error."""

    limits = ProtocolLimits(max_rpc_message_bytes=512)
    state = harness.state
    service = RemoteFillService(state, limits)
    response = decode_response(service.handle_bytes(b"x" * 513), limits)

    assert response.code is ResultCode.INVALID_MESSAGE
    assert response.descriptors == ()


def test_oversized_reservation_response_is_rejected_before_allocation(
    harness,
) -> None:
    """A valid request cannot allocate pages when its reply would overflow."""

    harness.client.execute(harness.requests.open())
    pages = tuple(
        msgspec.structs.replace(
            page,
            canonical_key=f"key-{index}-" + "x" * 4000,
        )
        for index, page in enumerate(harness.requests.pages())
    )
    request = seal_request(harness.requests.reserve(pages))
    generous = ProtocolLimits()
    request_bytes = encode_request(request, generous)
    limits = ProtocolLimits(max_rpc_message_bytes=len(request_bytes) + 64)
    encoded = encode_request(request, limits)
    service = RemoteFillService(harness.state, limits)
    before = service.metrics_snapshot()

    response = decode_response(service.handle_bytes(encoded), limits)
    later = seal_request(
        harness.requests.reserve(
            harness.requests.pages(start_chunk=1),
            window_id=1,
        )
    )
    later_response = decode_response(
        service.handle_bytes(encode_request(later, limits)),
        limits,
    )
    after = service.metrics_snapshot()

    assert response.code is ResultCode.RESOURCE_EXHAUSTED
    assert response.descriptors == ()
    assert later_response.code is ResultCode.TERMINAL
    assert harness.lifecycle.prepare_calls == 0
    assert after.active_windows == before.active_windows == 0
    assert after.active_bytes == before.active_bytes == 0
    assert after.reserved_bytes_total == before.reserved_bytes_total == 0
    assert after.capacity_rejections_total == before.capacity_rejections_total + 1


def test_response_preflight_bound_always_fits_real_reservation(harness) -> None:
    """The smallest accepted worst-case bound fits the real success reply."""

    harness.client.execute(harness.requests.open())
    request = seal_request(harness.requests.reserve())
    generous = ProtocolLimits()
    encoded = encode_request(request, generous)
    low = len(encoded)
    high = generous.max_rpc_message_bytes
    while low < high:
        candidate = (low + high) // 2
        limits = msgspec.structs.replace(
            generous,
            max_rpc_message_bytes=candidate,
        )
        try:
            harness.state.validate_response_capacity(request, limits)
        except ProtocolValidationError:
            low = candidate + 1
        else:
            high = candidate

    limits = msgspec.structs.replace(generous, max_rpc_message_bytes=low)
    harness.lifecycle.next_address = (1 << 64) - 0x800000
    service = RemoteFillService(harness.state, limits)

    response = decode_response(service.handle_bytes(encoded), limits)

    assert response.code is ResultCode.OK
    assert response.descriptors
    assert harness.lifecycle.prepare_calls == 1


def test_oversized_stale_request_cannot_abandon_active_transaction(harness) -> None:
    """Preflight rejection mutates only an otherwise eligible reservation."""

    harness.client.execute(harness.requests.open())
    pages = tuple(
        msgspec.structs.replace(
            page,
            canonical_key=f"stale-{index}-" + "x" * 4000,
        )
        for index, page in enumerate(harness.requests.pages())
    )
    request = harness.requests.reserve(pages)
    request = msgspec.structs.replace(
        request,
        common=msgspec.structs.replace(
            request.common,
            destination_engine_epoch=8,
        ),
    )
    request = seal_request(request)
    generous = ProtocolLimits()
    request_bytes = encode_request(request, generous)
    limits = msgspec.structs.replace(
        generous,
        max_rpc_message_bytes=len(request_bytes) + 64,
    )
    service = RemoteFillService(harness.state, limits)

    rejected = decode_response(
        service.handle_bytes(encode_request(request, limits)),
        limits,
    )
    accepted = harness.client.execute(harness.requests.reserve())

    assert rejected.code is ResultCode.RESOURCE_EXHAUSTED
    assert accepted.code is ResultCode.OK
    assert accepted.transaction_state is not None
    assert harness.lifecycle.prepare_calls == 1


def test_response_preflight_rejection_does_not_run_page_maintenance(harness) -> None:
    """Oversized rejection never releases or reallocates lifecycle pages."""

    harness.client.execute(harness.requests.open())
    first = harness.requests.reserve()
    assert harness.client.execute(first).code is ResultCode.OK
    harness.clock.advance(6.0)
    pages = tuple(
        msgspec.structs.replace(
            page,
            canonical_key=f"next-{index}-" + "x" * 4000,
        )
        for index, page in enumerate(harness.requests.pages(start_chunk=1))
    )
    request = seal_request(harness.requests.reserve(pages, window_id=1))
    generous = ProtocolLimits()
    encoded = encode_request(request, generous)
    limits = msgspec.structs.replace(
        generous,
        max_rpc_message_bytes=len(encoded) + 64,
    )
    service = RemoteFillService(harness.state, limits)

    response = decode_response(service.handle_bytes(encoded), limits)

    assert response.code is ResultCode.RESOURCE_EXHAUSTED
    assert harness.lifecycle.prepare_calls == 1
    assert harness.lifecycle.released == []


def test_descriptor_hmac_and_set_digest_cover_noncontiguous_pages(harness) -> None:
    """Every returned physical page has an authenticated distinct capability."""

    harness.client.execute(harness.requests.open())
    reserve = harness.requests.reserve(harness.requests.pages(4))
    response = harness.client.execute(reserve)

    assert len(response.descriptors) == 8
    pointers = [descriptor.destination_ptr for descriptor in response.descriptors]
    assert len(set(pointers)) == 8
    assert all(
        verify_descriptor(b"test deployment secret", descriptor)
        for descriptor in response.descriptors
    )
    assert (
        destination_descriptor_digest(response.descriptors)
        == response.destination_descriptor_digest
    )
    changed = msgspec.structs.replace(
        response.descriptors[0],
        destination_length=response.descriptors[0].destination_length + 1,
    )
    assert not verify_descriptor(b"test deployment secret", changed)


def test_response_descriptor_digest_detects_pointer_mutation(harness) -> None:
    """The decoded descriptor set remains bound to its aggregate digest."""

    harness.client.execute(harness.requests.open())
    response = harness.client.execute(harness.requests.reserve())
    builtins = msgspec.to_builtins(response)
    builtins["descriptors"][0]["destination_ptr"] += 1

    with pytest.raises(ProtocolValidationError, match="descriptor digest mismatch"):
        decode_response(msgspec.msgpack.encode(builtins), ProtocolLimits())


def test_incomplete_two_group_chunk_is_rejected(harness) -> None:
    """A group-0-only page cannot form a direct-fill control window."""

    pages = harness.requests.pages()[:1]
    request = seal_request(harness.requests.reserve(pages))

    with pytest.raises(ProtocolValidationError, match="exactly groups 0 and 1"):
        encode_request(request, ProtocolLimits())


def test_control_page_interval_must_equal_valid_tokens(harness) -> None:
    """Wire validation rejects ambiguous full-span partial-page metadata."""

    pages = tuple(
        msgspec.structs.replace(page, valid_tokens=512)
        for page in harness.requests.pages()
    )
    request = seal_request(harness.requests.reserve(pages))

    with pytest.raises(ProtocolValidationError, match="interval must equal"):
        encode_request(request, ProtocolLimits())


def test_payload_digest_detects_mutation(harness) -> None:
    """Changing a sealed request without resealing is rejected locally."""

    request = seal_request(harness.requests.open())
    changed = msgspec.structs.replace(request, request_id="another-request")

    with pytest.raises(ProtocolValidationError, match="payload digest mismatch"):
        encode_request(changed, ProtocolLimits())


def test_negotiation_binds_builtin_hash_seed(harness) -> None:
    """P and D reject equal layouts with inconsistent built-in hash identity."""

    assert harness.client.execute(harness.requests.negotiate()).code is ResultCode.OK
    mismatch = harness.requests.negotiate(
        token_hash_algorithm="builtin",
        python_hash_seed="7",
    )
    assert harness.client.execute(mismatch).code is ResultCode.RESERVATION_REJECTED


def test_nonbuiltin_hash_rejects_python_seed(harness) -> None:
    """A PYTHONHASHSEED identity is carried only for built-in hashing."""

    request = seal_request(
        harness.requests.negotiate(
            token_hash_algorithm="sha256_cbor",
            python_hash_seed="0",
        )
    )
    with pytest.raises(ProtocolValidationError, match="only for builtin"):
        encode_request(request, ProtocolLimits())
