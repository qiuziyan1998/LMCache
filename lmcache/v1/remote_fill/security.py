# SPDX-License-Identifier: Apache-2.0
"""Canonical digests and capabilities for remote-fill control messages."""

# Standard
from hashlib import blake2b
from hmac import compare_digest, digest
import json
from typing import Any

# Third Party
import msgspec

# Local
from .protocol import (
    ControlPage,
    DestinationPageDescriptor,
    OperationIdentity,
    RemoteFillRequest,
)


def canonical_bytes(value: Any) -> bytes:
    """Encode a msgspec-compatible value into deterministic JSON bytes.

    Args:
        value: Struct or built-in value to encode.

    Returns:
        UTF-8 encoded canonical JSON with sorted mapping keys.
    """

    builtins = msgspec.to_builtins(value, builtin_types=(str, int, float, bool))
    return json.dumps(
        builtins,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    """Compute a compact BLAKE2b digest for a canonical value.

    Args:
        value: Struct or built-in value to digest.

    Returns:
        A 64-character hexadecimal digest.
    """

    return blake2b(canonical_bytes(value), digest_size=32).hexdigest()


def request_payload_digest(request: RemoteFillRequest) -> str:
    """Compute a request digest with the digest field itself blanked.

    Args:
        request: Request whose payload is authenticated for idempotency.

    Returns:
        A stable hexadecimal digest over all other request fields.
    """

    unsigned_common = msgspec.structs.replace(request.common, payload_digest="")
    unsigned_request = msgspec.structs.replace(request, common=unsigned_common)
    return content_digest(unsigned_request)


def seal_request(request: RemoteFillRequest) -> RemoteFillRequest:
    """Return a request carrying its correct payload digest.

    Args:
        request: Request to seal. Its existing digest is ignored.

    Returns:
        A new immutable request with ``common.payload_digest`` populated.
    """

    payload_digest = request_payload_digest(request)
    common = msgspec.structs.replace(request.common, payload_digest=payload_digest)
    return msgspec.structs.replace(request, common=common)


def manifest_digest(pages: tuple[ControlPage, ...]) -> str:
    """Digest an ordered, pointer-free window page manifest.

    Args:
        pages: Control pages in their serialized order.

    Returns:
        A stable hexadecimal manifest digest.
    """

    return content_digest(pages)


def transaction_manifest_digest(
    manifest_digest_seed: str,
    window_manifests: tuple[tuple[int, str], ...],
    required_store_end: int,
    final_partial_valid_tokens: int,
) -> str:
    """Bind the final transaction extent to all ordered window manifests.

    Args:
        manifest_digest_seed: Per-transaction seed carried by ``OPEN``.
        window_manifests: ``(window_id, manifest_digest)`` pairs.
        required_store_end: Authoritative final store-plan token end.
        final_partial_valid_tokens: Valid tokens in the final partial page.

    Returns:
        A stable hexadecimal digest over the exact final manifest summary.
    """

    ordered = tuple(sorted(window_manifests))
    return content_digest(
        {
            "manifest_digest_seed": manifest_digest_seed,
            "window_manifests": ordered,
            "required_store_end": required_store_end,
            "final_partial_valid_tokens": final_partial_valid_tokens,
        }
    )


def descriptor_capability_mac(
    secret: bytes,
    descriptor: DestinationPageDescriptor,
) -> str:
    """Compute the HMAC capability for one destination descriptor.

    Args:
        secret: Deployment-scoped shared secret.
        descriptor: Descriptor to authenticate. Its existing MAC is ignored.

    Returns:
        A hexadecimal SHA-256 HMAC.
    """

    unsigned = msgspec.structs.replace(descriptor, capability_mac="")
    return digest(secret, canonical_bytes(unsigned), "sha256").hex()


def seal_descriptor(
    secret: bytes,
    descriptor: DestinationPageDescriptor,
) -> DestinationPageDescriptor:
    """Return a descriptor carrying its capability HMAC.

    Args:
        secret: Deployment-scoped shared secret.
        descriptor: Descriptor to seal.

    Returns:
        A new immutable descriptor with ``capability_mac`` populated.
    """

    capability_mac = descriptor_capability_mac(secret, descriptor)
    return msgspec.structs.replace(descriptor, capability_mac=capability_mac)


def verify_descriptor(
    secret: bytes,
    descriptor: DestinationPageDescriptor,
) -> bool:
    """Verify a descriptor capability without exposing its address.

    Args:
        secret: Deployment-scoped shared secret.
        descriptor: Descriptor received from the destination.

    Returns:
        ``True`` only when the HMAC matches.
    """

    return compare_digest(
        descriptor.capability_mac,
        descriptor_capability_mac(secret, descriptor),
    )


def destination_descriptor_digest(
    descriptors: tuple[DestinationPageDescriptor, ...],
) -> str:
    """Digest exact per-page capabilities for whole-window arming.

    Args:
        descriptors: Ordered descriptors returned by reservation.

    Returns:
        A stable hexadecimal descriptor-set digest.
    """

    return content_digest(descriptors)


def identity_without_digest(identity: OperationIdentity) -> OperationIdentity:
    """Return an operation identity with a blank payload digest.

    Args:
        identity: Identity to copy.

    Returns:
        An immutable identity suitable for constructing an unsealed request.
    """

    return msgspec.structs.replace(identity, payload_digest="")
