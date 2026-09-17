# SPDX-License-Identifier: Apache-2.0
"""Bounded msgspec codec and schema validation for remote fill."""

# Standard
from string import hexdigits
import math

# Third Party
import msgspec

# Local
from .protocol import (
    PROTOCOL_VERSION,
    AbortRequest,
    ArmWindowRequest,
    ControlPage,
    FinishRequest,
    NegotiateRequest,
    OpenRequest,
    OperationKind,
    ProtocolLimits,
    RemoteFillRequest,
    RemoteFillResponse,
    ReportTransferCompleteRequest,
    ReserveWindowRequest,
    StatusRequest,
)
from .security import (
    destination_descriptor_digest,
    manifest_digest,
    request_payload_digest,
)


class ProtocolValidationError(ValueError):
    """A safe validation failure that never includes destination pointers."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


def encode_request(
    request: RemoteFillRequest,
    limits: ProtocolLimits,
) -> bytes:
    """Validate and encode one request within the configured byte bound.

    Args:
        request: Sealed request to serialize.
        limits: Protocol bounds.

    Returns:
        Msgpack-encoded request bytes.

    Raises:
        ProtocolValidationError: If validation or the byte bound fails.
    """

    validate_request(request, limits)
    encoded = msgspec.msgpack.encode(request)
    _check_encoded_size(encoded, limits)
    return encoded


def decode_request(
    encoded: bytes,
    limits: ProtocolLimits,
) -> RemoteFillRequest:
    """Decode and validate one bounded request.

    Args:
        encoded: Msgpack request bytes.
        limits: Protocol bounds checked before decoding.

    Returns:
        A validated tagged request.

    Raises:
        ProtocolValidationError: If size, decoding, or schema validation fails.
    """

    _check_encoded_size(encoded, limits)
    try:
        request = msgspec.msgpack.decode(encoded, type=RemoteFillRequest)
    except (msgspec.DecodeError, msgspec.ValidationError) as exc:
        raise ProtocolValidationError("invalid remote-fill message") from exc
    validate_request(request, limits)
    return request


def encode_response(
    response: RemoteFillResponse,
    limits: ProtocolLimits,
) -> bytes:
    """Encode a bounded response.

    Args:
        response: Response to encode.
        limits: Protocol bounds.

    Returns:
        Msgpack response bytes.

    Raises:
        ProtocolValidationError: If the byte bound is exceeded.
    """

    _validate_response(response, limits)
    encoded = msgspec.msgpack.encode(response)
    _check_encoded_size(encoded, limits)
    return encoded


def decode_response(
    encoded: bytes,
    limits: ProtocolLimits,
) -> RemoteFillResponse:
    """Decode a bounded response.

    Args:
        encoded: Msgpack response bytes.
        limits: Protocol bounds checked before decoding.

    Returns:
        Decoded response.

    Raises:
        ProtocolValidationError: If size or decoding fails.
    """

    _check_encoded_size(encoded, limits)
    try:
        response = msgspec.msgpack.decode(encoded, type=RemoteFillResponse)
    except (msgspec.DecodeError, msgspec.ValidationError) as exc:
        raise ProtocolValidationError("invalid remote-fill response") from exc
    _validate_response(response, limits)
    return response


def validate_request(
    request: RemoteFillRequest,
    limits: ProtocolLimits,
) -> None:
    """Validate common fields and operation-specific invariants.

    Args:
        request: Decoded request.
        limits: Protocol and resource bounds.

    Raises:
        ProtocolValidationError: If any invariant is violated.
    """

    common = request.common
    if common.protocol_version != PROTOCOL_VERSION:
        raise ProtocolValidationError("unsupported protocol version")
    _bounded_text(common.operation_id, "operation_id", limits.max_string_bytes)
    _bounded_text(common.transfer_id, "transfer_id", limits.max_string_bytes)
    if common.operation_sequence < 0:
        raise ProtocolValidationError("operation_sequence must be nonnegative")
    if common.request_attempt < 0:
        raise ProtocolValidationError("request_attempt must be nonnegative")
    if common.destination_engine_epoch < 0:
        raise ProtocolValidationError("engine epoch must be nonnegative")
    if common.shared_cache_generation < 0:
        raise ProtocolValidationError("cache generation must be nonnegative")
    _hex_digest(common.payload_digest, "payload_digest")
    if request_payload_digest(request) != common.payload_digest:
        raise ProtocolValidationError("request payload digest mismatch")

    if isinstance(request, NegotiateRequest):
        _validate_negotiate(request, limits)
    elif isinstance(request, OpenRequest):
        _validate_open(request, limits)
    elif isinstance(request, ReserveWindowRequest):
        _validate_reserve(request, limits)
    elif isinstance(request, ArmWindowRequest):
        _validate_arm(request, limits)
    elif isinstance(request, ReportTransferCompleteRequest):
        _validate_report(request, limits)
    elif isinstance(request, FinishRequest):
        _validate_finish(request, limits)
    elif isinstance(request, AbortRequest):
        _bounded_text(request.reason, "reason", limits.max_string_bytes)
    elif isinstance(request, StatusRequest):
        if request.window_id < -1:
            raise ProtocolValidationError("window_id must be -1 or nonnegative")
    else:
        raise ProtocolValidationError("unsupported remote-fill operation")


def response_for_invalid_message(message: str) -> RemoteFillResponse:
    """Create a pointer-free response for an undecodable request.

    Args:
        message: Safe fixed diagnostic text.

    Returns:
        A generic invalid-message response.
    """

    from .protocol import ResultCode

    return RemoteFillResponse(
        operation=OperationKind.STATUS,
        operation_id="",
        code=ResultCode.INVALID_MESSAGE,
        message=message,
    )


def _validate_negotiate(
    request: NegotiateRequest,
    limits: ProtocolLimits,
) -> None:
    for name, value in (
        ("cache_namespace_tag", request.cache_namespace_tag),
        ("layout_tag", request.layout_tag),
        ("model_artifact_id", request.model_artifact_id),
        ("model_layout", request.model_layout),
        ("token_hash_algorithm", request.token_hash_algorithm),
    ):
        _bounded_text(value, name, limits.max_string_bytes)
    if request.chunk_size <= 0 or request.layer_count <= 0:
        raise ProtocolValidationError("chunk size and layer count must be positive")
    if not request.group_dimensions or any(v <= 0 for v in request.group_dimensions):
        raise ProtocolValidationError("group dimensions must be positive")
    if request.tp_size <= 0 or request.dp_size <= 0:
        raise ProtocolValidationError("TP and DP sizes must be positive")
    if request.token_hash_algorithm == "builtin":
        _bounded_text(
            request.python_hash_seed,
            "python_hash_seed",
            limits.max_string_bytes,
        )
    elif request.python_hash_seed:
        raise ProtocolValidationError(
            "python_hash_seed is valid only for builtin token hashing"
        )


def _validate_open(request: OpenRequest, limits: ProtocolLimits) -> None:
    for name, value in (
        ("request_id", request.request_id),
        ("source_engine_id", request.source_engine_id),
        ("destination_engine_id", request.destination_engine_id),
        ("cache_namespace_tag", request.cache_namespace_tag),
        ("layout_tag", request.layout_tag),
        ("model_artifact_id", request.model_artifact_id),
        ("manifest_digest_seed", request.manifest_digest_seed),
    ):
        _bounded_text(value, name, limits.max_string_bytes)
    _hex_digest(request.manifest_digest_seed, "manifest_digest_seed")
    if request.destination_dp_rank < 0:
        raise ProtocolValidationError("destination DP rank must be nonnegative")
    if request.required_store_end_hint < 0:
        raise ProtocolValidationError("required store end must be nonnegative")
    if request.planned_window_count_hint < 0:
        raise ProtocolValidationError("planned window count must be nonnegative")
    if request.planned_window_count_hint > limits.max_windows_per_transaction:
        raise ProtocolValidationError("planned window count exceeds limit")


def _validate_reserve(
    request: ReserveWindowRequest,
    limits: ProtocolLimits,
) -> None:
    if request.window_id < 0 or request.source_generation < 0:
        raise ProtocolValidationError(
            "window and source generation must be nonnegative"
        )
    if not request.control_pages:
        raise ProtocolValidationError("a reservation requires control pages")
    if len(request.control_pages) > limits.max_control_pages_per_window:
        raise ProtocolValidationError("control page count exceeds limit")
    _hex_digest(request.manifest_digest, "manifest_digest")
    if manifest_digest(request.control_pages) != request.manifest_digest:
        raise ProtocolValidationError("window manifest digest mismatch")

    seen: set[tuple[int, int]] = set()
    seen_keys: set[str] = set()
    groups_by_chunk: dict[int, set[int]] = {}
    metadata_by_chunk: dict[int, tuple[int, int, int, int, str]] = {}
    total_bytes = 0
    for page in request.control_pages:
        _validate_control_page(page, limits)
        selector = (page.kv_group, page.chunk_index)
        if selector in seen:
            raise ProtocolValidationError("duplicate group/chunk control page")
        if page.canonical_key in seen_keys:
            raise ProtocolValidationError("duplicate canonical control-page key")
        seen.add(selector)
        seen_keys.add(page.canonical_key)
        groups_by_chunk.setdefault(page.chunk_index, set()).add(page.kv_group)
        metadata = (
            page.chunk_start,
            page.chunk_end,
            page.valid_tokens,
            page.layer_count,
            page.layout_tag,
        )
        prior_metadata = metadata_by_chunk.setdefault(page.chunk_index, metadata)
        if prior_metadata != metadata:
            raise ProtocolValidationError("two-group chunk metadata is inconsistent")
        total_bytes += page.expected_bytes
    observed_group_sets = {
        frozenset(groups) for groups in groups_by_chunk.values()
    }
    if len(observed_group_sets) != 1:
        raise ProtocolValidationError(
            "all chunks require one uniform direct-group set"
        )
    observed_groups = next(iter(observed_group_sets))
    if observed_groups not in (frozenset({0}), frozenset({0, 1})):
        raise ProtocolValidationError(
            "each chunk requires exactly group 0 or groups 0 and 1"
        )
    if total_bytes > limits.max_window_bytes:
        raise ProtocolValidationError("window byte count exceeds limit")


def _validate_control_page(page: ControlPage, limits: ProtocolLimits) -> None:
    _bounded_text(page.canonical_key, "canonical_key", limits.max_key_bytes)
    _bounded_text(page.layout_tag, "layout_tag", limits.max_string_bytes)
    if page.kv_group not in (0, 1):
        raise ProtocolValidationError("kv_group must be 0 or 1")
    if page.chunk_index < 0 or page.chunk_start < 0:
        raise ProtocolValidationError("chunk index and start must be nonnegative")
    if page.chunk_end <= page.chunk_start:
        raise ProtocolValidationError("chunk end must exceed chunk start")
    chunk_tokens = page.chunk_end - page.chunk_start
    if page.valid_tokens <= 0 or page.valid_tokens != chunk_tokens:
        raise ProtocolValidationError("control-page interval must equal valid tokens")
    if page.destination_tp_rank != 0:
        raise ProtocolValidationError("prototype destination TP rank must be zero")
    if page.expected_bytes <= 0 or page.layer_count <= 0:
        raise ProtocolValidationError("page bytes and layer count must be positive")


def _validate_arm(request: ArmWindowRequest, limits: ProtocolLimits) -> None:
    if request.window_id < 0:
        raise ProtocolValidationError("window_id must be nonnegative")
    _bounded_text(
        request.native_transfer_attempt_id,
        "native_transfer_attempt_id",
        limits.max_string_bytes,
    )
    _hex_digest(request.manifest_digest, "manifest_digest")
    _hex_digest(
        request.destination_descriptor_digest,
        "destination_descriptor_digest",
    )


def _validate_report(
    request: ReportTransferCompleteRequest,
    limits: ProtocolLimits,
) -> None:
    if request.window_id < 0 or request.completed_bytes < 0:
        raise ProtocolValidationError("window and completed bytes must be nonnegative")
    _bounded_text(
        request.native_transfer_attempt_id,
        "native_transfer_attempt_id",
        limits.max_string_bytes,
    )
    _hex_digest(request.manifest_digest, "manifest_digest")


def _validate_finish(request: FinishRequest, limits: ProtocolLimits) -> None:
    if request.required_store_end < 0 or request.persistent_common_end < 0:
        raise ProtocolValidationError("finish token counts must be nonnegative")
    if request.final_partial_valid_tokens < 0:
        raise ProtocolValidationError("final partial token count must be nonnegative")
    _hex_digest(request.final_manifest_digest, "final_manifest_digest")


def _bounded_text(value: str, name: str, max_bytes: int) -> None:
    encoded_size = len(value.encode("utf-8"))
    if encoded_size == 0 or encoded_size > max_bytes:
        raise ProtocolValidationError(f"{name} is empty or exceeds its bound")


def _hex_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(ch not in hexdigits for ch in value):
        raise ProtocolValidationError(f"{name} must be a 64-character hex digest")


def _check_encoded_size(encoded: bytes, limits: ProtocolLimits) -> None:
    if len(encoded) > limits.max_rpc_message_bytes:
        raise ProtocolValidationError("remote-fill message exceeds byte limit")


def _validate_response(
    response: RemoteFillResponse,
    limits: ProtocolLimits,
) -> None:
    _bounded_text_optional(
        response.operation_id,
        "response operation_id",
        limits.max_string_bytes,
    )
    _bounded_text_optional(
        response.message,
        "response message",
        limits.max_string_bytes,
    )
    _bounded_text_optional(
        response.remote_session,
        "response remote_session",
        limits.max_string_bytes,
    )
    _bounded_text_optional(
        response.native_transfer_attempt_id,
        "response native attempt",
        limits.max_string_bytes,
    )
    if response.destination_engine_epoch < 0 or response.shared_cache_generation < 0:
        raise ProtocolValidationError("invalid response epoch or generation")
    if response.d_local_end_open < 0:
        raise ProtocolValidationError("invalid response LocalCPU prefix")
    if len(response.descriptors) > limits.max_control_pages_per_window:
        raise ProtocolValidationError("response descriptor count exceeds limit")
    if len(response.page_results) > limits.max_control_pages_per_window:
        raise ProtocolValidationError("response page-result count exceeds limit")
    if len(response.windows) > limits.max_windows_per_transaction:
        raise ProtocolValidationError("response window count exceeds limit")
    if response.descriptors:
        _hex_digest(
            response.destination_descriptor_digest,
            "response destination_descriptor_digest",
        )
        if (
            destination_descriptor_digest(response.descriptors)
            != response.destination_descriptor_digest
        ):
            raise ProtocolValidationError("response descriptor digest mismatch")
    elif response.destination_descriptor_digest:
        raise ProtocolValidationError("descriptor digest has no descriptors")
    max_address = (1 << 64) - 1
    for descriptor in response.descriptors:
        for name, value, bound in (
            (
                "descriptor canonical_key",
                descriptor.canonical_key,
                limits.max_key_bytes,
            ),
            (
                "descriptor remote_session",
                descriptor.remote_session,
                limits.max_string_bytes,
            ),
            (
                "descriptor reservation_id",
                descriptor.reservation_id,
                limits.max_string_bytes,
            ),
            ("descriptor transfer_id", descriptor.transfer_id, limits.max_string_bytes),
            (
                "descriptor native attempt",
                descriptor.native_transfer_attempt_id,
                limits.max_string_bytes,
            ),
        ):
            _bounded_text(value, name, bound)
        _hex_digest(descriptor.manifest_digest, "descriptor manifest_digest")
        _hex_digest(descriptor.capability_mac, "descriptor capability_mac")
        if (
            descriptor.destination_ptr <= 0
            or descriptor.destination_length <= 0
            or descriptor.destination_ptr > max_address
            or descriptor.destination_ptr + descriptor.destination_length > max_address
            or descriptor.chunk_index < 0
            or descriptor.window_id < 0
            or descriptor.kv_group not in (0, 1)
            or descriptor.request_attempt < 0
            or descriptor.destination_dp_rank < 0
            or descriptor.destination_tp_rank < 0
            or descriptor.destination_engine_epoch < 0
            or descriptor.shared_cache_generation < 0
            or not math.isfinite(descriptor.expires_at)
            or descriptor.expires_at <= 0
        ):
            raise ProtocolValidationError("invalid destination descriptor bounds")
    for page_result in response.page_results:
        _bounded_text(
            page_result.canonical_key,
            "page-result canonical_key",
            limits.max_key_bytes,
        )
        if page_result.kv_group not in (0, 1) or page_result.chunk_index < 0:
            raise ProtocolValidationError("invalid page-result selector")
    for window in response.windows:
        if (
            window.window_id < 0
            or window.page_count < 0
            or window.page_count > limits.max_control_pages_per_window
            or window.total_bytes < 0
            or window.total_bytes > limits.max_window_bytes
        ):
            raise ProtocolValidationError("invalid response window bounds")


def _bounded_text_optional(value: str, name: str, max_bytes: int) -> None:
    if len(value.encode("utf-8")) > max_bytes:
        raise ProtocolValidationError(f"{name} exceeds its bound")
