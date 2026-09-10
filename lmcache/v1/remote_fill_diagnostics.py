# SPDX-License-Identifier: Apache-2.0
"""Small, device-free diagnostics for RemoteFill lifecycle failures."""

# Standard
import json
import re
from typing import Any, Literal, Optional

REMOTE_FILL_DIAGNOSTIC_MARKER = "[LMCACHE_REMOTE_FILL_DIAGNOSTIC]"
# Backward-compatible import alias. New records use the diagnostic marker.
REMOTE_FILL_VALIDATION_MARKER = REMOTE_FILL_DIAGNOSTIC_MARKER
_MAX_DIAGNOSTIC_TEXT = 512
_ADDRESS_PATTERN = re.compile(r"0x[0-9a-fA-F]{4,}")
_POINTER_FIELD_PATTERN = re.compile(
    r"(?i)\b((?:source|destination|remote)_?)?(?:ptr|pointer|address)\s*[=:]\s*\d+"
)
_DIAGNOSTIC_NAMES = {
    "RF-D-000": "decoder_startup_failure",
    "RF-D-001": "decoder_retained_prefix_missing",
    "RF-D-002": "decoder_paired_prefix_lookup_failure",
    "RF-D-003": "decoder_retained_plan_invalid",
    "RF-D-004": "decoder_materialization_failure",
    "RF-D-005": "decoder_control_service_failure",
    "RF-D-900": "decoder_memory_safety_uncertain",
    "RF-P-001": "producer_handoff_malformed",
    "RF-P-002": "producer_handoff_incompatible",
    "RF-P-003": "producer_circuit_open",
    "RF-P-004": "producer_persistent_fallback",
    "RF-P-900": "producer_memory_safety_uncertain",
}


def _bounded_text(value: object) -> str:
    """Return a single-line, bounded representation safe for production logs."""

    try:
        text = str(value)
    except Exception:
        text = f"<{type(value).__name__} string conversion failed>"
    text = _ADDRESS_PATTERN.sub("<redacted-address>", text)
    text = _POINTER_FIELD_PATTERN.sub("<redacted-pointer>", text)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= _MAX_DIAGNOSTIC_TEXT:
        return text
    return text[: _MAX_DIAGNOSTIC_TEXT - 3] + "..."


def log_remote_fill_diagnostic(
    logger: Any,
    *,
    event: str,
    code: str,
    stage: str,
    action: str,
    req_id: Optional[str] = None,
    transfer_id: Optional[str] = None,
    reason: Optional[str] = None,
    error: Optional[BaseException] = None,
    memory_safety_uncertain: bool = False,
    severity: Literal["warning", "error", "critical"] = "error",
) -> None:
    """Emit one bounded, pointer-free RemoteFill lifecycle record.

    This helper deliberately performs no tensor access, hashing, synchronization,
    or traceback formatting. It is called only on an enabled RemoteFill failure
    path, so the disabled and successful paths have no additional work.
    """

    fields: dict[str, object] = {
        "schema": 1,
        "event": _bounded_text(event),
        "code": _bounded_text(code),
        "diagnostic_name": _DIAGNOSTIC_NAMES.get(
            code, "remote_fill_unclassified"
        ),
        "stage": _bounded_text(stage),
        "action": _bounded_text(action),
        "memory_safety_uncertain": bool(memory_safety_uncertain),
    }
    if req_id:
        fields["req_id"] = _bounded_text(req_id)
    if transfer_id:
        fields["transfer_id"] = _bounded_text(transfer_id)
    if reason:
        fields["reason"] = _bounded_text(reason)
    if error is not None:
        fields["error_type"] = type(error).__name__
        fields["error"] = _bounded_text(error)

    log = getattr(logger, severity)
    log(
        "%s %s",
        REMOTE_FILL_DIAGNOSTIC_MARKER,
        json.dumps(fields, separators=(",", ":"), sort_keys=True),
    )


def log_remote_fill_validation_failure(logger: Any, **fields: Any) -> None:
    """Compatibility wrapper for callers using the former public helper."""

    log_remote_fill_diagnostic(
        logger,
        event="remote_fill_diagnostic",
        **fields,
    )
