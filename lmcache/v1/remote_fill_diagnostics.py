# SPDX-License-Identifier: Apache-2.0
"""Small, device-free diagnostics for RemoteFill correctness failures."""

# Standard
import json
import re
from typing import Any, Literal, Optional

REMOTE_FILL_VALIDATION_MARKER = "[LMCACHE_REMOTE_FILL_VALIDATION]"
_MAX_DIAGNOSTIC_TEXT = 512
_ADDRESS_PATTERN = re.compile(r"0x[0-9a-fA-F]{4,}")
_POINTER_FIELD_PATTERN = re.compile(
    r"(?i)\b((?:source|destination|remote)_?)?(?:ptr|pointer|address)\s*[=:]\s*\d+"
)


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


def log_remote_fill_validation_failure(
    logger: Any,
    *,
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
    """Emit one bounded, pointer-free RemoteFill validation record.

    This helper deliberately performs no tensor access, hashing, synchronization,
    or traceback formatting. It is called only on an enabled RemoteFill failure
    path, so the disabled and successful paths have no additional work.
    """

    fields: dict[str, object] = {
        "schema": 1,
        "event": "remote_fill_validation_failure",
        "code": _bounded_text(code),
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
        REMOTE_FILL_VALIDATION_MARKER,
        json.dumps(fields, separators=(",", ":"), sort_keys=True),
    )
