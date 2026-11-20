# SPDX-License-Identifier: Apache-2.0
"""Bounded zero-lookup evidence, independent of PD_SERVING_PERF."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
from typing import Any
import json
import socket
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.rpc.storage_protocol import OperationIdentity

logger = init_logger(__name__)
_HOST = socket.gethostname()
_LOOKUP_FAILURE: ContextVar[dict[str, Any] | None] = ContextVar(
    "lmcache_lookup_failure", default=None
)


def page_key_fingerprint(key: str) -> str:
    """Fingerprint an exact physical storage key for P/D log correlation."""
    return sha256(key.encode("utf-8")).hexdigest()


@contextmanager
def collect_lookup_failure() -> Iterator[dict[str, Any]]:
    """Collect the first backend failure in this lookup thread, without I/O."""
    evidence: dict[str, Any] = {}
    token = _LOOKUP_FAILURE.set(evidence)
    try:
        yield evidence
    finally:
        _LOOKUP_FAILURE.reset(token)


def record_lookup_failure(
    reason: str, *, page_key: str | None = None, **fields: Any
) -> None:
    """Retain existing backend results; callers must not query storage again."""
    evidence = _LOOKUP_FAILURE.get()
    if evidence is None or evidence:
        return
    evidence.update(backend_reason=reason, **fields)
    if page_key is not None:
        evidence["first_unavailable_page_fp"] = page_key_fingerprint(page_key)


def lookup_failure_pending() -> bool:
    """Allow backends to skip diagnostic key inspection outside a lookup scope."""
    evidence = _LOOKUP_FAILURE.get()
    return evidence is not None and not evidence


def log_lookup_zero(
    req_id: str,
    reason: str,
    *,
    status: str,
    elapsed_ms: float,
    identity: OperationIdentity | None = None,
    error: str | None = None,
    **fields: Any,
) -> None:
    """Log one terminal zero result using host metadata, even with perf logs off."""
    if identity is not None:
        fields.update(
            operation_id=identity.operation_id,
            service_epoch=identity.service_epoch,
            client_session=identity.client_session,
            request_attempt=identity.request_attempt,
        )
    logger.info(
        "[LMCACHE_LOOKUP_ZERO] %s",
        json.dumps(
            dict(
                schema=1,
                req_id=req_id,
                reason=reason,
                status=status,
                elapsed_ms=round(elapsed_ms, 3),
                wall_time_ns=time.time_ns(),
                host=_HOST,
                error=error[:256] if error else None,
                **fields,
            ),
            separators=(",", ":"),
        ),
    )
