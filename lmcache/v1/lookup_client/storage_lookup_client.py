# SPDX-License-Identifier: Apache-2.0
"""Scheduler lookup facade; CPU sources and their exact location plans stay in M."""

# Standard
from concurrent.futures import TimeoutError
from dataclasses import dataclass
from typing import Any
from uuid import uuid4
import threading
import time

# Third Party
import msgspec

# First Party
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.rpc.storage_client import StorageClient, StorageFuture
from lmcache.v1.rpc.storage_protocol import (
    LookupReply,
    StorageLookupTicket,
    content_digest,
    wire_hash,
)
from lmcache.v1.token_database import ChunkedTokenDatabase

_TRANSPORTS: dict[tuple[str, str], StorageClient] = {}
_TRANSPORT_LOCK = threading.Lock()


def close_scheduler_storage(engine_id: str) -> None:
    """Close the scheduler session at manager shutdown, retaining unknown evidence."""
    with _TRANSPORT_LOCK:
        for key, client in tuple(_TRANSPORTS.items()):
            if key[0] == engine_id and client.close():
                del _TRANSPORTS[key]


@dataclass
class _Lookup:
    future: StorageFuture
    started: float
    expected_plan: tuple = ()
    ticket: StorageLookupTicket | None = None
    cleanup: StorageFuture | None = None
    cleanup_phase: str = "cancel"


class StorageLookupClient(LookupClientInterface):
    """Preserve int/None lookup results while keeping service tickets out of that API.

    Discovery is lazy because scheduler construction may precede worker bootstrap.
    Local status clearing transfers a ticket; explicit cleanup cancels its lookup
    hold and never retires the independent prepared-source lease.
    """

    def __init__(
        self,
        config: Any,
        metadata: Any,
        *,
        storage_client: Any = None,
        token_database: Any = None,
    ) -> None:
        if (
            config.external_lookup_client is not None
            or config.enable_blending
            or config.pd_role != "receiver"
            or not metadata.use_mla
            or not getattr(metadata, "mla_cache_tp_replicated", False)
            or not config.get_extra_config_value("save_only_first_rank", True)
        ):
            raise ValueError(
                "Decoder process lookup requires replicated MLA receiver keys "
                "and internal chunked lookup"
            )
        self.config = config
        self.metadata = metadata
        self.token_database = token_database or ChunkedTokenDatabase(config, metadata)
        self._client = storage_client
        self._shared_transport = False
        self._lookups: dict[str, _Lookup] = {}
        self._transferred: dict[str, list[_Lookup]] = {}
        self._cleaning: dict[str, list[_Lookup]] = {}
        self.reqs_status: dict[str, int] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._cleanup_wake = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, name="storage-lookup-cleanup", daemon=True
        )
        self._cleanup_thread.start()

    def get_storage_client(self) -> StorageClient:
        """Share the existing authenticated scheduler session for explicit admin."""
        return self._get_client()

    def lookup_cache(self, lookup_id: str) -> int | None:
        """Return -1 for a new lookup, None while pending, or its matched count."""
        with self._lock:
            if lookup_id in self.reqs_status:
                return self.reqs_status[lookup_id]
            if not self._poll_cleanup(lookup_id):
                return None
            entry = self._lookups.get(lookup_id)
            if entry is None:
                return -1
            if not entry.future.done():
                if (
                    time.monotonic() - entry.started
                ) * 1000 > self.config.lookup_timeout_ms:
                    self.cleanup_lookup(lookup_id)
                    self.reqs_status[lookup_id] = 0
                return None
            return self._accept(lookup_id, entry)

    def lookup(
        self, token_ids: Any, lookup_id: str, request_configs: dict | None = None
    ) -> int | None:
        """Hash the query once and submit only canonical ranges/hashes to M."""
        with self._lock:
            if lookup_id in self._lookups:
                return self.lookup_cache(lookup_id)
            if not self._poll_cleanup(lookup_id):
                return None
            chunks = tuple(
                self.token_database.process_tokens(token_ids, make_key=False)
            )
            if not chunks:
                self.reqs_status[lookup_id] = 0
                return 0
            kind = "bytes" if self.token_database.chunk_hash_type is bytes else "int"
            width = (
                self.token_database.chunk_hash_bytes * 8
                if kind == "bytes"
                else 256
                if "sha256" in self.config.pre_caching_hash_algorithm
                else 64
            )
            payload = {
                "starts": tuple(start for start, _, _ in chunks),
                "ends": tuple(end for _, end, _ in chunks),
                "hashes": tuple(wire_hash(value, width) for _, _, value in chunks),
                "hash_kind": kind,
                "hash_width": width,
                "request_configs": dict(request_configs or {}),
            }
            client = self._get_client()
            try:
                future = client.request(
                    "lookup",
                    payload,
                    request_id=lookup_id,
                    request_attempt=uuid4().hex,
                    load_generation=0,
                    purpose="lookup",
                )
            except BufferError:
                self.reqs_status[lookup_id] = 0  # Existing cache-miss/recompute policy.
                return 0
            entry = _Lookup(
                future,
                time.monotonic(),
                (
                    payload["starts"],
                    payload["ends"],
                    payload["hashes"],
                    content_digest(payload["request_configs"]),
                ),
            )
            self._lookups[lookup_id] = entry
        if self.config.enable_async_loading:
            return None
        try:
            future.result(self.config.lookup_timeout_ms / 1000)
        except TimeoutError:
            self.cleanup_lookup(lookup_id)
            with self._lock:
                self.reqs_status[lookup_id] = 0
            return 0  # The unresolved lookup remains retained for reconciliation.
        with self._lock:
            return self._accept(lookup_id, entry)

    def peek_ticket(self, lookup_id: str) -> StorageLookupTicket | None:
        """Return the immutable handoff without dropping the lookup's M hold."""
        with self._lock:
            entry = self._lookups.get(lookup_id)
            return entry.ticket if entry is not None else None

    def clear_lookup_status(self, lookup_id: str) -> None:
        """Forget local status while retaining an exported ticket for worker claim."""
        with self._lock:
            self.reqs_status.pop(lookup_id, None)
            entry = self._lookups.pop(lookup_id, None)
            if entry is not None:
                self._transferred.setdefault(lookup_id, []).append(entry)

    def cleanup_lookup(self, lookup_id: str) -> None:
        """Cancel only lookup ownership; a claimed preparation keeps its own holds."""
        with self._lock:
            self.reqs_status.pop(lookup_id, None)
            entries = self._transferred.pop(lookup_id, [])
            entry = self._lookups.pop(lookup_id, None)
            if entry is not None:
                entries.append(entry)
            self._cleaning.setdefault(lookup_id, []).extend(entries)
            for entry in entries:
                entry.cleanup = self._get_client().cancel(entry.future.identity)
                entry.cleanup.add_done_callback(
                    lambda _, key=lookup_id: self._continue_cleanup(key)
                )
            self._poll_cleanup(lookup_id)

    def supports_producer_reuse(self) -> bool:
        """The decoder retains ordinary lookup reuse semantics."""
        return True

    def is_healthy(self) -> bool:
        """Read cached transport health without discovering or polling M."""
        return not self._closed and (self._client is None or self._client.is_healthy())

    def close(self) -> None:
        """Queue lookup cleanup and retain unresolved service ownership on close."""
        with self._lock:
            if self._closed:
                return
            for lookup_id in set(self._lookups) | set(self._transferred):
                self.cleanup_lookup(lookup_id)
            client = self._client
        deadline = time.monotonic() + min(5.0, self.config.blocking_timeout_secs)
        while time.monotonic() < deadline:
            with self._lock:
                if not self._cleaning:
                    break
            self._cleanup_wake.set()
            time.sleep(0.01)
        with self._lock:
            self._closed = True
            self._cleanup_wake.set()
        if client is not None and not self._shared_transport:
            client.close()
        if threading.current_thread() is not self._cleanup_thread:
            self._cleanup_thread.join(1)

    def _get_client(self) -> StorageClient:
        if self._closed:
            raise RuntimeError("Storage lookup client is closed")
        if self._client is None:
            from lmcache.v1.rpc.storage_bootstrap import discover_storage_service

            port = (self.metadata.kv_connector_extra_config or {}).get(
                "lmcache_rpc_port", 0
            )
            descriptor = discover_storage_service(
                self.metadata.engine_id,
                port,
                timeout_s=self.config.blocking_timeout_secs,
            )
            key = (descriptor["cache_instance_id"], descriptor["service_epoch"])
            with _TRANSPORT_LOCK:
                if any(
                    instance == key[0] and epoch != key[1]
                    for instance, epoch in _TRANSPORTS
                ):
                    raise RuntimeError(
                        "Storage epoch changed; scheduler restart is required"
                    )
                client = _TRANSPORTS.get(key)
                if client is None:
                    client = StorageClient(
                        descriptor["work_endpoint"],
                        descriptor["lifecycle_endpoint"],
                        cache_instance_id=key[0],
                        service_epoch=key[1],
                        client_session="scheduler",
                        session_nonce=bytes.fromhex(
                            descriptor["sessions"]["scheduler"]
                        ),
                    )
                    _TRANSPORTS[key] = client
                if not client.is_healthy():
                    raise RuntimeError(
                        "Storage scheduler session failed; restart is required"
                    )
                self._client = client
                self._shared_transport = True
        return self._client

    def _accept(self, lookup_id: str, entry: _Lookup) -> int:
        result = entry.future.result()
        if result.status in ("BUSY", "CANCELLED_NOT_STARTED", "FAILED_SAFE"):
            self.cleanup_lookup(lookup_id)
            self.reqs_status[lookup_id] = 0
            return 0
        if result.status != "COMPLETED" or result.error:
            raise RuntimeError(
                result.error or f"Storage lookup returned {result.status}"
            )
        reply = msgspec.convert(result.value, type=LookupReply)
        starts, ends, hashes, config_digest = entry.expected_plan
        count = len(reply.starts)
        if (
            reply.matched_tokens < 0
            or len(reply.key_plan_digest) != 32
            or len(reply.starts) != len(reply.ends)
            or len(reply.starts) != len(reply.hashes)
            or reply.starts != starts[:count]
            or reply.ends != ends[:count]
            or reply.hashes != hashes[:count]
            or content_digest(reply.request_configs) != config_digest
            or (reply.matched_tokens == 0 and (count or reply.ticket_id))
            or (
                reply.matched_tokens > 0
                and (not reply.ends or reply.ends[-1] != reply.matched_tokens)
            )
        ):
            raise ValueError("Storage lookup returned an invalid canonical source plan")
        entry.ticket = StorageLookupTicket(
            reply, entry.future.identity, entry.future.payload_digest
        )
        self.reqs_status[lookup_id] = reply.matched_tokens
        return reply.matched_tokens

    def _poll_cleanup(self, lookup_id: str) -> bool:
        entries = self._cleaning.get(lookup_id, [])
        remaining = []
        for entry in entries:
            control = entry.cleanup
            if control is None or not control.done():
                remaining.append(entry)
                continue
            result = control.result()
            if result.status == "ACKNOWLEDGED":
                continue
            if not entry.future.done():
                # A cancelled unsent/delayed lookup still needs its recorded result.
                self._get_client().replay(entry.future.identity)
                entry.future.add_done_callback(
                    lambda _, key=lookup_id: self._continue_cleanup(key)
                )
                remaining.append(entry)
                continue
            if entry.cleanup_phase == "ack" and result.status == "PENDING":
                remaining.append(entry)
                continue
            summary = result.value if isinstance(result.value, dict) else {}
            digest = summary.get("terminal_result_digest")
            payload = {"terminal_result_digest": digest} if digest is not None else None
            entry.cleanup = self._get_client().acknowledge_terminal(
                entry.future.identity, payload
            )
            entry.cleanup_phase = "ack"
            entry.cleanup.add_done_callback(
                lambda _, key=lookup_id: self._continue_cleanup(key)
            )
            remaining.append(entry)
        if remaining:
            self._cleaning[lookup_id] = remaining
            return False
        self._cleaning.pop(lookup_id, None)
        return True

    def _continue_cleanup(self, lookup_id: str) -> None:
        self._cleanup_wake.set()

    def _cleanup_loop(self) -> None:
        while True:
            self._cleanup_wake.wait()
            self._cleanup_wake.clear()
            with self._lock:
                if self._closed:
                    return
                for lookup_id in tuple(self._cleaning):
                    try:
                        self._poll_cleanup(lookup_id)
                    except Exception as error:
                        self._get_client().mark_fatal(
                            f"Storage lookup cleanup failed: {error}"
                        )
                        return
