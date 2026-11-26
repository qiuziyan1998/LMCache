# SPDX-License-Identifier: Apache-2.0
"""Real lookup IPC against the CPU runtime during delayed ticket retirement."""

# Standard
from typing import Any, Iterator
import gc
import socket
import threading
import time
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.storage_lookup_client import StorageLookupClient
from lmcache.v1.rpc.storage_client import StorageClient
from lmcache.v1.rpc.storage_protocol import PrepareSpec
from lmcache.v1.storage_service import StorageService

# Local
import test_cpu_cache_runtime as cpu_runtime
from test_cpu_cache_runtime import control

setup = cpu_runtime.setup


def endpoint() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{sock.getsockname()[1]}"


class Tokens:
    chunk_hash_type = int
    chunk_hash_bytes = None

    def process_tokens(
        self, tokens: Any, make_key: bool = False
    ) -> Iterator[tuple[int, int, int]]:
        yield 0, 4, 7
        yield 4, 5, 8


@pytest.mark.parametrize("claimed", [False, True])
def test_cleanup_reconciles_pending_ticket_after_late_completion(
    setup: Any, claimed: bool
) -> None:
    owner, runtime, work, bootstrap, populate = setup
    bootstrap()
    populate()
    work_endpoint, control_endpoint = endpoint(), endpoint()
    service = StorageService(
        runtime,
        work_endpoint,
        control_endpoint,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions=runtime.registry.sessions,
    )
    transport = StorageClient(
        work_endpoint,
        control_endpoint,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="scheduler",
        session_nonce=b"s",
    )
    client = StorageLookupClient(
        LMCacheEngineConfig.from_defaults(
            decoder_storage_mode="process",
            pd_role="receiver",
            enable_async_loading=True,
            blocking_timeout_secs=1,
        ),
        SimpleNamespace(use_mla=True, mla_cache_tp_replicated=True),
        storage_client=transport,
        token_database=Tokens(),
    )
    blocked, release = threading.Event(), threading.Event()
    original = runtime._resolve_shared_rank0_layer_pages
    result = []
    prepare_thread = None
    pending_ack = threading.Event()
    original_handle = runtime.handle

    def handle(request: Any) -> Any:
        reply = original_handle(request)
        if request.method == "acknowledge_terminal" and reply.status == "PENDING":
            pending_ack.set()
        return reply

    runtime.handle = handle
    service.start()
    enabled = gc.isenabled()
    gc.disable()
    try:
        if not claimed:
            check_cancelled = runtime.registry.cancellation_requested

            def held_lookup_check(identity: Any) -> bool:
                cancelled = check_cancelled(identity)
                blocked.set()
                assert release.wait(3)
                return cancelled

            runtime.registry.cancellation_requested = held_lookup_check
        assert client.lookup(list(range(5)), "req") is None
        if not claimed:
            assert blocked.wait(2)
            client.cleanup_lookup("req")
            deadline = time.monotonic() + 2
            while not runtime.registry.snapshot()["operations"]:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            # Wait until CANCEL has executed, before allowing lookup publication.
            while not any(op.cancelled for op in runtime.registry._operations.values()):
                assert time.monotonic() < deadline
                time.sleep(0.005)
            release.set()
            assert pending_ack.wait(2)
        deadline = time.monotonic() + 2
        if claimed:
            while client.lookup_cache("req") is None:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            ticket = client.peek_ticket("req")
            assert ticket.reply.matched_tokens == 5
            client.clear_lookup_status("req")

            def held_read(**kwargs: Any) -> Any:
                blocked.set()
                assert release.wait(3)
                return original(**kwargs)

            runtime._resolve_shared_rank0_layer_pages = held_read
            spec = PrepareSpec(
                ticket.reply.ticket_id,
                ticket.reply.key_plan_digest,
                5,
                runtime.layout.digest,
                ("worker-0", "worker-1"),
            )
            prepare_thread = threading.Thread(
                target=lambda: result.append(
                    work(
                        "worker-0",
                        "prepare",
                        spec,
                        attempt=ticket.identity.request_attempt,
                        request_id="req",
                    )
                )
            )
            prepare_thread.start()
            assert blocked.wait(2)
            client.cleanup_lookup("req")
            assert pending_ack.wait(2)
            time.sleep(0.15)
            assert ticket.reply.ticket_id in runtime._tickets
            release.set()
            prepare_thread.join(2)
            assert not prepare_thread.is_alive()
        # No foreground lookup polling: background cleanup must retire the record.
        deadline = time.monotonic() + 1
        while transport._records or client._cleaning:
            if time.monotonic() >= deadline:
                pytest.fail("lookup remains retained after its ticket is releasable")
            time.sleep(0.01)
        assert not client._cleaning
        assert not runtime._tickets
        assert client.lookup_cache("req") == -1
    finally:
        release.set()
        if prepare_thread is not None:
            prepare_thread.join(3)
        if result:
            prepared, request = result[0]
            assert prepared.status == "COMPLETED"
            for group in prepared.value.groups:
                for participant in ("worker-0", "worker-1"):
                    retired = control(
                        runtime,
                        request,
                        "retire",
                        {"lease_id": group.lease_id, "proof": "never_imported"},
                        participant=participant,
                    )
                    assert retired.status == "RETIRED"
        client.close()
        # Test force cleanup after the pre-fix failure so service shutdown is safe.
        for ticket_id, entry in tuple(runtime._tickets.items()):
            runtime.drop_lookup(ticket_id, entry.identity)
        service.close()
        if enabled:
            gc.enable()
