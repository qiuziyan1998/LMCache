# SPDX-License-Identifier: Apache-2.0
"""Actual scheduler facade/factory tests with CPU transport and operation registry."""

# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import ast
import gc
import socket
import time

# Third Party
import msgspec
import pytest

# First Party
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.cpu_cache import StorageOperationRegistry
from lmcache.v1.lookup_client.factory import LookupClientFactory
from lmcache.v1.lookup_client.storage_lookup_client import StorageLookupClient
from lmcache.v1.manager import LMCacheManager
from lmcache.v1.rpc.storage_client import StorageClient
from lmcache.v1.rpc.storage_protocol import (
    LookupReply,
    StorageRequest,
    StorageResult,
    content_digest,
)
from lmcache.v1.storage_service import StorageService


class TokenDatabase:
    chunk_hash_type = int
    chunk_hash_bytes = None

    def __init__(self) -> None:
        self.calls = 0

    def process_tokens(self, tokens, make_key: bool = True):
        self.calls += 1
        for start in range(0, len(tokens), 4):
            yield start, min(start + 4, len(tokens)), -(start + 1)


class LookupRuntime:
    def __init__(self) -> None:
        self.registry = StorageOperationRegistry(
            "cache", "epoch", {"scheduler": b"nonce"}
        )
        self.tickets: dict = {}
        self.lookups = 0

    def handle(self, request: StorageRequest) -> StorageResult:
        def dispatch(work: StorageRequest) -> LookupReply:
            self.lookups += 1
            payload = work.payload
            time.sleep(payload["request_configs"].get("delay", 0))
            if self.registry.cancellation_requested(work.identity):
                return LookupReply(
                    0,
                    "",
                    self.lookups,
                    content_digest(payload),
                    (),
                    (),
                    (),
                    payload["request_configs"],
                )
            ticket_id = f"ticket-{self.lookups}"
            self.tickets[work.identity.key] = ticket_id
            self.registry.set_ticket(work.identity, ticket_id)
            return LookupReply(
                payload["ends"][-1],
                ticket_id,
                self.lookups,
                content_digest(payload),
                tuple(payload["starts"]),
                tuple(payload["ends"]),
                tuple(payload["hashes"]),
                payload["request_configs"],
            )

        result = self.registry.handle(request, dispatch)
        if request.method == "cancel":
            self.tickets.pop(request.identity.key, None)
            self.registry.set_ticket(request.identity, None)
        return result

    def is_healthy(self) -> bool:
        return self.registry.is_healthy()

    def close(self) -> None:
        assert not self.tickets


def endpoint() -> str:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        return f"tcp://127.0.0.1:{reservation.getsockname()[1]}"


def config(**overrides: Any) -> LMCacheEngineConfig:
    return LMCacheEngineConfig.from_defaults(
        decoder_storage_mode="process", pd_role="receiver", **overrides
    )


def metadata() -> SimpleNamespace:
    return SimpleNamespace(
        use_mla=True,
        mla_cache_tp_replicated=True,
        role="scheduler",
        engine_id="cache",
        kv_connector_extra_config={},
    )


@pytest.mark.parametrize("async_loading", [False, True])
def test_lookup_hashes_once_and_hands_off_ticket_before_cleanup(
    async_loading: bool,
) -> None:
    runtime = LookupRuntime()
    work, lifecycle = endpoint(), endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"scheduler": b"nonce"},
    )
    service.start()
    transport = StorageClient(
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="scheduler",
        session_nonce=b"nonce",
    )
    database = TokenDatabase()
    client = StorageLookupClient(
        config(enable_async_loading=async_loading),
        metadata(),
        storage_client=transport,
        token_database=database,
    )
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        assert client.lookup_cache("req") == -1
        result = client.lookup(list(range(9)), "req", {"lmcache.tag": "exact"})
        if async_loading:
            assert result is None
            deadline = time.monotonic() + 2
            while client.lookup_cache("req") is None:
                assert time.monotonic() < deadline
                time.sleep(0.005)
        assert client.lookup_cache("req") == 9
        ticket = client.peek_ticket("req")
        assert ticket is not None
        assert ticket.reply.hashes == (-1, -5, -9)
        assert ticket.reply.starts == (0, 4, 8)
        assert ticket.reply.ends == (4, 8, 9)
        for _ in range(1000):
            assert client.lookup_cache("req") == 9
            assert client.peek_ticket("req") is ticket
        assert database.calls == runtime.lookups == 1
        client.clear_lookup_status("req")
        assert client.peek_ticket("req") is None
        assert runtime.tickets  # Clearing local result did not cancel M ownership.
        client.cleanup_lookup("req")
        deadline = time.monotonic() + 2
        while runtime.registry.snapshot()["operations"]:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert not runtime.tickets
    finally:
        client.close()
        service.close()
        if was_enabled:
            gc.enable()


def test_scheduler_bypass_process_mode_never_constructs_engine(monkeypatch) -> None:
    from lmcache.observability import PrometheusLogger
    from lmcache.v1.cache_engine import LMCacheEngineBuilder

    get = Mock(side_effect=AssertionError("Scheduler must not get an engine"))
    create = Mock(side_effect=AssertionError("Scheduler must not allocate CPU storage"))
    monkeypatch.setattr(LMCacheEngineBuilder, "get", get)
    monkeypatch.setattr(LMCacheEngineBuilder, "get_or_create", create)
    monkeypatch.setattr(PrometheusLogger, "GetOrCreate", Mock())
    factory = VllmServiceFactory(
        config(enable_scheduler_bypass_lookup=True), None, "scheduler"
    )
    factory.metadata = metadata()
    assert factory.get_or_create_lmcache_engine() is None
    get.assert_not_called()
    create.assert_not_called()


def test_factory_admin_process_paths_do_not_create_lookup_server_or_discover(
    monkeypatch,
) -> None:
    import lmcache.v1.rpc.storage_bootstrap as bootstrap

    discover = Mock(
        side_effect=AssertionError("No discovery for construction or dryrun")
    )
    monkeypatch.setattr(bootstrap, "discover_storage_service", discover)
    settings = config(enable_scheduler_bypass_lookup=True)
    client = LookupClientFactory.create_lookup_client(settings, metadata(), Mock())
    try:
        assert isinstance(client, StorageLookupClient)
        assert client.is_healthy()
        assert (
            LookupClientFactory.create_lookup_server(
                SimpleNamespace(config=settings), metadata()
            )
            is None
        )
        manager = LMCacheManager.__new__(LMCacheManager)
        manager._config = settings
        assert "error" in manager.create_lookup_server(dryrun=True)
        discover.assert_not_called()
    finally:
        client.close()


def test_manager_propagates_process_constructor_failure() -> None:
    factory = SimpleNamespace(
        get_or_create_metadata=lambda: metadata(),
        get_or_create_lmcache_engine=Mock(side_effect=RuntimeError("M startup failed")),
    )
    with pytest.raises(RuntimeError, match="M startup failed"):
        LMCacheManager(config(), factory)


def test_actual_scheduler_update_transfers_ticket_before_clear() -> None:
    path = Path(__file__).parents[2] / "lmcache/integration/vllm/vllm_v1_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LMCacheConnectorV1Impl"
    )
    method = next(
        node
        for node in adapter.body
        if isinstance(node, ast.FunctionDef) and node.name == "update_state_after_alloc"
    )
    method.decorator_list = []
    namespace = {"Any": Any}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    ticket = object()
    spec = SimpleNamespace(cpu_source_ticket=None, can_load=True)
    lookup = SimpleNamespace(peek_ticket=lambda req_id: ticket)

    def clear(req_id: str) -> None:
        assert spec.cpu_source_ticket is ticket

    lookup.clear_lookup_status = clear
    worker = SimpleNamespace(
        config=config(),
        lookup_client=lookup,
        load_specs={"req": spec},
        _resume_lookup_queries={},
        _unfinished_requests={},
    )
    namespace[method.name](worker, SimpleNamespace(request_id="req"), 0)
    assert spec.cpu_source_ticket is ticket
    assert spec.can_load is False


def test_busy_lookup_acknowledges_cancelled_tombstone_result() -> None:
    runtime = LookupRuntime()
    work, lifecycle = endpoint(), endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"scheduler": b"nonce"},
        max_work=1,
    )
    service.start()
    transport = StorageClient(
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="scheduler",
        session_nonce=b"nonce",
    )
    client = StorageLookupClient(
        config(enable_async_loading=True),
        metadata(),
        storage_client=transport,
        token_database=TokenDatabase(),
    )
    try:
        client.lookup([1, 2, 3, 4], "slow", {"delay": 0.3})
        deadline = time.monotonic() + 2
        while runtime.lookups == 0:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        client.lookup([1, 2, 3, 4], "busy")
        while client.lookup_cache("busy") is None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert client.lookup_cache("busy") == 0
        while client.lookup_cache("slow") is None:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        client.cleanup_lookup("slow")
        while runtime.registry.snapshot()["operations"]:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert runtime.lookups == 1
    finally:
        client.close()
        service.close()


def test_admin_recreate_reuses_scheduler_session_operation_sequence(
    monkeypatch,
) -> None:
    import lmcache.v1.rpc.storage_bootstrap as bootstrap
    from lmcache.v1.lookup_client.storage_lookup_client import close_scheduler_storage

    runtime = LookupRuntime()
    work, lifecycle = endpoint(), endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"scheduler": b"nonce"},
    )
    service.start()
    monkeypatch.setattr(
        bootstrap,
        "discover_storage_service",
        lambda *args, **kwargs: {
            "cache_instance_id": "cache",
            "service_epoch": "epoch",
            "work_endpoint": work,
            "lifecycle_endpoint": lifecycle,
            "sessions": {"scheduler": b"nonce".hex()},
        },
    )
    first = StorageLookupClient(config(), metadata(), token_database=TokenDatabase())
    second = None
    try:
        assert first.lookup([1, 2, 3, 4], "req") == 4
        old_operation = first.peek_ticket("req").identity.operation_id
        first.close()
        assert runtime.registry.snapshot()["watermarks"]["scheduler"] == old_operation
        second = StorageLookupClient(
            config(), metadata(), token_database=TokenDatabase()
        )
        assert second.lookup([1, 2, 3, 4], "req") == 4
        assert second.peek_ticket("req").identity.operation_id > old_operation
    finally:
        first.close()
        if second is not None:
            second.close()
        close_scheduler_storage("cache")
        service.close()


def test_sync_lookup_timeout_stays_cached_miss_until_clear() -> None:
    runtime = LookupRuntime()
    work, lifecycle = endpoint(), endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"scheduler": b"nonce"},
    )
    service.start()
    transport = StorageClient(
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="scheduler",
        session_nonce=b"nonce",
    )
    client = StorageLookupClient(
        config(lookup_timeout_ms=50),
        metadata(),
        storage_client=transport,
        token_database=TokenDatabase(),
    )
    try:
        assert client.lookup([1, 2, 3, 4], "req", {"delay": 0.2}) == 0
        assert client.lookup_cache("req") == 0
        assert client.peek_ticket("req") is None
        deadline = time.monotonic() + 2
        while runtime.registry.snapshot()["operations"]:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert not runtime.tickets
    finally:
        client.close()
        service.close()


def test_lookup_rejects_changed_canonical_hash_prefix() -> None:
    class WrongPlanRuntime(LookupRuntime):
        def handle(self, request: StorageRequest) -> StorageResult:
            result = super().handle(request)
            if request.method == "lookup":
                result = msgspec.structs.replace(
                    result,
                    value=msgspec.structs.replace(
                        result.value, hashes=(999,) * len(result.value.hashes)
                    ),
                )
            return result

    runtime = WrongPlanRuntime()
    work, lifecycle = endpoint(), endpoint()
    service = StorageService(
        runtime,
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        sessions={"scheduler": b"nonce"},
    )
    service.start()
    transport = StorageClient(
        work,
        lifecycle,
        cache_instance_id="cache",
        service_epoch="epoch",
        client_session="scheduler",
        session_nonce=b"nonce",
    )
    client = StorageLookupClient(
        config(), metadata(), storage_client=transport, token_database=TokenDatabase()
    )
    try:
        with pytest.raises(ValueError, match="canonical source plan"):
            client.lookup([1, 2, 3, 4], "req")
        assert client.peek_ticket("req") is None
    finally:
        client.close()
        service.close()
