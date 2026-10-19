# SPDX-License-Identifier: Apache-2.0
"""Actual scheduler methods and storage lookup lifecycle over real CPU sockets."""

# Standard
from pathlib import Path
from types import SimpleNamespace as NS
import ast
import time

# Third Party
import pytest

# First Party
from lmcache.v1.lookup_client.storage_lookup_client import StorageLookupClient
from lmcache.v1.rpc.storage_client import StorageClient
from lmcache.v1.storage_service import StorageService

# Local
from test_checkpoint_adapter import method
from tests.v1.test_storage_lookup_client import (
    LookupRuntime,
    TokenDatabase,
    config,
    endpoint,
    metadata,
)


def wait_for(condition):
    deadline = time.monotonic() + 2
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def scheduler_adapter_class(ascend):
    base = type(
        "Base",
        (),
        {
            "request_finished": method("request_finished"),
            "update_connector_output": method("update_connector_output"),
        },
    )
    if not ascend:
        return base
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend/lmcache_ascend/integration/vllm/vllm_v1_adapter.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    implementation = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LMCacheAscendConnectorV1Impl"
    )
    selected = [
        node
        for node in implementation.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("request_finished", "update_connector_output")
    ]
    module = ast.parse("from __future__ import annotations")
    module.body.append(
        ast.ClassDef(
            name="ActualAscend",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=selected,
            decorator_list=[],
        )
    )
    namespace = {"Base": base}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["ActualAscend"]


@pytest.mark.parametrize("ascend", [False, True])
def test_aborted_emitted_direct_load_releases_scheduler_ticket_on_terminal_receive(
    ascend,
):
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
        max_pending=4,
    )
    settings = config()
    client = StorageLookupClient(
        settings, metadata(), storage_client=transport, token_database=TokenDatabase()
    )
    adapter = scheduler_adapter_class(ascend)()
    adapter.__dict__.update(
        config=settings,
        lookup_client=client,
        _manager=NS(lmcache_engine=None, lookup_client=client),
        _cold_perf_lookup_started={},
        _resume_lookup_queries={},
        _dsa_kv_policy_states={},
        _request_trackers={},
        _scheduler_live_sources={},
        store_async=True,
        kv_role="kv_both",
        load_specs={},
        _release_request_lookup_pins=lambda req_id: client.cleanup_lookup(req_id),
        _clear_request_marker=lambda name, req_id: (
            delattr(adapter, name) if getattr(adapter, name, None) == req_id else None
        ),
        _discard_request_set=lambda *args: None,
        _drop_worker_retrieve_state=lambda *args, **kwargs: None,
    )
    try:
        prior_attempt = None
        for index in range(6):
            req_id = "reused-request-id"
            assert client.lookup([1, 2, 3, 4], req_id) == 4
            ticket = client.peek_ticket(req_id)
            assert ticket.identity.request_attempt != prior_attempt
            prior_attempt = ticket.identity.request_attempt
            # Allocation transferred the ticket. M's actual prepare finally
            # releases the lookup hold independently of its prepared leases.
            client.clear_lookup_status(req_id)
            runtime.tickets.pop(ticket.identity.key)
            runtime.registry.set_ticket(ticket.identity, None)
            adapter._dsa_group1_direct_hbm_active_req_id = req_id
            request = NS(
                request_id=req_id, kv_transfer_params=None, dsa_compact_allocated=True
            )
            assert adapter.request_finished(request, []) == (True, None)
            assert transport._records  # Keep the emitted load's lookup until terminal.
            adapter.update_connector_output(NS(finished_recving={req_id}))
            wait_for(lambda: not transport._records)
            assert runtime.registry.snapshot()["operations"] == 0
    finally:
        client.close()
        service.close()


def test_actual_scheduler_rejects_request_id_reuse_while_delayed_receiver_is_owned():
    path = Path(__file__).resolve().parents[3] / "vllm/vllm/v1/core/sched/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "add_request"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    namespace = {
        "StreamingUpdate": NS(from_request=lambda _: object()),
        "RequestStatus": NS(WAITING_FOR_STREAMING_REQ="streaming"),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    scheduler = NS(requests={"r": NS(status="finished-aborted", streaming_queue=None)})
    with pytest.raises(AssertionError, match="duplicate request id"):
        namespace["add_request"](scheduler, NS(request_id="r"))


@pytest.mark.parametrize("mode", ["local", "process"])
def test_terminal_cleanup_keeps_empty_decode_and_local_completion_free_of_rpc(mode):
    cleanup = []
    adapter = NS(
        config=NS(decoder_storage_mode=mode),
        _release_request_lookup_pins=cleanup.append,
        _clear_request_marker=lambda *args: None,
    )
    update = method("update_connector_output")
    for _ in range(1000):
        update(adapter, NS(finished_recving=()))
    assert cleanup == []
    update(adapter, NS(finished_recving={"r"}))
    assert cleanup == (["r"] if mode == "process" else [])


@pytest.mark.parametrize("hit,total", [(129, 129), (4097, 4097), (1024, 2048)])
def test_process_admission_preserves_exact_ticket_and_existing_cold_policy(hit, total):
    ticket = object()
    request = NS(
        request_id="r",
        num_tokens=total,
        prompt_token_ids=list(range(total)),
        all_token_ids=list(range(total)),
        num_preemptions=0,
        status="waiting",
    )
    adapter = NS(
        _resident_cold_load_enabled=True,
        kv_role="kv_both",
        lookup_client=NS(lookup_cache=lambda **kw: hit, peek_ticket=lambda _: ticket),
        _cold_perf_lookup_started={},
        _lmcache_chunk_size=1024,
        _block_size=128,
        _dsa_scratch_capacity=4096,
        _dsa_kv_policy_threshold=0,
        config=NS(
            decoder_storage_mode="process",
            pd_role="receiver",
            min_retrieve_tokens=0,
            dsa_group1_load_mode="persistent_direct_hbm",
        ),
        enable_sparse_attention=True,
        supports_dsa_cold_compact_load=lambda: True,
        load_specs={},
        _requests_priority={},
    )
    matched = method(
        "get_num_new_matched_tokens", cdiv=lambda a, b: (a + b - 1) // b, LoadSpec=NS
    )(adapter, request, 0)
    spec = adapter.load_specs["r"]
    assert matched == hit - int(hit == total)
    assert spec.cpu_source_ticket is ticket
    assert getattr(spec, "dsa_cold_compact_load", False) == (hit == total)
    assert getattr(spec, "dsa_cold_resident_load", False) == (hit == total <= 4096)
