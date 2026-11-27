# SPDX-License-Identifier: Apache-2.0
"""Actual scheduler methods and storage lookup lifecycle over real CPU sockets."""

# Standard
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import ast
import runpy
import time

# Third Party
import pytest

# First Party
from lmcache.v1.lookup_client.abstract_client import LookupBusyTimeout
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
@pytest.mark.parametrize("dispatched", [False, True])
@pytest.mark.parametrize("required_prefix", [0, 3])
def test_rejection_or_abort_releases_lookup_ticket_at_its_ownership_boundary(
    ascend: bool, dispatched: bool, required_prefix: int,
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
        _requests_priority={},
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
            if dispatched:
                # Allocation transferred the ticket. Prepare releases the lookup
                # hold independently of the worker's prepared leases.
                client.clear_lookup_status(req_id)
                runtime.tickets.pop(ticket.identity.key)
                runtime.registry.set_ticket(ticket.identity, None)
                adapter._dsa_group1_direct_hbm_active_req_id = req_id
            adapter._requests_priority[req_id] = 0
            request = NS(
                request_id=req_id, kv_transfer_params={"ret_first_tok": True},
                output_token_ids=[], dsa_compact_allocated=dispatched,
                min_kv_prefix_tokens=required_prefix,
            )
            assert adapter.request_finished(request, []) == (ascend or dispatched, None)
            assert req_id not in adapter._requests_priority
            if dispatched:
                assert transport._records  # Keep the issued lookup until terminal.
                adapter.update_connector_output(NS(finished_recving={req_id}))
            wait_for(lambda: not transport._records)
            assert not runtime.tickets
            assert runtime.registry.snapshot()["operations"] == 0
    finally:
        client.close()
        service.close()


def test_layerwise_prefiller_returns_first_token_after_local_prefill() -> None:
    adapter = scheduler_adapter_class(True)()
    adapter.__dict__.update(
        config=NS(pd_role="sender", get_extra_config_value=lambda _, default: default),
        _layerwise_prefill_p_node=True,
        kv_role="kv_both",
        store_async=True,
        lookup_client=NS(clear_lookup_status=lambda _: None),
        _cold_perf_lookup_started={},
        _resume_lookup_queries={},
        _dsa_kv_policy_states={},
        _scheduler_live_sources={},
        _release_request_lookup_pins=lambda _: None,
        _discard_request_set=lambda *_: None,
        _drop_worker_retrieve_state=lambda *_, **kw: None,
        supports_dsa_live_split=lambda: False,
    )
    request = NS(
        request_id="prefill",
        kv_transfer_params={"ret_first_tok": True, "do_remote_decode": True},
        output_token_ids=[0, 17],
        num_cached_tokens=0,
    )
    assert adapter.request_finished(request, []) == (
        True,
        {"first_tok": 0, "remote_prefill_cached_tokens": 0},
    )


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


@pytest.mark.parametrize("expired", [False, True])
def test_busy_receiver_only_parks_or_rejects_the_sixteenth_row(expired: bool) -> None:
    """Exercise the connector, deployed waiting-loop guard and per-step output."""
    workspace = Path(__file__).resolve().parents[3]
    guard = runpy.run_path(
        str(workspace / "vllm/tests/v1/core/test_cold_compact_admission_guard.py")
    )
    scheduler, req, released, schema = guard["_scheduler"](length=31842)
    req.prompt_token_ids = range(req.num_tokens)
    peers = [
        NS(
            request_id=str(i),
            num_tokens=42,
            num_computed_tokens=42,
            use_structured_output=False,
            num_output_placeholders=0,
            has_encoder_inputs=False,
        )
        for i in range(15)
    ]
    scheduler.running.extend(peers)
    scheduler.requests.update((p.request_id, p) for p in peers)
    lookup = Mock()
    lookup.lookup_cache.side_effect = [
        None,
        None,
        LookupBusyTimeout(req.request_id) if expired else req.num_tokens,
    ]
    adapter = NS(
        kv_role="kv_both",
        lookup_client=lookup,
        config=config(),
        _cold_perf_lookup_started={},
        load_specs={},
        enable_sparse_attention=False,
        supports_dsa_cold_compact_load=lambda: False,
    )
    matched = method(
        "get_num_new_matched_tokens",
        LookupBusyTimeout=LookupBusyTimeout,
        RequestStatus=schema["RequestStatus"],
        logger=Mock(),
        LoadSpec=NS,
    )
    scheduler.connector = NS(
        get_num_new_matched_tokens=lambda r, local: (matched(adapter, r, local), False)
    )
    scheduler.kv_cache_manager = NS(get_computed_blocks=lambda r: (None, 0))
    for tick in range(3):
        step = guard["_run_waiting_lookup"](
            scheduler,
            guard["ASCEND"] / "recompute_scheduler.py",
            "RecomputeScheduler",
            "_finish_kv_admission_error",
        )
        if tick < 2:
            assert step["admitted"] == [] and step["step_skipped_waiting"] == [req]
            scheduler.waiting.extend(step["step_skipped_waiting"])
        output = NS(
            num_scheduled_tokens={p.request_id: 1 for p in peers},
            bootstrap_sample_req_ids=None,
            has_structured_output_requests=False,
        )
        scheduler._update_after_schedule(output)
        assert all(p.num_computed_tokens == 43 + tick for p in peers)
    if expired:
        assert not step["admitted"] and released == [req.request_id]
        [(_, error)] = output._dsa_admission_errors
        assert error.finish_reason == schema["FinishReason"].ERROR
        assert error.stop_reason == "storage_lookup_busy_timeout"
    else:
        assert step["admitted"] == [req.request_id] and not released
        assert req.num_external_computed_tokens == 31841
