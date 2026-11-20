# SPDX-License-Identifier: Apache-2.0
"""Execute actual scheduler admission code without model or NPU imports."""

# Standard
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import ast
import importlib.util

# Third Party
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCHEDULERS = [
    ("vllm/vllm/v1/core/sched/scheduler.py", "Scheduler"),
    ("vllm-ascend/vllm_ascend/core/recompute_scheduler.py", "RecomputeScheduler"),
    (
        "vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py",
        "SchedulerDynamicBatch",
    ),
]


class Queue(list):
    def pop_request(self):
        return self.pop(0)

    def prepend_request(self, request):
        self.insert(0, request)


def source_class(path, name):
    return next(
        node
        for node in ast.parse((ROOT / path).read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def admission(
    path,
    name,
    *,
    hit=256,
    length=1024,
    local=0,
    compact=False,
    extent=None,
    legacy=False,
    free=1000,
    bootstrap=False,
    load_async=True,
    legacy_capability=True,
):
    """Execute contiguous waiting admission, allocation, and async parking."""
    cls = source_class(path, name)
    schedule = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "schedule"
    )
    body = next(
        node.body
        for node in ast.walk(schedule)
        if isinstance(node, (ast.For, ast.While))
        and any(
            isinstance(stmt, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "num_external_computed_tokens"
                for t in stmt.targets
            )
            for stmt in node.body
        )
        and any(
            isinstance(stmt, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "new_blocks" for t in stmt.targets
            )
            for stmt in node.body
        )
    )
    first = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "num_external_computed_tokens"
            for t in node.targets
        )
    )
    last = next(
        i
        for i, node in enumerate(body)
        if i > first
        and isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "load_kv_async"
        and any(isinstance(stmt, ast.Continue) for stmt in node.body)
    )
    request = NS(
        request_id="incoming",
        num_tokens=length,
        num_prompt_tokens=length,
        num_preemptions=0,
        num_computed_tokens=0,
        has_encoder_inputs=False,
        dsa_compact_allocated=False,
        bootstrap_sample_pending=bootstrap,
        bootstrap_final_hidden=None,
        status="waiting",
    )
    connector = NS(
        get_num_new_matched_tokens=Mock(return_value=(hit, load_async)),
        supports_dsa_compact_external_load=legacy_capability,
        update_state_after_alloc=Mock(),
    )
    if not legacy:
        connector.is_dsa_cold_compact_load = Mock(return_value=compact)
        connector.get_dsa_cold_load_num_tokens = Mock(
            return_value=local + hit if extent is None else extent
        )
    allocated = []

    def allocate(req, count, **kwargs):
        assert req is request
        if load_async:
            assert count == 0
        end = (
            kwargs["num_new_computed_tokens"]
            + kwargs["num_external_computed_tokens"]
            + kwargs["num_lookahead_tokens"]
            + count
        )
        blocks_per_group = (end + 127) // 128
        if blocks_per_group > free:
            return None
        allocated[:] = [list(range(blocks_per_group)), list(range(blocks_per_group))]
        return allocated

    manager = NS(
        get_computed_blocks=lambda _: (None, local),
        allocate_slots=Mock(side_effect=allocate),
        get_blocks=lambda _: allocated,
    )
    waiting, skipped = Queue([request]), Queue()
    reject = Mock(return_value=False)
    obj = NS(
        block_size=128,
        num_lookahead_tokens=1,
        connector=connector,
        kv_cache_manager=manager,
        _reject_impossible_initial_dsa_request=reject,
        scheduler_config=NS(
            long_prefill_token_threshold=0, enable_chunked_prefill=True
        ),
        need_mamba_block_aligned_split=False,
        _checkpoint_capture_enabled=False,
        is_mtp_kv_consumer=False,
        _dsa_query_rows=2,
        is_encoder_decoder=False,
        connector_prefix_cache_stats=None,
        waiting=waiting,
    )
    scope = dict(
        self=obj,
        requests=[request],
        request_id="incoming",
        request_queue=waiting,
        step_skipped_waiting=skipped,
        skipped_waiting_requests=skipped,
        SERVING_PERF_ENABLED=False,
        bootstrap_only_step=False,
        bootstrap_sample_req_ids=[],
        num_scheduled_tokens={} if bootstrap else {"ongoing": 2},
        token_budget=32,
        encoder_compute_budget=0,
        logger=Mock(),
        RequestStatus=NS(WAITING_FOR_REMOTE_KVS="remote"),
    )
    loop = ast.parse("for request in requests: pass").body[0]
    loop.body = body[first : last + 1]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])),
            str(ROOT / path),
            "exec",
        ),
        scope,
    )
    return request, manager, connector, allocated, scope, reject


@pytest.mark.parametrize("path,name", SCHEDULERS)
@pytest.mark.parametrize("length,hit,local", [(1024, 256, 0), (1024, 384, 128)])
def test_async_partial_load_uses_dense_groups_without_query_compute(
    path, name, length, hit, local
):
    req, manager, connector, groups, scope, reject = admission(
        path,
        name,
        length=length,
        hit=hit,
        local=local,
    )
    kwargs = manager.allocate_slots.call_args.kwargs
    assert kwargs["dsa_compact_external_load"] is False
    assert kwargs["delay_cache_blocks"] is True
    assert kwargs["num_lookahead_tokens"] == 0
    assert req.num_computed_tokens == local + hit and req.status == "remote"
    assert (
        len(groups) == 2
        and len(groups[0]) == len(groups[1]) == (local + hit + 127) // 128
    )
    assert scope["num_scheduled_tokens"] == {"ongoing": 2}
    connector.update_state_after_alloc.assert_called_once()
    assert reject.call_args.kwargs["compact_external_load"] is False


@pytest.mark.parametrize("path,name", SCHEDULERS)
@pytest.mark.parametrize("length", [129, 4097, 81665])
@pytest.mark.parametrize("compact", [False, True])
def test_restore_extent_reserves_kb_plus_one_without_advancing_frontier(
    path, name, length, compact
):
    req, manager, _, groups, _, _ = admission(
        path,
        name,
        length=length,
        hit=length - 1,
        compact=compact,
        extent=length,
    )
    kwargs = manager.allocate_slots.call_args.kwargs
    assert kwargs["num_lookahead_tokens"] == 1
    assert kwargs["num_external_computed_tokens"] == length - 1
    assert kwargs["dsa_compact_external_load"] is compact
    assert req.num_computed_tokens == length - 1
    assert len(groups[0]) == len(groups[1]) == (length + 127) // 128


@pytest.mark.parametrize("path,name", SCHEDULERS)
def test_dense_capacity_failure_defers_without_publishing_metadata(path, name):
    req, manager, connector, groups, scope, _ = admission(
        path,
        name,
        length=1024,
        hit=512,
        free=3,
    )
    assert manager.allocate_slots.call_args.kwargs["dsa_compact_external_load"] is False
    assert not groups and req.status == "waiting" and req.num_computed_tokens == 0
    connector.update_state_after_alloc.assert_not_called()
    assert scope["num_scheduled_tokens"] == {"ongoing": 2}


@pytest.mark.parametrize("path,name", SCHEDULERS)
def test_legacy_connectors_keep_connector_wide_compact_policy(path, name):
    req, manager, _, _, _, _ = admission(path, name, legacy=True)
    assert manager.allocate_slots.call_args.kwargs["dsa_compact_external_load"] is True
    assert req.status == "remote"


def test_async_ordinary_bootstrap_cannot_force_compact_layout():
    req, manager, _, _, _, reject = admission(
        *SCHEDULERS[0],
        hit=129,
        length=129,
        extent=129,
        bootstrap=True,
    )
    assert manager.allocate_slots.call_args.kwargs["dsa_compact_external_load"] is False
    assert reject.call_args.kwargs["compact_external_load"] is False
    assert req.status == "remote" and req.num_computed_tokens == 129


def methods_class(path, name, methods, bases=()):
    cls = source_class(path, name)
    cls.bases = [ast.Name(id=base, ctx=ast.Load()) for base in bases]
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    scope = {base: object for base in bases}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), scope)
    return scope[name]


def test_actual_dynamic_wrapper_delegates_request_queries_and_ascend_inherits():
    wrapper = methods_class(
        "LMCache-NPU/lmcache/integration/vllm/lmcache_connector_v1.py",
        "LMCacheConnectorV1Dynamic",
        {"is_dsa_cold_compact_load", "get_dsa_cold_load_num_tokens"},
    )()
    wrapper._lmcache_engine = NS(
        is_dsa_cold_compact_load=Mock(return_value=False),
        get_dsa_cold_load_num_tokens=Mock(return_value=129),
    )
    assert wrapper.is_dsa_cold_compact_load("r") is False
    assert wrapper.get_dsa_cold_load_num_tokens("r") == 129
    wrapper._lmcache_engine.is_dsa_cold_compact_load.assert_called_once_with("r")
    ascend = source_class(
        "LMCache-Ascend/lmcache_ascend/integration/vllm/lmcache_ascend_connector_v1.py",
        "LMCacheAscendConnectorV1Dynamic",
    )
    assert any(
        isinstance(base, ast.Name) and base.id == "LMCacheConnectorV1Dynamic"
        for base in ascend.bases
    )


@pytest.mark.parametrize("selected", [0, 1])
def test_multi_query_uses_selected_child_and_preserves_legacy_fallback(selected):
    cls = methods_class(
        "vllm/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py",
        "MultiConnector",
        {"is_dsa_cold_compact_load", "get_dsa_cold_load_num_tokens"},
    )
    obj = cls()
    obj._connectors = [
        NS(
            is_dsa_cold_compact_load=lambda _: False,
            get_dsa_cold_load_num_tokens=lambda _: 129,
        ),
        NS(),
    ]
    obj._requests_to_connector = {"r": selected}
    obj.supports_dsa_compact_external_load = True
    assert obj.is_dsa_cold_compact_load("r") is (False if selected == 0 else None)
    assert obj.get_dsa_cold_load_num_tokens("r") == (129 if selected == 0 else 0)
    assert obj.is_dsa_cold_compact_load("absent") is None
    assert obj.get_dsa_cold_load_num_tokens("absent") == 0
    ascend = source_class(
        "vllm-ascend/vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py",
        "AscendMultiConnector",
    )
    assert any(
        isinstance(base, ast.Name) and base.id == "MultiConnector"
        for base in ascend.bases
    )


def test_async_variants_resolve_updated_schedule_by_python_mro():
    class Base:
        def schedule(self):
            return "base"

    async_node = source_class(
        "vllm/vllm/v1/core/sched/async_scheduler.py", "AsyncScheduler"
    )
    recompute_node = source_class(SCHEDULERS[1][0], "RecomputeScheduler")
    async_recompute_node = source_class(SCHEDULERS[1][0], "AsyncRecomputeScheduler")
    assert [base.id for base in async_node.bases] == ["Scheduler"]
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "schedule"
        for node in async_node.body
    )
    assert [base.id for base in recompute_node.bases] == ["Scheduler"]
    assert [base.id for base in async_recompute_node.bases] == [
        "AsyncScheduler",
        "RecomputeScheduler",
    ]
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "schedule"
        for node in async_recompute_node.body
    )
    async_cls = type("Async", (Base,), {})
    recompute_cls = type("Recompute", (Base,), {"schedule": lambda self: "recompute"})
    combined = type("AsyncRecompute", (async_cls, recompute_cls), {})
    assert async_cls().schedule() == "base"
    assert combined().schedule() == "recompute"


@pytest.mark.parametrize("path,name", SCHEDULERS)
def test_sync_loading_keeps_existing_prefill_and_does_not_query_async_metadata(
    path, name
):
    _, manager, connector, _, _, _ = admission(path, name, load_async=False)
    assert manager.allocate_slots.call_args.args[1] == 32
    assert manager.allocate_slots.call_args.kwargs["delay_cache_blocks"] is False
    assert manager.allocate_slots.call_args.kwargs["dsa_compact_external_load"] is False
    connector.is_dsa_cold_compact_load.assert_not_called()
    connector.get_dsa_cold_load_num_tokens.assert_not_called()


def test_actual_recompute_schedule_continues_decode_while_lookup_is_pending():
    # Execute the complete deployed schedule method with a recording allocator.
    # Its model/device bootstrap is deliberately excluded by the existing fixture.
    path = ROOT / "vllm-ascend/tests/standalone/test_sparse_recovery_budget.py"
    spec = importlib.util.spec_from_file_location("nonblocking_schedule_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    ongoing = fixture.request("ongoing", computed=128, end=129, compact=False)
    incoming = fixture.request(
        "incoming", computed=0, end=1024, compact=False, status="WAITING"
    )
    obj, allocations = fixture.scheduler(running=[ongoing], waiting=[incoming])
    obj.connector_prefix_cache_stats = None
    obj.kv_cache_manager.get_computed_blocks = lambda _: (
        obj.kv_cache_manager.empty_kv_cache_blocks,
        0,
    )

    def pending_lookup(*args):
        return None, False

    obj.connector = NS(
        get_num_new_matched_tokens=Mock(side_effect=pending_lookup),
        update_state_after_alloc=Mock(
            side_effect=AssertionError("pending lookup allocated")
        ),
        build_connector_meta=lambda _: NS(),
    )
    output = obj.schedule()
    assert output.num_scheduled_tokens == {"ongoing": 1}
    assert allocations == [("ongoing", 128, 1)]
    assert incoming.num_computed_tokens == 0 and incoming.status == "WAITING"
    obj.connector.get_num_new_matched_tokens.assert_called_once_with(incoming, 0)
    obj.connector.update_state_after_alloc.assert_not_called()


@pytest.mark.parametrize("path,name", SCHEDULERS)
def test_optional_multi_query_none_keeps_legacy_wide_policy(path, name):
    _, manager, _, _, _, _ = admission(path, name, compact=None)
    assert manager.allocate_slots.call_args.kwargs["dsa_compact_external_load"] is True


def test_legacy_bootstrap_fallback_survives_optional_multi_query_none():
    _, manager, _, _, _, reject = admission(
        *SCHEDULERS[0],
        hit=129,
        length=129,
        compact=None,
        extent=129,
        bootstrap=True,
        legacy_capability=False,
    )
    assert manager.allocate_slots.call_args.kwargs["dsa_compact_external_load"] is True
    assert reject.call_args.kwargs["compact_external_load"] is True
