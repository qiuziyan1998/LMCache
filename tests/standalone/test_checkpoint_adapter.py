# SPDX-License-Identifier: Apache-2.0
"""Execute production adapter methods with CPU-only scheduler collaborators."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lmcache/integration/vllm/vllm_v1_adapter.py"
spec = importlib.util.spec_from_file_location(
    "checkpoint_control_test", SOURCE.with_name("preemption_checkpoint.py")
)
control = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = control
spec.loader.exec_module(control)


def method(name, **extra):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    ns = dict(vars(control))
    ns.update(
        RequestStatus=NS(PREEMPTED="preempted"),
        serving_perf_enabled=lambda: False,
        logger=NS(debug=lambda *a, **kw: None),
        extract_mm_features=lambda r: ([], []),
        extract_request_configs=lambda p: None,
    )
    ns.update(extra)
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[prefix, node], type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        ns,
    )
    return ns[name]


def request():
    return NS(
        request_id="r",
        num_tokens=12,
        all_token_ids=list(range(12)),
        prompt_token_ids=list(range(4)),
        num_preemptions=1,
        status="preempted",
        sampling_params=None,
        is_finished=lambda: False,
    )


def test_capture_is_not_sealed_until_accepted_output_is_available():
    req = request()
    tracker = NS(
        prompt_len=4,
        decode_window_save_committed_end=4,
        dsa_nonresident_frontier=6,
        skip_save=False,
        request_configs=None,
    )
    adapter = NS(
        _lmcache_chunk_size=4,
        _decode_window_save_window_size=8,
        _request_trackers={"r": tracker},
        kv_role="kv_both",
        config=NS(dsa_two_groups=True),
        _preemption_checkpoints={},
        _resume_lookup_queries={},
        _unfinished_requests={"r": req},
    )
    output = NS(
        preemption_snapshots=(("r", 1, ((1, 2, 3), (4, 5, 6)), 14),),
        finished_req_ids=set(),
    )
    meta = NS()
    build = method("_build_preemption_controls")
    build(adapter, meta, output, output.preemption_snapshots)
    assert len(meta.preemption_captures) == 1 and meta.preemption_seals == ()
    pending = adapter._preemption_checkpoints["r"]
    pending.accept(control.CheckpointResult("r", 1, "captured", 14))
    output.preemption_snapshots = ()
    build(adapter, meta, output, output.preemption_snapshots)
    assert meta.preemption_seals[0].tokens == tuple(range(11))
    assert pending.status == "persisting" and pending.end == 11
    build(adapter, meta, output, output.preemption_snapshots)
    assert meta.preemption_seals == ()


def test_pending_checkpoint_parks_without_querying_storage():
    pending = control.PendingCheckpoint(
        control.CaptureSpec("r", 1, 4, 14, 6, ((1,), (2,)))
    )
    adapter = NS(
        kv_role="kv_both",
        _preemption_checkpoints={"r": pending},
        config=NS(blocking_timeout_secs=30),
    )
    assert method("get_num_new_matched_tokens")(adapter, request(), 0) is None


def test_ready_checkpoint_queries_exact_generated_partial_history():
    req = request()
    seen = []
    capture = control.CaptureSpec("r", 1, 4, 14, 6, ((1,), (2,)))
    pending = control.PendingCheckpoint(capture, "ready", 11)
    lookup = NS(
        lookup_cache=lambda **kw: -1, lookup=lambda ids, **kw: seen.append(tuple(ids))
    )
    adapter = NS(
        kv_role="kv_both",
        _preemption_checkpoints={"r": pending},
        _resume_lookup_queries={},
        _request_trackers={},
        _requests_priority={},
        lookup_client=lookup,
        skip_last_n_tokens=0,
    )
    assert method("get_num_new_matched_tokens")(adapter, req, 0) is None
    assert seen == [tuple(range(11))]
    assert adapter._resume_lookup_queries["r"][1] == "preemption_checkpoint"


def test_cold_promotion_is_one_shot_and_preserves_exact_frontier():
    take = method("_take_completed_cold_load")
    load = NS(
        can_load=False,
        dsa_cold_compact_load=True,
        dsa_remap_frontier=11,
        dsa_cold_load_generation=7,
    )
    adapter = NS(_dsa_cold_loaded_req_ids={"r"})
    assert take(adapter, "r", load)
    assert load.can_load and load.dsa_cold_compact_resume
    assert load.dsa_remap_frontier == 11 and load.dsa_cold_load_generation == 7
    assert not take(adapter, "r", load)


def test_preemption_cannot_reuse_other_generation_prepared_state():
    check = method("completed_cold_resume_state")
    req = NS(
        load_spec=NS(
            dsa_cold_compact_resume=True,
            dsa_cold_load_generation=7,
            lmcache_cached_tokens=11,
        )
    )
    state = NS(
        completed_cold_load_generation=6,
        token_count=11,
        indexer_npu_resident=True,
        prepared_sparse_sources={0: object()},
    )
    assert not check(req, state)
    state.completed_cold_load_generation = 7
    assert check(req, state)
    state.indexer_npu_resident = False
    assert not check(req, state)


def test_stale_checkpoint_result_does_not_invalidate_inflight_lookup():
    pending = control.PendingCheckpoint(
        control.CaptureSpec("r", 2, 4, 14, 6, ((1,), (2,))), "ready", 11
    )
    adapter = NS(
        _preemption_checkpoints={"r": pending},
        _resume_lookup_queries={"r": (11, "checkpoint", 0)},
        _unfinished_requests={"r": NS(num_preemptions=2)},
        lookup_client=NS(
            clear_lookup_status=lambda key: pytest.fail(
                "stale reply cleared live lookup"
            )
        ),
    )
    method("accept_preemption_result")(
        adapter, control.CheckpointResult("r", 1, "failed")
    )
    assert adapter._resume_lookup_queries["r"][0] == 11


def test_old_attempt_cannot_seal_after_request_generation_changes():
    req = request()
    req.num_preemptions = 2
    pending = control.PendingCheckpoint(
        control.CaptureSpec("r", 1, 4, 14, 6, ((1,), (2,))), "captured"
    )
    adapter = NS(
        _lmcache_chunk_size=4,
        _decode_window_save_window_size=8,
        _preemption_checkpoints={"r": pending},
        _unfinished_requests={"r": req},
    )
    output = NS(preemption_snapshots=(), finished_req_ids=set())
    meta = NS()
    method("_build_preemption_controls")(
        adapter, meta, output, output.preemption_snapshots
    )
    assert meta.preemption_seals == ()
    assert meta.preemption_cancels == (("r", 1),)


def test_failed_checkpoint_restore_invalidates_its_lookup_proof():
    req = request()
    req.kv_resume_checkpoint = (1, 12, 11)
    pending = control.PendingCheckpoint(
        control.CaptureSpec("r", 1, 4, 14, 6, ((1,), (2,))), "ready", 11
    )
    adapter = NS(
        _preemption_checkpoints={"r": pending},
        _unfinished_requests={"r": req},
        _dsa_cold_indexer_block_ids={"r": {7, 8}},
        _lmcache_chunk_size=4,
        _resume_lookup_queries={"r": (11, "preemption_checkpoint", 0)},
        lookup_client=NS(clear_lookup_status=lambda key: None),
    )
    method("update_connector_output")(
        adapter,
        NS(
            finished_recving=(), invalid_block_ids={8}, completed_decode_window_saves={}
        ),
    )
    assert pending.status == "ready" and pending.end == 8
    assert req.kv_resume_checkpoint is None
    assert "r" not in adapter._resume_lookup_queries


def test_request_skip_save_is_respected_by_checkpoint_capture():
    req = request()
    tracker = NS(
        prompt_len=4,
        decode_window_save_committed_end=4,
        dsa_nonresident_frontier=6,
        skip_save=False,
        request_configs={"lmcache.skip_save": True},
    )
    adapter = NS(
        _lmcache_chunk_size=4,
        _decode_window_save_window_size=8,
        _request_trackers={"r": tracker},
        kv_role="kv_both",
        config=NS(dsa_two_groups=True),
        _preemption_checkpoints={},
        _resume_lookup_queries={},
        _unfinished_requests={"r": req},
    )
    meta = NS()
    method("_build_preemption_controls")(
        adapter,
        meta,
        NS(
            preemption_snapshots=(("r", 1, ((1, 2), (3, 4)), 11),),
            finished_req_ids=set(),
        ),
    )
    assert meta.preemption_captures == ()


def test_missing_capture_acknowledgement_expires_to_prompt_lookup():
    import time

    req = request()
    pending = control.PendingCheckpoint(
        control.CaptureSpec("r", 1, 4, 14, 6, ((1,), (2,)))
    )
    pending.started_at = time.monotonic() - 10
    seen = []
    lookup = NS(
        clear_lookup_status=lambda key: None,
        lookup_cache=lambda **kw: -1,
        lookup=lambda ids, **kw: seen.append(tuple(ids)),
    )
    adapter = NS(
        kv_role="kv_both",
        _preemption_checkpoints={"r": pending},
        config=NS(blocking_timeout_secs=1),
        _arm_preemption_controls=lambda: seen.append("armed"),
        _resume_lookup_queries={"r": (11, "preemption_checkpoint", 0)},
        _request_trackers={},
        _requests_priority={},
        lookup_client=lookup,
        skip_last_n_tokens=0,
    )
    assert method("get_num_new_matched_tokens")(adapter, req, 0) is None
    assert pending.status == "failed" and pending.cancel_pending
    assert seen == ["armed", tuple(range(4))]


def test_completed_resume_still_checks_shared_cache_generation():
    req = NS(req_id="r", resumed_from_preemption=True, is_sparse_decode=True)
    state = NS(
        req_id="r",
        token_count=11,
        shared_request_active=True,
        shared_generation=1,
        prepared_sparse_sources={0: NS(total_tokens=11)},
    )
    adapter = NS(
        _completed_cold_resume=lambda request: True,
        _worker_retrieve_state={"r": state},
        _worker_retrieve_state_can_extend=lambda *args: False,
        lmcache_engine=NS(shared_cpu_cache_generation=2),
    )
    assert method("_should_invalidate_worker_retrieve_state")(adapter, req, 11)


def test_long_tail_is_offered_without_decode_save_and_partial_capture_is_sealed():
    req = request()
    tracker = NS(
        prompt_len=4,
        decode_window_save_committed_end=0,
        dsa_nonresident_frontier=4,
        skip_save=False,
        request_configs=None,
    )
    adapter = NS(
        _lmcache_chunk_size=4,
        _decode_window_save_window_size=0,
        _request_trackers={"r": tracker},
        kv_role="kv_both",
        config=NS(dsa_two_groups=True),
        _preemption_checkpoints={},
        _resume_lookup_queries={},
        _unfinished_requests={"r": req},
    )
    output = NS(finished_req_ids=set())
    meta = NS()
    method("_build_preemption_controls")(
        adapter, meta, output, [("r", 1, ((1,), (2,)), 80)]
    )
    capture = meta.preemption_captures[0]
    assert capture.end == 80 and capture.prefix_end == 4
    pending = adapter._preemption_checkpoints["r"]
    assert pending.accept(control.CheckpointResult("r", 1, "captured", 9))
    method("_build_preemption_controls")(adapter, meta, output, [])
    assert meta.preemption_seals[0].tokens == tuple(range(9))
    assert pending.end == 9


def test_local_lookup_marker_is_only_sent_for_the_checkpoint_generation():
    req = request()
    captured = control.CaptureSpec("r", 1, 4, 12, 4, ((1,), (2,)), prefix_end=4)
    pending = control.PendingCheckpoint(captured, "ready", 11)
    calls = []
    adapter = NS(
        kv_role="kv_both",
        _preemption_checkpoints={"r": pending},
        _resume_lookup_queries={},
        _request_trackers={},
        _requests_priority={},
        lookup_client=NS(
            lookup_cache=lambda **kw: -1, lookup=lambda ids, **kw: calls.append(kw)
        ),
        skip_last_n_tokens=0,
    )
    assert method("get_num_new_matched_tokens")(adapter, req, 0) is None
    assert calls[0]["request_configs"][control.LOCAL_CHECKPOINT_CONFIG] == 1


@pytest.mark.parametrize("hit", [8, 11])
def test_partial_local_hit_sets_the_actual_restore_frontier(hit):
    req = request()
    captured = control.CaptureSpec("r", 1, 4, 12, 4, ((1,), (2,)), prefix_end=4)
    adapter = NS(
        kv_role="kv_both",
        _preemption_checkpoints={"r": control.PendingCheckpoint(captured, "ready", 11)},
        _resume_lookup_queries={},
        _request_trackers={},
        _requests_priority={},
        skip_last_n_tokens=0,
        lookup_client=NS(lookup_cache=lambda **kw: hit),
        _cold_perf_lookup_started={},
        _lmcache_chunk_size=4,
        _block_size=4,
        _dsa_scratch_capacity=4,
        _dsa_kv_policy_threshold=4,
        config=NS(min_retrieve_tokens=0, dsa_group1_load_mode="persistent_direct_hbm"),
        enable_sparse_attention=True,
        supports_dsa_cold_compact_load=lambda: True,
        load_specs={},
    )
    matched = method(
        "get_num_new_matched_tokens", cdiv=lambda a, b: (a + b - 1) // b, LoadSpec=NS
    )(adapter, req, 0)
    assert matched == hit
    spec = adapter.load_specs["r"]
    assert spec.checkpoint_generation == 1 and spec.checkpoint_prefix_end == 4
    assert spec.dsa_remap_frontier == hit


def test_restore_retry_is_strictly_shorter_and_bounded():
    pending = control.PendingCheckpoint(
        control.CaptureSpec("r", 1, 4, 80, 4, ((1,), (2,)), prefix_end=4), "ready", 11
    )
    pending.retry_shorter(4)
    assert pending.status == "ready" and pending.end == 8
    pending.retry_shorter(4)
    assert pending.status == "failed" and pending.restore_retries == 2


def test_failed_restore_retry_is_shorter_than_the_actual_evicted_frontier():
    req = request()
    req.kv_resume_checkpoint = (1, 12, 8)
    pending = control.PendingCheckpoint(
        control.CaptureSpec("r", 1, 0, 12, 0, ((1,), (2,)), prefix_end=0), "ready", 11
    )
    adapter = NS(
        _preemption_checkpoints={"r": pending},
        _unfinished_requests={"r": req},
        _dsa_cold_indexer_block_ids={"r": {7}},
        _lmcache_chunk_size=4,
        _resume_lookup_queries={"r": (8, "preemption_checkpoint", 0)},
        lookup_client=NS(clear_lookup_status=lambda key: None),
    )
    method("update_connector_output")(
        adapter,
        NS(
            finished_recving=(), invalid_block_ids={7}, completed_decode_window_saves={}
        ),
    )
    assert pending.end == 4 and pending.status == "ready"


def test_all_worker_completion_schedules_release_even_after_request_removal():
    calls = []
    adapter = NS(
        _checkpoint_restore_attempts={"r": (1, 7)},
        _arm_preemption_controls=lambda: calls.append("arm"),
        _clear_request_marker=lambda *a: None,
        _lmcache_chunk_size=4,
        _preemption_checkpoints={},
        _unfinished_requests={},
    )
    method("update_connector_output")(
        adapter, NS(finished_recving={"r"}, completed_decode_window_saves={})
    )
    assert calls == ["arm"] and not adapter._checkpoint_restore_attempts
    meta = NS()
    method("_build_preemption_controls")(adapter, meta, NS(finished_req_ids=set()), [])
    assert meta.preemption_releases == (("r", 1, 7),)
    assert "_checkpoint_restore_releases" not in adapter.__dict__
