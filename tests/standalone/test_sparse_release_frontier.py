# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for cold-hit release acknowledgements.

Execute the real adapter methods/dataclasses without importing vLLM or NPU
extensions. Only lookup and diagnostics are stubbed; release validation,
window scheduling and save slot mappings use production code.
"""

# Standard
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import ast
import logging

# Third Party
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "lmcache/integration/vllm/vllm_v1_adapter.py"
COMMIT_PATH = ROOT / "lmcache/integration/vllm/decode_window_commit.py"


@pytest.fixture(scope="module")
def callbacks() -> SimpleNamespace:
    """Load production definitions without initializing external services."""
    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"))
    adapter_methods = {
        "supports_dsa_cold_compact_load",
        "get_num_new_matched_tokens",
        "_mark_initial_sparse_release_ready",
        "get_completed_decode_window_saves",
        "update_connector_output",
        "_eligible_dsa_release_frontier",
        "_init_decode_window_save_start",
        "_should_decode_window_save",
        "_add_decode_window_save_metas",
    }
    classes = {
        "LoadSpec",
        "SaveSpec",
        "RequestTracker",
        "ReqMeta",
        "LMCacheConnectorMetadata",
        "LMCacheConnectorV1Impl",
    }
    functions = {"_apply_mm_hashes", "_build_slot_mapping_window"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            if node.name == "LMCacheConnectorV1Impl":
                node.body = [
                    n for n in node.body if getattr(n, "name", None) in adapter_methods
                ]
                assert {n.name for n in node.body} == adapter_methods
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    commit = ast.parse(COMMIT_PATH.read_text(encoding="utf-8"))
    selected.extend(n for n in commit.body if isinstance(n, ast.FunctionDef))
    namespace = dict(
        __name__=__name__,
        dataclass=dataclass,
        field=field,
        deque=deque,
        torch=torch,
        KVConnectorMetadata=object,
        logger=logging.getLogger(__name__),
        cdiv=lambda n, d: (n + d - 1) // d,
        utils=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
        _lmcache_nvtx_annotate=lambda fn: fn,
        _dsa_sparse_decode_d_node_enabled=lambda: True,
        cold_start_perf_enabled=lambda: False,
        cold_start_perf_log=Mock(),
        _mtp_dw_diag_enabled=lambda: False,
        _mtp_dw_deep_diag_enabled=lambda: False,
        _mtp_dw_event=Mock(),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ADAPTER_PATH), "exec"), namespace
    )
    return SimpleNamespace(**namespace)


def make_request(
    callbacks: SimpleNamespace,
    prompt: int = 819200,
    *,
    cached: int | None = None,
    window: int = 256,
    delay: int = 0,
    cold_enabled: bool = True,
) -> tuple[Any, Any, Any]:
    """Construct the scheduler/worker state of a complete external hit."""
    cached = prompt if cached is None else cached
    impl = callbacks.LMCacheConnectorV1Impl()
    impl.kv_role = "kv_both"
    impl.enable_sparse_attention = True
    impl.config = SimpleNamespace(
        min_retrieve_tokens=0,
        enable_dsa_cold_compact_load=cold_enabled,
        enable_sparse_attention=True,
        dsa_two_groups=True,
        enable_shared_cpu_cache=True,
        use_layerwise=True,
    )
    impl.lookup_client = SimpleNamespace(lookup_cache=lambda **kw: cached)
    impl._cold_perf_lookup_started = {}
    impl._d_node_lookup_wait_started = {}
    impl._d_node_lookup_failures = {}
    impl._block_size = 128
    impl._lmcache_chunk_size = 256
    impl._dsa_scratch_capacity = 4096
    impl._dsa_kv_policy_threshold = 10000
    impl._decode_window_save_window_size = window
    impl._decode_window_save_commit_delay_windows = delay
    impl.load_specs = {}
    impl._completed_decode_window_saves = {}
    impl._trace_decode_window_decision = Mock()
    impl._windowed_sparse_layerwise_save_enabled = lambda: True
    impl._is_dsa_two_groups = lambda: True
    impl._shared_cpu_config_value = lambda name, default: True

    impl.get_num_new_matched_tokens(
        SimpleNamespace(request_id="repro", num_tokens=prompt), 0
    )
    spec = impl.load_specs["repro"]
    spec.can_load = True
    tracker = callbacks.RequestTracker(
        req_id="repro",
        prompt_len=prompt,
        token_ids=[7] * (prompt + 1),
        allocated_block_ids=list(range((prompt + 2 * max(window, 256)) // 128 + 1)),
        num_saved_tokens=cached,
        num_lmcache_cached_tokens=cached,
        is_decode_phase=True,
        decode_window_save_committed_end=(
            spec.dsa_release_frontier
            if spec.dsa_release_frontier is not None
            else cached
        ),
    )
    tracker.allocated_block_ids_indexer = tracker.allocated_block_ids.copy()
    if getattr(spec, "dsa_cold_compact_load", False):
        # These existing fields are set by build_connector_meta on cold resume.
        tracker.sparse_remap_frontier = spec.dsa_remap_frontier
        tracker.dsa_nonresident_frontier = spec.dsa_remap_frontier
    impl._request_trackers = {tracker.req_id: tracker}
    return impl, tracker, spec


def receive(impl: Any, tracker: Any, end: int) -> dict[str, int]:
    """Return the release authorization sent to the vLLM scheduler."""
    output = SimpleNamespace(completed_decode_window_saves={tracker.req_id: end})
    impl.update_connector_output(output)
    return output.completed_decode_window_saves


@pytest.mark.parametrize("prompt", [819199, 819200, 819201])
@pytest.mark.parametrize("window", [256, 512])
@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("delay", [0, 2])
def test_cold_hit_release_preserves_save_cursor(
    callbacks: SimpleNamespace, prompt: int, window: int, initialized: bool, delay: int
) -> None:
    """The worker's real initial ack never rewinds or re-stores the prompt."""
    impl, tracker, spec = make_request(callbacks, prompt, window=window, delay=delay)
    anchor = prompt // 256 * 256
    release = (prompt - 1) // 256 * 256
    if initialized:
        impl._init_decode_window_save_start(tracker)

    request = callbacks.ReqMeta(
        req_id=tracker.req_id, token_ids=[], is_sparse_decode=True, load_spec=spec
    )
    impl._mark_initial_sparse_release_ready(request)
    output = SimpleNamespace(
        completed_decode_window_saves=impl.get_completed_decode_window_saves()
    )
    assert output.completed_decode_window_saves == {tracker.req_id: release}
    impl.update_connector_output(output)
    assert output.completed_decode_window_saves == {tracker.req_id: release}
    assert tracker.dsa_current_released_frontier == release
    assert tracker.dsa_nonresident_frontier == prompt - 1
    assert tracker.num_saved_tokens == prompt
    assert tracker.num_lmcache_cached_tokens == prompt
    assert tracker.decode_window_save_next_start == anchor
    assert tracker.decode_window_save_anchor == anchor
    assert not tracker.decode_window_save_pending_commits
    assert receive(impl, tracker, release) == {}  # Duplicate ack is harmless.

    # Ordinary sparse metadata still skips the entire already-cached prompt.
    tracker.sparse_token_ids = [7] * prompt
    tracker.sparse_meta_frontier = prompt
    metadata = callbacks.ReqMeta.from_request_tracker(
        tracker, 128, 256, load_spec=spec, is_sparse_decode=True, dsa_two_groups=True
    )
    assert not metadata.save_spec.can_save
    assert metadata.save_spec.skip_leading_tokens == prompt

    # Save only when the original window becomes full. A partial prompt may
    # already cross that boundary on its first decode token; an aligned one
    # must not re-save the retained prompt chunk.
    saves = callbacks.LMCacheConnectorMetadata()
    impl._add_decode_window_save_metas(saves, tracker)
    end = anchor + window
    if len(tracker.token_ids) < end:
        assert len(saves.requests) == 0
        tracker.token_ids.extend([7] * (end - len(tracker.token_ids)))
        impl._add_decode_window_save_metas(saves, tracker)
    assert len(saves.requests) == 1
    save = saves.requests[0]
    assert save.decode_window_start == anchor
    assert save.decode_window_end == end
    assert save.save_spec.skip_leading_tokens == anchor
    assert save.save_slot_mapping_base == anchor
    assert save.save_slot_mapping[0].numel() == window
    assert save.save_indexer_slot_mapping[0].numel() == window
    assert tracker.decode_window_save_inflight_end == end

    # An old ack must not complete this new in-flight save or move its cursor.
    assert receive(impl, tracker, release) == {}
    assert tracker.decode_window_save_next_start == end
    assert tracker.decode_window_save_inflight_end == end
    assert not tracker.decode_window_save_pending_commits
    receive(impl, tracker, end)
    assert tracker.decode_window_save_inflight_end is None
    pending = list(tracker.decode_window_save_pending_commits)
    committed = tracker.decode_window_save_committed_end
    assert receive(impl, tracker, release) == {}
    assert tracker.decode_window_save_committed_end == committed
    assert list(tracker.decode_window_save_pending_commits) == pending


@pytest.mark.parametrize("prompt", [256, 4096, 10000])
def test_short_prompt_does_not_get_cold_hit_adjustment(
    callbacks: SimpleNamespace, prompt: int
) -> None:
    """The adjustment is not a blanket rule for chunk-aligned prompt lengths."""
    impl, tracker, spec = make_request(callbacks, prompt)
    assert not getattr(spec, "dsa_cold_compact_load", False)
    assert spec.dsa_release_frontier is None
    assert not hasattr(tracker, "sparse_remap_frontier")
    impl._init_decode_window_save_start(tracker)
    with pytest.raises(RuntimeError, match="unexpected frontier"):
        receive(impl, tracker, prompt // 256 * 256 - 256)


def test_normal_full_hit_does_not_get_cold_hit_adjustment(
    callbacks: SimpleNamespace,
) -> None:
    """A full hit without compact remapping keeps the normal frontier rules."""
    impl, tracker, _ = make_request(callbacks, cold_enabled=False)
    impl._init_decode_window_save_start(tracker)
    with pytest.raises(RuntimeError, match="unexpected frontier"):
        receive(impl, tracker, 818944)
    assert receive(impl, tracker, 819200) == {tracker.req_id: 819200}


@pytest.mark.parametrize("remap_marker", [False, True])
def test_partial_hit_keeps_cached_prefix_release(
    callbacks: SimpleNamespace, remap_marker: bool
) -> None:
    """An appended prompt must use the hit length, not prompt-minus-one."""
    impl, tracker, _ = make_request(callbacks, cached=818944)
    if remap_marker:
        tracker.sparse_remap_frontier = 818944
        tracker.dsa_nonresident_frontier = 818944
    impl._init_decode_window_save_start(tracker)
    assert receive(impl, tracker, 818944) == {tracker.req_id: 818944}
    with pytest.raises(RuntimeError, match="unexpected frontier"):
        receive(impl, tracker, 818688)


@pytest.mark.parametrize("end", [818688, 818943, 818945, 819201])
def test_cold_hit_still_rejects_unexpected_frontiers(
    callbacks: SimpleNamespace, end: int
) -> None:
    """Recognizing the one exact initial ack does not relax save validation."""
    impl, tracker, _ = make_request(callbacks)
    impl._init_decode_window_save_start(tracker)
    with pytest.raises(RuntimeError, match="unexpected frontier"):
        receive(impl, tracker, end)


def test_late_initial_ack_uses_original_hit_not_live_remap(
    callbacks: SimpleNamespace,
) -> None:
    """A late initial ack cannot rewind saved/released progress or pending saves."""
    impl, tracker, _ = make_request(callbacks)
    tracker.token_ids.extend([7] * 512)
    tracker.num_saved_tokens = 819456
    tracker.decode_window_save_committed_end = 819456
    tracker.decode_window_save_next_start = 819712
    tracker.decode_window_save_anchor = 819200
    tracker.decode_window_save_inflight_end = 819712
    tracker.sparse_remap_frontier = 819456
    tracker.dsa_nonresident_frontier = 819456
    tracker.dsa_current_released_frontier = 819456
    tracker.decode_window_save_pending_commits.append(819456)
    before = vars(tracker).copy()
    before["decode_window_save_pending_commits"] = deque([819456])
    assert receive(impl, tracker, 818944) == {}
    assert vars(tracker) == before


def test_window_disabled_keeps_initial_release_handling(
    callbacks: SimpleNamespace,
) -> None:
    """No save-window configuration is needed for the existing release path."""
    impl, tracker, _ = make_request(callbacks, window=0)
    assert receive(impl, tracker, 818944) == {tracker.req_id: 818944}
    assert tracker.decode_window_save_next_start is None
    assert tracker.num_saved_tokens == 819200
