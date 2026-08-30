# SPDX-License-Identifier: Apache-2.0
"""P2: scheduler-side sparse decode metadata and zombie request behavior."""

# Standard
from concurrent.futures import Future
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm import vllm_v1_adapter as adapter_module
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    LoadSpec,
    RequestTracker,
    WorkerRetrieveState,
    _has_live_group1_source_for_dp,
    _has_live_latent_source_for_dp,
    _live_split_source_dp_rank,
)
from tests.v1.connector_test_utils import make_worker_impl


@dataclass
class StubCachedRequestData:
    req_ids: list[str]
    new_token_ids: list[list[int]]
    new_block_ids: list[list[int]]
    resumed_req_ids: set[str] = field(default_factory=set)


@dataclass
class StubSchedulerOutput:
    finished_req_ids: set[str]
    scheduled_new_reqs: list
    scheduled_cached_reqs: StubCachedRequestData
    num_scheduled_tokens: dict[str, int]


def _make_scheduler_impl() -> LMCacheConnectorV1Impl:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl.enable_sparse_attention = True
    impl.config = MagicMock()
    impl.config.save_decode_cache = False
    impl.config.save_full_chunk_in_decode = False
    impl.config.dsa_two_groups = False
    impl.config.enable_dsa_cold_compact_load = False
    impl.config.enable_sparse_attention = True
    impl.config.enable_shared_cpu_cache = False
    impl.config.use_layerwise = True
    impl.config.dsa_group1_load_mode = "p2p_preferred"
    impl.config.priority_limit = None
    impl.kv_role = "kv_both"
    impl.force_skip_save = False
    impl._block_size = 16
    impl._lmcache_chunk_size = 256
    impl._decode_window_save_window_size = 0
    impl._decode_window_save_commit_delay_windows = 0
    impl._dsa_scratch_capacity = 4096
    impl._dsa_kv_policy_threshold = 0
    impl._dsa_kv_policy_log = False
    impl._dsa_kv_policy_states = {}
    impl._cold_perf_lookup_started = {}
    impl._resume_lookup_queries = {}
    impl._discard_partial_chunks = True
    impl._request_trackers = {}
    impl._unfinished_requests = {}
    impl.load_specs = {}
    impl._requests_priority = {}
    impl._cold_perf_lookup_started = {}
    impl.skip_last_n_tokens = 0
    return impl


def _make_preempted_lookup_request(
    req_id: str,
    prompt_len: int,
    output_tokens: list[int],
) -> SimpleNamespace:
    prompt = list(range(prompt_len))
    return SimpleNamespace(
        request_id=req_id,
        status=adapter_module.RequestStatus.PREEMPTED,
        priority=0,
        prompt_token_ids=prompt,
        num_prompt_tokens=prompt_len,
        all_token_ids=prompt + output_tokens,
        num_tokens=prompt_len + len(output_tokens),
        num_output_tokens=len(output_tokens),
        sampling_params=None,
    )


def _make_vllm_request(
    req_id: str,
    prompt_len: int,
    num_computed: int,
    decode_token: int,
) -> SimpleNamespace:
    prompt = list(range(prompt_len))
    return SimpleNamespace(
        request_id=req_id,
        num_prompt_tokens=prompt_len,
        prompt_token_ids=prompt,
        num_computed_tokens=num_computed,
        all_token_ids=prompt + [decode_token],
    )


def test_dsa_kv_policy_log_uses_storage_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _make_scheduler_impl()
    impl._dsa_kv_policy_log = True
    log = MagicMock()
    monkeypatch.setattr(adapter_module, "logger", log)

    impl._log_dsa_kv_policy("req-1", False, 1024, 1025)
    impl._log_dsa_kv_policy("req-1", False, 1024, 1026)
    impl._log_dsa_kv_policy("req-1", True, 1024, 1027)

    assert log.info.call_count == 2
    first = log.info.call_args_list[0].args
    assert first[:4] == (
        "[DSA_KV_POLICY] req=%s kv_policy=%s%s prompt_len=%d computed=%d threshold=%d",
        "req-1",
        "full_resident",
        " (first)",
    )
    second = log.info.call_args_list[1].args
    assert second[2:4] == (
        "sparse_managed",
        " (switched from full_resident)",
    )
    assert "path=dense" not in first[0]


def test_shutdown_clears_dsa_kv_policy_state() -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl._dsa_kv_policy_states = {"finished": "full_resident"}
    impl._manager = SimpleNamespace(stop_services=MagicMock())

    impl.shutdown()

    assert impl._dsa_kv_policy_states == {}
    impl._manager.stop_services.assert_called_once_with()


@pytest.mark.parametrize("kv_role", ["kv_consumer", "kv_both"])
@pytest.mark.parametrize(
    ("hit_tokens", "expected_committed"),
    [(3000, 2816), (8192, 8192)],
)
def test_dsa_prefix_hit_uses_full_allocation_and_chunk_aligned_committed_end(
    kv_role: str, hit_tokens: int, expected_committed: int
) -> None:
    impl = _make_scheduler_impl()
    impl.kv_role = kv_role
    impl.config.min_retrieve_tokens = hit_tokens + 1
    lookup_client = MagicMock()
    lookup_client.lookup_cache.return_value = hit_tokens
    impl._manager = SimpleNamespace(lookup_client=lookup_client)
    request = SimpleNamespace(
        request_id=f"{kv_role}-{hit_tokens}",
        num_tokens=hit_tokens,
    )

    # A full hit recomputes the final prompt token, so vLLM needs slots for
    # hit_tokens - 1 external tokens here and the scheduled token completes
    # the fully resident prompt layout.
    assert impl.get_num_new_matched_tokens(request, 0) == hit_tokens - 1
    load_spec = impl.load_specs[request.request_id]
    assert load_spec.lmcache_cached_tokens == hit_tokens
    assert load_spec.dsa_committed_end == expected_committed
    assert load_spec.dsa_scratch_capacity == 4096
    assert not hasattr(request, "dsa_external_tail_chunk_start")


def test_dsa_cold_compact_async_requires_complete_prompt_hit() -> None:
    impl = _make_scheduler_impl()
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.min_retrieve_tokens = 0
    lookup_client = MagicMock()
    lookup_client.lookup_cache.return_value = 8192
    impl._manager = SimpleNamespace(lookup_client=lookup_client)
    request = SimpleNamespace(request_id="cold-full-hit", num_tokens=8192)

    assert impl.get_num_new_matched_tokens(request, 0) == 8191
    assert impl.should_load_kv_async(request.request_id)
    assert impl.load_specs[request.request_id].dsa_cold_compact_load
    assert impl.load_specs[request.request_id].dsa_committed_end == 8192
    assert impl.load_specs[request.request_id].dsa_remap_frontier == 8191
    assert (
        impl.load_specs[request.request_id].dsa_release_frontier
        == 7936
    )

    lookup_client.lookup_cache.return_value = 8194
    unaligned = SimpleNamespace(request_id="cold-unaligned", num_tokens=8194)
    assert impl.get_num_new_matched_tokens(unaligned, 0) == 8193
    assert impl.load_specs[unaligned.request_id].dsa_committed_end == 8194
    assert impl.load_specs[unaligned.request_id].dsa_remap_frontier == 8193
    assert (
        impl.load_specs[unaligned.request_id].dsa_release_frontier
        == 8192
    )

    lookup_client.lookup_cache.return_value = 8193
    new_block = SimpleNamespace(request_id="cold-new-block", num_tokens=8193)
    assert impl.get_num_new_matched_tokens(new_block, 0) == 8192
    assert not impl.should_load_kv_async(new_block.request_id)
    assert not hasattr(
        impl.load_specs[new_block.request_id], "dsa_cold_compact_load"
    )

    lookup_client.lookup_cache.return_value = 4096
    partial = SimpleNamespace(request_id="cold-partial-hit", num_tokens=8192)
    assert impl.get_num_new_matched_tokens(partial, 0) == 4096
    assert not impl.should_load_kv_async(partial.request_id)
    partial_spec = impl.load_specs[partial.request_id]
    assert not hasattr(partial_spec, "dsa_cold_compact_load")
    assert partial_spec.dsa_release_frontier is None


def test_dsa_cold_compact_is_skipped_when_frontier_below_scratch_capacity() -> None:
    """Regression: 0804-4. Cold compact materializes the prefix in indexer
    blocks that only the SFA compact-scratch remap can read, and the remap
    requires a frontier of zero or >= scratch_capacity. A short prompt would
    resume with the raw unclamped frontier (hit_tokens - 1, e.g. 49), which
    the staged-SFA route rejects as FATAL frontier_too_short. Prompts whose
    cold-compact frontier would land below scratch_capacity must fall back to
    the normal dense-prefix load path instead."""
    impl = _make_scheduler_impl()
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.min_retrieve_tokens = 0
    lookup_client = MagicMock()
    lookup_client.lookup_cache.return_value = 50
    impl._manager = SimpleNamespace(lookup_client=lookup_client)
    request = SimpleNamespace(request_id="short-prompt", num_tokens=50)

    assert impl.get_num_new_matched_tokens(request, 0) == 49
    assert not impl.should_load_kv_async(request.request_id)
    load_spec = impl.load_specs[request.request_id]
    assert not hasattr(load_spec, "dsa_cold_compact_load")
    assert load_spec.dsa_release_frontier is None
    assert load_spec.dsa_committed_end == 0
    assert load_spec.dsa_scratch_capacity == 4096

    # The band between zero and scratch_capacity must also skip cold compact:
    # the raw frontier (2999) is below the scratch capacity (4096).
    lookup_client.lookup_cache.return_value = 3000
    mid = SimpleNamespace(request_id="mid-prompt", num_tokens=3000)
    assert impl.get_num_new_matched_tokens(mid, 0) == 2999
    assert not impl.should_load_kv_async(mid.request_id)
    assert not hasattr(impl.load_specs[mid.request_id], "dsa_cold_compact_load")
    assert impl.load_specs[mid.request_id].dsa_committed_end == 2999 // 256 * 256

    # A long prompt whose frontier reaches scratch_capacity still engages the
    # compact load (see test_dsa_cold_compact_async_requires_complete_prompt_hit).
    lookup_client.lookup_cache.return_value = 8192
    long = SimpleNamespace(request_id="long-prompt", num_tokens=8192)
    assert impl.get_num_new_matched_tokens(long, 0) == 8191
    assert impl.should_load_kv_async(long.request_id)
    assert impl.load_specs[long.request_id].dsa_cold_compact_load


def test_dsa_cold_compact_is_skipped_within_kv_policy_threshold() -> None:
    """Regression: 0808-2 (方案 A threshold gate). A prompt within the KV-policy
    threshold uses resident main blocks, so cold compact (compact KV
    materialized in indexer blocks) is pure overhead. Such prompts take the
    normal dense-prefix load path; only prompts beyond the threshold keep the
    compact load."""
    impl = _make_scheduler_impl()
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.min_retrieve_tokens = 0
    impl._dsa_kv_policy_threshold = 10000
    lookup_client = MagicMock()
    impl._manager = SimpleNamespace(lookup_client=lookup_client)

    # Within threshold, with a cold-compact frontier that would satisfy the
    # scratch-capacity gate: the threshold gate must skip cold compact.
    lookup_client.lookup_cache.return_value = 8378
    short = SimpleNamespace(
        request_id="short-within-threshold", num_tokens=8378
    )
    assert impl.get_num_new_matched_tokens(short, 0) == 8377
    assert not impl.should_load_kv_async(short.request_id)
    short_spec = impl.load_specs[short.request_id]
    assert not hasattr(short_spec, "dsa_cold_compact_load")
    assert short_spec.dsa_release_frontier is None
    assert short_spec.dsa_committed_end == 8378 // 256 * 256

    # Beyond threshold: the compact load still engages.
    lookup_client.lookup_cache.return_value = 12000
    long = SimpleNamespace(
        request_id="long-beyond-threshold", num_tokens=12000
    )
    assert impl.get_num_new_matched_tokens(long, 0) == 11999
    assert impl.should_load_kv_async(long.request_id)
    assert impl.load_specs[long.request_id].dsa_cold_compact_load

    # Threshold disabled (0): the same prompt engages the compact load again
    # (baseline behavior preserved when the dense fast-path is off).
    impl._dsa_kv_policy_threshold = 0
    lookup_client.lookup_cache.return_value = 8378
    control = SimpleNamespace(
        request_id="threshold-disabled", num_tokens=8378
    )
    assert impl.get_num_new_matched_tokens(control, 0) == 8377
    assert impl.should_load_kv_async(control.request_id)
    assert impl.load_specs[control.request_id].dsa_cold_compact_load


@pytest.mark.parametrize(
    (
        "prompt_len",
        "expect_cold_compact",
        "expected_committed",
        "expected_remap",
        "expected_release",
    ),
    [
        (10_000, False, 9_984, None, None),
        (10_001, True, 10_001, 10_000, 9_984),
        (10_240, True, 10_240, 10_239, 9_984),
        (10_241, False, 10_240, None, None),
    ],
)
def test_dsa_cold_compact_threshold_alignment_boundaries(
    prompt_len: int,
    expect_cold_compact: bool,
    expected_committed: int,
    expected_remap: int | None,
    expected_release: int | None,
) -> None:
    """The compact-load gate and its raw/aligned frontiers are independent."""
    impl = _make_scheduler_impl()
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.min_retrieve_tokens = 0
    impl._block_size = 128
    impl._dsa_kv_policy_threshold = 10_000
    lookup_client = MagicMock()
    lookup_client.lookup_cache.return_value = prompt_len
    impl._manager = SimpleNamespace(lookup_client=lookup_client)
    request = SimpleNamespace(
        request_id=f"threshold-{prompt_len}",
        num_tokens=prompt_len,
    )

    assert impl.get_num_new_matched_tokens(request, 0) == prompt_len - 1
    load_spec = impl.load_specs[request.request_id]
    assert (
        getattr(load_spec, "dsa_cold_compact_load", False)
        is expect_cold_compact
    )
    assert load_spec.dsa_committed_end == expected_committed
    assert load_spec.dsa_remap_frontier == expected_remap
    assert load_spec.dsa_release_frontier == expected_release


@pytest.mark.parametrize("prompt_len", [10_001, 10_240, 16_457])
def test_cold_compact_nonresident_frontier_survives_threshold_rollback(
    prompt_len: int,
) -> None:
    """Cold compact omits main latent KV even before any block release."""
    impl = _make_scheduler_impl()
    impl.config.dsa_two_groups = True
    impl._dsa_kv_policy_threshold = 10_000
    req_id = f"cold-rollback-{prompt_len}"
    remap_frontier = prompt_len - 1
    release = (
        remap_frontier
        // impl._lmcache_chunk_size
        * impl._lmcache_chunk_size
    )
    num_blocks = (prompt_len + impl._block_size - 1) // impl._block_size
    latent_blocks = list(range(num_blocks))
    indexer_blocks = list(range(10_000, 10_000 + num_blocks))
    prompt_tokens = list(range(prompt_len))
    decode_token = prompt_len + 100
    vllm_request = _make_vllm_request(
        req_id,
        prompt_len,
        prompt_len - 1,
        decode_token,
    )
    impl._unfinished_requests[req_id] = vllm_request
    load_spec = LoadSpec(
        vllm_cached_tokens=prompt_len - 1,
        lmcache_cached_tokens=prompt_len,
        can_load=True,
        dsa_committed_end=prompt_len,
        dsa_remap_frontier=remap_frontier,
        dsa_scratch_capacity=impl._dsa_scratch_capacity,
        dsa_release_frontier=release,
        dsa_current_released_frontier=0,
    )
    load_spec.dsa_cold_compact_load = True
    impl.load_specs[req_id] = load_spec
    impl._dsa_cold_loaded_req_ids = {req_id}

    # vLLM invokes this callback again when an asynchronous cold load resumes,
    # with zero new external tokens because the lookup already completed.
    impl._manager = SimpleNamespace(lookup_client=MagicMock())
    impl.update_state_after_alloc(vllm_request, 0)
    assert load_spec.can_load

    new_request = SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=prompt_tokens,
        block_ids=[latent_blocks, indexer_blocks],
        num_computed_tokens=prompt_len - 1,
        sampling_params=SimpleNamespace(extra_args=None),
    )
    new_output = StubSchedulerOutput(
        finished_req_ids=set(),
        scheduled_new_reqs=[new_request],
        scheduled_cached_reqs=StubCachedRequestData([], [], []),
        num_scheduled_tokens={req_id: 1},
    )

    first_meta = impl.build_connector_meta(new_output).requests[0]
    tracker = impl._request_trackers[req_id]
    assert tracker.dsa_nonresident_frontier == remap_frontier
    assert tracker.dsa_current_released_frontier == 0
    assert first_meta.is_sparse_decode
    assert first_meta.dsa_nonresident_frontier == remap_frontier
    assert first_meta.dsa_current_released_frontier == 0

    # Simulate invalid-block recovery rolling logical progress back into the
    # dense-threshold range before any scheduler release was acknowledged.
    vllm_request.num_computed_tokens = 9_984
    rollback_output = StubSchedulerOutput(
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=StubCachedRequestData(
            req_ids=[req_id],
            new_token_ids=[[vllm_request.all_token_ids[9_984]]],
            new_block_ids=[[]],
        ),
        num_scheduled_tokens={req_id: 1},
    )

    rollback_meta = impl.build_connector_meta(rollback_output).requests[0]
    assert len(tracker.token_ids) <= impl._dsa_kv_policy_threshold
    assert tracker.dsa_nonresident_frontier == remap_frontier
    assert tracker.dsa_current_released_frontier == 0
    assert rollback_meta.is_sparse_decode
    assert rollback_meta.dsa_nonresident_frontier == remap_frontier
    assert rollback_meta.load_spec is not None
    assert rollback_meta.load_spec.can_load
    assert rollback_meta.load_spec.dsa_committed_end >= remap_frontier
    assert rollback_meta.load_spec.lmcache_cached_tokens >= remap_frontier


def test_dsa_cold_compact_is_disabled_with_vllm_prefix_caching() -> None:
    impl = _make_scheduler_impl()
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True)
    )

    assert not impl.supports_dsa_cold_compact_load()


def test_live_split_does_not_require_decoder_cold_load() -> None:
    impl = _make_scheduler_impl()
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.use_layerwise = True
    impl.config.get_extra_config_value.side_effect = (
        lambda key, default=False: key == "mooncake_direct_npu_prefill_store"
    )
    impl._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        kv_transfer_config=SimpleNamespace(
            get_from_extra_config=lambda side, default: {
                "prefill": {"tp_size": 2},
                "decode": {"tp_size": 2},
            }.get(side, default)
        ),
    )
    impl.use_layerwise = False
    impl.async_loading = False
    impl._release_request_lookup_pins = MagicMock()
    impl._drop_worker_retrieve_state = MagicMock()
    request = SimpleNamespace(
        request_id="live-prefill",
        status=adapter_module.RequestStatus.FINISHED_STOPPED,
        kv_transfer_params={"do_remote_decode": True},
    )

    assert impl.supports_dsa_live_split()
    assert not impl.supports_dsa_live_latent_split()
    assert not impl.supports_dsa_cold_compact_load()
    impl.request_finished(request, [])
    assert request.kv_transfer_params["request_live_split"] is True


@pytest.mark.parametrize("compact", [False, True])
def test_request_finished_delays_only_compact_block_release(compact: bool) -> None:
    impl = _make_scheduler_impl()
    impl._resume_lookup_queries["finished"] = (22_016, "decode_committed", 22_016)
    impl.use_layerwise = False
    impl.async_loading = False
    impl._release_request_lookup_pins = MagicMock()
    impl._drop_worker_retrieve_state = MagicMock()
    request = SimpleNamespace(
        request_id="finished",
        status=adapter_module.RequestStatus.FINISHED_STOPPED,
        kv_transfer_params=None,
        dsa_compact_allocated=compact,
    )

    delay_free, _ = impl.request_finished(request, [])

    assert delay_free is compact
    assert "finished" not in impl._resume_lookup_queries


def test_preempted_lookup_without_decode_cache_uses_prompt_boundary() -> None:
    impl = _make_scheduler_impl()
    impl.config.save_decode_cache = False
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.min_retrieve_tokens = 0
    request = _make_preempted_lookup_request(
        "partial-tail-resume",
        22_012,
        [100_001, 100_002],
    )
    lookup_client = MagicMock()
    lookup_client.lookup_cache.return_value = -1
    lookup_client.lookup.side_effect = lambda tokens, **_: (
        len(request.prompt_token_ids)
        if len(tokens) == len(request.prompt_token_ids)
        else 21_504
    )
    impl._manager = SimpleNamespace(lookup_client=lookup_client)

    matched = impl.get_num_new_matched_tokens(request, 0)

    assert matched == 22_012
    assert lookup_client.lookup.call_args.args[0] == request.prompt_token_ids
    load_spec = impl.load_specs[request.request_id]
    assert load_spec.lmcache_cached_tokens == 22_012
    assert load_spec.dsa_cold_compact_load
    assert load_spec.dsa_committed_end == 22_012
    assert load_spec.dsa_remap_frontier == 22_012
    assert request.num_tokens - matched == 2


def test_preempted_lookup_with_decode_cache_recomputes_output_suffix() -> None:
    impl = _make_scheduler_impl()
    impl.config.save_decode_cache = True
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.min_retrieve_tokens = 0
    request = _make_preempted_lookup_request(
        "decode-cache-resume",
        22_012,
        [100_001, 100_002],
    )
    lookup_client = MagicMock()
    lookup_client.lookup_cache.return_value = -1
    lookup_client.lookup.side_effect = lambda tokens, **_: (
        len(request.prompt_token_ids)
        if len(tokens) == len(request.prompt_token_ids)
        else 21_504
    )
    impl._manager = SimpleNamespace(lookup_client=lookup_client)

    matched = impl.get_num_new_matched_tokens(request, 0)

    assert impl.config.save_decode_cache
    assert matched == len(request.prompt_token_ids)
    assert lookup_client.lookup.call_args.args[0] == request.prompt_token_ids
    assert impl.load_specs[request.request_id].dsa_cold_compact_load
    assert request.num_tokens - matched == request.num_output_tokens


def test_preempted_lookup_uses_published_decode_window() -> None:
    impl = _make_scheduler_impl()
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.min_retrieve_tokens = 0
    impl._decode_window_save_window_size = impl._lmcache_chunk_size
    request = _make_preempted_lookup_request(
        "decode-window-resume",
        22_012,
        [100_001, 100_002, 100_003, 100_004, 100_005, 100_006],
    )
    committed_end = 22_016
    impl._request_trackers[request.request_id] = SimpleNamespace(
        prompt_len=len(request.prompt_token_ids),
        token_ids=list(request.all_token_ids),
        decode_window_save_committed_end=committed_end,
        disagg_spec=None,
        skip_save=False,
        request_configs=None,
        is_decode_phase=True,
    )
    lookup_client = MagicMock()
    lookup_client.lookup_cache.return_value = -1
    lookup_client.lookup.side_effect = lambda tokens, **_: len(tokens)
    impl._manager = SimpleNamespace(lookup_client=lookup_client)

    matched = impl.get_num_new_matched_tokens(request, 0)

    assert matched == committed_end
    assert (
        lookup_client.lookup.call_args.args[0]
        == request.all_token_ids[:committed_end]
    )
    assert impl.load_specs[request.request_id].dsa_cold_compact_load
    assert request.num_tokens - matched == 2


def test_preempted_lookup_never_claims_unverified_decode_window() -> None:
    impl = _make_scheduler_impl()
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.min_retrieve_tokens = 0
    impl._decode_window_save_window_size = impl._lmcache_chunk_size
    request = _make_preempted_lookup_request(
        "decode-window-short-hit",
        22_012,
        [100_001, 100_002, 100_003, 100_004, 100_005, 100_006],
    )
    committed_end = 22_016
    actual_hit = 21_760
    impl._request_trackers[request.request_id] = SimpleNamespace(
        prompt_len=len(request.prompt_token_ids),
        token_ids=list(request.all_token_ids),
        decode_window_save_committed_end=committed_end,
        disagg_spec=None,
        skip_save=False,
        request_configs=None,
        is_decode_phase=True,
    )
    lookup_client = MagicMock()
    lookup_client.lookup_cache.return_value = -1
    lookup_client.lookup.return_value = actual_hit
    impl._manager = SimpleNamespace(lookup_client=lookup_client)

    matched = impl.get_num_new_matched_tokens(request, 0)

    assert matched == actual_hit
    assert impl.load_specs[request.request_id].lmcache_cached_tokens == actual_hit
    assert not getattr(
        impl.load_specs[request.request_id], "dsa_cold_compact_load", False
    )


def test_preempted_async_lookup_freezes_decode_window_query() -> None:
    impl = _make_scheduler_impl()
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.min_retrieve_tokens = 0
    impl._decode_window_save_window_size = impl._lmcache_chunk_size
    request = _make_preempted_lookup_request(
        "decode-window-async",
        22_012,
        list(range(100_001, 100_263)),
    )
    initial_end = 22_016
    tracker = SimpleNamespace(
        prompt_len=len(request.prompt_token_ids),
        token_ids=list(request.all_token_ids),
        decode_window_save_committed_end=initial_end,
        disagg_spec=None,
        skip_save=False,
        request_configs=None,
        is_decode_phase=True,
    )
    impl._request_trackers[request.request_id] = tracker
    lookup_client = MagicMock()
    lookup_client.lookup_cache.side_effect = [-1, initial_end]
    lookup_client.lookup.return_value = None
    impl._manager = SimpleNamespace(lookup_client=lookup_client)

    assert impl.get_num_new_matched_tokens(request, 0) is None
    tracker.decode_window_save_committed_end = 22_272
    matched = impl.get_num_new_matched_tokens(request, 0)

    assert matched == initial_end
    lookup_client.lookup.assert_called_once()
    assert lookup_client.lookup.call_args.args[0] == request.all_token_ids[:initial_end]
    assert impl._resume_lookup_queries[request.request_id][:2] == (
        initial_end,
        "decode_committed",
    )


def test_live_split_honors_shared_cpu_flag_from_extra_config() -> None:
    impl = _make_scheduler_impl()
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = False
    impl.config.extra_config = {"enable_shared_cpu_cache": True}
    impl.config.get_extra_config_value.side_effect = (
        lambda key, default=False: key == "mooncake_direct_npu_prefill_store"
    )
    impl._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False)
    )

    assert impl.supports_dsa_live_split()


@pytest.mark.parametrize(
    "mode", ("persistent_serial", "persistent_parallel_prefetch")
)
def test_persistent_group1_modes_disable_live_split(mode: str) -> None:
    impl = _make_scheduler_impl()
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.dsa_group1_load_mode = mode
    impl.config.get_extra_config_value.side_effect = (
        lambda key, default=False: key == "mooncake_direct_npu_prefill_store"
    )

    assert not impl.supports_dsa_live_split()


def test_live_latent_split_requires_explicit_opt_in() -> None:
    impl = _make_scheduler_impl()
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.use_layerwise = True
    impl.config.get_extra_config_value.side_effect = (
        lambda key, default=False: key
        in {
            "mooncake_direct_npu_prefill_store",
            "enable_dsa_live_latent_split",
            "mooncake_reuse_vllm_transfer_engine",
        }
    )
    impl._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        kv_transfer_config=SimpleNamespace(
            get_from_extra_config=lambda side, default: {
                "prefill": {"tp_size": 2},
                "decode": {"tp_size": 2},
            }.get(side, default)
        ),
    )

    assert impl.supports_dsa_live_latent_split()


def test_live_latent_source_activation_requires_transport_negotiation() -> None:
    impl = _make_scheduler_impl()
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.use_layerwise = True
    impl.config.get_extra_config_value.side_effect = (
        lambda key, default=False: key
        in {
            "mooncake_direct_npu_prefill_store",
            "enable_dsa_live_latent_split",
            "mooncake_reuse_vllm_transfer_engine",
        }
    )
    impl._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False)
    )
    impl._live_latent_split_requested = False

    assert impl.supports_dsa_live_latent_split()
    assert impl._live_latent_split_requested is False

    impl.configure_live_latent_source(True)
    assert impl._live_latent_split_requested is True

    impl.configure_live_latent_source(False)
    assert impl._live_latent_split_requested is False


def test_live_latent_split_requires_shared_transfer_engine() -> None:
    impl = _make_scheduler_impl()
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.use_layerwise = True
    impl.config.get_extra_config_value.side_effect = (
        lambda key, default=False: key
        in {
            "mooncake_direct_npu_prefill_store",
            "enable_dsa_live_latent_split",
        }
    )
    impl._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False)
    )

    assert not impl.supports_dsa_live_latent_split()


def test_live_latent_decoder_does_not_require_prefill_direct_store() -> None:
    impl = _make_scheduler_impl()
    impl.kv_role = "kv_consumer"
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.use_layerwise = True
    impl.config.get_extra_config_value.side_effect = (
        lambda key, default=False: key
        in {
            "enable_dsa_live_latent_split",
            "mooncake_reuse_vllm_transfer_engine",
        }
    )
    impl._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False)
    )

    assert not impl.supports_dsa_live_split()
    assert impl.supports_dsa_cold_compact_load()
    assert impl.supports_dsa_live_latent_split()


def test_live_latent_producer_still_requires_direct_store() -> None:
    impl = _make_scheduler_impl()
    impl.kv_role = "kv_producer"
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.use_layerwise = True
    impl.config.get_extra_config_value.side_effect = (
        lambda key, default=False: key
        in {
            "enable_dsa_live_latent_split",
            "mooncake_reuse_vllm_transfer_engine",
        }
    )
    impl._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False)
    )

    assert not impl.supports_dsa_live_split()
    assert not impl.supports_dsa_live_latent_split()


def test_live_latent_kv_both_accepts_each_local_protocol_half() -> None:
    impl = _make_scheduler_impl()
    impl.kv_role = "kv_both"
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.use_layerwise = True
    enabled = {
        "mooncake_direct_npu_prefill_store",
        "enable_dsa_live_latent_split",
        "mooncake_reuse_vllm_transfer_engine",
    }
    impl.config.get_extra_config_value.side_effect = (
        lambda key, default=False: key in enabled
    )
    impl._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        kv_transfer_config=SimpleNamespace(
            get_from_extra_config=lambda side, default: {
                "prefill": {"tp_size": 2},
                "decode": {"tp_size": 2},
            }.get(side, default)
        ),
    )

    assert impl.supports_dsa_live_split()
    assert impl.supports_dsa_cold_compact_load()
    assert impl.supports_dsa_live_latent_split()

    impl.config.enable_dsa_cold_compact_load = False
    assert impl.supports_dsa_live_latent_split()

    impl.config.enable_dsa_cold_compact_load = True
    enabled.remove("mooncake_direct_npu_prefill_store")
    assert impl.supports_dsa_live_latent_split()

    impl.config.enable_dsa_cold_compact_load = False
    assert not impl.supports_dsa_live_latent_split()


def test_live_latent_source_gate_requires_exact_tp0_dp_descriptor() -> None:
    hybrid = {
        "format": "layer_slot_runs_v1",
        "tp_rank": 0,
        "dp_rank": 2,
        "group_byte_totals": [0, 64],
        "latent_group_byte_total": 128,
        "compact_layout": {
            "group_id": 1,
            "token_count": 4,
            "layers": [{"layer_id": 0}],
            "runs": [{"token_count": 4}],
        },
        "latent_layout": {
            "group_id": 0,
            "token_count": 4,
            "layers": [{"layer_id": 0}],
            "pages": [{"token_count": 4}],
        },
    }
    params = {
        "ascend_live_split_source_v1": {"descriptors": [hybrid]}
    }

    assert _has_live_latent_source_for_dp(params, 2, 4)
    assert not _has_live_latent_source_for_dp(params, 1, 4)
    assert not _has_live_latent_source_for_dp(params, 2, 5)
    assert not _has_live_latent_source_for_dp({}, 2, 4)
    assert not _has_live_latent_source_for_dp(
        {
            "ascend_live_split_source_v1": {
                "descriptors": [{**hybrid, "latent_layout": None}]
            }
        },
        2,
        4,
    )
    for malformed in (
        {**hybrid, "tp_rank": None},
        {**hybrid, "tp_rank": 0.0},
        {**hybrid, "dp_rank": "bad"},
        {
            **hybrid,
            "latent_layout": {
                **hybrid["latent_layout"],
                "token_count": "bad",
            },
        },
        {
            **hybrid,
            "latent_layout": {
                **hybrid["latent_layout"],
                "token_count": 4.0,
            },
        },
    ):
        assert not _has_live_latent_source_for_dp(
            {
                "ascend_live_split_source_v1": {
                    "descriptors": [malformed]
                }
            },
            2,
            4,
        )


def test_live_group1_source_gate_requires_complete_tp_set() -> None:
    def descriptor(tp_rank: int, *, tokens: int = 4) -> dict:
        return {
            "tp_rank": tp_rank,
            "dp_rank": 1,
            "group_byte_totals": [0, 64],
            "compact_layout": {
                "group_id": 1,
                "token_count": tokens,
                "layers": [{"layer_id": 0}],
                "runs": [{"token_count": tokens}],
            },
        }

    complete = {
        "ascend_live_split_source_v1": {
            "descriptors": [descriptor(0), descriptor(1)]
        }
    }
    assert _has_live_group1_source_for_dp(complete, 1, 4, 2)
    assert not _has_live_group1_source_for_dp({}, 1, 4, 2)
    assert not _has_live_group1_source_for_dp(
        {
            "ascend_live_split_source_v1": {
                "descriptors": [descriptor(0)]
            }
        },
        1,
        4,
        2,
    )
    assert not _has_live_group1_source_for_dp(complete, 0, 4, 2)
    assert not _has_live_group1_source_for_dp(
        {
            "ascend_live_split_source_v1": {
                "descriptors": [descriptor(0), descriptor(1, tokens=3)]
            }
        },
        1,
        4,
        2,
    )


@pytest.mark.parametrize(
    ("negotiated", "mode", "expected"),
    [
        (False, "p2p_preferred", False),
        (True, "p2p_preferred", True),
        (True, "persistent_serial", False),
        (True, "persistent_parallel_prefetch", False),
    ],
)
def test_cold_meta_requires_two_sided_latent_activation(
    negotiated: bool,
    mode: str,
    expected: bool,
) -> None:
    impl = _make_scheduler_impl()
    impl._block_size = 2
    impl._dsa_cold_indexer_block_ids = {}
    impl._live_latent_split_requested = negotiated
    impl.config.dsa_group1_load_mode = mode
    impl._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=1)
    )
    tokens = [1, 2, 3, 4]
    request = SimpleNamespace(
        request_id="req-live",
        all_token_ids=tokens,
        prompt_token_ids=tokens,
        sampling_params=SimpleNamespace(extra_args=None),
        mm_features=None,
        mm_hashes=None,
        kv_transfer_params={
            "live_split_capabilities": (
                "ascend_live_split_v2",
                "ascend_live_split_compact_v1",
                "ascend_live_split_latent_cpu_v1",
            ),
            "remote_block_ids": [1, 2],
            "remote_dp_rank": 0,
            "ascend_live_split_source_v1": {
                "descriptors": [{
                    "tp_rank": 0,
                    "dp_rank": 0,
                    "format": "layer_slot_runs_v1",
                    "group_byte_totals": [0, 64],
                    "latent_group_byte_total": 128,
                    "compact_layout": {
                        "group_id": 1,
                        "token_count": len(tokens),
                        "layers": [{"layer_id": 0}],
                        "runs": [{"token_count": len(tokens)}],
                    },
                    "latent_layout": {
                        "group_id": 0,
                        "token_count": len(tokens),
                        "layers": [{"layer_id": 0}],
                        "pages": [{"token_count": len(tokens)}],
                    },
                }]
            },
        },
    )
    load_spec = LoadSpec(
        vllm_cached_tokens=0,
        lmcache_cached_tokens=len(tokens),
        can_load=True,
    )
    load_spec.dsa_cold_load_generation = 1
    blocks = SimpleNamespace(
        get_unhashed_block_ids_all_groups=lambda: [[], [10, 11]]
    )

    meta = impl._build_dsa_cold_compact_meta(request, blocks, load_spec)

    assert getattr(meta, "live_split_requested", False) is (
        mode == "p2p_preferred"
    )
    assert meta.live_split_latent_cpu is expected


def test_cold_meta_does_not_negotiate_live_split_without_source() -> None:
    impl = _make_scheduler_impl()
    impl._block_size = 2
    impl._dsa_cold_indexer_block_ids = {}
    impl.config.dsa_group1_load_mode = "p2p_preferred"
    impl._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            data_parallel_size=1,
            tensor_parallel_size=1,
        )
    )
    tokens = [1, 2, 3, 4]
    request = SimpleNamespace(
        request_id="req-persistent-fallback",
        all_token_ids=tokens,
        prompt_token_ids=tokens,
        sampling_params=SimpleNamespace(extra_args=None),
        mm_features=None,
        mm_hashes=None,
        kv_transfer_params={
            "live_split_capabilities": (
                "ascend_live_split_v2",
                "ascend_live_split_compact_v1",
            ),
            "live_split_transfer_id": "capability-without-source",
            "remote_block_ids": [1, 2],
            "remote_dp_rank": 0,
        },
    )
    load_spec = LoadSpec(
        vllm_cached_tokens=0,
        lmcache_cached_tokens=len(tokens),
        can_load=True,
    )
    blocks = SimpleNamespace(
        get_unhashed_block_ids_all_groups=lambda: [[], [10, 11]]
    )

    meta = impl._build_dsa_cold_compact_meta(request, blocks, load_spec)

    assert getattr(meta, "live_split_requested", False) is False
    assert getattr(meta, "live_split_remote_block_ids", None) is None


def test_dp2_live_split_requires_explicit_global_source_route() -> None:
    parallel = SimpleNamespace(data_parallel_size=2, data_parallel_index=1)
    base = {
        "live_split_capabilities": ("ascend_live_split_v2",),
        "remote_dp_rank": 0,
    }

    assert _live_split_source_dp_rank(base, parallel) is None
    assert _live_split_source_dp_rank(
        {
            **base,
            "live_split_capabilities": (
                "ascend_live_split_v2",
                "ascend_live_split_dp_routing_v1",
            ),
        },
        parallel,
    ) == 0
    assert _live_split_source_dp_rank(
        {**base, "remote_dp_rank": True}, parallel
    ) is None
    assert _live_split_source_dp_rank(
        {**base, "remote_dp_rank": 2}, parallel
    ) is None


def test_dsa_cold_compact_alloc_metadata_has_only_indexer_slots() -> None:
    impl = _make_scheduler_impl()
    impl.config.enable_dsa_cold_compact_load = True
    impl.config.dsa_two_groups = True
    impl.config.enable_shared_cpu_cache = True
    impl._pending_dsa_cold_load_metas = {}
    impl._dsa_cold_indexer_block_ids = {}
    impl._dsa_cold_load_generation = 0
    lookup_client = MagicMock()
    impl._manager = SimpleNamespace(lookup_client=lookup_client)
    req_id = "cold-meta"
    impl.load_specs[req_id] = LoadSpec(
        vllm_cached_tokens=0,
        lmcache_cached_tokens=8192,
        can_load=False,
        dsa_committed_end=8192,
        dsa_remap_frontier=8191,
    )
    impl.load_specs[req_id].dsa_cold_compact_load = True
    request = SimpleNamespace(
        request_id=req_id,
        num_tokens=8192,
        all_token_ids=list(range(8192)),
        prompt_token_ids=list(range(8192)),
        sampling_params=SimpleNamespace(extra_args=None),
        kv_transfer_params=None,
    )
    blocks = SimpleNamespace(
        get_unhashed_block_ids_all_groups=lambda: [
            list(range(16)),
            list(range(100, 612)),
        ]
    )

    impl.update_state_after_alloc(request, 8191, blocks)
    req_meta = impl._pending_dsa_cold_load_metas[req_id]
    assert req_meta.slot_mapping == []
    assert len(req_meta.indexer_slot_mapping[0]) == 8192
    assert req_meta.dsa_current_released_frontier == 0
    assert req_meta.dsa_nonresident_frontier == 8191
    assert req_meta.load_spec.dsa_cold_load_generation == 1


def test_dsa_cold_compact_submit_captures_current_npu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl._block_size = 16
    impl._dsa_cold_load_futures = {}
    impl._run_dsa_cold_compact_load = MagicMock()
    executor = SimpleNamespace(submit=MagicMock(return_value=Future()))
    impl._get_dsa_cold_load_executor = lambda: executor
    fake_npu = SimpleNamespace(current_device=MagicMock(return_value=5))
    monkeypatch.setattr(adapter_module.torch, "npu", fake_npu, raising=False)
    request = SimpleNamespace(
        req_id="cold-device-submit",
        load_spec=SimpleNamespace(dsa_cold_load_generation=1),
        indexer_slot_mapping=[torch.tensor([160, 161])],
    )

    impl._submit_dsa_cold_compact_load(request)

    executor.submit.assert_called_once_with(
        impl._run_dsa_cold_compact_load,
        request,
        5,
    )


def test_staged_sfa_native_barrier_waits_for_cold_compact_loads() -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    first = MagicMock(done=MagicMock(return_value=False))
    second = MagicMock(done=MagicMock(return_value=True))
    impl._dsa_cold_load_futures = {
        "cold-pending": (1, first, object(), set(), 0.0),
        "cold-complete": (1, second, object(), set(), 0.0),
    }

    impl.synchronize_staged_sfa_capture_unsafe_loads()

    first.result.assert_called_once_with()
    second.result.assert_called_once_with()
    assert set(impl._dsa_cold_load_futures) == {
        "cold-pending",
        "cold-complete",
    }


def test_staged_sfa_native_barrier_defers_cold_load_failure() -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    failed = Future()
    failed.set_exception(RuntimeError("cold load failed"))
    request = SimpleNamespace(
        load_spec=SimpleNamespace(dsa_cold_load_generation=1)
    )
    impl._dsa_cold_load_futures = {
        "cold-failed": (1, failed, request, {100}, 0.0),
    }
    impl._synchronize_dsa_cold_dense_load = MagicMock()
    impl._release_unadopted_shared_request_objects = MagicMock()
    impl._release_shared_worker_retrieve_state = MagicMock()
    impl._invalid_block_ids = set()
    impl.lmcache_engine = object()

    impl.synchronize_staged_sfa_capture_unsafe_loads()

    impl._synchronize_dsa_cold_dense_load.assert_called_once_with()
    assert "cold-failed" in impl._dsa_cold_load_futures
    assert impl._drain_dsa_cold_load_futures() == {"cold-failed"}
    assert impl._invalid_block_ids == {100}
    assert not hasattr(impl, "_dsa_cold_load_futures")


def test_staged_sfa_native_barrier_rejects_active_failed_stream() -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    failed = Future()
    failed.set_exception(RuntimeError("cold load failed"))
    impl._dsa_cold_load_futures = {
        "cold-failed": (1, failed, object(), set(), 0.0),
    }
    impl._synchronize_dsa_cold_dense_load = MagicMock(
        side_effect=RuntimeError("stream still active")
    )

    with pytest.raises(RuntimeError, match="stream still active"):
        impl.synchronize_staged_sfa_capture_unsafe_loads()


def test_dsa_cold_compact_worker_restores_submitted_npu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl.lmcache_engine = object()
    impl.num_layers = 1
    impl._num_layers_for_group = lambda _kv_group: 0
    fake_npu = SimpleNamespace(set_device=MagicMock())
    monkeypatch.setattr(adapter_module.torch, "npu", fake_npu, raising=False)
    request = SimpleNamespace(load_spec=object())

    with pytest.raises(RuntimeError, match="matching latent/indexer"):
        impl._run_dsa_cold_compact_load(request, 6)

    fake_npu.set_device.assert_called_once_with(6)


def test_dsa_cold_compact_worker_retains_sources_when_stream_sync_fails() -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl.num_layers = 1
    impl._num_layers_for_group = lambda _kv_group: 1
    impl._kvcaches_for_group = lambda _kv_group: []
    impl._sparse_retrieve_kwargs = MagicMock(return_value=({}, None, None))
    impl._synchronize_dsa_cold_dense_load = MagicMock(
        side_effect=RuntimeError("sync failed")
    )
    impl._release_unadopted_shared_request_objects = MagicMock()
    impl._release_shared_worker_retrieve_state = MagicMock()
    impl.lmcache_engine = SimpleNamespace(
        retrieve_layer_head_token_wise=MagicMock(
            side_effect=ValueError("retrieve failed")
        )
    )
    request = SimpleNamespace(
        req_id="cold-sync-failed",
        load_spec=SimpleNamespace(lmcache_cached_tokens=1),
        token_ids=[1],
    )

    with pytest.raises(ValueError, match="retrieve failed") as raised:
        impl._run_dsa_cold_compact_load(request, None)

    assert isinstance(
        raised.value._lmcache_dsa_cold_state,
        WorkerRetrieveState,
    )
    impl._release_unadopted_shared_request_objects.assert_not_called()
    impl._release_shared_worker_retrieve_state.assert_not_called()


def test_dsa_cold_compact_finished_signal_waits_for_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    future = Future()
    request = SimpleNamespace(
        load_spec=SimpleNamespace(
            lmcache_cached_tokens=8192,
            dsa_cold_load_generation=1,
        )
    )
    state = WorkerRetrieveState(req_id="cold-future")
    state.location = "LocalCPUBackend"
    state._dsa_cold_load_completed_at = 2.0
    impl._dsa_cold_load_futures = {
        "cold-future": (1, future, request, {100, 101}, 0.0)
    }
    impl._publish_worker_retrieve_state = MagicMock()
    impl._invalid_block_ids = set()
    events = []
    monkeypatch.setattr(
        adapter_module,
        "cold_start_perf_log",
        lambda _logger, event, **fields: events.append((event, fields)),
    )
    monkeypatch.setattr(adapter_module, "cold_start_perf_now", lambda: 3.0)

    assert impl._drain_dsa_cold_load_futures() is None
    assert "cold-future" in impl._dsa_cold_load_futures

    future.set_result(state)
    assert impl._drain_dsa_cold_load_futures() == {"cold-future"}
    impl._publish_worker_retrieve_state.assert_called_once()
    assert state._dsa_cold_prune_protected
    event, fields = events[-1]
    assert event == "worker_load_complete"
    assert fields["mode"] == "dsa_cold_compact"
    assert fields["background_ms"] == 2000.0
    assert fields["scheduler_poll_ms"] == 1000.0


def test_dsa_cold_compact_generation_mismatch_releases_returned_state() -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    future = Future()
    request = SimpleNamespace(
        load_spec=SimpleNamespace(dsa_cold_load_generation=2)
    )
    state = WorkerRetrieveState(req_id="cold-stale-generation")
    future.set_result(state)
    impl._dsa_cold_load_futures = {
        "cold-stale-generation": (1, future, request, {100}, 0.0)
    }
    impl._worker_retrieve_state = {}
    impl._synchronize_dsa_cold_dense_load = MagicMock()
    impl._release_unadopted_shared_request_objects = MagicMock()
    impl._release_shared_worker_retrieve_state = MagicMock()
    impl._publish_worker_retrieve_state = MagicMock()
    impl._invalid_block_ids = set()
    impl.lmcache_engine = object()

    assert impl._drain_dsa_cold_load_futures() == {
        "cold-stale-generation"
    }
    impl._publish_worker_retrieve_state.assert_not_called()
    # The completed future owns a concrete state even when its generation is
    # stale, so the rejected result must be released rather than leaked.
    impl._release_unadopted_shared_request_objects.assert_called_once_with(
        state, request
    )
    impl._release_shared_worker_retrieve_state.assert_called_once_with(
        state, impl.lmcache_engine
    )
    assert impl._invalid_block_ids == {100}


def test_dsa_cold_compact_failed_state_releases_after_sync_retry() -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    future = Future()
    request = SimpleNamespace(
        load_spec=SimpleNamespace(dsa_cold_load_generation=1)
    )
    state = WorkerRetrieveState(req_id="cold-sync-retry")
    error = RuntimeError("load failed")
    error._lmcache_dsa_cold_state = state
    future.set_exception(error)
    impl._dsa_cold_load_futures = {
        "cold-sync-retry": (1, future, request, {100, 101}, 0.0)
    }
    impl._synchronize_dsa_cold_dense_load = MagicMock(
        side_effect=[RuntimeError("still active"), None]
    )
    impl._release_unadopted_shared_request_objects = MagicMock()
    impl._release_shared_worker_retrieve_state = MagicMock()
    impl._invalid_block_ids = set()
    impl.lmcache_engine = object()

    assert impl._drain_dsa_cold_load_futures() is None
    assert "cold-sync-retry" in impl._dsa_cold_load_futures
    impl._release_shared_worker_retrieve_state.assert_not_called()

    assert impl._drain_dsa_cold_load_futures() == {"cold-sync-retry"}
    impl._release_unadopted_shared_request_objects.assert_called_once_with(
        state, request
    )
    impl._release_shared_worker_retrieve_state.assert_called_once_with(
        state, impl.lmcache_engine
    )
    assert impl._invalid_block_ids == {100, 101}


def test_dsa_cold_ready_state_survives_cross_rank_completion_gap() -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    state = WorkerRetrieveState(
        req_id="cold-ready-gap",
        metadata_warm=True,
        token_count=8192,
        shared_request_active=True,
        shared_latent_status="present",
        shared_index_status="present",
        indexer_npu_resident=True,
    )
    state._dsa_cold_prune_protected = True
    impl._worker_retrieve_state = {"cold-ready-gap": state}
    impl._worker_retrieve_registry_version = 1
    impl._release_shared_worker_retrieve_state = MagicMock()
    impl._release_request_lookup_pins = MagicMock()

    # Another TP may still be loading while this TP participates in forwards
    # that contain only unrelated RUNNING requests.
    impl._prune_worker_retrieve_state(set())

    assert impl._worker_retrieve_state["cold-ready-gap"] is state
    impl._release_shared_worker_retrieve_state.assert_not_called()
    request = SimpleNamespace(
        load_spec=SimpleNamespace(lmcache_cached_tokens=8192)
    )
    assert (
        impl._shared_sparse_decode_indexer_retrieve_mode(request, state, 8192)
        == adapter_module.INDEXER_RETRIEVE_RESIDENT_SKIP
    )


def test_dsa_cold_compact_abort_releases_unpublished_cpu_state() -> None:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    future = Future()
    request = SimpleNamespace(
        load_spec=SimpleNamespace(
            lmcache_cached_tokens=8192,
            dsa_cold_load_generation=1,
        )
    )
    state = WorkerRetrieveState(req_id="cold-aborted")
    future.set_result(state)
    impl._dsa_cold_load_futures = {
        "cold-aborted": (1, future, request, {100, 101}, 0.0)
    }
    impl._dsa_cold_aborted_req_ids = {"cold-aborted"}
    impl.lmcache_engine = object()
    impl._publish_worker_retrieve_state = MagicMock()
    impl._release_unadopted_shared_request_objects = MagicMock()
    impl._release_shared_worker_retrieve_state = MagicMock()
    impl._release_request_lookup_pins = MagicMock()
    impl._invalid_block_ids = set()

    assert impl._drain_dsa_cold_load_futures() == {"cold-aborted"}
    impl._publish_worker_retrieve_state.assert_not_called()
    impl._release_unadopted_shared_request_objects.assert_called_once_with(
        state, request
    )
    impl._release_shared_worker_retrieve_state.assert_called_once()
    impl._release_request_lookup_pins.assert_called_once_with("cold-aborted")
    assert not hasattr(impl, "_dsa_cold_aborted_req_ids")


def test_dsa_cold_compact_failure_does_not_mark_request_ready() -> None:
    impl = _make_scheduler_impl()
    impl._dsa_cold_loaded_req_ids = set()
    impl._dsa_cold_indexer_block_ids = {
        "cold-failed": {100, 101},
        "cold-ready": {200, 201},
    }
    for req_id in ("cold-failed", "cold-ready"):
        impl.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=8192,
            can_load=True,
        )
        impl.load_specs[req_id].dsa_cold_compact_load = True

    impl.update_connector_output(
        SimpleNamespace(
            finished_recving={"cold-failed", "cold-ready"},
            invalid_block_ids={100, 101},
            completed_decode_window_saves={},
        )
    )

    assert impl._dsa_cold_loaded_req_ids == {"cold-ready"}
    assert not hasattr(impl, "_dsa_cold_indexer_block_ids")


def test_first_sparse_step_publishes_initial_release_frontier_once() -> None:
    impl = _make_scheduler_impl()
    impl._completed_decode_window_saves = {}
    request = SimpleNamespace(
        req_id="initial-sparse-release",
        is_sparse_decode=True,
        resumed_from_preemption=False,
        dsa_current_released_frontier=0,
        dsa_nonresident_frontier=0,
        load_spec=SimpleNamespace(
            dsa_committed_end=8192,
            dsa_release_frontier=8192,
            dsa_current_released_frontier=0,
        ),
    )

    impl._mark_initial_sparse_release_ready(request)
    assert impl.get_completed_decode_window_saves() == {
        request.req_id: 8192,
    }

    # The one-shot marker prevents a later sparse step from publishing the
    # initial frontier again after the completion output has been drained.
    impl._mark_initial_sparse_release_ready(request)
    assert impl.get_completed_decode_window_saves() == {}

    request.resumed_from_preemption = True
    impl._mark_initial_sparse_release_ready(request)
    assert impl.get_completed_decode_window_saves() == {
        "initial-sparse-release": 8192
    }


def test_chunk_size_must_be_integer_multiple_of_block_size() -> None:
    impl = _make_scheduler_impl()
    impl._lmcache_chunk_size = 250

    with pytest.raises(
        AssertionError,
        match=r"chunk_size=250, block_size=16.*N \* block_size",
    ):
        impl._get_decode_window_save_window_size(impl.config)


def test_decode_window_decisions_cover_second_window_and_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _make_scheduler_impl()
    impl._decode_window_save_window_size = 256
    tracker = RequestTracker(
        req_id="window-progress",
        prompt_len=256,
        token_ids=list(range(400)),
        allocated_block_ids=list(range(64)),
        num_saved_tokens=256,
    )
    tracker.is_decode_phase = True
    tracker.decode_window_save_next_start = 256
    events: list[dict] = []
    monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "1")
    monkeypatch.setattr(
        adapter_module,
        "_mtp_dw_event",
        lambda _stage, **fields: events.append(fields),
    )

    impl._trace_decode_window_decision(tracker)
    tracker.token_ids.extend(range(400, 508))
    impl._trace_decode_window_decision(tracker)
    tracker.token_ids.extend(range(508, 512))
    impl._trace_decode_window_decision(tracker)
    impl._trace_decode_window_decision(
        tracker, decision="emitted", reason="window_ready"
    )
    tracker.decode_window_save_next_start = 512
    tracker.token_ids.extend(range(512, 764))
    impl._trace_decode_window_decision(tracker)
    tracker.token_ids.extend(range(764, 768))
    impl._trace_decode_window_decision(tracker)
    impl._trace_decode_window_decision(
        tracker, decision="request_finish", reason="request_finished"
    )

    decisions = [event["decision"] for event in events]
    assert decisions == [
        "initial",
        "near_boundary",
        "boundary_reached",
        "emitted",
        "near_boundary",
        "boundary_reached",
        "request_finish",
    ]
    assert events[-2]["next_end"] == 768
    assert events[-2]["tracker_len"] == 768
    assert events[-1]["reason"] == "request_finished"


def test_decode_window_decision_reports_blocked_unfinished_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _make_scheduler_impl()
    impl._decode_window_save_window_size = 256
    tracker = RequestTracker(
        req_id="window-blocked",
        prompt_len=256,
        token_ids=list(range(256)),
        allocated_block_ids=list(range(16)),
        num_saved_tokens=256,
    )
    events: list[dict] = []
    monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "1")
    monkeypatch.setattr(
        adapter_module,
        "_mtp_dw_event",
        lambda _stage, **fields: events.append(fields),
    )

    impl._trace_decode_window_decision(tracker)
    impl._trace_decode_window_decision(
        tracker, decision="request_finish", reason="request_finished"
    )

    assert events[0]["decision"] == "initial"
    assert events[0]["reason"] == "not_decode_phase"
    assert events[0]["should_save"] is False
    assert events[1]["decision"] == "request_finish"
    assert events[1]["tracker_len"] == 256


def test_decode_window_call_site_allocates_no_state_when_diag_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _make_scheduler_impl()
    impl._decode_window_save_window_size = 256
    tracker = RequestTracker(
        req_id="window-disabled-diag",
        prompt_len=256,
        token_ids=list(range(256)),
        allocated_block_ids=list(range(16)),
        num_saved_tokens=256,
    )
    meta = MagicMock()
    monkeypatch.delenv("VLLM_ASCEND_MTP_DW_DIAG", raising=False)

    impl._add_decode_window_save_metas(meta, tracker)

    assert not hasattr(impl, "_mtp_dw_window_decision_states")
    meta.add_request.assert_not_called()


def test_decode_window_anchor_is_partial_lmcache_chunk_start() -> None:
    impl = _make_scheduler_impl()
    impl._decode_window_save_window_size = 512
    tracker = RequestTracker(
        req_id="anchored-window",
        prompt_len=300,
        token_ids=list(range(1400)),
        allocated_block_ids=list(range(120)),
        num_saved_tokens=300,
    )
    tracker.is_decode_phase = True
    tracker.decode_window_save_committed_end = 300

    assert impl._init_decode_window_save_start(tracker) == 256
    assert tracker.decode_window_save_anchor == 256
    assert tracker.decode_window_save_committed_end == 256


def test_decode_window_merges_all_complete_windows_that_are_ready() -> None:
    impl = _make_scheduler_impl()
    impl._decode_window_save_window_size = 512
    tracker = RequestTracker(
        req_id="one-window-at-a-time",
        prompt_len=300,
        token_ids=list(range(1400)),
        allocated_block_ids=list(range(120)),
        num_saved_tokens=300,
    )
    tracker.is_decode_phase = True
    meta = MagicMock()

    impl._add_decode_window_save_metas(meta, tracker)

    meta.add_request.assert_called_once()
    emitted = meta.add_request.call_args.args[0]
    assert emitted.decode_window_start == 256
    assert emitted.decode_window_end == 1280
    assert tracker.decode_window_save_inflight_end == 1280
    assert tracker.decode_window_save_next_start == 1280

    impl._add_decode_window_save_metas(meta, tracker)
    meta.add_request.assert_called_once()

    impl._request_trackers[tracker.req_id] = tracker
    impl.update_connector_output(
        SimpleNamespace(completed_decode_window_saves={tracker.req_id: 1280})
    )
    assert tracker.decode_window_save_committed_end == 1280
    assert tracker.decode_window_save_inflight_end is None

    tracker.token_ids.extend(range(1400, 1824))
    impl._add_decode_window_save_metas(meta, tracker)
    assert meta.add_request.call_count == 2
    next_emitted = meta.add_request.call_args.args[0]
    assert next_emitted.decode_window_start == 1280
    assert next_emitted.decode_window_end == 1792


def test_prefill_completion_publishes_only_chunk_aligned_frontier() -> None:
    impl = _make_scheduler_impl()
    impl._completed_decode_window_saves = {}
    request = SimpleNamespace(
        req_id="prefill-partial",
        token_ids=list(range(300)),
        is_last_prefill=True,
        is_sparse_decode=False,
        is_decode_window_save=False,
    )

    impl._mark_prefill_committed(request, len(request.token_ids))

    assert impl._completed_decode_window_saves == {"prefill-partial": 256}


class TestRequestTrackerPhase:
    def test_one_token_prefill_boundary_remains_prefill(self) -> None:
        tracker = RequestTracker(
            req_id="phase-boundary",
            prompt_len=4,
            token_ids=[0, 1, 2],
            allocated_block_ids=[0],
        )

        tracker.update([3], [])
        assert not tracker.is_decode_phase

        tracker.update([4], [])
        assert tracker.is_decode_phase


class TestDisaggSpecOwnership:
    def test_build_meta_copies_spec_from_scheduler_owned_request(self) -> None:
        impl = _make_scheduler_impl()
        req_id = "vllm-req"
        impl._unfinished_requests[req_id] = SimpleNamespace(
            kv_transfer_params={
                "disagg_spec": {
                    "req_id": "transfer-req",
                    "receiver_host": "decode-host",
                    "receiver_init_port": 9000,
                    "receiver_alloc_port": 9001,
                }
            }
        )
        new_request = SimpleNamespace(
            req_id=req_id,
            prompt_token_ids=[1],
            block_ids=[0],
            num_computed_tokens=0,
            sampling_params=SimpleNamespace(extra_args=None),
        )
        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[new_request],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[],
                new_token_ids=[],
                new_block_ids=[],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        impl.build_connector_meta(scheduler_output)

        tracker = impl._request_trackers[req_id]
        assert tracker.disagg_spec is not None
        assert tracker.disagg_spec.receiver_id == "decode-host9000"


class TestBuildConnectorMetaSparseSyntheticLoadSpec:
    def test_cold_compact_resume_marker_is_one_shot(self) -> None:
        impl = _make_scheduler_impl()
        impl._manager = SimpleNamespace(lookup_client=MagicMock())
        req_id = "cold-resume"
        prompt_len = 8193
        request = SimpleNamespace(
            request_id=req_id,
            req_id=req_id,
            prompt_token_ids=list(range(prompt_len)),
            block_ids=list(range(513)),
            num_tokens=prompt_len,
            num_computed_tokens=prompt_len - 1,
            sampling_params=SimpleNamespace(extra_args=None),
        )
        impl._unfinished_requests[req_id] = _make_vllm_request(
            req_id,
            prompt_len,
            prompt_len - 1,
            prompt_len,
        )
        impl._dsa_cold_loaded_req_ids = {req_id}
        impl.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=prompt_len,
            can_load=False,
            dsa_committed_end=prompt_len,
            dsa_remap_frontier=prompt_len - 1,
        )
        impl.load_specs[req_id].dsa_cold_compact_load = True
        cold_meta = SimpleNamespace(req_id=req_id)
        impl._build_dsa_cold_compact_meta = MagicMock(return_value=cold_meta)

        # The first allocation callback admits the asynchronous full-prefix
        # load and emits its no-forward worker metadata.
        impl.update_state_after_alloc(request, prompt_len - 1, object())
        assert impl.load_specs[req_id].can_load is True
        initial = impl.build_connector_meta(
            StubSchedulerOutput(
                finished_req_ids=set(),
                scheduled_new_reqs=[],
                scheduled_cached_reqs=StubCachedRequestData([], [], []),
                num_scheduled_tokens={},
            )
        )
        assert initial.dsa_cold_compact_load_pending is True
        assert initial.requests == [cold_meta]

        # A completed async load is scheduled a second time with no newly
        # allocated external tokens. This is the real transition that used to
        # clear can_load immediately before the first sparse resume.
        impl._dsa_cold_loaded_req_ids = {req_id}
        impl.update_state_after_alloc(request, 0)
        assert impl.load_specs[req_id].can_load is False

        first = impl.build_connector_meta(
            StubSchedulerOutput(
                finished_req_ids=set(),
                scheduled_new_reqs=[request],
                scheduled_cached_reqs=StubCachedRequestData([], [], []),
                num_scheduled_tokens={req_id: 1},
            )
        ).requests[0]

        assert first.is_sparse_decode
        assert first.load_spec.can_load is True
        assert first.load_spec.dsa_committed_end == prompt_len
        assert first.load_spec.dsa_remap_frontier == prompt_len - 1
        assert first.dsa_nonresident_frontier == prompt_len - 1
        assert first.load_spec.dsa_cold_compact_resume is True
        assert not hasattr(first.load_spec, "dsa_cold_compact_load")

        second = impl.build_connector_meta(
            StubSchedulerOutput(
                finished_req_ids=set(),
                scheduled_new_reqs=[],
                scheduled_cached_reqs=StubCachedRequestData(
                    [req_id], [[prompt_len]], [[]]
                ),
                num_scheduled_tokens={req_id: 1},
            )
        ).requests[0]
        assert not hasattr(second.load_spec, "dsa_cold_compact_resume")

    def test_default_zero_threshold_keeps_sparse_decode(
        self,
    ) -> None:
        """方案 A: with the default threshold 0 (unset or '0') the dense
        fast-path is disabled, so even short prompts keep sparse decode."""
        impl = _make_scheduler_impl()
        impl._dsa_kv_policy_threshold = 0
        req_id = "zero-threshold"
        prompt_len = 300
        decode_token = 999
        vllm_req = _make_vllm_request(
            req_id, prompt_len, prompt_len, decode_token
        )

        impl._unfinished_requests[req_id] = vllm_req
        impl._request_trackers[req_id] = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=list(range(19)),
            num_saved_tokens=prompt_len,
        )
        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_token]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert len(meta.requests) == 1
        req_meta = meta.requests[0]
        assert req_meta.is_sparse_decode
        assert req_meta.load_spec is not None

    def test_short_prompt_within_kv_policy_threshold_is_full_resident(
        self,
    ) -> None:
        """方案 A: requests within the KV-policy threshold use the
        full-resident policy (is_sparse_decode=False).

        Regression: 0806-4. The FIRST compute step (token_ids == prompt_len,
        is_decode_phase still False, e.g. right after a cold-compact load) must
        also stay in the connector metadata (is_sparse_decode=False,
        load_spec=None). Otherwise the step classifies as
        MISSING_CONNECTOR_METADATA, falls back to the eager path, and the
        torch_npu fx compiler captures ACL graphs at serving time — which is
        illegal to overlap with the async cold-compact load thread's
        synchronized device copies (device error 507057). Keeping every
        full-resident step staged prevents serving-time graph capture."""
        impl = _make_scheduler_impl()
        impl._dsa_kv_policy_threshold = 2048
        req_id = "short-prompt"
        prompt_len = 300
        decode_token = 999
        vllm_req = _make_vllm_request(
            req_id, prompt_len, prompt_len, decode_token
        )

        impl._unfinished_requests[req_id] = vllm_req
        impl._request_trackers[req_id] = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=list(range(19)),
            num_saved_tokens=prompt_len,
            is_decode_phase=False,
        )
        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_token]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert len(meta.requests) == 1
        req_meta = meta.requests[0]
        assert not req_meta.is_sparse_decode
        assert req_meta.load_spec is None
        assert req_meta.save_spec is not None
        assert not req_meta.save_spec.can_save

    def test_long_prompt_beyond_kv_policy_threshold_is_sparse_managed(
        self,
    ) -> None:
        """方案 A: requests beyond the KV-policy threshold keep the
        sparse-managed behavior."""
        impl = _make_scheduler_impl()
        impl._dsa_kv_policy_threshold = 2048
        req_id = "long-prompt"
        prompt_len = 4096
        decode_token = 999
        vllm_req = _make_vllm_request(
            req_id, prompt_len, prompt_len, decode_token
        )
        impl._unfinished_requests[req_id] = vllm_req
        impl._request_trackers[req_id] = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=list(range(256)),
            num_saved_tokens=prompt_len,
        )
        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_token]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert len(meta.requests) == 1
        req_meta = meta.requests[0]
        assert req_meta.is_sparse_decode
        assert req_meta.load_spec is not None
        assert req_meta.load_spec.lmcache_cached_tokens == prompt_len

    def test_short_prompt_growing_past_threshold_switches_to_sparse(
        self,
    ) -> None:
        """方案 A: the KV-policy decision follows the CURRENT context length. A
        short prompt that grows past the threshold switches from full-resident
        to sparse-managed loading/release behavior."""
        impl = _make_scheduler_impl()
        impl._dsa_kv_policy_threshold = 2048
        impl._decode_window_save_window_size = 256
        req_id = "grown-prompt"
        prompt_len = 300
        context_len = 5000
        decode_token = 999
        vllm_req = _make_vllm_request(
            req_id, prompt_len, context_len, decode_token
        )

        impl._unfinished_requests[req_id] = vllm_req
        impl._request_trackers[req_id] = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(context_len)),
            allocated_block_ids=list(range(313)),
            num_saved_tokens=prompt_len,
        )
        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_token]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert len(meta.requests) >= 1
        req_meta = next(
            request
            for request in meta.requests
            if not getattr(request, "is_decode_window_save", False)
        )
        assert req_meta.is_sparse_decode
        assert req_meta.load_spec is not None
        assert not req_meta.load_spec.can_load
        assert req_meta.load_spec.dsa_committed_end == 0

    def test_first_decode_step_keeps_short_prompt_resident(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        req_id = "first-decode"
        prompt_len = 300
        decode_token = 999
        vllm_req = _make_vllm_request(
            req_id, prompt_len, prompt_len, decode_token
        )

        impl._unfinished_requests[req_id] = vllm_req
        impl._request_trackers[req_id] = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=list(range(19)),
            num_saved_tokens=prompt_len,
        )
        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_token]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert len(meta.requests) == 1
        req_meta = meta.requests[0]
        assert req_meta.is_sparse_decode
        assert req_meta.load_spec is not None
        assert req_meta.load_spec.can_load is False
        assert req_meta.load_spec.lmcache_cached_tokens == 0
        assert req_meta.load_spec.dsa_committed_end == 0

    def test_sparse_decode_steps_synthesize_load_spec_and_sparse_tokens(self) -> None:
        impl = _make_scheduler_impl()
        req_id = "sparse-req"
        prompt_len = 256
        vllm_req = _make_vllm_request(req_id, prompt_len, prompt_len + 1, 999)
        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=list(range(16)),
            num_saved_tokens=prompt_len,
        )
        tracker.is_decode_phase = True
        tracker.sparse_token_ids = list(range(prompt_len))
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[999]],
                new_block_ids=[[16]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        assert len(meta.requests) == 1
        req_meta = meta.requests[0]

        assert req_meta.is_sparse_decode
        assert req_meta.load_spec is not None
        assert req_meta.load_spec.can_load is True
        assert req_meta.load_spec.lmcache_cached_tokens == prompt_len
        assert req_meta.token_ids == list(range(prompt_len))
        assert req_meta.save_spec is not None
        assert req_meta.save_spec.can_save is False
        assert not hasattr(req_meta, "cached_keys")

    def test_sparse_decode_does_not_load_partial_only_prompt(self) -> None:
        impl = _make_scheduler_impl()
        req_id = "sparse-partial"
        prompt_len = 128
        vllm_req = _make_vllm_request(req_id, prompt_len, prompt_len + 1, 999)

        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=list(range(8)),
            num_saved_tokens=prompt_len,
        )
        tracker.is_decode_phase = True
        tracker.sparse_token_ids = list(range(prompt_len))
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[999]],
                new_block_ids=[[8]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        req_meta = meta.requests[0]

        assert req_meta.is_sparse_decode
        assert req_meta.load_spec is not None
        assert req_meta.load_spec.can_load is False
        assert req_meta.load_spec.lmcache_cached_tokens == 0
        assert req_meta.token_ids == []
        assert req_meta.save_spec is not None
        assert req_meta.save_spec.can_save is False

    def test_sparse_decode_seeds_prompt_but_loads_aligned_prefix(self) -> None:
        impl = _make_scheduler_impl()
        req_id = "sparse-req"
        prompt_len = 18879
        tracker_token_len = 2495
        decode_token = 1001
        vllm_req = _make_vllm_request(
            req_id, prompt_len, prompt_len + 1, decode_token
        )

        impl._unfinished_requests[req_id] = vllm_req
        num_blocks = (prompt_len + impl._block_size - 1) // impl._block_size
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(tracker_token_len)),
            allocated_block_ids=list(range(num_blocks)),
            num_saved_tokens=tracker_token_len,
        )
        tracker.is_decode_phase = True
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_token]],
                new_block_ids=[[num_blocks]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        req_meta = meta.requests[0]

        assert req_meta.is_sparse_decode
        assert req_meta.load_spec is not None
        aligned_prompt_len = prompt_len - prompt_len % impl._lmcache_chunk_size
        assert req_meta.load_spec.lmcache_cached_tokens == aligned_prompt_len
        assert len(req_meta.token_ids) == aligned_prompt_len
        assert req_meta.token_ids[0] == 0
        assert req_meta.token_ids[-1] == aligned_prompt_len - 1
        assert len(tracker.sparse_token_ids) == aligned_prompt_len
        assert req_meta.save_spec is not None
        assert req_meta.save_spec.can_save is False

    def test_cold_compact_kv_both_keeps_partial_prompt_prepared_frontier(
        self,
    ) -> None:
        impl = _make_scheduler_impl()
        impl.config.dsa_two_groups = True
        impl._decode_window_save_window_size = 256
        impl._dsa_scratch_capacity = 4096

        req_id = "cold-kv-both-partial"
        prompt_len = 4355
        release_frontier = 4352
        decode_token = 10_001
        vllm_req = _make_vllm_request(
            req_id,
            prompt_len,
            prompt_len,
            decode_token,
        )
        impl._unfinished_requests[req_id] = vllm_req

        num_blocks = (prompt_len + 1 + impl._block_size - 1) // impl._block_size
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=list(range(num_blocks)),
            allocated_block_ids_indexer=list(range(10_000, 10_000 + num_blocks)),
            num_saved_tokens=prompt_len,
            decode_window_save_committed_end=release_frontier,
        )
        tracker.is_decode_phase = True
        tracker.sparse_token_ids = list(range(prompt_len))
        tracker.sparse_meta_frontier = prompt_len
        tracker.sparse_remap_frontier = prompt_len - 1
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_token]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        req_meta = next(req for req in meta.requests if req.is_sparse_decode)

        assert req_meta.load_spec is not None
        assert req_meta.load_spec.lmcache_cached_tokens == prompt_len
        assert req_meta.load_spec.dsa_committed_end == prompt_len - 1
        assert req_meta.load_spec.dsa_remap_frontier == prompt_len - 1
        assert (
            req_meta.load_spec.dsa_release_frontier
            == release_frontier
        )
        assert req_meta.sparse_warm_ref
        assert req_meta.token_ids == []
        assert req_meta.slot_mapping == []
        assert req_meta.indexer_slot_mapping == []

    def test_multi_step_sparse_decode_reuses_tracker_sparse_token_ids(self) -> None:
        impl = _make_scheduler_impl()
        req_id = "sparse-req"
        prompt_len = 256
        sparse_tokens = list(range(prompt_len))
        prompt = list(range(prompt_len))
        vllm_req = SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=prompt_len,
            prompt_token_ids=prompt,
            num_computed_tokens=prompt_len + 2,
            all_token_ids=prompt + [1000, 1001],
        )

        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)) + [1000],
            allocated_block_ids=list(range(17)),
            num_saved_tokens=prompt_len,
        )
        tracker.is_decode_phase = True
        tracker.sparse_token_ids = sparse_tokens.copy()
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[1001]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        req_meta = meta.requests[0]
        assert req_meta.token_ids == sparse_tokens
        assert tracker.sparse_token_ids == sparse_tokens
        assert req_meta.save_spec is not None
        assert req_meta.save_spec.can_save is False

        vllm_req.num_computed_tokens += 1
        vllm_req.all_token_ids.append(1002)
        scheduler_output.scheduled_cached_reqs.new_token_ids = [[1002]]
        meta = impl.build_connector_meta(scheduler_output)
        req_meta = meta.requests[0]

        assert req_meta.sparse_warm_ref
        assert req_meta.retrieve_token_count() == prompt_len
        assert req_meta.token_ids == []
        assert req_meta.slot_mapping == []
        assert req_meta.indexer_slot_mapping == []
        assert req_meta.decode_token_mask is None
        assert req_meta.decode_ret_mask is None
        assert tracker.sparse_token_ids == sparse_tokens

    def test_decode_window_sparse_load_uses_committed_window_end(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        impl._dsa_scratch_capacity = 256

        req_id = "sparse-window"
        prompt_len = 356
        prompt = list(range(prompt_len))
        decode_tokens = list(range(10_000, 10_254))
        all_tokens = prompt + decode_tokens
        vllm_req = SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=prompt_len,
            prompt_token_ids=prompt,
            num_computed_tokens=608,
            all_token_ids=all_tokens,
        )

        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=all_tokens[:608],
            allocated_block_ids=list(range(40)),
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True
        tracker.decode_window_save_committed_end = 256
        tracker.sparse_token_ids = all_tokens[:352]
        tracker.sparse_slot_mapping = [torch.arange(352)]
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[all_tokens[608]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        req_meta = next(req for req in meta.requests if req.is_sparse_decode)

        assert req_meta.load_spec is not None
        assert req_meta.load_spec.can_load is True
        assert req_meta.load_spec.lmcache_cached_tokens == 256
        assert req_meta.load_spec.dsa_committed_end == 256
        assert req_meta.token_ids == all_tokens[:256]
        assert tracker.sparse_token_ids == all_tokens[:256]
        assert req_meta.slot_mapping[0].numel() == 256
        assert tracker.sparse_slot_mapping[0].numel() == 256

        impl.update_connector_output(
            SimpleNamespace(completed_decode_window_saves={req_id: 512})
        )
        vllm_req.num_computed_tokens = 609
        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[all_tokens[609]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        req_meta = next(req for req in meta.requests if req.is_sparse_decode)

        assert req_meta.load_spec is not None
        assert req_meta.load_spec.can_load is True
        assert req_meta.load_spec.lmcache_cached_tokens == 512
        assert req_meta.load_spec.dsa_committed_end == 512
        assert not req_meta.sparse_warm_ref
        assert req_meta.token_ids == all_tokens[:512]
        assert tracker.sparse_token_ids == all_tokens[:512]
        assert req_meta.slot_mapping[0].numel() == 512
        assert tracker.sparse_slot_mapping[0].numel() == 512

    def test_decode_window_completion_rejects_unemitted_frontier(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        req_id = "sparse-window"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=256,
            token_ids=list(range(600)),
            allocated_block_ids=list(range(40)),
            num_saved_tokens=256,
        )
        tracker.decode_window_save_next_start = 512
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        with pytest.raises(RuntimeError, match="exceeds request frontier"):
            impl.update_connector_output(
                SimpleNamespace(completed_decode_window_saves={req_id: 999})
            )

    def test_initial_cached_prefix_release_precedes_appended_prompt_window(
        self,
    ) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        req_id = "appended-prompt"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=1024,
            token_ids=list(range(1025)),
            allocated_block_ids=list(range(65)),
            num_saved_tokens=512,
            num_lmcache_cached_tokens=512,
            decode_window_save_committed_end=1024,
        )
        tracker.is_decode_phase = True
        tracker.decode_window_save_anchor = 1024
        tracker.decode_window_save_next_start = 1024
        impl._request_trackers[req_id] = tracker
        output = SimpleNamespace(completed_decode_window_saves={req_id: 512})

        impl.update_connector_output(output)

        assert output.completed_decode_window_saves == {}
        assert tracker.dsa_nonresident_frontier == 0
        assert tracker.decode_window_save_committed_end == 1024
        assert tracker.decode_window_save_next_start == 1024

        with pytest.raises(RuntimeError, match="unexpected frontier"):
            impl.update_connector_output(
                SimpleNamespace(completed_decode_window_saves={req_id: 768})
            )

    def test_decode_window_completion_ignored_before_save_frontier_exists(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        req_id = "sparse-window"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=256,
            token_ids=list(range(600)),
            allocated_block_ids=list(range(40)),
            num_saved_tokens=256,
        )
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        impl.update_connector_output(
            SimpleNamespace(completed_decode_window_saves={req_id: 512})
        )

        assert tracker.decode_window_save_committed_end == 256

    def test_decode_window_completion_rejects_unknown_token_frontier(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        req_id = "sparse-window"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=256,
            token_ids=list(range(384)),
            allocated_block_ids=list(range(40)),
            num_saved_tokens=256,
        )
        tracker.decode_window_save_next_start = 512
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        with pytest.raises(RuntimeError, match="exceeds request frontier"):
            impl.update_connector_output(
                SimpleNamespace(completed_decode_window_saves={req_id: 512})
            )

    def test_decode_window_completion_rejects_block_only_aligned_frontier(
        self,
    ) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        req_id = "sparse-window"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=256,
            token_ids=list(range(400)),
            allocated_block_ids=list(range(40)),
            num_saved_tokens=256,
        )
        # 272 is block aligned (16) but not LMCache-chunk aligned (256).
        tracker.decode_window_save_anchor = 256
        tracker.decode_window_save_next_start = 272
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        with pytest.raises(RuntimeError, match="lmcache_chunk_size=256"):
            impl.update_connector_output(
                SimpleNamespace(completed_decode_window_saves={req_id: 272})
            )

    def test_decode_window_completion_uses_partial_chunk_anchor(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 512
        req_id = "sparse-window"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=300,
            token_ids=list(range(800)),
            allocated_block_ids=list(range(60)),
            num_saved_tokens=300,
        )
        tracker.decode_window_save_anchor = 256
        tracker.decode_window_save_next_start = 768
        tracker.decode_window_save_inflight_end = 768
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        impl.update_connector_output(
            SimpleNamespace(completed_decode_window_saves={req_id: 768})
        )

        assert tracker.decode_window_save_committed_end == 768
        assert tracker.decode_window_save_inflight_end is None

    def test_decode_window_commit_delay_keeps_save_pipeline_moving(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        impl._decode_window_save_commit_delay_windows = 1
        req_id = "sparse-window"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=256,
            token_ids=list(range(1024)),
            allocated_block_ids=list(range(64)),
            num_saved_tokens=256,
        )
        tracker.decode_window_save_anchor = 256
        tracker.decode_window_save_next_start = 512
        tracker.decode_window_save_inflight_end = 512
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        first_output = SimpleNamespace(
            completed_decode_window_saves={req_id: 512}
        )
        impl.update_connector_output(first_output)

        assert tracker.decode_window_save_inflight_end is None
        assert tracker.decode_window_save_committed_end == 256
        assert list(tracker.decode_window_save_pending_commits) == [512]
        # vLLM releases from this mapping after the connector callback.
        assert first_output.completed_decode_window_saves == {}

        # The cleared in-flight marker allows the scheduler to emit this next
        # save while committed_end and split_boundary still remain at 256.
        tracker.decode_window_save_next_start = 768
        tracker.decode_window_save_inflight_end = 768
        second_output = SimpleNamespace(
            completed_decode_window_saves={req_id: 768}
        )
        impl.update_connector_output(second_output)

        assert tracker.decode_window_save_inflight_end is None
        assert tracker.decode_window_save_committed_end == 512
        assert list(tracker.decode_window_save_pending_commits) == [768]
        assert second_output.completed_decode_window_saves == {}

    def test_decode_window_commit_delay_two(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        impl._decode_window_save_commit_delay_windows = 2
        req_id = "sparse-window"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=256,
            token_ids=list(range(1280)),
            allocated_block_ids=list(range(80)),
            num_saved_tokens=256,
        )
        tracker.decode_window_save_anchor = 256
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        published = []
        for window_end in (512, 768, 1024):
            tracker.decode_window_save_next_start = window_end
            tracker.decode_window_save_inflight_end = window_end
            output = SimpleNamespace(
                completed_decode_window_saves={req_id: window_end}
            )
            impl.update_connector_output(output)
            published.append(dict(output.completed_decode_window_saves))
            assert tracker.decode_window_save_inflight_end is None

        assert published == [{}, {}, {}]
        assert tracker.decode_window_save_committed_end == 512
        assert list(tracker.decode_window_save_pending_commits) == [768, 1024]

    def test_decode_window_commit_delay_does_not_hold_initial_frontier(
        self,
    ) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        impl._decode_window_save_commit_delay_windows = 2
        req_id = "sparse-window"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=300,
            token_ids=list(range(512)),
            allocated_block_ids=list(range(32)),
            num_saved_tokens=0,
        )
        # build_connector_meta may initialize next_start before the worker
        # publishes the initial sparse/prefill frontier.
        tracker.decode_window_save_anchor = 256
        tracker.decode_window_save_next_start = 256
        tracker.decode_window_save_committed_end = 0
        impl._request_trackers[req_id] = tracker
        output = SimpleNamespace(
            completed_decode_window_saves={req_id: 256}
        )

        impl.update_connector_output(output)

        assert tracker.decode_window_save_committed_end == 256
        assert list(tracker.decode_window_save_pending_commits) == []
        assert output.completed_decode_window_saves == {}

    def test_decode_window_commit_delay_counts_catch_up_save_once(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        impl._decode_window_save_commit_delay_windows = 1
        req_id = "sparse-window"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=256,
            token_ids=list(range(1536)),
            allocated_block_ids=list(range(96)),
            num_saved_tokens=256,
        )
        tracker.decode_window_save_anchor = 256
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        tracker.decode_window_save_next_start = 1024
        tracker.decode_window_save_inflight_end = 1024
        first_output = SimpleNamespace(
            completed_decode_window_saves={req_id: 1024}
        )
        impl.update_connector_output(first_output)
        assert first_output.completed_decode_window_saves == {}

        tracker.decode_window_save_next_start = 1280
        tracker.decode_window_save_inflight_end = 1280
        second_output = SimpleNamespace(
            completed_decode_window_saves={req_id: 1280}
        )
        impl.update_connector_output(second_output)

        assert second_output.completed_decode_window_saves == {}
        assert tracker.decode_window_save_committed_end == 1024

    def test_decode_window_completion_rejects_partial_value(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 512
        req_id = "sparse-window"
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=300,
            token_ids=list(range(700)),
            allocated_block_ids=list(range(60)),
            num_saved_tokens=300,
        )
        tracker.decode_window_save_next_start = 768
        tracker.decode_window_save_anchor = 256
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        with pytest.raises(RuntimeError, match="unexpected frontier"):
            impl.update_connector_output(
                SimpleNamespace(completed_decode_window_saves={req_id: 700})
            )

    def test_decode_window_sparse_load_uses_nonzero_save_frontier(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 512
        req_id = "sparse-window"
        all_tokens = list(range(900))
        vllm_req = SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=300,
            prompt_token_ids=all_tokens[:300],
            num_computed_tokens=800,
            all_token_ids=all_tokens,
        )
        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=300,
            token_ids=all_tokens[:799],
            allocated_block_ids=list(range(80)),
            num_saved_tokens=300,
        )
        tracker.is_decode_phase = True
        tracker.decode_window_save_next_start = 768
        tracker.decode_window_save_committed_end = 768
        tracker.sparse_token_ids = all_tokens[:300]
        tracker.sparse_slot_mapping = [torch.arange(300)]
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[all_tokens[799]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)
        req_meta = next(req for req in meta.requests if req.is_sparse_decode)

        assert req_meta.load_spec is not None
        assert req_meta.load_spec.lmcache_cached_tokens == 768


class TestZombieRequestInMetadata:
    def test_finished_request_still_in_metadata_keeps_worker_state(self) -> None:
        impl = make_worker_impl()
        impl._worker_retrieve_state = {
            "zombie": WorkerRetrieveState(metadata_warm=True, cached_keys=[["k"]]),
            "active": WorkerRetrieveState(metadata_warm=True),
        }

        impl._prune_worker_retrieve_state({"zombie", "active"})
        assert "zombie" in impl._worker_retrieve_state
        assert "active" in impl._worker_retrieve_state

    def test_request_removed_from_metadata_keeps_warm_worker_state(self) -> None:
        impl = make_worker_impl()
        engine = MagicMock()
        impl._manager = SimpleNamespace(lmcache_engine=engine)
        impl._worker_retrieve_state = {
            "zombie": WorkerRetrieveState(
                metadata_warm=True, cached_keys=[["k"]]
            ),
            "active": WorkerRetrieveState(
                metadata_warm=True, cached_keys=[["k2"]]
            ),
        }

        impl._prune_worker_retrieve_state({"active"})

        assert "zombie" in impl._worker_retrieve_state
        assert "active" in impl._worker_retrieve_state
        engine.lookup_unpin.assert_not_called()

    def test_scheduler_clears_tracker_on_finished_but_worker_needs_prune(self) -> None:
        impl = _make_scheduler_impl()
        req_id = "finished-req"
        impl._request_trackers[req_id] = RequestTracker(
            req_id=req_id,
            prompt_len=32,
            token_ids=list(range(32)),
            allocated_block_ids=[0],
        )
        impl._unfinished_requests[req_id] = SimpleNamespace(request_id=req_id)
        impl._dsa_kv_policy_states[req_id] = "sparse_managed"

        scheduler_output = StubSchedulerOutput(
            finished_req_ids={req_id},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[],
                new_token_ids=[],
                new_block_ids=[],
            ),
            num_scheduled_tokens={},
        )

        impl.build_connector_meta(scheduler_output)
        assert req_id not in impl._request_trackers
        assert req_id not in impl._unfinished_requests
        assert req_id not in impl._dsa_kv_policy_states


class TestDecodeWindowSaveMetadata:
    @staticmethod
    def _build_decode_window_case(
        *,
        shared_cpu: bool,
        indexer_blocks: bool,
    ) -> tuple[LMCacheConnectorV1Impl, RequestTracker, StubSchedulerOutput]:
        impl = _make_scheduler_impl()
        impl.enable_sparse_attention = False
        impl.config.dsa_two_groups = True
        impl.config.extra_config = (
            {"enable_shared_cpu_cache": True} if shared_cpu else {}
        )
        impl._decode_window_save_window_size = 256

        req_id = "decode-window"
        prompt_len = 356
        prompt = list(range(prompt_len))
        # partial chunk [256, 356) plus 156 decode tokens reaches the
        # first complete decode-save frontier at 512 exactly.
        decode_tokens = list(range(10_000, 10_156))
        vllm_req = SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=prompt_len,
            prompt_token_ids=prompt,
            num_computed_tokens=prompt_len + 155,
            all_token_ids=prompt + decode_tokens,
        )
        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=prompt + decode_tokens[:-1],
            allocated_block_ids=list(range(40)),
            allocated_block_ids_indexer=(
                list(range(40)) if indexer_blocks else None
            ),
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_tokens[-1]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )
        return impl, tracker, scheduler_output

    def test_decode_window_save_includes_prefill_tail_at_boundary(self) -> None:
        impl = _make_scheduler_impl()
        impl.enable_sparse_attention = False
        impl._decode_window_save_window_size = 256

        req_id = "decode-window"
        prompt_len = 356
        prompt = list(range(prompt_len))
        # partial chunk [256, 356) plus 156 decode tokens reaches the
        # first complete decode-save frontier at 512 exactly.
        decode_tokens = list(range(10_000, 10_156))
        vllm_req = SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=prompt_len,
            prompt_token_ids=prompt,
            num_computed_tokens=prompt_len + 155,
            all_token_ids=prompt + decode_tokens,
        )
        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=prompt + decode_tokens[:-1],
            allocated_block_ids=list(range(40)),
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_tokens[-1]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert len(meta.requests) == 2
        req_meta = next(
            request
            for request in meta.requests
            if request.is_decode_window_save
        )
        main_meta = next(
            request
            for request in meta.requests
            if not request.is_decode_window_save
        )
        assert main_meta.dsa_current_released_frontier == 0
        assert main_meta.dsa_nonresident_frontier == 0
        assert not main_meta.is_sparse_decode
        assert req_meta.is_decode_window_save is True
        assert req_meta.decode_window_start == 256
        assert req_meta.decode_window_end == 512
        assert req_meta.decode_window_size == 256
        assert req_meta.save_spec is not None
        assert req_meta.save_spec.skip_leading_tokens == 256
        assert req_meta.token_ids == prompt + decode_tokens
        assert len(req_meta.slot_mapping[0]) == 512
        assert tracker.num_saved_tokens == 256
        assert tracker.decode_window_save_next_start == 512
        assert tracker.decode_window_save_committed_end == 256

    def test_decode_window_save_waits_until_boundary(self) -> None:
        impl = _make_scheduler_impl()
        impl.enable_sparse_attention = False
        impl._decode_window_save_window_size = 256

        req_id = "decode-window"
        prompt_len = 356
        prompt = list(range(prompt_len))
        # One token short of the first chunk-anchored frontier at 512.
        decode_tokens = list(range(10_000, 10_155))
        vllm_req = SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=prompt_len,
            prompt_token_ids=prompt,
            num_computed_tokens=prompt_len + 154,
            all_token_ids=prompt + decode_tokens,
        )
        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=prompt_len,
            token_ids=prompt + decode_tokens[:-1],
            allocated_block_ids=list(range(40)),
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True
        tracker.decode_window_save_committed_end = 256
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[decode_tokens[-1]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert len(meta.requests) == 1
        assert not meta.requests[0].is_decode_window_save
        assert meta.requests[0].dsa_current_released_frontier == 0
        assert meta.requests[0].dsa_nonresident_frontier == 0
        assert tracker.num_saved_tokens == 256
        assert tracker.decode_window_save_next_start == 256

    def test_decode_window_frontier_resets_after_preemption(self) -> None:
        tracker = RequestTracker(
            req_id="decode-window",
            prompt_len=356,
            token_ids=list(range(768)),
            allocated_block_ids=list(range(32)),
            num_saved_tokens=256,
        )
        tracker.decode_window_save_next_start = 768
        tracker.decode_window_save_committed_end = 768
        tracker.decode_window_save_pending_commits.extend([512, 768])
        tracker.sparse_meta_frontier = 768

        tracker.update(
            new_token_ids=[999],
            new_block_ids=[],
            preempted=True,
            lmcache_cached_tokens=256,
            vllm_cached_tokens=0,
            all_token_ids=list(range(400)),
        )

        assert tracker.decode_window_save_next_start is None
        assert tracker.decode_window_save_committed_end == 256
        assert tracker.num_lmcache_cached_tokens == 256
        assert list(tracker.decode_window_save_pending_commits) == []
        assert tracker.sparse_meta_frontier is None

    def test_decode_window_frontier_resets_after_token_rollback(self) -> None:
        impl = _make_scheduler_impl()
        impl.enable_sparse_attention = False
        impl._decode_window_save_window_size = 256

        req_id = "decode-window"
        all_tokens = list(range(600))
        vllm_req = SimpleNamespace(
            request_id=req_id,
            num_prompt_tokens=356,
            prompt_token_ids=all_tokens[:356],
            num_computed_tokens=400,
            all_token_ids=all_tokens,
        )
        impl._unfinished_requests[req_id] = vllm_req
        tracker = RequestTracker(
            req_id=req_id,
            prompt_len=356,
            token_ids=list(range(768)),
            allocated_block_ids=list(range(32)),
            num_saved_tokens=512,
        )
        tracker.is_decode_phase = True
        tracker.decode_window_save_next_start = 768
        tracker.decode_window_save_committed_end = 768
        tracker.decode_window_save_pending_commits.extend([512, 768])
        tracker.sparse_meta_frontier = 256
        impl._request_trackers[req_id] = tracker

        scheduler_output = StubSchedulerOutput(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData(
                req_ids=[req_id],
                new_token_ids=[[all_tokens[400]]],
                new_block_ids=[[]],
            ),
            num_scheduled_tokens={req_id: 1},
        )

        impl.build_connector_meta(scheduler_output)

        assert tracker.decode_window_save_next_start == 256
        assert tracker.decode_window_save_committed_end == 256
        assert list(tracker.decode_window_save_pending_commits) == []
        assert tracker.sparse_meta_frontier is None

    def test_released_frontier_survives_preemption(self) -> None:
        tracker = RequestTracker(
            req_id="released",
            prompt_len=356,
            token_ids=list(range(768)),
            allocated_block_ids=list(range(48)),
            num_saved_tokens=768,
            dsa_nonresident_frontier=768,
            dsa_current_released_frontier=768,
        )

        tracker.update(
            new_token_ids=[999],
            new_block_ids=[],
            preempted=True,
            lmcache_cached_tokens=256,
            vllm_cached_tokens=0,
            all_token_ids=list(range(900)),
        )

        assert tracker.dsa_nonresident_frontier == 768
        assert tracker.decode_window_save_committed_end == 768
        assert tracker.dsa_current_released_frontier == 0

    def test_cold_compact_nonresident_survives_preemption_without_release(
        self,
    ) -> None:
        tracker = RequestTracker(
            req_id="cold-preempted",
            prompt_len=10_001,
            token_ids=list(range(10_001)),
            allocated_block_ids=list(range(80)),
            num_saved_tokens=10_001,
            dsa_nonresident_frontier=10_000,
            dsa_current_released_frontier=0,
        )
        tracker.sparse_remap_frontier = 10_000

        tracker.update(
            new_token_ids=[10_001],
            new_block_ids=list(range(80, 160)),
            preempted=True,
            lmcache_cached_tokens=10_001,
            vllm_cached_tokens=0,
            all_token_ids=list(range(10_002)),
        )

        assert tracker.dsa_nonresident_frontier == 10_000
        assert tracker.dsa_current_released_frontier == 0
        assert tracker.sparse_remap_frontier == 10_000

    def test_decode_window_release_is_gated_by_kv_policy_threshold(self) -> None:
        impl = _make_scheduler_impl()
        impl._decode_window_save_window_size = 256
        impl._dsa_kv_policy_threshold = 10000
        tracker = RequestTracker(
            req_id="threshold-gate",
            prompt_len=5120,
            token_ids=list(range(5376)),
            allocated_block_ids=list(range(336)),
            num_saved_tokens=5120,
            decode_window_save_committed_end=5120,
        )
        tracker.is_decode_phase = True
        tracker.decode_window_save_anchor = 5120
        tracker.decode_window_save_next_start = 5376
        tracker.decode_window_save_inflight_end = 5376
        impl._request_trackers[tracker.req_id] = tracker
        output = SimpleNamespace(
            completed_decode_window_saves={tracker.req_id: 5376}
        )

        impl.update_connector_output(output)

        assert tracker.decode_window_save_committed_end == 5376
        assert tracker.dsa_nonresident_frontier == 0
        assert tracker.dsa_current_released_frontier == 0
        assert output.completed_decode_window_saves == {}

        tracker.token_ids.extend(range(5376, 10240))
        tracker.decode_window_save_next_start = 10240
        tracker.decode_window_save_inflight_end = 10240
        output.completed_decode_window_saves[tracker.req_id] = 10240

        impl.update_connector_output(output)

        assert tracker.dsa_nonresident_frontier == 10240
        assert tracker.dsa_current_released_frontier == 10240
        assert output.completed_decode_window_saves == {tracker.req_id: 10240}

    def test_preempted_block_table_can_republish_sticky_frontier(self) -> None:
        impl = _make_scheduler_impl()
        impl._dsa_kv_policy_threshold = 10000
        tracker = RequestTracker(
            req_id="preempted-release",
            prompt_len=12000,
            token_ids=list(range(12288)),
            allocated_block_ids=list(range(768)),
            num_saved_tokens=12288,
            decode_window_save_committed_end=12288,
            dsa_nonresident_frontier=12288,
            dsa_current_released_frontier=0,
        )
        impl._request_trackers[tracker.req_id] = tracker
        output = SimpleNamespace(
            completed_decode_window_saves={tracker.req_id: 12288}
        )

        impl.update_connector_output(output)

        assert tracker.dsa_nonresident_frontier == 12288
        assert tracker.dsa_current_released_frontier == 12288
        assert output.completed_decode_window_saves == {
            tracker.req_id: 12288
        }

    def test_two_group_decode_window_save_without_shared_cpu_requires_indexer(
        self,
    ) -> None:
        impl, tracker, scheduler_output = self._build_decode_window_case(
            shared_cpu=False,
            indexer_blocks=False,
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert meta.requests == []
        assert tracker.decode_window_save_next_start == 256

    def test_deep_window_group_plan_is_not_emitted_for_partial_groups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        impl, tracker, scheduler_output = self._build_decode_window_case(
            shared_cpu=False,
            indexer_blocks=False,
        )
        events: list[dict] = []
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "1")
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "1")
        monkeypatch.setattr(
            adapter_module,
            "_mtp_dw_event",
            lambda stage, **fields: events.append({"stage": stage, **fields}),
        )

        impl.build_connector_meta(scheduler_output)

        plans = [
            event for event in events if event.get("event") == "window_group_plan"
        ]
        assert plans == []

        finished = StubSchedulerOutput(
            finished_req_ids={tracker.req_id},
            scheduled_new_reqs=[],
            scheduled_cached_reqs=StubCachedRequestData([], [], []),
            num_scheduled_tokens={},
        )
        impl.build_connector_meta(finished)
        assert not hasattr(impl, "_mtp_dw_deep_window_group_planned_reqs")

    def test_deep_window_group_plan_requires_both_gates_and_dedupes_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        impl, tracker, scheduler_output = self._build_decode_window_case(
            shared_cpu=False,
            indexer_blocks=False,
        )
        events: list[dict] = []
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "1")
        monkeypatch.delenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", raising=False)
        monkeypatch.setattr(
            adapter_module,
            "_mtp_dw_event",
            lambda stage, **fields: events.append({"stage": stage, **fields}),
        )

        impl.build_connector_meta(scheduler_output)

        assert not any(
            event.get("event") == "window_group_plan" for event in events
        )
        assert not hasattr(impl, "_mtp_dw_deep_window_group_planned_reqs")

        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "1")
        deep_impl, _, deep_output = self._build_decode_window_case(
            shared_cpu=False,
            indexer_blocks=True,
        )
        deep_impl.build_connector_meta(deep_output)
        deep_impl.build_connector_meta(deep_output)

        plans = [
            event for event in events if event.get("event") == "window_group_plan"
        ]
        assert len(plans) == 1

    def test_shared_cpu_two_group_decode_window_save_requires_indexer_slots(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        impl, tracker, scheduler_output = self._build_decode_window_case(
            shared_cpu=True,
            indexer_blocks=False,
        )
        events: list[dict] = []
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DIAG", "1")
        monkeypatch.setenv("VLLM_ASCEND_MTP_DW_DEEP_DIAG", "1")
        monkeypatch.setattr(
            adapter_module,
            "_mtp_dw_event",
            lambda stage, **fields: events.append({"stage": stage, **fields}),
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert len(meta.requests) == 1
        assert not meta.requests[0].is_decode_window_save
        assert meta.requests[0].dsa_current_released_frontier == 0
        assert meta.requests[0].dsa_nonresident_frontier == 0
        assert tracker.decode_window_save_next_start == 256
        assert not any(
            event.get("event") == "window_group_plan" for event in events
        )
        assert not hasattr(impl, "_mtp_dw_deep_window_group_planned_reqs")

    def test_shared_cpu_two_group_decode_window_save_allows_matching_indexer(
        self,
    ) -> None:
        impl, tracker, scheduler_output = self._build_decode_window_case(
            shared_cpu=True,
            indexer_blocks=True,
        )

        meta = impl.build_connector_meta(scheduler_output)

        assert len(meta.requests) == 2
        req_meta = next(
            request
            for request in meta.requests
            if request.is_decode_window_save
        )
        assert req_meta.is_decode_window_save is True
        assert req_meta.save_spec is not None
        assert req_meta.save_spec.can_save_indexer is True
        assert req_meta.indexer_slot_mapping
        assert tracker.decode_window_save_next_start == 512
