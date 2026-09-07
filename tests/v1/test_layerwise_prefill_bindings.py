# SPDX-License-Identifier: Apache-2.0
"""CPU scheduler tests using the fork's actual request and output metadata."""

# Standard
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.sampling_params import SamplingParams
from vllm.v1.core.dsa_shared_pool import DSABlockAllocationMode
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    SaveSpec,
)

# Local
from tests.v1.test_vllm_kv_cache_config_cardinality import (
    _kv_cache_config,
    _patch_connector_startup,
)

BankIds = tuple[tuple[list[int], ...], ...]


@pytest.fixture
def connector(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> LMCacheConnectorV1Impl:
    """Run real scheduler initialization, replacing only external services."""
    initialize = LMCacheConnectorV1Impl._init_connector_state
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "true")
    config, vllm_config, _ = _patch_connector_startup(
        monkeypatch, dsa_two_groups=True, model_num_layers=79
    )
    monkeypatch.setattr(LMCacheConnectorV1Impl, "_init_connector_state", initialize)
    vllm_config.cache_config.block_size = 16
    vllm_config.kv_transfer_config.get_from_extra_config = lambda _key, default: default
    method = getattr(request, "param", None)
    vllm_config.speculative_config = (
        SimpleNamespace(method=method, num_speculative_tokens=1)
        if method is not None
        else None
    )
    config.save_unfull_chunk = True
    config.use_layerwise = True
    config.enable_sparse_attention = True
    result = LMCacheConnectorV1Impl(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        object(),
        kv_cache_config=_kv_cache_config(79, 22),
    )
    result._manager.lookup_client = Mock()
    # Shared-bank correctness must not depend on ordinary save policy.
    result.kv_role = "kv_consumer"
    result.force_skip_save = True
    config.priority_limit = 0
    return result


def _request(
    req_id: str = "req",
    prompt_len: int = 530,
    *,
    prompt_token_ids: list[int] | None = None,
) -> Request:
    return Request(
        request_id=req_id,
        prompt_token_ids=(
            list(range(prompt_len)) if prompt_token_ids is None else prompt_token_ids
        ),
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_transfer_params": {
                    "lmcache.tag": req_id,
                    "lmcache.skip_save": True,
                    "unrelated": "ignored",
                }
            },
        ),
        pooling_params=None,
    )


def _banks(count: int, offset: int = 0) -> BankIds:
    return tuple(
        tuple(
            list(
                range(
                    offset + 1 + bank * 200 + group * 100,
                    offset + 1 + bank * 200 + group * 100 + count,
                )
            )
            for group in range(2)
        )
        for bank in range(2)
    )


def _new(
    connector: LMCacheConnectorV1Impl,
    request: Request,
    *,
    start: int = 0,
    generation: int = 1,
    banks: BankIds | None = None,
) -> NewRequestData:
    request.num_computed_tokens = start
    load = connector.load_specs.get(request.request_id)
    external = 0
    if load is not None:
        external = max(start - load.vllm_cached_tokens, 0)
    connector.update_state_after_alloc(request, external)
    connector._requests_priority[request.request_id] = 100
    if banks is None:
        banks = _banks(34)
    return NewRequestData.from_request(
        request,
        banks[0],
        block_ids_by_bank=banks,
        block_allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
        allocation_generation=generation,
    )


def _cached(
    req_ids: list[str],
    starts: list[int],
    *,
    banks: list[BankIds | None] | None = None,
    generations: list[int] | None = None,
    resumed: set[str] | None = None,
) -> CachedRequestData:
    banks = banks if banks is not None else [None] * len(req_ids)
    return CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=resumed or set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[bank[0] if bank is not None else None for bank in banks],
        num_computed_tokens=starts,
        num_output_tokens=[0] * len(req_ids),
        new_block_ids_by_bank=banks,
        new_block_allocation_modes=[
            DSABlockAllocationMode.PREFILL_CHILD if bank is not None else None
            for bank in banks
        ],
        allocation_generations=generations or [1] * len(req_ids),
    )


def _output(
    scheduled: dict[str, int],
    *,
    new: list[NewRequestData] | None = None,
    cached: CachedRequestData | None = None,
    finished: set[str] | None = None,
) -> SchedulerOutput:
    return replace(
        SchedulerOutput.make_empty(),
        scheduled_new_reqs=new or [],
        scheduled_cached_reqs=cached or CachedRequestData.make_empty(),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        finished_req_ids=finished or set(),
    )


def test_p_flag_is_captured_and_empty_metadata_is_explicit(
    connector: LMCacheConnectorV1Impl, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", "false")
    meta = connector.build_connector_meta(SchedulerOutput.make_empty())
    assert meta.layerwise_prefill_requests == []
    assert meta.requests == []
    assert LMCacheConnectorMetadata().layerwise_prefill_requests is None


@pytest.mark.parametrize("value", ["1", "yes", "", "invalid"])
def test_constructor_rejects_non_strict_p_flag(
    connector: LMCacheConnectorV1Impl, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE", value)
    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        LMCacheConnectorV1Impl(
            connector._vllm_config, KVConnectorRole.SCHEDULER, object()
        )


@pytest.mark.parametrize("prompt_len", [16, 300, 530])
@pytest.mark.parametrize("connector", [None, "mtp"], indirect=True)
def test_full_hit_retains_original_partial_hash_extent(
    connector: LMCacheConnectorV1Impl, prompt_len: int
) -> None:
    connector.config.enable_dsa_cold_compact_load = True
    connector.config.enable_shared_cpu_cache = True
    connector._dsa_scratch_capacity = 16
    assert not connector.supports_dsa_cold_compact_load()
    request = _request(prompt_len=prompt_len)
    connector.lookup_client.lookup_cache.return_value = -1
    connector.lookup_client.lookup.return_value = prompt_len
    assert connector.get_num_new_matched_tokens(request, 0) == prompt_len - 1
    connector.update_state_after_alloc(request, prompt_len - 1)
    request.num_computed_tokens = prompt_len - 1
    banks = _banks((prompt_len + 15) // 16)
    new = NewRequestData.from_request(
        request,
        banks[0],
        block_ids_by_bank=banks,
        block_allocation_mode=DSABlockAllocationMode.PREFILL_CHILD,
        allocation_generation=7,
    )
    # schedule() advances the live Request after constructing this snapshot.
    request.num_computed_tokens = prompt_len
    output = _output({"req": 1}, new=[new])
    before = deepcopy(output)
    meta = connector.build_connector_meta(output)
    assert output == before
    (binding,) = meta.layerwise_prefill_requests
    assert (binding.compute_start, binding.compute_end, binding.restore_end) == (
        prompt_len - 1,
        prompt_len,
        prompt_len,
    )
    assert binding.token_ids == tuple(range(prompt_len))
    assert binding.allocation_generation == 7
    (req_meta,) = meta.requests
    assert req_meta.save_spec == SaveSpec(prompt_len - 1, True, True, True)
    assert req_meta.is_last_prefill
    assert req_meta.load_spec.lmcache_cached_tokens == prompt_len
    assert not hasattr(req_meta.load_spec, "dsa_cold_compact_load")
    assert req_meta.slot_mapping == req_meta.indexer_slot_mapping == []
    assert req_meta.token_ids == list(binding.token_ids)
    assert binding.request_configs == {"lmcache.tag": "req", "lmcache.skip_save": True}
    assert connector.load_specs == {}
    assert connector._request_trackers == {}
    with pytest.raises(FrozenInstanceError):
        binding.compute_end = 0


def test_two_partial_chunks_300_to_530_use_snapshot_and_bank_deltas(
    connector: LMCacheConnectorV1Impl,
) -> None:
    request = _request()
    new = _new(connector, request, banks=_banks(19))
    request.num_computed_tokens = 300
    first = connector.build_connector_meta(_output({"req": 300}, new=[new]))
    (initial,) = first.layerwise_prefill_requests
    assert (initial.compute_start, initial.compute_end, initial.restore_end) == (
        0,
        300,
        0,
    )
    assert initial.token_ids == tuple(range(300))
    assert not first.requests[0].is_last_prefill

    delta = _banks(15, 19)
    cached = _cached(["req"], [300], banks=[delta])
    request.num_computed_tokens = 530
    output = _output({"req": 230}, cached=cached)
    before = deepcopy(output)
    second = connector.build_connector_meta(output)
    (binding,) = second.layerwise_prefill_requests
    assert (binding.compute_start, binding.compute_end, binding.restore_end) == (
        300,
        530,
        300,
    )
    assert binding.token_ids == tuple(range(530))
    assert binding.block_ids_by_bank == tuple(
        tuple(tuple(group) for group in bank) for bank in _banks(34)
    )
    assert initial.block_ids_by_bank[0][0] == tuple(range(1, 20))
    assert output == before
    assert second.requests[0].save_spec == SaveSpec(300, True, True, True)
    assert second.requests[0].is_last_prefill
    assert second.requests[0].load_spec is None
    assert connector._request_trackers == {}
    assert connector._requests_priority == {}


@pytest.mark.parametrize(
    "connector,can_load,start,restore_end",
    [
        (None, False, 0, 0),
        (None, True, 300, 300),
        ("mtp", False, 0, 0),
        ("mtp", True, 299, 300),
    ],
    indirect=["connector"],
)
def test_partial_external_hit_is_used_only_on_initial_chunk(
    connector: LMCacheConnectorV1Impl, can_load: bool, start: int, restore_end: int
) -> None:
    connector.load_specs["req"] = LoadSpec(0, 300, can_load)
    request = _request()
    first = connector.build_connector_meta(
        _output({"req": 400 - start}, new=[_new(connector, request, start=start)])
    )
    (binding,) = first.layerwise_prefill_requests
    assert (binding.compute_start, binding.compute_end, binding.restore_end) == (
        start,
        400,
        restore_end,
    )
    second = connector.build_connector_meta(
        _output({"req": 130}, cached=_cached(["req"], [400]))
    )
    (binding,) = second.layerwise_prefill_requests
    assert (binding.compute_start, binding.compute_end, binding.restore_end) == (
        400,
        530,
        400,
    )
    assert second.requests[0].load_spec is None


@pytest.mark.parametrize("connector", ["mtp", "deepseek_mtp"], indirect=True)
@pytest.mark.parametrize("local_cached", [0, 128])
def test_divergent_prompts_recompute_mtp_partial_prefix_boundary(
    connector: LMCacheConnectorV1Impl, local_cached: int
) -> None:
    first_request = _request("first")
    first = connector.build_connector_meta(
        _output({"first": 530}, new=[_new(connector, first_request)])
    )
    first_binding = first.layerwise_prefill_requests[0]
    second_request = _request(
        "second", prompt_token_ids=list(range(256)) + list(range(1000, 1274))
    )
    second_request.sampling_params.extra_args = deepcopy(
        first_request.sampling_params.extra_args
    )
    assert first_request.prompt_token_ids[:256] == second_request.prompt_token_ids[:256]
    assert first_request.prompt_token_ids[256] != second_request.prompt_token_ids[256]
    connector.lookup_client.lookup_cache.return_value = -1
    connector.lookup_client.lookup.return_value = 256

    matched = connector.get_num_new_matched_tokens(second_request, local_cached)
    assert matched == 255 - local_cached
    connector.lookup_client.lookup.assert_called_once_with(
        second_request.all_token_ids,
        lookup_id="second",
        request_configs=first_binding.request_configs,
    )
    load = connector.load_specs["second"]
    assert load.lmcache_cached_tokens == 256
    assert load.vllm_cached_tokens == local_cached
    assert not load.can_load
    new = _new(connector, second_request, start=local_cached + matched)
    assert load.can_load
    second_request.num_computed_tokens = 400
    meta = connector.build_connector_meta(
        _output({"second": 145}, new=[new], finished={"first"})
    )
    (binding,) = meta.layerwise_prefill_requests
    assert (binding.compute_start, binding.compute_end, binding.restore_end) == (
        255,
        400,
        256,
    )
    assert binding.token_ids == tuple(second_request.prompt_token_ids[:400])
    assert binding.token_ids[:256] == first_binding.token_ids[:256]
    assert binding.token_ids[256] != first_binding.token_ids[256]
    assert binding.request_configs == first_binding.request_configs
    assert meta.requests[0].load_spec.lmcache_cached_tokens == 256
    assert meta.requests[0].save_spec.skip_leading_tokens == 255


@pytest.mark.parametrize(
    "connector,p_node,hit,local_cached,expected",
    [
        ("mtp", True, 0, 0, 0),
        ("mtp", True, 1, 0, 0),
        ("mtp", True, 256, 255, 0),
        ("mtp", True, 256, 256, 0),
        ("mtp", False, 256, 128, 128),
        (None, True, 256, 128, 128),
        ("eagle", True, 256, 128, 128),
        ("mtp", False, 530, 128, 401),
    ],
    indirect=["connector"],
)
def test_external_boundary_is_limited_to_p_mtp_and_positive_hits(
    connector: LMCacheConnectorV1Impl,
    p_node: bool,
    hit: int,
    local_cached: int,
    expected: int,
) -> None:
    connector._layerwise_prefill_p_node = p_node
    request = _request()
    connector.lookup_client.lookup_cache.return_value = hit
    assert connector.get_num_new_matched_tokens(request, local_cached) == expected
    connector.update_state_after_alloc(request, expected)
    load = connector.load_specs[request.request_id]
    assert load.lmcache_cached_tokens == hit
    assert load.can_load == (expected > 0)


def test_four_interleaved_requests_keep_independent_full_tables(
    connector: LMCacheConnectorV1Impl,
) -> None:
    requests = [_request(f"req-{i}") for i in range(4)]
    news = [
        _new(connector, req, generation=i + 1, banks=_banks(19, i * 1000))
        for i, req in enumerate(requests)
    ]
    connector.build_connector_meta(
        _output({req.req_id: 300 for req in news[:2]}, new=news[:2])
    )
    mixed = connector.build_connector_meta(
        _output(
            {"req-2": 300, "req-3": 300, "req-1": 100},
            new=news[2:],
            cached=_cached(["req-1"], [300], banks=[_banks(6, 1019)], generations=[2]),
        )
    )
    assert [b.request_id for b in mixed.layerwise_prefill_requests] == [
        "req-2",
        "req-3",
        "req-1",
    ]
    output = _output(
        {"req-3": 230, "req-0": 230, "req-1": 130, "req-2": 230},
        cached=_cached(
            ["req-3", "req-0", "req-1", "req-2"],
            [300, 300, 400, 300],
            banks=[_banks(15, 3019), _banks(15, 19), _banks(9, 1025), _banks(15, 2019)],
            generations=[4, 1, 2, 3],
        ),
    )
    before = deepcopy(output)
    final = connector.build_connector_meta(output)
    assert output == before
    for binding in final.layerwise_prefill_requests:
        i = int(binding.request_id[-1])
        assert binding.allocation_generation == i + 1
        assert binding.compute_end == 530
        assert binding.restore_end == (400 if i == 1 else 300)
        assert binding.block_ids_by_bank == tuple(
            tuple(tuple(group) for group in bank) for bank in _banks(34, i * 1000)
        )


@pytest.mark.parametrize(
    "connector,external_hit,start",
    [
        (None, 0, 0),
        (None, 300, 300),
        (None, 530, 529),
        ("mtp", 0, 0),
        ("mtp", 300, 299),
        ("mtp", 530, 529),
    ],
    indirect=["connector"],
)
def test_resume_replaces_banks_and_resets_generation(
    connector: LMCacheConnectorV1Impl, external_hit: int, start: int
) -> None:
    request = _request()
    connector.build_connector_meta(
        _output({"req": 300}, new=[_new(connector, request)])
    )
    if external_hit:
        connector.load_specs["req"] = LoadSpec(0, external_hit, True)
    output = _output(
        {"req": 530 - start},
        cached=_cached(
            ["req"], [start], banks=[_banks(34, 1000)], generations=[2], resumed={"req"}
        ),
    )
    request.num_computed_tokens = 530
    meta = connector.build_connector_meta(output)
    (binding,) = meta.layerwise_prefill_requests
    assert binding.allocation_generation == 2
    assert binding.compute_start == start
    assert binding.restore_end == external_hit
    assert binding.block_ids_by_bank[0][0] == tuple(range(1001, 1035))
    assert meta.requests[0].resumed_from_preemption
    assert meta.requests[0].save_spec.skip_leading_tokens == start


@pytest.mark.parametrize(
    "resumed,generation", [(False, 0), (False, 2), (True, 1), (True, 0)]
)
def test_generation_mismatch_fails_without_changing_state(
    connector: LMCacheConnectorV1Impl, resumed: bool, generation: int
) -> None:
    connector.build_connector_meta(
        _output({"req": 300}, new=[_new(connector, _request())])
    )
    before = connector._layerwise_prefill_scheduler_requests.copy()
    with pytest.raises(ValueError, match="generation"):
        connector.build_connector_meta(
            _output(
                {"req": 230},
                cached=_cached(
                    ["req"],
                    [300],
                    banks=[_banks(34, 1000)] if resumed else None,
                    generations=[generation],
                    resumed={"req"} if resumed else None,
                ),
            )
        )
    assert connector._layerwise_prefill_scheduler_requests == before


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "mode",
        "primary",
        "empty",
        "null",
        "negative",
        "short",
        "alias",
        "length",
        "groups",
        "generation",
    ],
)
def test_invalid_new_banks_fail_closed(
    connector: LMCacheConnectorV1Impl, bad: str
) -> None:
    new = _new(connector, _request())
    if bad == "missing":
        new.block_ids_by_bank = None
    elif bad == "mode":
        new.block_allocation_mode = DSABlockAllocationMode.FULL_PARENT
    elif bad == "primary":
        new.block_ids = ([999], [999])
    elif bad == "generation":
        new.allocation_generation = None
    elif bad == "empty":
        new.block_ids_by_bank = (([], []), ([], []))
        new.block_ids = ([], [])
    elif bad == "groups":
        new.block_ids_by_bank = ((_banks(34)[0][0],), (_banks(34)[1][0],))
    else:
        banks = new.block_ids_by_bank
        if bad == "null":
            banks[1][1][0] = 0
        elif bad == "negative":
            banks[1][1][0] = -1
        elif bad == "short":
            new.block_ids_by_bank = _banks(18)
            new.block_ids = new.block_ids_by_bank[0]
        elif bad == "alias":
            banks[1][0][0] = banks[0][0][0]
        elif bad == "length":
            banks[1][0].pop()
    with pytest.raises(ValueError, match="[Ll]ayerwise"):
        connector.build_connector_meta(_output({"req": 300}, new=[new]))
    assert connector._layerwise_prefill_scheduler_requests == {}


@pytest.mark.parametrize(
    "field",
    [
        "new_block_ids",
        "num_computed_tokens",
        "num_output_tokens",
        "new_block_ids_by_bank",
        "new_block_allocation_modes",
        "allocation_generations",
        "new_token_ids",
    ],
)
def test_cached_parallel_arrays_must_have_equal_lengths(
    connector: LMCacheConnectorV1Impl, field: str
) -> None:
    connector.build_connector_meta(
        _output({"req": 300}, new=[_new(connector, _request())])
    )
    cached = _cached(["req"], [300])
    setattr(cached, field, [None, None])
    with pytest.raises(ValueError, match="arrays differ in length"):
        connector.build_connector_meta(_output({"req": 230}, cached=cached))


def test_missing_delta_bank_does_not_publish_any_request_in_batch(
    connector: LMCacheConnectorV1Impl,
) -> None:
    connector.build_connector_meta(
        _output({"req": 300}, new=[_new(connector, _request())])
    )
    cached = _cached(["req"], [300], banks=[_banks(15, 34)])
    cached.new_block_ids_by_bank = [None]
    new = _new(connector, _request("new"), generation=2, banks=_banks(34, 1000))
    connector.load_specs["new"] = LoadSpec(0, 0, False)
    before = connector._layerwise_prefill_scheduler_requests.copy()
    with pytest.raises(ValueError, match="missing PREFILL_CHILD banks"):
        connector.build_connector_meta(
            _output({"new": 300, "req": 230}, new=[new], cached=cached)
        )
    assert connector._layerwise_prefill_scheduler_requests == before
    assert "new" in connector.load_specs


@pytest.mark.parametrize(
    "start,scheduled", [(530, 1), (531, 1), (0, 0), (0, -1), (-1, 10)]
)
def test_non_prefill_ranges_fail_closed(
    connector: LMCacheConnectorV1Impl, start: int, scheduled: int
) -> None:
    new = _new(connector, _request(), start=start)
    with pytest.raises(ValueError, match="positive prompt range"):
        connector.build_connector_meta(_output({"req": scheduled}, new=[new]))


@pytest.mark.parametrize(
    "connector,start,hit",
    [
        (None, 0, 300),
        (None, 300, 530),
        (None, 299, 300),
        (None, 529, 531),
        ("mtp", 256, 256),
        ("mtp", 254, 256),
    ],
    indirect=["connector"],
)
def test_external_hit_must_match_snapshot(
    connector: LMCacheConnectorV1Impl, start: int, hit: int
) -> None:
    new = _new(connector, _request(), start=start)
    connector.load_specs["req"] = LoadSpec(0, hit, True)
    with pytest.raises(ValueError, match="external hit"):
        connector.build_connector_meta(_output({"req": 1}, new=[new]))


def test_finished_requests_clear_scheduler_state_and_load_specs(
    connector: LMCacheConnectorV1Impl,
) -> None:
    connector.build_connector_meta(
        _output({"req": 300}, new=[_new(connector, _request())])
    )
    connector.load_specs["req"] = LoadSpec(0, 300, True)
    connector._requests_priority["req"] = 10
    meta = connector.build_connector_meta(_output({}, finished={"req"}))
    assert meta.layerwise_prefill_requests == []
    assert connector._layerwise_prefill_scheduler_requests == {}
    assert connector._unfinished_requests == {}
    assert connector.load_specs == {}
    assert connector._requests_priority == {}
    with pytest.raises(ValueError, match="has no state"):
        connector.build_connector_meta(
            _output({"req": 230}, cached=_cached(["req"], [300]))
        )


@pytest.mark.parametrize("connector", ["mtp"], indirect=True)
def test_finished_id_reuse_preserves_fresh_request_and_external_hit(
    connector: LMCacheConnectorV1Impl,
) -> None:
    connector.build_connector_meta(
        _output({"req": 300}, new=[_new(connector, _request())])
    )
    fresh = _request(prompt_token_ids=list(range(256)) + list(range(1000, 1274)))
    connector.lookup_client.lookup_cache.return_value = 256
    assert connector.get_num_new_matched_tokens(fresh, 0) == 255
    new = _new(connector, fresh, start=255, generation=2, banks=_banks(34, 1000))
    meta = connector.build_connector_meta(
        _output({"req": 145}, new=[new], finished={"req"})
    )
    assert meta.layerwise_prefill_finished == {"req"}
    (binding,) = meta.layerwise_prefill_requests
    assert (binding.compute_start, binding.compute_end, binding.restore_end) == (
        255,
        400,
        256,
    )
    assert binding.allocation_generation == 2
    assert binding.block_ids_by_bank[0][0] == tuple(range(1001, 1035))
    assert meta.requests[0].load_spec.lmcache_cached_tokens == 256
    assert connector._unfinished_requests["req"] is fresh
    assert connector.load_specs == {}
    assert connector._requests_priority == {}
    continuation = connector.build_connector_meta(
        _output({"req": 130}, cached=_cached(["req"], [400], generations=[2]))
    )
    assert continuation.layerwise_prefill_requests[0].token_ids == tuple(
        fresh.prompt_token_ids
    )


def test_normal_path_keeps_none_sentinel(connector: LMCacheConnectorV1Impl) -> None:
    connector._layerwise_prefill_p_node = False
    meta = connector.build_connector_meta(SchedulerOutput.make_empty())
    assert meta.layerwise_prefill_requests is None


def test_new_request_uses_its_prompt_snapshot_without_legacy_tracker(
    connector: LMCacheConnectorV1Impl,
) -> None:
    new = _new(connector, _request(), start=529)
    connector._unfinished_requests.clear()
    meta = connector.build_connector_meta(_output({"req": 2}, new=[new]))
    (binding,) = meta.layerwise_prefill_requests
    assert (binding.compute_start, binding.compute_end) == (529, 530)
    assert binding.token_ids == tuple(range(530))
    assert meta.requests[0].is_last_prefill


@pytest.mark.parametrize("invalid", ["gap", "rollback", "decode", "missing_request"])
def test_cached_snapshot_must_be_a_known_prefill_continuation(
    connector: LMCacheConnectorV1Impl, invalid: str
) -> None:
    connector.build_connector_meta(
        _output({"req": 300}, new=[_new(connector, _request())])
    )
    cached = _cached(["req"], [300])
    if invalid == "gap":
        cached.num_computed_tokens = [301]
    elif invalid == "rollback":
        cached.num_computed_tokens = [299]
    elif invalid == "decode":
        cached.num_output_tokens = [1]
    else:
        connector._unfinished_requests.clear()
    with pytest.raises(ValueError, match="Layerwise-prefill"):
        connector.build_connector_meta(_output({"req": 230}, cached=cached))
