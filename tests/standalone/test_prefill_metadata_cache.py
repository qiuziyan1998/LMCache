# SPDX-License-Identifier: Apache-2.0
"""CPU equivalence and bounded-work tests for request-owned prefill plans."""

# Standard
import ast
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.prefill_metadata import PrefillMetadataCache, PrefillSequenceView
from lmcache.v1.token_database import ChunkedTokenDatabase


def database(*, save_tail=True, page=False):
    config = LMCacheEngineConfig.from_legacy(
        chunk_size=4,
        backend="cpu",
        save_unfull_chunk=save_tail,
    )
    config.dsa_two_groups = True
    metadata = LMCacheMetadata("model", 8, 8, 1, 1, torch.float16, (2, 1, 4, 1, 2))
    db = ChunkedTokenDatabase(config, metadata)
    if page:
        db.mooncake_payload_layout = "test-page-layout"
    return db


@pytest.mark.parametrize("save_tail", [False, True])
@pytest.mark.parametrize("page", [False, True])
def test_plans_match_real_database_for_growth_short_queries_and_masks(save_tail, page):
    db, cache = database(save_tail=save_tail, page=page), PrefillMetadataCache()
    tokens = list(range(100))
    config = {"lmcache.tag.adapter": "a"}
    for end in (0, 2, 3, 4, 7, 8, 33, 32, 11, 100, 7, 100):
        for group in (0, 1):
            for skip in (0, end // 4 * 4):
                plan = cache.prepare(
                    db,
                    tokens[:end],
                    request_configs=config,
                    kv_group=group,
                    num_layers=3,
                    skip_tokens=skip,
                )
                mask = torch.arange(end) >= skip
                expected = list(
                    db.process_tokens(
                        tokens=tokens[:end],
                        mask=mask,
                        kv_group=group,
                        request_configs=config,
                    )
                )
                assert list(plan.candidates) == expected
                assert list(plan.starts) == [item[0] for item in expected]
                assert list(plan.ends) == [item[1] for item in expected]
                split = [item[2].split_layers(3) for item in expected]
                assert list(map(list, plan.keys_chunk_major)) == split
                assert list(map(list, plan.keys_layer_major)) == (
                    list(map(list, zip(*split, strict=True))) if split else [[], [], []]
                )
                assert all(
                    list(row) == list(plan.base_keys)
                    for row in plan.page_keys_layer_major
                )


def test_long_prefix_only_hashes_new_chunks_and_partial_tail(monkeypatch):
    db, cache = database(), PrefillMetadataCache()
    token_counts, split_calls = [], []
    original_hash = db.hash_func
    original_split = CacheEngineKey.split_layers

    def measured_hash(value):
        token_counts.append(len(value[1]))
        return original_hash(value)

    def measured_split(key, count):
        split_calls.append(count)
        return original_split(key, count)

    db.hash_func = measured_hash
    monkeypatch.setattr(CacheEngineKey, "split_layers", measured_split)
    tokens = list(range(40009))
    for group in (0, 1):
        cache.prepare(db, tokens[:40001], kv_group=group, num_layers=2)
    token_counts.clear()
    split_calls.clear()
    group0 = cache.prepare(db, tokens[:40005], num_layers=2)
    group1 = cache.prepare(db, tokens[:40005], kv_group=1, num_layers=2)
    assert token_counts == [4, 1]
    assert split_calls == [2, 2, 2, 2]
    assert (group0.hashes_new, group1.hashes_new) == (2, 0)
    assert (group0.keys_new, group1.keys_new) == (4, 4)
    assert group0.keys_kept == 20000
    # A retrieve of the preceding store frontier is cached, not a rollback.
    token_counts.clear()
    split_calls.clear()
    short = cache.prepare(db, tokens[:40001], num_layers=2)
    assert not token_counts and not split_calls
    assert short.hashes_new == short.keys_new == 0


def test_paused_load_plan_keeps_fixed_length_and_partial_tail_after_store_growth():
    db, cache = database(), PrefillMetadataCache()
    load = cache.prepare(db, list(range(7)), num_layers=2)
    before = list(load.candidates)
    layers = list(map(list, load.keys_layer_major))
    cache.prepare(db, list(range(19)), num_layers=2)
    cache.prepare(db, list(range(19)), kv_group=1, num_layers=2)
    assert list(load.candidates) == before
    assert list(map(list, load.keys_layer_major)) == layers
    assert list(load.ends) == [4, 7]
    assert len(load.starts[:]) == 2


def test_config_mutation_invalidates_without_mutating_existing_views():
    db, cache = database(), PrefillMetadataCache()
    config = {"lmcache.tag.adapter": "old"}
    old = cache.prepare(db, [1, 2, 3, 4], num_layers=2, request_configs=config)
    old_keys = list(old.base_keys)
    config["lmcache.tag.adapter"] = "new"
    new = cache.prepare(db, [5, 6, 7, 8], num_layers=2, request_configs=config)
    assert new.hashes_new == 1 and new.keys_new == 2
    assert list(old.base_keys) == old_keys
    assert old.base_keys[0] != new.base_keys[0]


@pytest.mark.parametrize("values", [list(range(6)), "abcdef", tuple(range(6))])
def test_sequence_views_follow_sequence_index_and_slice_semantics(values):
    view = PrefillSequenceView(values, 1, 5)
    assert list(view) == list(values[1:5])
    assert view[-1] == values[4]
    for selection in (
        slice(None),
        slice(1, -1),
        slice(None, None, -1),
        slice(3, 1),
        slice(None, None, 2),
    ):
        assert list(view[selection]) == list(values[1:5][selection])
    with pytest.raises(IndexError):
        _ = view[4]
    if isinstance(values, list):
        sub = view[:]
        values.extend([6, 7])
        assert len(view) == len(sub) == 4


SOURCE = (
    Path(__file__).resolve().parents[2] / "lmcache/integration/vllm/vllm_v1_adapter.py"
)


def adapter_types():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    state = next(
        node for node in tree.body if getattr(node, "name", "") == "WorkerRetrieveState"
    )
    adapter = next(
        node
        for node in tree.body
        if getattr(node, "name", "") == "LMCacheConnectorV1Impl"
    )
    methods = {
        "_prefill_metadata_cache_enabled",
        "_prefill_metadata_cache_for_request",
        "_prefill_metadata_scope_changed",
        "_layerwise_store_kwargs",
        "_prepare_dense_prefix_retrieve_state",
        "_set_worker_retrieve_state",
        "_mark_worker_retrieve_registry_changed",
        "_release_shared_worker_retrieve_state",
        "_prefill_retrieve_skip_tokens",
        "_load_token_mask_for_retrieve",
        "_merge_cache_group_by_ranges",
        "_merge_store_result_into_worker_state",
        "_ensure_layer_cache_shape",
        "_retain_shared_store_seed_state",
    }
    adapter.body = [
        node for node in adapter.body if getattr(node, "name", "") in methods
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias("annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, state, adapter], type_ignores=[])
    )
    namespace = dict(
        dataclass=dataclass,
        field=field,
        torch=torch,
        deepcopy=deepcopy,
        PrefillMetadataCache=PrefillMetadataCache,
        prefill_reuse_debug_enabled=lambda rank: False,
        LayerwisePointerTable=lambda: NS(clear=lambda: None, truncate=lambda _: None),
    )
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["LMCacheConnectorV1Impl"], namespace["WorkerRetrieveState"]


Adapter, State = adapter_types()


def adapter_request():
    adapter = Adapter()
    Adapter._release_dense_load_source_owners = staticmethod(
        lambda *args, **kwargs: None
    )
    adapter._layerwise_prefill_p_node = adapter._layerwise_prefill_dma = True
    adapter._deferred_layerwise_prefill_load_active = True
    adapter.kv_role = "kv_both"
    adapter._worker_retrieve_state = {}
    adapter._is_decode_window_save_request = lambda request: False
    adapter._is_dsa_two_groups = lambda: True
    adapter.lmcache_engine = NS(
        enable_shared_cpu_cache=True,
        shared_cpu_cache_generation=7,
        supports_dense_sparse_cache_retention=lambda: True,
        token_database=database(),
    )
    request = NS(
        req_id="r",
        token_ids=list(range(7)),
        request_configs={},
        block_allocation_mode="prefill_child",
        is_sparse_decode=False,
        layerwise_prefill_bank_offset=0,
        resumed_from_preemption=False,
        prefill_metadata_mm_signature=(),
    )
    return adapter, request


def test_first_store_and_two_retrieve_groups_share_request_cache():
    adapter, request = adapter_request()
    store0 = adapter._layerwise_store_kwargs(request, 0)
    store1 = adapter._layerwise_store_kwargs(request, 1)
    cache = store0["_prefill_metadata_cache"]
    assert store1["_prefill_metadata_cache"] is cache
    state = adapter._worker_retrieve_state["r"]
    request.token_ids.extend(range(7, 12))
    state2, reuse = adapter._prepare_dense_prefix_retrieve_state(
        request, state, retain_dense_seed=True, dsa_two_groups=True, token_count=7
    )
    assert reuse and state2 is state and state2.prefill_metadata_cache is cache
    assert state2.dense_prefix_generation == 7


@pytest.mark.parametrize(
    "change", ["generation", "preempt", "rollback", "config", "mm"]
)
def test_request_scope_changes_replace_metadata_cache(change):
    adapter, request = adapter_request()
    first = adapter._layerwise_store_kwargs(request, 0)["_prefill_metadata_cache"]
    old_state = adapter._worker_retrieve_state[request.req_id]
    if change == "generation":
        adapter.lmcache_engine.shared_cpu_cache_generation += 1
    elif change == "preempt":
        request = NS(**vars(request))
        request.resumed_from_preemption = True
    elif change == "rollback":
        request.token_ids = [1, 2]
    elif change == "config":
        request.request_configs["lmcache.tag.adapter"] = "new"
    else:
        request.prefill_metadata_mm_signature = (("new-image",), ((0, 3),))
    second = adapter._layerwise_store_kwargs(request, 0)["_prefill_metadata_cache"]
    assert second is not first
    assert adapter._worker_retrieve_state[request.req_id] is not old_state
    assert old_state.prefill_metadata_cache is None
    assert (
        adapter._layerwise_store_kwargs(request, 1)["_prefill_metadata_cache"] is second
    )


@pytest.mark.parametrize(
    "path", ["d", "sparse", "consumer", "blend", "no_dma", "no_shared"]
)
def test_non_raw_p_paths_keep_existing_store_arguments(path):
    adapter, request = adapter_request()
    if path == "d":
        adapter._layerwise_prefill_p_node = False
    elif path == "sparse":
        request.is_sparse_decode = True
    elif path == "consumer":
        adapter.kv_role = "kv_consumer"
    elif path == "blend":
        adapter.enable_blending = True
    elif path == "no_dma":
        adapter._layerwise_prefill_dma = False
    else:
        adapter.lmcache_engine.enable_shared_cpu_cache = False
    assert "_prefill_metadata_cache" not in adapter._layerwise_store_kwargs(request, 0)
    assert not adapter._worker_retrieve_state


def test_release_drops_cache_and_the_last_metadata_step():
    adapter, request = adapter_request()
    adapter._layerwise_store_kwargs(request, 0)
    state = adapter._worker_retrieve_state[request.req_id]
    Adapter._release_dense_load_source_owners = staticmethod(
        lambda *args, **kwargs: None
    )
    adapter._release_shared_worker_retrieve_state(state)
    assert state.prefill_metadata_cache is None
    assert state.prefill_metadata_scope is state.prefill_metadata_step is None
    assert state.prefill_metadata_frontier == 0


@pytest.mark.parametrize("prefix", [0, 1, 4, 7, 8, 12])
def test_cached_skip_count_matches_the_real_dense_mask(prefix):
    adapter, request = adapter_request()
    request.load_spec = NS(vllm_cached_tokens=prefix)
    request.decode_token_mask = None
    mask = adapter._load_token_mask_for_retrieve(request, 7, 4)
    assert adapter._prefill_retrieve_skip_tokens(request, 7, 4) == (
        len(mask) - int(mask.sum())
    )


def test_promotion_passes_exact_partial_tail_frontier_and_only_changed_group():
    adapter, request = adapter_request()
    state = State(req_id="r", dense_prefix_generation=7)
    state.cached_starts[:] = [0, 4]
    state.cached_ends[:] = [4, 6]
    state.cached_memory_objs[:] = [[object(), object()] for _ in range(2)]
    state.cached_keys[:] = [["first", "partial"] for _ in range(2)]
    state.cached_memory_objs_indexer[:] = [[object()]]
    first_owners = [row[0] for row in state.cached_memory_objs]
    result = NS(
        kv_group=0,
        starts=[4, 8],
        ends=[8, 10],
        keys=[["full", "tail"] for _ in range(2)],
        memory_objs=[[object(), object()] for _ in range(2)],
        tensors=[],
        chunk_dev_ptrs=[],
        chunk_ptrs=[],
    )
    assert adapter._merge_store_result_into_worker_state(state, result, request) == 2
    assert state.shared_source_append_from == {0: 1}
    assert [row[0] for row in state.cached_memory_objs] == first_owners
    calls = []
    adapter.lmcache_engine.retain_shared_cpu_store_seed = lambda *args, **kwargs: (
        calls.append((args, kwargs))
    )
    adapter._retain_shared_store_seed_state(state)
    assert list(calls[0][0][1]) == [0]
    assert calls[0][1] == {"append_from": {0: 1}}
    assert not state.shared_source_append_from


@pytest.mark.parametrize("change", ["config", "mm"])
def test_dense_load_scope_change_invalidates_retained_source_prefix(change):
    adapter, request = adapter_request()
    adapter._layerwise_store_kwargs(request, 0)
    old = adapter._worker_retrieve_state[request.req_id]
    old.dense_prefix_seed = True
    old.cached_memory_objs[:] = [[object()]]
    if change == "config":
        request.request_configs["lmcache.tag.adapter"] = "new"
    else:
        request.prefill_metadata_mm_signature = (("new-image",), ((0, 3),))
    new, reuse = adapter._prepare_dense_prefix_retrieve_state(
        request,
        old,
        retain_dense_seed=True,
        dsa_two_groups=True,
        token_count=7,
    )
    assert reuse and new is not old
    assert not new.cached_memory_objs
    assert not old.cached_memory_objs
    assert old.prefill_metadata_cache is None
