# SPDX-License-Identifier: Apache-2.0
"""Startup sizing uses registered layouts, before Ascend detects device layout."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.indexer_c8 import IndexerC8Layout
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager
from lmcache.v1.metadata import LMCacheMetadata


def make_engine(tp: int = 2, direct: bool = True) -> LMCacheEngine:
    groups = KVLayerGroupsManager()
    groups.kv_layer_groups = [
        KVLayerGroupInfo(
            [f"{group}.{i}" for i in range(layers)],
            list(range(layers)),
            torch.Size([1, 128, width]),
            torch.bfloat16,
        )
        for group, layers, width in [(0, 79, 576), (1, 22, 128)]
    ]
    engine = object.__new__(LMCacheEngine)
    engine.metadata = LMCacheMetadata(
        model_name="glm",
        world_size=tp,
        local_world_size=tp,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(79, 1, 1024, 1, 576),
        use_mla=True,
        chunk_size=1024,
        max_model_len=1048576,
        runtime_kv_group_layer_counts=(79, 22),
        kv_layer_groups_manager=groups,
    )
    engine.config = SimpleNamespace(
        enable_sparse_attention=True,
        chunk_size=1024,
        max_local_cpu_size=125,
        extra_config={"vllm_max_num_seqs": 12},
        dsa_group1_load_mode="persistent_direct_hbm" if direct else "p2p_preferred",
    )
    engine.config.get_extra_config_value = engine.config.extra_config.get
    engine.num_layers = 79
    engine.dsa_two_groups = True
    engine.enable_shared_cpu_cache = engine.save_only_first_rank = True
    engine._layerwise_prefill_p_node = False
    # Actual Ascend get_shape() fallback before its lazy layout initialization.
    engine.gpu_connector = SimpleNamespace(
        get_shape=lambda tokens, kv_group=0: torch.Size([tokens, 2, 576])
    )
    return engine


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("prefiller", [False, True])
def test_registered_mla_fits_125_gib_without_double_kv(tp, prefiller):
    engine = make_engine(tp)
    engine._layerwise_prefill_p_node = prefiller
    engine._report_shared_cpu_sparse_capacity_sanity()
    estimate = engine.config.extra_config["shared_cpu_sparse_startup_capacity_estimate"]
    assert estimate["bytes_per_chunk_all_layers"] == 93192192
    assert estimate["one_max_request_bytes"] == 95428804608
    assert estimate["max_full_length_requests"] == 1
    assert estimate["kv_groups"] == [0]


@pytest.mark.parametrize(
    "policy", [None, IndexerC8Layout(), IndexerC8Layout(c8_layers=(False, True) * 11)]
)
@pytest.mark.parametrize("direct", [False, True])
def test_group1_counts_physical_owners_and_scale_bytes(policy, direct):
    engine = make_engine(direct=direct)
    engine.metadata.indexer_c8_layout = policy
    engine._report_shared_cpu_sparse_capacity_sanity()
    index_bytes = (
        22 * 128 * 2
        if policy is None
        else sum(policy.token_bytes_for(i) for i in range(22))
    )
    expected = 1048576 * (79 * 576 * 2 + (0 if direct else index_bytes))
    estimate = engine.config.extra_config["shared_cpu_sparse_startup_capacity_estimate"]
    assert estimate["one_max_request_bytes"] == expected


def test_real_capacity_limit_and_chunk_rounding_remain():
    engine = make_engine()
    engine.config.max_local_cpu_size = 88.875
    engine._report_shared_cpu_sparse_capacity_sanity()
    engine.metadata.max_model_len = 1048577
    with pytest.raises(ValueError, match="one maximum request cannot fit"):
        engine._report_shared_cpu_sparse_capacity_sanity()
    estimate = engine.config.extra_config["shared_cpu_sparse_startup_capacity_estimate"]
    assert estimate["chunks_per_seq"] == 1025
    assert estimate["one_max_request_bytes"] == 1025 * 93192192


def test_unregistered_layout_preserves_legacy_estimation():
    engine = make_engine()
    engine.metadata.kv_layer_groups_manager.kv_layer_groups.clear()
    with pytest.raises(ValueError, match="one maximum request cannot fit"):
        engine._report_shared_cpu_sparse_capacity_sanity()
    estimate = engine.config.extra_config["shared_cpu_sparse_startup_capacity_estimate"]
    assert estimate["one_max_request_bytes"] == 190857609216


def test_registered_runtime_cardinality_mismatch_still_rejected():
    engine = make_engine()
    engine.metadata.runtime_kv_group_layer_counts = (79, 23)
    with pytest.raises(ValueError, match="disagrees"):
        engine._report_shared_cpu_sparse_capacity_sanity()
