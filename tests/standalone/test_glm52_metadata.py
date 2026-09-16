# SPDX-License-Identifier: Apache-2.0
"""Production compatibility for model-specific page identities and waits."""
import ast
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_layout import mooncake_payload_layout

ROOT = Path(__file__).resolve().parents[2]


def metadata(counts):
    return LMCacheMetadata(
        model_name="model", world_size=1, local_world_size=1, worker_id=0,
        local_worker_id=0, kv_dtype=torch.bfloat16, kv_shape=(4, 1, 256, 1, 576),
        use_mla=True, runtime_kv_group_layer_counts=counts,
        runtime_kv_group_layer_names=tuple(tuple(f"layers.{i}.g{g}" for i in range(n))
                                         for g, n in enumerate(counts)),
    )


def config():
    return LMCacheEngineConfig.from_defaults(
        use_layerwise=True,
        dsa_two_groups=True,
        remote_fill_model_artifact_id="artifact",
        remote_fill_cache_namespace="namespace",
        extra_config={"mooncake_dsa_raw_token_dims": {0: 576, 1: 128}},
    )


def test_equal_groups_keep_the_production_payload_identity():
    current = metadata((4, 4))
    legacy = replace(current, runtime_kv_group_layer_counts=None,
                     runtime_kv_group_layer_names=None)
    assert mooncake_payload_layout(config(), current) == mooncake_payload_layout(
        config(), legacy
    )


def test_unequal_topology_and_physical_row_order_have_distinct_identities():
    original = metadata((4, 2))
    names = original.runtime_kv_group_layer_names
    reordered = replace(
        original, runtime_kv_group_layer_names=(names[0], names[1][::-1])
    )
    assert (
        mooncake_payload_layout(config(), original)[0]
        != mooncake_payload_layout(config(), reordered)[0]
    )
    assert (
        mooncake_payload_layout(config(), original)[0]
        != mooncake_payload_layout(config(), metadata((4, 4)))[0]
    )
    with pytest.raises(ValueError, match="ordered runtime"):
        mooncake_payload_layout(
            config(), replace(original, runtime_kv_group_layer_names=None)
        )


def adapter():
    path = ROOT / "lmcache/integration/vllm/vllm_v1_adapter.py"
    cls = next(n for n in ast.parse(path.read_text(encoding="utf8")).body
               if isinstance(n, ast.ClassDef) and n.name == "LMCacheConnectorV1Impl")
    names = {
        "_refresh_kvcaches_list",
        "_is_dsa_two_groups",
        "_layerwise_required_wait_groups",
        "_shared_indexer_required_wait_groups",
        "_indexer_model_layers",
        "_layerwise_has_indexer_model_layer",
        "_layerwise_layer_id_from_name",
        "_layerwise_wait_should_advance",
        "_build_kv_layer_groups",
        "_normalize_dsa_kv_layer_groups",
    }
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    ns = dict(
        cached_property=cached_property, KVConnectorRole=NS(SCHEDULER="scheduler"),
        logger=NS(info=lambda *a, **kw: None),
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    obj = ns[cls.name]()
    obj.config = NS(dsa_two_groups=True)
    obj._layerwise_retriever_is_sparse = [False]
    obj._layerwise_waited_groups = set()
    return obj


def bind(obj, producers):
    obj.kv_caches = {f"model.layers.{i}.self_attn.attn": object() for i in range(4)}
    obj.kv_caches.update(
        {f"model.layers.{i}.self_attn.indexer.k_cache": object() for i in producers}
    )
    obj._refresh_kvcaches_list()


def test_shared_consumer_advances_without_masking_missing_producer_wait():
    obj = adapter()
    bind(obj, (0, 2))
    obj.current_layer = 0
    assert not obj._layerwise_wait_should_advance(0)
    assert obj._layerwise_wait_should_advance(1)
    obj.current_layer = 1
    assert obj._layerwise_wait_should_advance(0)
    obj.current_layer = 2
    assert obj._layerwise_required_wait_groups() == {0, 1}
    # No callback-time scan after the immutable producer map has been resolved.
    obj._layerwise_layer_id_from_name = lambda _: pytest.fail("per-layer name parsing")
    assert obj._layerwise_has_indexer_model_layer(2)
    assert not obj._layerwise_has_indexer_model_layer(3)


def test_full_indexer_model_keeps_the_original_cached_wait_dispatch():
    obj = adapter()
    bind(obj, range(4))
    assert "_layerwise_required_wait_groups" not in obj.__dict__
    required = obj._layerwise_required_wait_groups()
    obj._layerwise_retriever_is_sparse = None
    assert obj._layerwise_required_wait_groups() is required


def test_reregistration_cannot_advertise_old_row_order_for_new_buffers():
    obj = adapter()
    bind(obj, (0, 2))
    names = (tuple(obj._latent_layer_names), tuple(obj._indexer_layer_names))
    groups = [
        NS(layer_names=list(items), num_layers=len(items), dtype="bf16")
        for items in names
    ]
    manager = NS(kv_layer_groups=groups, build_kv_layer_groups=lambda caches: None)
    obj.num_layers = 4
    obj.lmcache_engine = NS(
        metadata=NS(
            kv_layer_groups_manager=manager,
            runtime_kv_group_layer_counts=(4, 2),
            runtime_kv_group_layer_names=names,
        ),
        num_layers_for_group=lambda group: len(names[group]),
    )
    obj._build_kv_layer_groups()
    obj.kv_caches = dict(reversed(list(obj.kv_caches.items())))
    with pytest.raises(ValueError, match="layer order"):
        obj._build_kv_layer_groups()
