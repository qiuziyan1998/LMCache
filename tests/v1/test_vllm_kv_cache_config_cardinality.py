# SPDX-License-Identifier: Apache-2.0
"""Tests for deriving LMCache KV group cardinality from vLLM."""

# Standard
import operator
from types import SimpleNamespace

# Third Party
import pytest
import torch
from vllm.config import ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
)

# First Party
from lmcache.integration.vllm import lmcache_connector_v1 as connector_module
from lmcache.integration.vllm import utils as vllm_utils
from lmcache.integration.vllm import vllm_v1_adapter as adapter_module
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.v1.config import LMCacheEngineConfig


def _kv_cache_config(
    latent_layers: int,
    indexer_layers: int,
    *,
    first_layer: int = 0,
    reverse_groups: bool = False,
) -> KVCacheConfig:
    producer_executions = (
        [0, 1, 2, *range(6, 75, 4), 78]
        if latent_layers == 79 and indexer_layers == 22 and first_layer == 0
        else list(range(indexer_layers))
    )
    latent_names = [
        f"model.layers.{layer}.self_attn.attn"
        for layer in range(first_layer, first_layer + latent_layers)
    ]
    indexer_names = [
        f"model.layers.{first_layer + execution}.self_attn.indexer.k_cache"
        for execution in producer_executions
    ]
    latent_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    indexer_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    groups = [
        KVCacheGroupSpec(latent_names, latent_spec),
        KVCacheGroupSpec(indexer_names, indexer_spec),
    ]
    if reverse_groups:
        groups.reverse()
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    rows_by_group = []
    for kv_group, layer_names in enumerate((latent_names, indexer_names)):
        executions = range(latent_layers) if kv_group == 0 else producer_executions
        rows_by_group.append(
            tuple(
                SimpleNamespace(
                    layer_name=layer_name,
                    execution_ordinal=execution_ordinal,
                    kv_group=kv_group,
                    row_ordinal=row_ordinal,
                    bank=row_ordinal % 2,
                )
                for row_ordinal, (layer_name, execution_ordinal) in enumerate(
                    zip(layer_names, executions, strict=True)
                )
            )
        )
    indexer_by_execution = {row.execution_ordinal: row for row in rows_by_group[1]}
    config.dsa_kv_topology = SimpleNamespace(  # type: ignore[assignment]
        rows_by_group=tuple(rows_by_group),
        executions=tuple(
            SimpleNamespace(
                execution_ordinal=execution_ordinal,
                latent=latent,
                indexer=indexer_by_execution.get(execution_ordinal),
            )
            for execution_ordinal, latent in enumerate(rows_by_group[0])
        ),
        signature=f"test-{latent_layers}-{indexer_layers}-{first_layer}",
    )
    return config


def _patch_connector_startup(
    monkeypatch,
    *,
    dsa_two_groups: bool,
    model_num_layers: int,
    pipeline_parallel_size: int = 1,
    rank: int = 0,
) -> tuple[LMCacheEngineConfig, SimpleNamespace, list[str]]:
    observed: list[str] = []
    config = LMCacheEngineConfig.from_defaults(
        dsa_two_groups=dsa_two_groups,
    )

    class MetadataOnlyManager:
        def __init__(self, _config, factory, connector):
            observed.append("manager")
            self.lmcache_engine_metadata = factory.get_or_create_metadata()
            self.lmcache_engine = None

        def start_services(self):
            observed.append("services")

    def init_connector_state(*_args):
        observed.append("layerwise")

    def setup_metrics(*_args):
        observed.append("metrics")

    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        lambda: config,
    )
    monkeypatch.setattr(adapter_module, "LMCacheManager", MetadataOnlyManager)
    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "_init_connector_state",
        init_connector_state,
    )
    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "_setup_metrics",
        setup_metrics,
    )
    monkeypatch.setattr(vllm_utils, "calculate_draft_layers", lambda _config: 0)
    monkeypatch.setattr(vllm_utils, "mla_enabled", lambda _config: True)
    monkeypatch.setattr(vllm_utils, "validate_mla_config", lambda *_args: None)

    model_config = SimpleNamespace(
        model="test-model",
        served_model_name="test-model",
        dtype=torch.float16,
        max_model_len=4096,
        get_num_layers=lambda _parallel_config: model_num_layers,
        get_num_kv_heads=lambda _parallel_config: 1,
        get_head_size=lambda: 576,
    )
    parallel_config = ParallelConfig(
        pipeline_parallel_size=pipeline_parallel_size,
        rank=rank,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        scheduler_config=SimpleNamespace(),
        device_config=SimpleNamespace(device="cpu"),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={},
        ),
        speculative_config=None,
    )
    return config, vllm_config, observed


@pytest.mark.parametrize("group_layers", [(79, 22), (79, 79)])
def test_runtime_kv_group_layer_counts_are_derived_from_vllm(group_layers) -> None:
    result = LMCacheConnectorV1Impl._derive_runtime_kv_group_layer_counts(
        True,
        _kv_cache_config(*group_layers),
    )

    assert result == group_layers


@pytest.mark.parametrize("group_layers", [(79, 79), (79, 22)])
def test_connector_carries_real_vllm_group_counts_into_runtime_metadata(
    monkeypatch,
    group_layers,
) -> None:
    _, vllm_config, observed = _patch_connector_startup(
        monkeypatch,
        dsa_two_groups=True,
        model_num_layers=79,
    )
    kv_cache_config = _kv_cache_config(*group_layers)

    connector = LMCacheConnectorV1Impl(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        object(),
        kv_cache_config=kv_cache_config,
    )

    assert isinstance(kv_cache_config, KVCacheConfig)
    assert all(
        isinstance(group, KVCacheGroupSpec) for group in kv_cache_config.kv_cache_groups
    )
    assert (
        connector.lmcache_engine_metadata.runtime_kv_group_layer_counts == group_layers
    )
    assert observed == ["manager", "services", "layerwise", "metrics"]


def test_connector_caches_authoritative_dsa_topology(monkeypatch) -> None:
    _, vllm_config, observed = _patch_connector_startup(
        monkeypatch,
        dsa_two_groups=True,
        model_num_layers=79,
    )
    kv_cache_config = _kv_cache_config(79, 22)
    topology = kv_cache_config.dsa_kv_topology
    assert topology is not None
    topology_logs = []

    def capture_info(message, *args, **_kwargs):
        if message.startswith("Received vLLM DSA KV topology"):
            topology_logs.append((message, args))

    monkeypatch.setattr(adapter_module.logger, "info", capture_info)

    connector = LMCacheConnectorV1Impl(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        object(),
        kv_cache_config=kv_cache_config,
    )

    cache = connector._dsa_kv_topology_cache
    assert cache is not None
    assert cache.descriptor is topology
    assert cache.group_cardinalities == (79, 22)
    assert tuple(map(len, cache.group_layer_names)) == (79, 22)
    assert len(cache.layer_name_to_row) == 101
    assert len(cache.execution_to_entry) == 79

    execution_6 = cache.execution_to_entry[6]
    assert execution_6.latent.row_ordinal == 6
    assert execution_6.indexer.row_ordinal == 3
    assert execution_6.indexer.bank == 1
    assert (
        cache.layer_name_to_row[execution_6.indexer.layer_name] is execution_6.indexer
    )

    execution_78 = cache.execution_to_entry[78]
    assert execution_78.latent.row_ordinal == 78
    assert execution_78.indexer.row_ordinal == 21
    assert execution_78.indexer.bank == 1
    assert (
        cache.layer_name_to_row[execution_78.indexer.layer_name] is execution_78.indexer
    )

    with pytest.raises(TypeError):
        operator.setitem(cache.layer_name_to_row, "replacement", execution_6.latent)
    with pytest.raises(TypeError):
        operator.setitem(cache.execution_to_entry, 6, execution_78)

    assert len(topology_logs) == 1
    assert topology_logs[0][1][0] == topology.signature
    assert topology_logs[0][1][1] == [79, 22]
    assert topology_logs[0][1][2][3] == 6
    assert topology_logs[0][1][2][-1] == 78
    assert observed == ["manager", "services", "layerwise", "metrics"]


@pytest.mark.parametrize(
    ("mismatch", "error"),
    [
        ("cardinality", "group cardinalities disagree"),
        ("membership", "group layer names disagree"),
    ],
)
def test_dsa_topology_rejects_config_group_mismatch_before_startup(
    monkeypatch,
    mismatch,
    error,
) -> None:
    _, vllm_config, observed = _patch_connector_startup(
        monkeypatch,
        dsa_two_groups=True,
        model_num_layers=79,
    )
    kv_cache_config = _kv_cache_config(79, 22)
    if mismatch == "cardinality":
        kv_cache_config.kv_cache_groups[1].layer_names.pop()
    else:
        kv_cache_config.kv_cache_groups[1].layer_names[-1] = "unexpected.indexer"

    with pytest.raises(ValueError, match=error):
        LMCacheConnectorV1Impl(
            vllm_config,
            KVConnectorRole.SCHEDULER,
            object(),
            kv_cache_config=kv_cache_config,
        )

    assert observed == []


@pytest.mark.parametrize(
    ("topology", "error"),
    [
        (None, "requires dsa_kv_topology"),
        (SimpleNamespace(rows_by_group=((), ())), "topology is partial"),
    ],
)
def test_dsa_topology_abi_fails_closed_before_startup(
    monkeypatch,
    topology,
    error,
) -> None:
    _, vllm_config, observed = _patch_connector_startup(
        monkeypatch,
        dsa_two_groups=True,
        model_num_layers=79,
    )
    kv_cache_config = _kv_cache_config(79, 22)
    kv_cache_config.dsa_kv_topology = topology

    with pytest.raises(ValueError, match=error):
        LMCacheConnectorV1Impl(
            vllm_config,
            KVConnectorRole.SCHEDULER,
            object(),
            kv_cache_config=kv_cache_config,
        )

    assert observed == []


def test_dsa_legacy_kv_cache_config_without_topology_remains_supported(
    monkeypatch,
) -> None:
    _, vllm_config, observed = _patch_connector_startup(
        monkeypatch,
        dsa_two_groups=True,
        model_num_layers=79,
    )
    current_config = _kv_cache_config(79, 22)
    legacy_config = SimpleNamespace(kv_cache_groups=current_config.kv_cache_groups)

    connector = LMCacheConnectorV1Impl(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        object(),
        kv_cache_config=legacy_config,
    )

    assert connector._dsa_kv_topology_cache is None
    assert connector.lmcache_engine_metadata.runtime_kv_group_layer_counts == (
        79,
        22,
    )
    assert observed == ["manager", "services", "layerwise", "metrics"]


def test_equal_cardinality_groups_require_latent_then_indexer_order() -> None:
    kv_cache_config = _kv_cache_config(79, 79)
    latent_group, indexer_group = kv_cache_config.kv_cache_groups

    assert all("indexer" not in name for name in latent_group.layer_names)
    assert all("indexer" in name for name in indexer_group.layer_names)
    assert LMCacheConnectorV1Impl._derive_runtime_kv_group_layer_counts(
        True,
        kv_cache_config,
    ) == (79, 79)


def test_reversed_equal_cardinality_groups_fail_closed() -> None:
    with pytest.raises(ValueError, match="latent then indexer order"):
        LMCacheConnectorV1Impl._derive_runtime_kv_group_layer_counts(
            True,
            _kv_cache_config(79, 79, reverse_groups=True),
        )


def test_runtime_kv_group_layer_counts_are_not_process_config_state() -> None:
    first = LMCacheConnectorV1Impl._derive_runtime_kv_group_layer_counts(
        True,
        _kv_cache_config(79, 22),
    )
    second = LMCacheConnectorV1Impl._derive_runtime_kv_group_layer_counts(
        True,
        _kv_cache_config(79, 79),
    )

    assert first == (79, 22)
    assert second == (79, 79)


@pytest.mark.parametrize("group_layers", [(79, 0), (0, 22)])
def test_runtime_kv_group_layer_counts_require_two_non_empty_groups(
    group_layers,
) -> None:
    with pytest.raises(ValueError, match="exactly two positive"):
        LMCacheConnectorV1Impl._derive_runtime_kv_group_layer_counts(
            True,
            _kv_cache_config(*group_layers),
        )


@pytest.mark.parametrize("num_groups", [1, 3])
def test_runtime_kv_group_layer_counts_require_exactly_two_groups(
    num_groups,
) -> None:
    kv_cache_config = _kv_cache_config(79, 22)
    if num_groups == 1:
        kv_cache_config.kv_cache_groups.pop()
    else:
        kv_cache_config.kv_cache_groups.append(kv_cache_config.kv_cache_groups[0])

    with pytest.raises(ValueError, match="exactly two positive"):
        LMCacheConnectorV1Impl._derive_runtime_kv_group_layer_counts(
            True,
            kv_cache_config,
        )


def test_runtime_kv_group_layer_counts_do_not_affect_non_dsa_models() -> None:
    result = LMCacheConnectorV1Impl._derive_runtime_kv_group_layer_counts(
        False,
        _kv_cache_config(79, 22),
    )

    assert result is None


def test_dsa_pipeline_parallel_nonzero_stage_fails_before_startup(
    monkeypatch,
) -> None:
    _, vllm_config, observed = _patch_connector_startup(
        monkeypatch,
        dsa_two_groups=True,
        model_num_layers=39,
        pipeline_parallel_size=2,
        rank=1,
    )
    kv_cache_config = _kv_cache_config(39, 11, first_layer=40)
    assert (
        kv_cache_config.kv_cache_groups[0].layer_names[0].startswith("model.layers.40.")
    )

    with pytest.raises(ValueError, match="pipeline_parallel_size must be 1"):
        LMCacheConnectorV1Impl(
            vllm_config,
            KVConnectorRole.WORKER,
            object(),
            kv_cache_config=kv_cache_config,
        )

    assert observed == []


def test_non_dsa_pipeline_parallel_startup_is_unchanged(monkeypatch) -> None:
    _, vllm_config, observed = _patch_connector_startup(
        monkeypatch,
        dsa_two_groups=False,
        model_num_layers=39,
        pipeline_parallel_size=2,
        rank=1,
    )

    connector = LMCacheConnectorV1Impl(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        object(),
        kv_cache_config=_kv_cache_config(39, 11, first_layer=40),
    )

    assert connector.lmcache_engine_metadata.runtime_kv_group_layer_counts is None
    assert observed == ["manager", "services", "layerwise", "metrics"]


@pytest.mark.parametrize(
    "initial_extra_config",
    [{}, {"unrelated": "preserved"}],
)
def test_runtime_kv_group_layer_counts_do_not_mutate_config(
    monkeypatch,
    initial_extra_config,
) -> None:
    observed = []

    class FakeConfig:
        dsa_two_groups = True

        def __init__(self):
            self.extra_config = dict(initial_extra_config)

    class FakeServiceFactory:
        def __init__(
            self,
            config,
            _vllm_config,
            _role,
            *,
            runtime_kv_group_layer_counts,
            dsa_kv_topology,
        ):
            self.runtime_kv_group_layer_counts = runtime_kv_group_layer_counts
            self.dsa_kv_topology = dsa_kv_topology
            observed.append(
                (
                    "factory",
                    dict(config.extra_config),
                    runtime_kv_group_layer_counts,
                    dsa_kv_topology,
                )
            )

    class FakeManager:
        def __init__(self, config, factory, connector):
            observed.append(
                (
                    "manager",
                    dict(config.extra_config),
                    factory.runtime_kv_group_layer_counts,
                    factory.dsa_kv_topology,
                )
            )
            self.config = config
            self.factory = factory
            self.lmcache_engine = None

        def start_services(self):
            observed.append(
                (
                    "start",
                    dict(self.config.extra_config),
                    self.factory.runtime_kv_group_layer_counts,
                    self.factory.dsa_kv_topology,
                )
            )

    config = FakeConfig()
    monkeypatch.setattr(adapter_module, "LMCacheEngineConfig", FakeConfig)
    monkeypatch.setattr(
        adapter_module,
        "lmcache_get_or_create_config",
        lambda: config,
    )
    monkeypatch.setattr(adapter_module, "VllmServiceFactory", FakeServiceFactory)
    monkeypatch.setattr(adapter_module, "LMCacheManager", FakeManager)
    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "_apply_extra_config",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both"),
        parallel_config=ParallelConfig(),
    )

    kv_cache_config = _kv_cache_config(79, 22)
    LMCacheConnectorV1Impl(
        vllm_config,
        SimpleNamespace(name="SCHEDULER"),
        object(),
        kv_cache_config=kv_cache_config,
    )

    assert observed == [
        (
            "factory",
            initial_extra_config,
            (79, 22),
            kv_cache_config.dsa_kv_topology,
        ),
        (
            "manager",
            initial_extra_config,
            (79, 22),
            kv_cache_config.dsa_kv_topology,
        ),
        (
            "start",
            initial_extra_config,
            (79, 22),
            kv_cache_config.dsa_kv_topology,
        ),
    ]
    assert config.extra_config == initial_extra_config


def test_service_factory_carries_runtime_counts_into_metadata(monkeypatch) -> None:
    monkeypatch.setattr(vllm_utils, "calculate_draft_layers", lambda _config: 0)
    monkeypatch.setattr(vllm_utils, "mla_enabled", lambda _config: False)
    monkeypatch.setattr(vllm_utils, "validate_mla_config", lambda *_args: None)
    model_config = SimpleNamespace(
        model="test-model",
        served_model_name="test-model",
        dtype=torch.float16,
        max_model_len=4096,
        get_num_layers=lambda _parallel_config: 79,
        get_num_kv_heads=lambda _parallel_config: 8,
        get_head_size=lambda: 128,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(rank=0, world_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    topology = object()
    factory = VllmServiceFactory(
        SimpleNamespace(chunk_size=256),
        vllm_config,
        "scheduler",
        runtime_kv_group_layer_counts=(79, 22),
        dsa_kv_topology=topology,
    )

    metadata = factory.get_or_create_metadata()

    assert metadata is not None
    assert metadata.runtime_kv_group_layer_counts == (79, 22)
    assert metadata.dsa_kv_topology is topology


def test_dynamic_connector_forwards_kv_cache_config(monkeypatch) -> None:
    captured = {}
    vllm_config = object()
    role = object()
    kv_cache_config = object()

    def fake_base_init(self, **kwargs):
        captured["base"] = kwargs

    def fake_impl(*args, **kwargs):
        captured["impl"] = (args, kwargs)
        return object()

    monkeypatch.setattr(KVConnectorBase_V1, "__init__", fake_base_init)
    monkeypatch.setattr(connector_module, "LMCacheConnectorV1Impl", fake_impl)

    connector = connector_module.LMCacheConnectorV1Dynamic(
        vllm_config,
        role,
        kv_cache_config,
    )

    assert captured["base"]["kv_cache_config"] is kv_cache_config
    assert captured["impl"][0] == (vllm_config, role, connector)
    assert captured["impl"][1]["kv_cache_config"] is kv_cache_config


def test_dynamic_worker_without_backend_keeps_only_resident_indexer_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, vllm_config, _ = _patch_connector_startup(
        monkeypatch,
        dsa_two_groups=True,
        model_num_layers=101,
    )
    monkeypatch.setattr(KVConnectorBase_V1, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(
        vllm_utils,
        "calculate_local_rank_and_world_size",
        lambda _config: (0, 1),
    )

    connector = connector_module.LMCacheConnectorV1Dynamic(
        vllm_config,
        KVConnectorRole.WORKER,
        _kv_cache_config(79, 22),
    )

    assert connector.supports_dsa_index_lmcache is True
    assert connector.supports_layerwise_prefill_eager_callbacks is False
    assert connector.supports_layerwise_prefill_transfer_window is False
    assert connector.supports_layerwise_prefill_p_node is False


def test_dynamic_connector_forwards_layerwise_prefill_contract(monkeypatch) -> None:
    calls = []
    engine = SimpleNamespace(
        supports_layerwise_prefill_eager_callbacks=True,
        supports_dsa_index_lmcache=True,
        supports_layerwise_prefill_transfer_window=True,
        wait_for_layerwise_prefill_load=lambda metadata: calls.append(
            ("wait", metadata)
        ),
        save_layerwise_prefill_kv_layer=lambda metadata, kv_layer, attn_metadata: (
            calls.append(("save", metadata, kv_layer, attn_metadata))
        ),
        submit_layerwise_prefill_save=lambda metadata, kv_layer, attn_metadata: (
            calls.append(("submit-save", metadata, kv_layer, attn_metadata))
        ),
        submit_layerwise_prefill_load=lambda metadata: calls.append(
            ("submit-load", metadata)
        ),
        finish_layerwise_prefill_save=lambda metadata: calls.append(
            ("finish", metadata)
        ),
    )
    monkeypatch.setattr(KVConnectorBase_V1, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(
        connector_module,
        "LMCacheConnectorV1Impl",
        lambda *args, **kwargs: engine,
    )

    connector = connector_module.LMCacheConnectorV1Dynamic(object(), object(), object())
    metadata = object()
    kv_layer = torch.empty(1)
    attn_metadata = object()

    assert connector.supports_layerwise_prefill_eager_callbacks is True
    assert connector.supports_dsa_index_lmcache is True
    assert connector.supports_layerwise_prefill_p_node is True
    assert connector.supports_layerwise_prefill_transfer_window is True

    connector.wait_for_layerwise_prefill_load(metadata)
    connector.save_layerwise_prefill_kv_layer(metadata, kv_layer, attn_metadata)
    connector.submit_layerwise_prefill_save(metadata, kv_layer, attn_metadata)
    connector.submit_layerwise_prefill_load(metadata)
    connector.finish_layerwise_prefill_save(metadata)

    assert [call[0] for call in calls] == [
        "wait",
        "save",
        "submit-save",
        "submit-load",
        "finish",
    ]
