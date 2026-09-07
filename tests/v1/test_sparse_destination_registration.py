# SPDX-License-Identifier: Apache-2.0
"""Final destination registration and adapter forwarding contracts."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm import vllm_v1_adapter as adapter_mod
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl


def _registration_setup() -> tuple[LMCacheConnectorV1Impl, MagicMock]:
    impl = object.__new__(LMCacheConnectorV1Impl)
    impl._role = adapter_mod.KVConnectorRole.WORKER
    impl.use_layerwise = impl.enable_sparse_attention = True
    impl.config = SimpleNamespace(
        dsa_two_groups=True,
        enable_remote_lmcache_store=True,
        pd_role="receiver",
        dsa_group1_load_mode="persistent_direct_hbm",
    )
    impl._vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False)
    )
    impl.kv_caches = {"latent": torch.zeros(2), "indexer": torch.zeros(2)}
    impl._sparse_destination_binding = None
    impl._refresh_kvcaches_list()
    seal = MagicMock(return_value=object())
    impl.lmcache_engine = SimpleNamespace(
        gpu_connector=SimpleNamespace(seal_sparse_destination_layout=seal)
    )
    return impl, seal


def test_destination_registration_preserves_canonical_list_and_rejects_rebind() -> None:
    impl, seal = _registration_setup()
    caches = impl._latent_kvcaches
    impl.seal_sparse_destination_layout()
    seal.assert_called_once_with(caches)
    assert impl._sparse_destination_binding is seal.return_value
    impl._refresh_kvcaches_list()
    assert impl._latent_kvcaches is caches
    assert impl._kvcaches_list is caches
    replacement = {"latent": torch.zeros(2)}
    with pytest.raises(RuntimeError, match="Cannot replace sealed"):
        impl.register_kv_caches(replacement)
    assert impl.kv_caches is not replacement
    context = SimpleNamespace(
        no_compile_layers={"new": SimpleNamespace(kv_cache=[torch.zeros(2)])},
        virtual_engine=0,
    )
    with pytest.raises(RuntimeError, match="Cannot extend sealed"):
        impl._init_kv_caches_from_forward_context(context)
    assert "new" not in impl.kv_caches
    seal.side_effect = RuntimeError("changed storage")
    with pytest.raises(RuntimeError, match="changed storage"):
        impl._refresh_kvcaches_list()
    assert impl._latent_kvcaches is caches


@pytest.mark.parametrize(
    "mode",
    [
        "sender",
        "sleep",
        "dense",
        "nonlayerwise",
        "single_group",
        "no_remote",
        "other_group1",
        "scheduler",
        "old_connector",
    ],
)
def test_destination_registration_keeps_unsupported_paths_unsealed(mode: str) -> None:
    impl, seal = _registration_setup()
    if mode == "sender":
        impl.config.pd_role = "sender"
    elif mode == "sleep":
        impl._vllm_config.model_config.enable_sleep_mode = True
    elif mode == "dense":
        impl.enable_sparse_attention = False
    elif mode == "nonlayerwise":
        impl.use_layerwise = False
    elif mode == "single_group":
        impl.config.dsa_two_groups = False
    elif mode == "no_remote":
        impl.config.enable_remote_lmcache_store = False
    elif mode == "other_group1":
        impl.config.dsa_group1_load_mode = "p2p_preferred"
    elif mode == "scheduler":
        impl._role = adapter_mod.KVConnectorRole.SCHEDULER
    else:
        impl.lmcache_engine.gpu_connector = object()
    impl.seal_sparse_destination_layout()
    seal.assert_not_called()
    assert impl._sparse_destination_binding is None


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("main_thread", [False, True])
def test_destination_binding_is_forwarded_only_to_prepared_group0(
    group: int, main_thread: bool
) -> None:
    impl, seal = _registration_setup()
    impl.seal_sparse_destination_layout()
    source = object()
    impl._prepared_sparse_source = MagicMock(return_value=source)
    request = SimpleNamespace(
        load_spec=object(), sparse_warm_ref=True, req_id="request", decode_ret_mask=None
    )
    state = SimpleNamespace(token_count=4, decode_ret_mask=None)
    extra = {"registered_destination_layout": seal.return_value} if main_thread else {}
    kwargs, _, prepared = impl._sparse_retrieve_kwargs(
        request,
        state,
        state,
        kvcaches=impl._kvcaches_for_group(group),
        slot_mapping=torch.arange(4),
        sync=False,
        kv_group=group,
        request_ordinal=0,
        dsa_two_groups=True,
        token_count=4,
        shared_cpu_enabled=False,
        shared_cpu_preflight_state=None,
        **extra,
    )
    assert prepared is source
    assert kwargs.get("registered_destination_layout") is (
        seal.return_value if group == 0 and main_thread else None
    )
