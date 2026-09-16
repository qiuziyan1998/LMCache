# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for vLLM connector lifecycle tests."""

# Standard
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

# Third Party
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    SaveSpec,
)


class _FakeParent:
    def __init__(self, metadata: LMCacheConnectorMetadata):
        self._connector_metadata = metadata

    def _get_connector_metadata(self) -> LMCacheConnectorMetadata:
        return self._connector_metadata


class _FakeEngine:
    def __init__(self) -> None:
        self.unpinned: list[str] = []

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)


class _FakeManager:
    def __init__(self, engine: _FakeEngine):
        self.lmcache_engine = engine


def make_sparse_req_meta(
    req_id: str,
    *,
    can_load: bool = True,
    can_save: bool = False,
    token_count: int = 4,
) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_ids=list(range(token_count)),
        slot_mapping=[torch.arange(token_count, dtype=torch.long)],
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=token_count,
            can_load=can_load,
        ),
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=can_save),
        is_sparse_decode=True,
    )


def make_non_sparse_req_meta(req_id: str, *, can_save: bool = True) -> ReqMeta:
    return ReqMeta(
        req_id=req_id,
        token_ids=[1, 2, 3, 4],
        slot_mapping=[torch.arange(4, dtype=torch.long)],
        load_spec=None,
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=can_save),
        is_sparse_decode=False,
    )


def make_worker_connector(
    requests: list[ReqMeta],
    engine: Optional[_FakeEngine] = None,
    *,
    use_layerwise: bool = False,
    kv_role: str = "kv_both",
) -> tuple[LMCacheConnectorV1Impl, LMCacheConnectorMetadata, _FakeEngine]:
    metadata = LMCacheConnectorMetadata(requests=requests)
    fake_engine = engine or _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    connector._manager = _FakeManager(fake_engine)
    connector.kv_role = kv_role
    connector.use_layerwise = use_layerwise
    connector.enable_blending = False
    connector.device = "cpu"
    connector._lmcache_chunk_size = 256
    connector._layerwise_save_storers = {}
    connector._worker_retrieve_state = {}
    connector._cold_perf_load_started = {}
    connector._cold_perf_dense_load_started = {}
    connector._cold_perf_dense_load_completed = {}
    connector._layerwise_sparse_shared_ordered = []
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector.config = MagicMock()
    connector.config.get_extra_config_value = lambda key, default=False: default
    connector.async_loading = False
    return connector, metadata, fake_engine


def make_worker_impl() -> LMCacheConnectorV1Impl:
    impl = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    impl._worker_retrieve_state = {}
    impl._cold_perf_load_started = {}
    impl._cold_perf_dense_load_started = {}
    impl._cold_perf_dense_load_completed = {}
    impl._layerwise_sparse_shared_ordered = []
    impl.lmcache_engine = None
    impl.kv_role = "kv_both"
    impl._lmcache_chunk_size = 256
    return impl


def make_stub_request(request_id: str) -> Any:
    return SimpleNamespace(
        request_id=request_id,
        status=SimpleNamespace(name="FINISHED"),
    )
