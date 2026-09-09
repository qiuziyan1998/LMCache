# SPDX-License-Identifier: Apache-2.0
"""Native topology qualification must not change TP/DP identities."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm import utils
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory


@pytest.mark.parametrize("tp,dp", [(8, 2), (4, 4), (1, 8)])
@pytest.mark.parametrize("role", ["scheduler", "worker"])
@pytest.mark.parametrize(
    "override,expected",
    [
        ({}, True),
        ({"pipeline_parallel_size": 2}, False),
        ({"prefill_context_parallel_size": 2}, False),
        ({"decode_context_parallel_size": 2}, False),
        ({"enable_elastic_ep": True}, False),
        ({"world_size": 32}, False),
    ],
)
def test_metadata_qualifies_only_replicated_native_topology(
    monkeypatch: pytest.MonkeyPatch,
    tp: int,
    dp: int,
    role: str,
    override: dict,
    expected: bool,
) -> None:
    """DP is independent; PP/CP/elastic and non-MLA remain unqualified."""
    parallel = SimpleNamespace(
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        world_size=tp,
        rank=tp - 1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_elastic_ep=False,
    )
    vars(parallel).update(override)
    model = SimpleNamespace(
        model="model",
        served_model_name="model",
        dtype=torch.float16,
        get_num_layers=lambda _: 79,
        get_num_kv_heads=lambda _: 1,
        get_head_size=lambda: 576,
        max_model_len=178000,
    )
    config = SimpleNamespace(
        model_config=model,
        parallel_config=parallel,
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    monkeypatch.setattr(utils, "calculate_draft_layers", lambda _: 0)
    monkeypatch.setattr(
        utils, "calculate_local_rank_and_world_size", lambda _: (tp - 1, tp)
    )
    monkeypatch.setattr(utils, "validate_mla_config", Mock())
    for use_mla in (True, False):
        monkeypatch.setattr(utils, "mla_enabled", lambda _, value=use_mla: value)
        factory = VllmServiceFactory(SimpleNamespace(chunk_size=1024), config, role)
        metadata = factory.get_or_create_metadata()
        assert metadata.mla_cache_tp_replicated is (expected and use_mla)
        assert metadata.world_size == parallel.world_size
        assert metadata.worker_id == parallel.rank
        assert parallel.data_parallel_size == dp
        assert factory.get_or_create_metadata() is metadata
