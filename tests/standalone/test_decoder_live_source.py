# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for disabling decoder source capture without losing KV work."""

# Standard
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS
from typing import Any
import ast
import logging

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.dcp import DcpWriteLayout
from lmcache.v1.config import LMCacheEngineConfig


@pytest.fixture(scope="module")
def definitions() -> NS:
    """Load real metadata code without importing the vLLM/NPU runtime."""
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    source = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_apply_mm_hashes",
        "_build_slot_mapping",
        "LoadSpec",
        "SaveSpec",
        "RequestTracker",
        "ReqMeta",
    }
    methods = {
        "_build_request_meta",
        "_dcp_write_layout",
        "_group1_p2p_preferred",
        "supports_dsa_live_split",
    }
    tree = ast.parse("from __future__ import annotations")
    for node in source.body:
        if getattr(node, "name", None) in names:
            tree.body.append(node)
        elif getattr(node, "name", None) == "LMCacheConnectorV1Impl":
            tree.body.extend(
                method
                for method in node.body
                if getattr(method, "name", None) in methods
            )

    def cdiv(n: int, d: int) -> int:
        return (n + d - 1) // d

    namespace: dict[str, Any] = dict(
        dataclass=dataclass,
        field=field,
        deque=deque,
        torch=torch,
        cdiv=cdiv,
        utils=NS(cdiv=cdiv),
        logger=logging.getLogger(__name__),
        _lmcache_nvtx_annotate=lambda fn: fn,
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return NS(**namespace)


def _adapter(
    definitions: NS,
    *,
    role: str | None = "receiver",
    disabled: bool = True,
    world: int = 4,
    tokens: int = 257,
) -> tuple[NS, Any, NS]:
    config = LMCacheEngineConfig.from_defaults(
        pd_role=role,
        disable_decoder_live_source=disabled,
        dsa_two_groups=True,
        decoder_storage_mode="process",
        enable_async_loading=True,
        dsa_group1_load_mode="persistent_direct_hbm",
    )
    tracker = definitions.RequestTracker(
        req_id="r",
        prompt_len=tokens,
        token_ids=list(range(tokens)),
        allocated_block_ids=[2, 3] if world > 1 else [2, 3, 4],
        allocated_block_ids_indexer=[20, 21, 22],
    )
    request = NS(
        kv_transfer_params={"do_remote_decode": True, "remote_request_id": "p-r"},
        dsa_compact_allocated=False,
    )
    adapter = NS(
        config=config,
        kv_role="kv_both",
        _block_size=128,
        _lmcache_chunk_size=1024,
        _discard_partial_chunks=False,
        _dsa_dcp_size=world if world > 1 else None,
        _unfinished_requests={"r": request},
        _windowed_sparse_layerwise_save_enabled=lambda: False,
        _supports_dsa_split_layout=lambda: True,
    )
    for name in (
        "_dcp_write_layout",
        "_group1_p2p_preferred",
        "supports_dsa_live_split",
    ):
        setattr(adapter, name, MethodType(getattr(definitions, name), adapter))
    return adapter, tracker, request


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("true", True), ("0", False), ("false", False)]
)
def test_environment_override(
    value: str, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag defaults off and overrides an existing config via the normal loader."""
    config = LMCacheEngineConfig.from_defaults(pd_role="receiver")
    assert config.disable_decoder_live_source is False
    monkeypatch.setenv("LMCACHE_DISABLE_DECODER_LIVE_SOURCE", value)
    config.update_config_from_env()
    assert config.disable_decoder_live_source is expected
    assert LMCacheEngineConfig.from_env().disable_decoder_live_source is expected


def test_disabled_switch_preserves_existing_dcp_failure(definitions: NS) -> None:
    """Reproduce the reported 3-logical-block/2-local-block reshape failure."""
    adapter, tracker, _ = _adapter(definitions, disabled=False)
    with pytest.raises(RuntimeError, match=r"shape '\[3, 1\]' is invalid"):
        definitions._build_request_meta(adapter, tracker, None)


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("tokens", [257, 280, 384])
def test_receiver_keeps_load_and_dcp_save(
    definitions: NS, world: int, tokens: int
) -> None:
    """Sharded D metadata retains both KV groups and the original handoff/load."""
    adapter, tracker, request = _adapter(definitions, world=world, tokens=tokens)
    handoff = dict(request.kv_transfer_params)
    load = definitions.LoadSpec(0, tokens, True, process_ordinary_load=True)
    meta = definitions._build_request_meta(adapter, tracker, load)
    assert meta is not None and meta.token_ids == list(range(tokens))
    assert not meta.live_source_requested
    assert meta.live_source_token_ids == []
    assert meta.live_source_slot_mapping == []
    assert meta.live_source_indexer_slot_mapping == []
    assert meta.load_spec is load and load.process_ordinary_load and load.can_load
    assert meta.slot_mapping == []
    assert meta.save_spec.can_save
    assert meta.save_spec.can_save_latent and meta.save_spec.can_save_indexer
    assert meta.dcp_write_layout == DcpWriteLayout(world, 128, (2, 3))
    assert meta.indexer_slot_mapping[0].tolist() == list(
        range(20 * 128, 20 * 128 + tokens)
    )
    assert request.kv_transfer_params == handoff
    assert adapter.config.decoder_storage_mode == "process"
    assert adapter.config.enable_async_loading
    assert adapter.config.dsa_group1_load_mode == "persistent_direct_hbm"


def test_receiver_keeps_resident_graph_metadata(definitions: NS) -> None:
    """A fully resident request remains visible after source capture is disabled."""
    adapter, tracker, _ = _adapter(definitions)
    tracker.skip_save = True
    meta = definitions._build_request_meta(adapter, tracker, None)
    assert meta is not None and meta.is_last_prefill
    assert not meta.is_sparse_decode and not meta.live_source_requested
    assert not meta.save_spec.can_save and meta.load_spec is None


@pytest.mark.parametrize("role", ["receiver", "sender", None])
@pytest.mark.parametrize("disabled", [False, True])
def test_flag_is_receiver_only(
    definitions: NS, role: str | None, disabled: bool
) -> None:
    """P keeps live-source capture and advertisement even with a shared environment."""
    adapter, tracker, _ = _adapter(definitions, role=role, disabled=disabled, world=1)
    # Enable the prerequisites so the capability assertion exercises this flag.
    adapter.config.dsa_group1_load_mode = "p2p_preferred"
    adapter.config.extra_config = {"mooncake_direct_npu_prefill_store": True}
    expected = not (disabled and role == "receiver")
    meta = definitions._build_request_meta(adapter, tracker, None)
    assert meta.live_source_requested is expected
    assert adapter.supports_dsa_live_split() is expected
    if expected:
        assert meta.live_source_token_ids == tracker.token_ids
        assert meta.live_source_slot_mapping[0].tolist() == list(range(256, 513))
        assert meta.live_source_indexer_slot_mapping[0].numel() == 257
