# SPDX-License-Identifier: Apache-2.0
"""Exercise real scheduler metadata with sharded blocks, without an NPU runtime."""

# Standard
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
import ast

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.dcp import DcpWriteLayout
from lmcache.v1.config import LMCacheEngineConfig


@pytest.fixture(scope="module")
def definitions() -> dict[str, Any]:
    """Load production metadata and capability methods without vLLM/NPU imports."""
    path = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/vllm_v1_adapter.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_apply_mm_hashes",
        "_build_slot_mapping",
        "_copy_block_ids_by_bank",
        "LoadSpec",
        "SaveSpec",
        "RequestTracker",
        "ReqMeta",
        "LMCacheConnectorV1Impl",
    }
    methods = {
        "_build_request_meta",
        "_dcp_write_layout",
        "_group1_p2p_preferred",
        "_supports_dsa_split_layout",
        "_supports_dsa_live_latent_common",
        "supports_dsa_live_split",
        "supports_dsa_live_latent_source",
        "supports_dsa_live_latent_destination",
        "supports_dsa_live_latent_split",
        "supports_dsa_cold_compact_load",
    }
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    adapter = next(node for node in body if node.name == "LMCacheConnectorV1Impl")
    adapter.body = [
        node for node in adapter.body if getattr(node, "name", None) in methods
    ]
    prefix = ast.parse("from __future__ import annotations").body

    def cdiv(n: int, d: int) -> int:
        return (n + d - 1) // d

    namespace = dict(
        dataclass=dataclass,
        field=field,
        deque=deque,
        torch=torch,
        cdiv=cdiv,
        utils=NS(cdiv=cdiv),
        _lmcache_nvtx_annotate=lambda fn: fn,
        logger=NS(debug=lambda *args: None),
    )
    exec(
        compile(ast.Module(body=prefix + body, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def make_adapter(
    definitions: dict[str, Any], world: int | None, mode: str = "persistent_direct_hbm"
) -> Any:
    """Create the scheduler collaborators for a P/D handoff request."""
    adapter = definitions["LMCacheConnectorV1Impl"]()
    # The production initializer sets this marker only for validated DCP receivers.
    adapter.__dict__.update(
        _dsa_dcp_size=world,
        _dsa_dcp_live_origin=32,
        _block_size=128,
        _lmcache_chunk_size=1024,
        _discard_partial_chunks=False,
        _windowed_sparse_layerwise_save_enabled=lambda: False,
        _unfinished_requests={
            "r": NS(
                dsa_compact_allocated=False,
                kv_transfer_params={"do_remote_decode": True, "remote_block_ids": [99]},
            )
        },
        config=LMCacheEngineConfig.from_defaults(
            pd_role="receiver" if world else "sender",
            decoder_storage_mode="process",
            save_decode_cache=False,
            dsa_two_groups=True,
            use_layerwise=True,
            enable_sparse_attention=True,
            enable_shared_cpu_cache=True,
            enable_dsa_cold_compact_load=True,
            dsa_group1_load_mode=mode,
            extra_config={
                "mooncake_direct_npu_prefill_store": True,
                "enable_dsa_live_latent_split": True,
                "mooncake_reuse_vllm_transfer_engine": True,
            },
        ),
        kv_role="kv_both",
    )
    return adapter


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("mode", ["p2p_preferred", "persistent_direct_hbm"])
@pytest.mark.parametrize(
    "load,save", [(False, False), (True, False), (False, True), (True, True)]
)
def test_decoder_handoff_preserves_load_save_and_graph_metadata(
    definitions: dict[str, Any],
    world: int,
    mode: str,
    load: bool,
    save: bool,
) -> None:
    """A handoff must not interpret two local latent blocks as three global blocks."""
    adapter = make_adapter(definitions, world, mode)
    tracker = definitions["RequestTracker"](
        req_id="r",
        prompt_len=257,
        token_ids=list(range(257)),
        allocated_block_ids=[2, 3],
        allocated_block_ids_indexer=[20, 21, 22],
        skip_save=not save,
    )
    spec = definitions["LoadSpec"](0, 257, True) if load else None
    metadata = adapter._build_request_meta(tracker, spec)

    assert metadata is not None  # Required for full-resident graph routing, too.
    assert not metadata.live_source_requested
    assert metadata.live_source_token_ids == []
    assert metadata.live_source_slot_mapping == []
    assert metadata.live_source_indexer_slot_mapping == []
    assert metadata.load_spec is spec
    assert metadata.slot_mapping == []
    assert metadata.save_spec.can_save == save
    assert metadata.save_spec.can_save_latent == save
    assert metadata.save_spec.can_save_indexer == save
    assert metadata.dcp_write_layout == (
        DcpWriteLayout(world, 128, (2, 3)) if save else None
    )
    assert not metadata.is_sparse_decode and metadata.is_last_prefill
    if load or save:
        assert metadata.token_ids == list(range(257))
        torch.testing.assert_close(
            metadata.indexer_slot_mapping[0], torch.arange(2560, 2817)
        )
    else:
        assert metadata.token_ids == [] and metadata.indexer_slot_mapping == []
    assert adapter._unfinished_requests["r"].kv_transfer_params == {
        "do_remote_decode": True,
        "remote_block_ids": [99],
    }


@pytest.mark.parametrize("world", [None, 4, 8])
@pytest.mark.parametrize("role", ["kv_consumer", "kv_producer", "kv_both"])
def test_capabilities_separate_dense_source_from_sharded_destination(
    definitions: dict[str, Any],
    world: int | None,
    role: str,
) -> None:
    """DCP disables source advertisement, not the decoder's existing import path."""
    adapter = make_adapter(definitions, world, "p2p_preferred")
    adapter.kv_role = role
    assert adapter.supports_dsa_live_split() == (world is None)
    assert adapter.supports_dsa_live_latent_source() == (world is None)
    assert adapter.supports_dsa_live_latent_destination()
    assert adapter.supports_dsa_live_latent_split() == (
        world is None or role != "kv_producer"
    )


@pytest.mark.parametrize("pd_role", ["sender", "receiver"])
@pytest.mark.parametrize("mode", ["p2p_preferred", "persistent_direct_hbm"])
def test_non_dcp_handoff_keeps_existing_dense_source_mapping(
    definitions: dict[str, Any],
    pd_role: str,
    mode: str,
) -> None:
    """Preserve both sender and receiver metadata behavior outside the DCP fix."""
    adapter = make_adapter(definitions, None, mode)
    adapter.config.pd_role = pd_role
    tracker = definitions["RequestTracker"](
        req_id="r",
        prompt_len=257,
        token_ids=list(range(257)),
        allocated_block_ids=[2, 3, 4],
        allocated_block_ids_indexer=[20, 21, 22],
        skip_save=True,
    )
    metadata = adapter._build_request_meta(tracker, None)
    assert metadata.live_source_requested
    assert metadata.live_source_token_ids == list(range(257))
    torch.testing.assert_close(
        metadata.live_source_slot_mapping[0], torch.arange(256, 513)
    )
    torch.testing.assert_close(
        metadata.live_source_indexer_slot_mapping[0], torch.arange(2560, 2817)
    )
    assert metadata.dcp_write_layout is None


@pytest.mark.parametrize("mode", ["persistent_direct_hbm", "p2p_preferred"])
@pytest.mark.parametrize("skip_save", [False, True])
def test_four_bank_sender_keeps_persistent_handoff_without_dense_source(
    definitions: dict[str, Any], mode: str, skip_save: bool
) -> None:
    adapter = make_adapter(definitions, None, mode)
    adapter._layerwise_prefill_p_node = adapter._layerwise_prefill_dma = True
    adapter._unfinished_requests["r"].num_computed_tokens = 0
    banks = tuple(
        ([1 + b, 5 + b, 9 + b], [21 + b, 25 + b, 29 + b]) for b in range(4)
    )
    tokens = list(range(257))
    tracker = definitions["RequestTracker"](
        req_id="r",
        prompt_len=len(tokens),
        token_ids=tokens,
        allocated_block_ids=banks[0][0],
        allocated_block_ids_indexer=banks[0][1],
        allocated_block_ids_by_bank=banks,
        block_allocation_mode="prefill_child",
        skip_save=skip_save,
    )
    metadata = adapter._build_request_meta(tracker, None)
    assert metadata.is_last_prefill and metadata.token_ids == tokens
    assert metadata.block_ids_by_bank == banks
    assert metadata.save_spec.can_save == (not skip_save)
    assert adapter._unfinished_requests["r"].kv_transfer_params["do_remote_decode"]
    assert not metadata.live_source_requested
    assert metadata.live_source_token_ids == []
    assert metadata.live_source_slot_mapping == []
    assert metadata.live_source_indexer_slot_mapping == []
    assert not adapter.supports_dsa_live_split()
    assert not adapter.supports_dsa_live_latent_source()


def test_banked_sender_keeps_canonical_handoff_and_reloads_short_local_table(definitions):
    adapter = make_adapter(definitions, None)
    adapter._layerwise_prefill_dma = True
    adapter._prefill_dcp_size = 8
    adapter._unfinished_requests["r"].num_computed_tokens = 257
    banks = tuple(([b+1], [10+b, 20+b, 30+b]) for b in range(4))
    tracker = definitions["RequestTracker"](
        req_id="r", prompt_len=1000, token_ids=list(range(260)),
        allocated_block_ids=banks[0][0], allocated_block_ids_indexer=banks[0][1],
        allocated_block_ids_by_bank=banks, block_allocation_mode="prefill_child",
    )
    meta = adapter._build_request_meta(tracker, None)
    assert meta.load_spec.can_load and meta.load_spec.lmcache_cached_tokens == 257
    assert meta.block_ids_by_bank == banks and not meta.live_source_requested
    assert meta.slot_mapping == [] and meta.dcp_write_layout is None
    assert len(meta.indexer_slot_mapping[0]) >= 257
