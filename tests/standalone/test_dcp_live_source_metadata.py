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
from lmcache.integration.vllm.kv_policy import load_spec_requires_retrieve
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
        "_build_slot_mapping_window",
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
        load_spec_requires_retrieve=load_spec_requires_retrieve,
        logger=NS(debug=lambda *args: None),
    )
    exec(
        compile(ast.Module(body=prefix + body, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def test_resident_decode_save_uses_live_tokens_and_both_physical_groups(
    definitions: dict[str, Any],
) -> None:
    """An old sparse prefix must not truncate the next resident decode save."""
    tracker = definitions["RequestTracker"](
        req_id="r",
        prompt_len=4,
        token_ids=list(range(8)),
        allocated_block_ids=[5, 2],
        allocated_block_ids_indexer=[11, 9],
        is_decode_phase=True,
        num_saved_tokens=4,
        sparse_token_ids=[0, 1, 2, 3],
        sparse_meta_frontier=4,
    )
    spec = definitions["LoadSpec"](0, 4, True, dsa_full_resident=True)
    meta = definitions["ReqMeta"].from_request_tracker(
        tracker, block_size=4, lmcache_chunk_size=4, load_spec=spec,
        save_decode_cache=True, is_sparse_decode=True, dsa_two_groups=True,
        windowed_sparse_layerwise_save=True,
    )
    assert meta.is_sparse_decode and meta.save_spec.can_save
    assert meta.token_ids == list(range(8)) and meta.save_slot_mapping_base == 4
    assert meta.slot_mapping[0].tolist() == [8, 9, 10, 11]
    assert meta.indexer_slot_mapping[0].tolist() == [36, 37, 38, 39]
    assert not meta.sparse_warm_ref and meta.decode_ret_mask is None


@pytest.mark.parametrize("can_load", [False, True])
def test_banked_prefill_fallback_restores_full_or_save_only_slots(
    definitions: dict[str, Any], can_load: bool
) -> None:
    """Banked DMA omits both groups; fallback rebuilds only the slots needed."""
    bank = ([5, 2], [11, 9])
    tracker = definitions["RequestTracker"](
        req_id="r",
        prompt_len=8,
        token_ids=list(range(8)),
        allocated_block_ids=bank[0],
        allocated_block_ids_indexer=bank[1],
        allocated_block_ids_by_bank=(bank, ([6, 3], [12, 10])),
        block_allocation_mode="prefill_child",
        num_saved_tokens=4,
    )
    meta = definitions["ReqMeta"].from_request_tracker(
        tracker,
        block_size=4,
        lmcache_chunk_size=4,
        load_spec=definitions["LoadSpec"](0, 4, can_load),
        dsa_two_groups=True,
        windowed_sparse_layerwise_save=True,
        layerwise_prefill_dma=True,
    )
    assert meta.prefill_dma_num_slots == 8 and meta.save_slot_mapping_base == 4
    assert meta.slot_mapping == meta.indexer_slot_mapping == []
    assert meta.save_slot_mapping == meta.save_indexer_slot_mapping == []

    meta.materialize_omitted_slot_mappings(block_size=4)
    latent_window = [8, 9, 10, 11]
    indexer_window = [36, 37, 38, 39]
    assert meta.save_slot_mapping[0].tolist() == latent_window
    assert meta.save_indexer_slot_mapping[0].tolist() == indexer_window
    assert meta.slot_mapping[0].tolist() == (
        [20, 21, 22, 23] + latent_window if can_load else latent_window
    )
    assert meta.indexer_slot_mapping[0].tolist() == (
        [44, 45, 46, 47] + indexer_window if can_load else indexer_window
    )


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
@pytest.mark.parametrize("full_resident", [False, True])
def test_non_dcp_handoff_keeps_existing_dense_source_mapping(
    definitions: dict[str, Any],
    pd_role: str,
    mode: str,
    full_resident: bool,
) -> None:
    """A staged resident row exports the same partial tail as a dense row."""
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
    spec = definitions["LoadSpec"](0, 0, False, dsa_full_resident=True)
    metadata = adapter._build_request_meta(
        tracker, spec if full_resident else None, is_sparse_decode=full_resident,
    )
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
