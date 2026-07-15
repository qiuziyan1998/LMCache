# SPDX-License-Identifier: Apache-2.0
"""Prefill full-hit recalc_last: keep tokens/slots for partial-chunk key match."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LoadSpec,
    LMCacheConnectorV1Impl,
    ReqMeta,
    RequestTracker,
)


def _block_ids(num_tokens: int, block_size: int) -> list[int]:
    return list(range((num_tokens + block_size - 1) // block_size))


def _make_prefill_req(*, prompt_len: int = 18879) -> ReqMeta:
    return ReqMeta(
        req_id="req-1",
        token_ids=list(range(prompt_len)),
        slot_mapping=[torch.arange(prompt_len, dtype=torch.long)],
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=prompt_len,
            can_load=True,
        ),
        is_sparse_decode=False,
    )


class TestFullHitRecalcLast:
    def test_detects_full_hit_prefill(self) -> None:
        spec = LoadSpec(0, 18879, True)
        assert LMCacheConnectorV1Impl._full_hit_recalc_last_token(
            spec, 18879, is_sparse_decode=False
        )
        assert not LMCacheConnectorV1Impl._full_hit_recalc_last_token(
            spec, 18879, is_sparse_decode=True
        )

    def test_bootstrap_hidden_full_hit_does_not_recalculate_last(self) -> None:
        spec = LoadSpec(0, 18879, True, bootstrap_sample=True)
        assert not LMCacheConnectorV1Impl._full_hit_recalc_last_token(
            spec, 18879, is_sparse_decode=False
        )

    def test_scheduler_keeps_all_tokens_for_bootstrap_hidden(self) -> None:
        prompt_len = 18879
        connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
        connector.kv_role = "kv_consumer"
        connector.enable_sparse_attention = True
        connector.lookup_client = MagicMock()
        connector.lookup_client.lookup_cache.return_value = prompt_len
        connector.config = SimpleNamespace(min_retrieve_tokens=0)
        connector.load_specs = {}
        connector._requests_priority = {}

        request = SimpleNamespace(
            request_id="bootstrap",
            num_tokens=prompt_len,
            bootstrap_sample_pending=True,
        )

        assert connector.get_num_new_matched_tokens(request, 0) == prompt_len
        assert connector.load_specs[request.request_id].bootstrap_sample

        connector.enable_sparse_attention = False
        assert connector.get_num_new_matched_tokens(request, 0) == prompt_len - 1
        assert not connector.load_specs[request.request_id].bootstrap_sample

    def test_preserves_tokens_and_slots_for_partial_chunk(self) -> None:
        req = _make_prefill_req()
        tokens = list(range(18879))
        slots = torch.arange(18879, dtype=torch.long)
        out_tokens, out_slots = LMCacheConnectorV1Impl._trim_prefill_for_recalc_last(
            req, tokens, slots
        )
        assert out_tokens is tokens
        assert out_slots is slots
        assert len(out_tokens) == 18879
        assert out_slots.numel() == 18879

    def test_sparse_decode_untouched(self) -> None:
        req = _make_prefill_req()
        req.is_sparse_decode = True
        tokens = list(range(18879))
        slots = torch.arange(18879, dtype=torch.long)
        out_tokens, out_slots = LMCacheConnectorV1Impl._trim_prefill_for_recalc_last(
            req, tokens, slots
        )
        assert out_tokens is tokens
        assert out_slots is slots

    def test_full_hit_new_request_keeps_prompt_tokens_for_restore(self) -> None:
        prompt_len = 18879
        block_size = 16
        prompt_tokens = list(range(prompt_len))
        new_request = SimpleNamespace(
            req_id="req-full-hit",
            prompt_token_ids=prompt_tokens,
            block_ids=(
                _block_ids(prompt_len, block_size),
                _block_ids(prompt_len, block_size),
            ),
            sampling_params=SimpleNamespace(extra_args=None),
        )

        tracker = RequestTracker.from_new_request(
            lmcache_config=None,
            new_request=new_request,
            num_tokens_to_compute=1,
            lmcache_cached_tokens=prompt_len,
            skip_save=False,
        )

        assert tracker.token_ids == prompt_tokens

    def test_dense_full_hit_req_meta_uses_lmcache_hit_length(self) -> None:
        prompt_len = 18879
        block_size = 16
        tracker = RequestTracker(
            req_id="req-full-hit",
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=_block_ids(prompt_len, block_size),
            allocated_block_ids_indexer=_block_ids(prompt_len, block_size),
            num_saved_tokens=prompt_len,
        )

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=block_size,
            lmcache_chunk_size=256,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=prompt_len,
                can_load=True,
            ),
            dsa_two_groups=True,
        )

        assert req_meta is not None
        assert len(req_meta.token_ids) == prompt_len
        assert req_meta.slot_mapping[0].numel() == prompt_len
        assert req_meta.indexer_slot_mapping[0].numel() == prompt_len
        assert req_meta.save_spec.can_save is False

    def test_sparse_decode_req_meta_builds_full_indexer_slots(self) -> None:
        prompt_len = 18879
        block_size = 16
        indexer_block_offset = 1000
        tracker = RequestTracker(
            req_id="req-sparse-hit",
            prompt_len=prompt_len,
            token_ids=list(range(prompt_len)),
            allocated_block_ids=_block_ids(prompt_len, block_size),
            allocated_block_ids_indexer=[
                block_id + indexer_block_offset
                for block_id in _block_ids(prompt_len, block_size)
            ],
            num_saved_tokens=prompt_len,
        )
        tracker.is_decode_phase = True

        req_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=block_size,
            lmcache_chunk_size=256,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=prompt_len,
                can_load=True,
            ),
            dsa_two_groups=True,
            is_sparse_decode=True,
        )

        assert req_meta is not None
        assert req_meta.slot_mapping[0].numel() == 2048
        assert req_meta.indexer_slot_mapping[0].numel() == prompt_len
        assert req_meta.indexer_slot_mapping[0][0].item() == (
            indexer_block_offset * block_size
        )
        assert req_meta.indexer_slot_mapping[0][-1].item() == (
            indexer_block_offset * block_size + prompt_len - 1
        )
        assert not torch.equal(
            req_meta.slot_mapping[0],
            req_meta.indexer_slot_mapping[0],
        )
        first_indexer_slots = req_meta.indexer_slot_mapping[0]

        req_meta_again = ReqMeta.from_request_tracker(
            tracker,
            block_size=block_size,
            lmcache_chunk_size=256,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=prompt_len,
                can_load=True,
            ),
            dsa_two_groups=True,
            is_sparse_decode=True,
        )

        assert req_meta_again is not None
        assert req_meta_again.sparse_warm_ref
        assert req_meta_again.token_ids == []
        assert req_meta_again.slot_mapping == []
        assert req_meta_again.indexer_slot_mapping == []
        assert tracker.sparse_indexer_slot_mapping[0] is first_indexer_slots

    def test_sparse_producer_keeps_full_metadata(self) -> None:
        tracker = RequestTracker(
            req_id="req-producer",
            prompt_len=256,
            token_ids=list(range(257)),
            allocated_block_ids=_block_ids(257, 16),
            num_saved_tokens=256,
        )
        tracker.is_decode_phase = True
        load_spec = LoadSpec(
            vllm_cached_tokens=0,
            lmcache_cached_tokens=256,
            can_load=True,
        )

        for _ in range(2):
            req_meta = ReqMeta.from_request_tracker(
                tracker,
                block_size=16,
                lmcache_chunk_size=256,
                load_spec=load_spec,
                is_sparse_decode=True,
                save_entire_prefix=True,
            )

            assert req_meta is not None
            assert not req_meta.sparse_warm_ref
            assert len(req_meta.token_ids) >= 256

    def test_sparse_indexer_slot_mapping_prefers_request_slots(self) -> None:
        impl = object.__new__(LMCacheConnectorV1Impl)
        impl.device = "cpu"
        latent_slots = torch.arange(4, dtype=torch.long)
        request_indexer_slots = torch.arange(100, 108, dtype=torch.long)
        attn = SimpleNamespace(indexer_slot_mapping=torch.arange(200, 208))

        result = LMCacheConnectorV1Impl._sparse_indexer_slot_mapping(
            impl,
            attn,
            latent_slots,
            lmcache_cached_tokens=8,
            request_indexer_slots=request_indexer_slots,
            strict=True,
        )

        assert torch.equal(result, request_indexer_slots)

    def test_sparse_indexer_slot_mapping_requires_full_prompt_slots(self) -> None:
        impl = object.__new__(LMCacheConnectorV1Impl)
        impl.device = "cpu"
        latent_slots = torch.arange(4, dtype=torch.long)
        request_indexer_slots = torch.arange(100, 104, dtype=torch.long)
        attn = SimpleNamespace(indexer_slot_mapping=torch.arange(200, 208))

        result = LMCacheConnectorV1Impl._sparse_indexer_slot_mapping(
            impl,
            attn,
            latent_slots,
            lmcache_cached_tokens=8,
            request_indexer_slots=request_indexer_slots,
            strict=True,
        )

        assert torch.equal(result, attn.indexer_slot_mapping)

    def test_sparse_indexer_slot_mapping_strict_rejects_latent_fallback(self) -> None:
        impl = object.__new__(LMCacheConnectorV1Impl)
        impl.device = "cpu"
        latent_slots = torch.arange(4, dtype=torch.long)
        attn = SimpleNamespace(slot_mapping=None, indexer_slot_mapping=None)

        with pytest.raises(RuntimeError, match="full DSA index slot mapping"):
            LMCacheConnectorV1Impl._sparse_indexer_slot_mapping(
                impl,
                attn,
                latent_slots,
                lmcache_cached_tokens=4,
                strict=True,
            )

    def test_indexer_save_slot_mapping_prefers_layer_metadata(self) -> None:
        impl = object.__new__(LMCacheConnectorV1Impl)
        request = _make_prefill_req(prompt_len=8)
        request.indexer_slot_mapping = [torch.arange(100, 108)]
        attn = SimpleNamespace(indexer_slot_mapping=torch.arange(200, 208))

        result = LMCacheConnectorV1Impl._indexer_save_slot_mapping(
            impl,
            request,
            attn,
            layer_name=None,
            token_count=8,
        )

        assert torch.equal(result, attn.indexer_slot_mapping)

    def test_indexer_save_slot_mapping_ignores_short_request_slots(self) -> None:
        impl = object.__new__(LMCacheConnectorV1Impl)
        request = _make_prefill_req(prompt_len=8)
        request.indexer_slot_mapping = [torch.arange(4)]
        attn = SimpleNamespace(indexer_slot_mapping=torch.arange(200, 208))

        result = LMCacheConnectorV1Impl._indexer_save_slot_mapping(
            impl,
            request,
            attn,
            layer_name=None,
            token_count=8,
        )

        assert torch.equal(result, attn.indexer_slot_mapping)

    def test_indexer_save_slot_mapping_does_not_fallback_to_request_slots(
        self,
    ) -> None:
        impl = object.__new__(LMCacheConnectorV1Impl)
        request = _make_prefill_req(prompt_len=8)
        request.indexer_slot_mapping = [torch.arange(100, 108)]
        attn = SimpleNamespace(indexer_slot_mapping=None, slot_mapping=None)

        result = LMCacheConnectorV1Impl._indexer_save_slot_mapping(
            impl,
            request,
            attn,
            layer_name=None,
            token_count=8,
        )

        assert result is None
