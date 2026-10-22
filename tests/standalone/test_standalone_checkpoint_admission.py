# SPDX-License-Identifier: Apache-2.0
"""Actual checkpoint admission for short/long standalone and P/D requests."""

from types import SimpleNamespace as NS

import pytest

from test_checkpoint_adapter import method, control


@pytest.mark.parametrize("length", [2, 127, 128, 129, 4096, 4097, 20000])
@pytest.mark.parametrize("role", [None, "receiver"])
def test_checkpoint_resume_uses_exact_resident_or_compact_frontier(length, role):
    prefix = min(4, length - 1)
    req = NS(
        request_id="r",
        num_tokens=length + 1,
        num_preemptions=1,
        prompt_token_ids=list(range(prefix)),
        all_token_ids=list(range(length + 1)),
        status="preempted",
    )
    capture = control.CaptureSpec(
        "r", 1, 0, length + 1, prefix, ((1,), (2,)), prefix_end=prefix
    )
    adapter = NS(
        kv_role="kv_both",
        _resident_cold_load_enabled=True,
        _preemption_checkpoints={
            "r": control.PendingCheckpoint(capture, "ready", length)
        },
        _resume_lookup_queries={},
        _request_trackers={},
        _requests_priority={},
        skip_last_n_tokens=0,
        lookup_client=NS(lookup_cache=lambda **kw: length),
        _cold_perf_lookup_started={},
        _lmcache_chunk_size=1024,
        _block_size=128,
        _dsa_scratch_capacity=4096,
        _dsa_kv_policy_threshold=0,
        config=NS(
            pd_role=role,
            decode_preemption_checkpoint=True,
            min_retrieve_tokens=0,
            dsa_group1_load_mode="p2p_preferred"
            if role is None
            else "persistent_direct_hbm",
        ),
        enable_sparse_attention=True,
        supports_dsa_cold_compact_load=lambda: True,
        load_specs={},
    )
    matched = method(
        "get_num_new_matched_tokens", cdiv=lambda a, b: (a + b - 1) // b, LoadSpec=NS
    )(adapter, req, 0)
    spec = adapter.load_specs["r"]
    assert matched == length and spec.checkpoint_generation == 1
    assert spec.lmcache_cached_tokens == length and spec.dsa_cold_compact_load
    assert getattr(spec, "dsa_cold_resident_load", False) == (length <= 4096)
    assert spec.dsa_remap_frontier == (0 if length <= 4096 else length)
    assert spec.dsa_committed_end == length
    if length <= 4096:
        assert spec.dsa_cold_resume_computed_end == length
    assert spec.dsa_group1_direct_hbm == (role == "receiver")
