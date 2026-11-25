# SPDX-License-Identifier: Apache-2.0
"""Owner-only persistent reads, checkpoint tail retirement and attempt matching."""

from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest
import torch

from lmcache.integration.vllm import storage_process as storage
from lmcache.integration.vllm.dcp import DcpWriteLayout


@pytest.mark.parametrize("owner", [0, 2])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_only_owner_reads_but_every_rank_votes_and_retires(
    monkeypatch, owner, checkpoint
):
    direct, tails, votes, retired = [], [], [], []
    for rank in range(4):
        gate = Future()
        gate.set_result(None)
        spec = NS(dsa_group1_direct_hbm=True)
        if checkpoint:
            spec.checkpoint_generation, spec.checkpoint_prefix_end = 2, 5
        request = NS(req_id="r", request_configs=None, load_spec=spec)
        engine = NS(
            load_group1_pages_direct=lambda *a, rank=rank, **kw: direct.append(
                (rank, len(a[0]))
            ),
            finish_checkpoint_prefix=lambda req, ready, rank=rank: votes.append(
                (rank, ready)
            )
            or ready,
            native_source_lifetime=NS(retire_source=lambda _: None),
        )
        source = NS(
            owner_lease=NS(
                release=lambda proof, rank=rank: retired.append((rank, proof))
            )
        )
        plan = dict(
            request=request,
            tokens=list(range(9)),
            indexer_slots_cpu=torch.arange(9),
            indexer_kvcaches=(),
            owns_indexer=rank == owner,
            token_mask=torch.ones(9),
            token_count=9,
            latent_shared_ready=gate,
            storage_sources={1: source},
            chunk_plan=lambda: [(0, 4, 1), (4, 8, 2), (8, 9, 3)],
        )
        monkeypatch.setattr(
            storage,
            "load_prepared_dense_source",
            lambda *a, rank=rank: tails.append((rank, a[-1])),
        )
        storage.run_indexer(
            NS(lmcache_engine=engine, _lmcache_chunk_size=4), plan, None
        )
    assert direct == [(owner, 4 if checkpoint else 9)]
    assert tails == ([(owner, 4)] if checkpoint else [])
    assert votes == ([(i, True) for i in range(4)] if checkpoint else [])
    assert [r for r, _ in retired] == (list(range(4)) if checkpoint else [])


def test_rank_specific_parent_numbers_and_null_snapshot():
    layout = DcpWriteLayout(2, 4, (51, 52), 0, ((3, 0), (1, 2)), 1, 15)
    assert list(layout.runs(0, 8, 0)) == [(0, 12, 4)]
    assert list(layout.runs(0, 16, 1)) == [(4, 4, 4), (12, 8, 4)]
    with pytest.raises(ValueError, match="null"):
        list(layout.runs(8, 12, 0))
