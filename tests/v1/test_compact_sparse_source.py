# SPDX-License-Identifier: Apache-2.0
"""CPU tests of explicit service leases and the real compact source builder."""

# Standard
import gc
from concurrent.futures import ThreadPoolExecutor

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.sparse import build_compact_prepared_sparse_source
from lmcache.v1.shared_cpu_cache import LocalLeaseHandle, SharedCPUCacheError


def make_lease():
    retired = []
    mapping = object()
    lease = LocalLeaseHandle(
        lease_id="epoch:lease:1",
        mapping=mapping,
        retire=lambda identity, proof: retired.append((identity, proof)),
    )
    return lease, mapping, retired


def test_unused_lease_retires_once_without_gc():
    lease, mapping, retired = make_lease()
    enabled = gc.isenabled()
    gc.disable()
    try:
        held = lease.retain()
        assert held is not lease and held.mapping is mapping
        lease.close()
        lease.close()
        assert not lease.is_valid() and held.is_valid() and not retired
        held.close()
        held.close()
        assert retired == [("epoch:lease:1", "never_imported")]
        with pytest.raises(SharedCPUCacheError, match="retain"):
            held.retain()
    finally:
        if enabled:
            gc.enable()


def test_each_imported_hold_requires_its_own_fence():
    lease, _, retired = make_lease()
    first, second = lease.retain(), lease.retain()
    first.mark_imported()
    second.mark_imported()
    lease.release()
    first.release(proof="fenced")
    for proof in (None, "never_imported"):
        with pytest.raises(SharedCPUCacheError, match="final-access fence"):
            second.release(proof)
    assert second.is_valid() and retired == []
    second.release(proof="fenced")
    assert retired == [("epoch:lease:1", "fenced")]


def test_retirement_callback_runs_outside_lock():
    seen = []
    lease = LocalLeaseHandle(
        lease_id="callback",
        mapping=object(),
        retire=lambda *args: seen.append(lease.is_valid()),
    )
    lease.release()
    assert seen == [False]


def test_concurrent_holds_retire_exactly_once():
    lease, _, retired = make_lease()
    holds = [lease.retain() for _ in range(16)]
    lease.close()
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda held: held.release(), holds + holds))
    assert retired == [("epoch:lease:1", "never_imported")]


def test_unknown_retirement_proof_keeps_lease():
    lease, _, retired = make_lease()
    with pytest.raises(ValueError, match="proof"):
        lease.release("timeout")
    assert lease.is_valid() and not retired
    lease.close()


def compact_source(lease, **kwargs):
    args = dict(
        num_layers=2,
        total_tokens=6,
        chunk_token_counts=(4, 2),
        owner_lease=lease,
        chunk_size=4,
        expected_pointer_device=torch.device("cpu"),
    )
    args.update(kwargs)
    return build_compact_prepared_sparse_source(
        [torch.tensor([101, 102]), torch.tensor([201, 202])], **args
    )


def test_compact_source_keeps_pointer_rows_without_materializing_views():
    lease, _, retired = make_lease()
    source = compact_source(lease)
    assert source.owner_lease is lease and source.graph_owners is None
    assert source.total_tokens == 6 and source.chunk_token_counts == (4, 2)
    assert source.validated_chunk_size == 4
    assert all(layer.memory_objs == layer.tensors == () for layer in source.layers)
    assert source.layers[1].chunk_ptrs_npu.tolist() == [201, 202]
    assert compact_source(lease).binding_token is not source.binding_token
    assert not retired
    lease.close()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_layers": 3}, "complete layers"),
        ({"chunk_token_counts": ()}, "token coverage"),
        ({"chunk_token_counts": (3, 3)}, "configured chunk layout"),
        ({"total_tokens": 5}, "match total_tokens"),
        ({"total_tokens": 7}, "complete layers"),
        ({"chunk_size": 0}, "positive"),
    ],
)
def test_compact_source_reuses_existing_coverage_validation(kwargs, match):
    lease, _, _ = make_lease()
    with pytest.raises(ValueError, match=match):
        compact_source(lease, **kwargs)
    lease.close()


def test_compact_source_rejects_released_ownership():
    lease, _, _ = make_lease()
    lease.close()
    with pytest.raises(ValueError, match="released"):
        compact_source(lease)


@pytest.mark.parametrize(
    "rows,match",
    [
        ([torch.tensor([1, 2], dtype=torch.int32)] * 2, "int64"),
        ([torch.tensor([1])] * 2, "pointer coverage"),
        ([torch.tensor([[1, 2]])] * 2, "1D"),
        ([torch.tensor([1, 2, 3, 4])[::2]] * 2, "contiguous"),
    ],
)
def test_compact_source_rejects_malformed_pointer_rows(rows, match):
    lease, _, _ = make_lease()
    with pytest.raises(ValueError, match=match):
        build_compact_prepared_sparse_source(
            rows, num_layers=2, total_tokens=6, chunk_token_counts=(4, 2),
            owner_lease=lease, chunk_size=4,
        )
    lease.close()
