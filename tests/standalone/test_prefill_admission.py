# SPDX-License-Identifier: Apache-2.0
"""Admission accounting without a model, accelerator, or remote cache."""

import ast
import importlib.util
from pathlib import Path
import sys
from threading import Lock
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prefill_admission_tested", ROOT / "lmcache/v1/prefill_admission.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Budget = module.PrefillAdmissionBudget


def reports(capacity=1100, available=None, retired=()):
    return tuple(
        dict(
            rank=r,
            pool=dict(
                capacity=capacity,
                available=capacity if available is None else available,
            )
            if r == 0
            else None,
            page_bytes=1 if r == 0 else 0,
            chunk_tokens=1024,
            max_batch_tokens=16384,
            retired=retired,
        )
        for r in range(4)
    )


def test_two_500k_requests_fit_third_waits_until_all_ranks_retire():
    budget = Budget(4)
    assert not budget.admit("a", 500000)[0]  # no guessed config capacity
    budget.update(reports())
    assert budget.admit("a", 500000)[0]
    assert budget.admit("b", 500000)[0]
    before = dict(budget.reservations)
    assert budget.admit("c", 500000)[:2] == (False, None)
    assert budget.reservations == before
    for _ in range(31):  # Existing 16k chunks never reserve their prefix again.
        assert budget.admit("a", 500000)[0]
    budget.finish("a")
    assert not budget.admit("c", 500000)[0]
    budget.update(reports(retired=("a",))[:3])
    assert not budget.admit("c", 500000)[0]
    budget.update(reports(retired=("a",))[3:])
    assert budget.admit("c", 500000)[0]


def test_long_request_reserves_future_chunks_and_both_components():
    budget = Budget(4)
    budget.update(reports())
    ok, _, info = budget.admit("a", 500000)
    assert ok and info["need_bytes"] == 489 + 16 + 2
    assert info["need_bytes"] > 16  # not merely the first chunk


def test_pinned_or_inflight_memory_is_deducted_even_after_scheduler_finish():
    budget = Budget(4)
    budget.update(reports())
    assert budget.admit("a", 500000)[0]
    budget.finish("a")
    budget.update(reports(available=400, retired=("a",)))
    assert not budget.admit("b", 500000)[0]
    budget.update(reports(available=600))
    assert budget.admit("b", 500000)[0]


def test_oversized_request_is_rejected_without_leaking_credit():
    budget = Budget(4)
    budget.update(reports(capacity=300))
    ok, error, info = budget.admit("a", 500000)
    assert not ok and "required_bytes=507" in error
    assert not budget.waiting and not budget.reservations
    assert budget.admit("small", 2048)[0]


def test_short_request_can_fit_after_long_request_was_deferred():
    budget = Budget(4)
    budget.update(reports(capacity=530))
    assert budget.admit("a", 500000)[0]
    assert not budget.admit("b", 500000)[0]
    assert budget.admit("c", 2048)[0]


def test_cleanup_ack_before_finish_is_not_lost():
    budget = Budget(4)
    budget.update(reports())
    budget.admit("a", 500000)
    budget.update(reports(retired=("a",)))
    assert "a" in budget.reservations
    budget.finish("a")
    assert "a" not in budget.reservations


def test_each_real_pool_must_have_space_and_every_worker_must_report():
    budget = Budget(4)
    rows = list(reports())
    rows[2] = dict(rows[2], pool=dict(capacity=1100, available=300), page_bytes=1)
    budget.update(tuple(rows[:3]))
    assert not budget.admit("a", 500000)[0]
    budget.update(tuple(rows[3:]))
    assert not budget.admit("a", 500000)[0]


@pytest.mark.parametrize("use_hot", [False, True])
def test_capacity_counts_only_free_and_evictable_unique_owners(use_hot):
    path = ROOT / "lmcache/v1/storage_backend/local_cpu_backend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "get_prefill_admission_capacity"
    )
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    evictable = NS(can_evict=True, get_physical_size=lambda: 100)
    pinned = NS(can_evict=False, get_physical_size=lambda: 200)
    inflight = NS(can_evict=False, get_physical_size=lambda: 300)
    backend = NS(
        cpu_lock=Lock(),
        use_hot=use_hot,
        get_allocator_capacity_bytes=lambda: (50, 1000),
        hot_cache={"a": evictable, "alias": evictable, "pin": pinned, "dma": inflight},
    )
    assert ns[fn.name](backend) == dict(capacity=1000, available=150 if use_hot else 50)
