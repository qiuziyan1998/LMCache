# SPDX-License-Identifier: Apache-2.0
"""Exercise real SFA preparation against LMCache under reversed thread order."""

# Standard
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from threading import Condition, Event, local
from types import SimpleNamespace as NS
from typing import Any, Callable
from unittest.mock import patch
import ast
import importlib.util
import multiprocessing
import sys
import time

# Third Party
import pytest
import torch
import torch.distributed as dist

# First Party
from lmcache.v1.control_group import LMCacheControlGroup


def runner_source() -> Path:
    """Find the companion checkout or installed Ascend runner without loading it."""
    roots = [Path(__file__).resolve().parents[3] / "vllm-ascend/vllm_ascend"]
    spec = importlib.util.find_spec("vllm_ascend")
    if spec and spec.submodule_search_locations:
        roots.extend(Path(root) for root in spec.submodule_search_locations)
    for root in roots:
        path = root / "worker/model_runner_v1.py"
        if path.is_file():
            return path
    pytest.skip("Requires the companion vllm-ascend source")


def preparation(path: Path, tp: Any, dp: Any) -> Callable:
    """Load the production failure vote without importing the NPU runner."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "_coordinate_sfa_full_graph_preparation"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    namespace = dict(
        torch=torch,
        dist=dist,
        get_tp_group=lambda: tp,
        get_dp_group=lambda: dp,
        uses_local_sfa_fail_stop=lambda _: False,
        record_function_or_nullcontext=lambda _: nullcontext(),
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize(
    "load_failure,graph_failure", [(None, None), (1, None), (None, 3), (1, 3)]
)
def test_reversed_threads_keep_ready_and_failed_votes_separate(
    monkeypatch: pytest.MonkeyPatch,
    load_failure: int | None,
    graph_failure: int | None,
) -> None:
    """A deterministic collective rendezvous verifies both protocols and counts."""
    condition, thread = Condition(), local()
    sequence, calls, slots = defaultdict(int), defaultdict(int), {}
    entered = [Event() for _ in range(4)]

    def create(*, ranks: list[int], **kwargs: Any) -> tuple:
        return "cache", tuple(ranks)

    def reduce(tensor: torch.Tensor, *, op: Any, group: tuple) -> None:
        rank = thread.rank
        with condition:
            key = group, sequence[rank, group]
            sequence[rank, group] += 1
            calls[rank, group] += 1
            slot = slots.setdefault(key, {})
            slot[rank] = tensor.clone(), op
            entered[rank].set()
            condition.notify_all()
            assert condition.wait_for(lambda: len(slot) == len(group[1]), timeout=5)
            # Mismatched MAX/MIN is a protocol error, not a defined reduction.
            assert {item[1] for item in slot.values()} == {op}
            values = torch.stack([item[0] for item in slot.values()])
            tensor.copy_(
                values.min(0).values
                if op == dist.ReduceOp.MIN
                else values.max(0).values
            )

    monkeypatch.setattr(dist, "new_group", create)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    controls, prepares = [], []
    source = runner_source()
    for rank in range(4):
        ranks = tuple(range(rank // 2 * 2, rank // 2 * 2 + 2))
        controls.append(LMCacheControlGroup(ranks))
        prepares.append(
            preparation(
                source,
                NS(world_size=2, cpu_group=("model-tp", ranks)),
                NS(world_size=2, cpu_group=("model-dp", (rank % 2, rank % 2 + 2))),
            )
        )

    def run(rank: int, loading: bool) -> bool:
        thread.rank = rank
        if loading:
            return controls[rank].collective_all_true(rank != load_failure)
        error = (
            RuntimeError("injected graph failure") if rank == graph_failure else None
        )
        runner = NS(_sfa_preparation_groups=None, vllm_config=NS(parallel_config=None))
        try:
            prepares[rank](runner, error, dummy_run=False)
            return False
        except RuntimeError as caught:
            assert "source/metadata preparation failed" in str(caught)
            assert caught.__cause__ is error
            return True

    with ThreadPoolExecutor(8) as executor:
        jobs = {
            (rank, bool(rank % 2)): executor.submit(run, rank, bool(rank % 2))
            for rank in range(4)
        }
        assert all(event.wait(5) for event in entered)
        jobs.update(
            {
                (rank, not bool(rank % 2)): executor.submit(
                    run, rank, not bool(rank % 2)
                )
                for rank in range(4)
            }
        )
        for (rank, loading), job in jobs.items():
            expected = (
                load_failure not in range(rank // 2 * 2, rank // 2 * 2 + 2)
                if loading
                else graph_failure is not None
            )
            assert job.result(timeout=10) == expected
    assert len(calls) == 12 and set(calls.values()) == {
        1
    }  # One load vote, TP MAX and DP MAX per rank.


def gloo_worker(rank: int, tp_size: int, store: str, source: str) -> None:
    """Run the same concurrency against real Gloo, including object fallback."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=store,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        ranks = list(
            range(rank // tp_size * tp_size, rank // tp_size * tp_size + tp_size)
        )
        model_tp = dist.new_group(ranks, backend="gloo", use_local_synchronization=True)
        dp_ranks = list(range(rank % tp_size, 4, tp_size))
        model_dp = (
            dist.new_group(dp_ranks, backend="gloo", use_local_synchronization=True)
            if len(dp_ranks) > 1
            else None
        )
        control = LMCacheControlGroup(ranks)
        prepare = preparation(
            Path(source),
            NS(world_size=tp_size, cpu_group=model_tp),
            NS(world_size=len(dp_ranks), cpu_group=model_dp),
        )
        original_reduce = dist.all_reduce
        for step, (load_failure, graph_failure) in enumerate(
            [(None, None), (1, None), (None, 3), (1, 3)] * 2
        ):
            entered = Event()

            def reduce(
                tensor: torch.Tensor, *, op: Any, group: Any, entered: Event = entered
            ) -> None:
                work = original_reduce(tensor, op=op, group=group, async_op=True)
                entered.set()
                work.wait(timeout=timedelta(seconds=20))

            def load(step: int = step, load_failure: int | None = load_failure) -> bool:
                src = step % tp_size
                marker = {"step": step, "source": ranks[src]}
                assert (
                    control.broadcast_object(
                        marker if rank == ranks[src] else None, src
                    )
                    == marker
                )
                return control.collective_all_true(rank != load_failure)

            def graph(graph_failure: int | None = graph_failure) -> bool:
                runner = NS(
                    _sfa_preparation_groups=None, vllm_config=NS(parallel_config=None)
                )
                error = (
                    RuntimeError("injected graph failure")
                    if rank == graph_failure
                    else None
                )
                try:
                    prepare(runner, error, dummy_run=True)
                    return False
                except RuntimeError as caught:
                    assert (
                        "source/metadata preparation failed" in str(caught)
                        and caught.__cause__ is error
                    )
                    return True

            with (
                patch.object(dist, "all_reduce", reduce),
                ThreadPoolExecutor(2) as executor,
            ):
                if (rank + step // 4) % 2:
                    loaded = executor.submit(load)
                    assert entered.wait(15)
                    prepared = executor.submit(graph)
                else:
                    prepared = executor.submit(graph)
                    assert entered.wait(15)
                    loaded = executor.submit(load)
                assert loaded.result(timeout=25) == (load_failure not in ranks)
                assert prepared.result(timeout=25) == (graph_failure is not None)
            dist.barrier()
        control.close()
        dist.barrier(
            group=model_tp
        )  # Closing cache control must leave model groups usable.
        control = LMCacheControlGroup(ranks)
        assert control.collective_all_true(True)
        control.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    sys.platform != "linux" or not dist.is_gloo_available(),
    reason="Requires Linux Gloo",
)
@pytest.mark.parametrize("tp_size", [2, 4])
def test_real_gloo_control_isolation(
    tmp_path: Path, tp_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise disjoint TP2 and TP4 cohorts without requiring an NPU."""
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    source = str(runner_source())
    context = multiprocessing.get_context("spawn")
    children = [
        context.Process(
            target=gloo_worker,
            args=(rank, tp_size, (tmp_path / "store").as_uri(), source),
        )
        for rank in range(4)
    ]
    try:
        for child in children:
            child.start()
        deadline = time.monotonic() + 90
        for child in children:
            child.join(max(0, deadline - time.monotonic()))
            assert child.exitcode == 0
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            if child.pid is not None:
                child.join(5)
