# SPDX-License-Identifier: Apache-2.0
"""Real Gloo regression for concurrent LMCache loading and SFA preparation."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace as NS
from typing import Any, Callable
from unittest.mock import patch
import ast
import multiprocessing
import time

# Third Party
import pytest
import torch
import torch.distributed as dist

# First Party
from lmcache.v1.control_group import LMCacheControlGroup


def _preparation_method(path: str, tp: NS, dp: NS) -> Callable:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_coordinate_sfa_full_graph_preparation"
    )
    tree = ast.parse("from __future__ import annotations")
    tree.body.append(method)
    namespace: dict[str, Any] = dict(
        torch=torch,
        dist=dist,
        get_tp_group=lambda: tp,
        get_dp_group=lambda: dp,
        record_function_or_nullcontext=lambda _: nullcontext(),
        uses_local_sfa_fail_stop=lambda _: False,
    )
    exec(compile(ast.fix_missing_locations(tree), path, "exec"), namespace)
    return namespace[method.name]


def _gloo_worker(rank: int, tp_size: int, store: str, source: str) -> None:
    # Some CPU test wheels expose Gloo but omit its backend entry point.
    register = getattr(dist.distributed_c10d, "_register_builtin_gloo_backend", None)
    if register is not None and "GLOO" not in getattr(dist.Backend, "_plugins", {}):
        register()
    world = 4
    dist.init_process_group(
        "gloo",
        init_method=store,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    try:
        start = rank // tp_size * tp_size
        ranks = list(range(start, start + tp_size))
        model_tp = dist.new_group(
            ranks,
            backend="gloo",
            use_local_synchronization=True,
            timeout=timedelta(seconds=30),
        )
        dp_ranks = list(range(rank % tp_size, world, tp_size))
        model_dp = (
            dist.new_group(
                dp_ranks,
                backend="gloo",
                use_local_synchronization=True,
                timeout=timedelta(seconds=30),
            )
            if len(dp_ranks) > 1
            else None
        )
        control = LMCacheControlGroup(ranks)
        prepare = _preparation_method(
            source,
            NS(world_size=tp_size, cpu_group=model_tp),
            NS(world_size=len(dp_ranks), cpu_group=model_dp),
        )
        runner = NS(_sfa_preparation_groups=None, vllm_config=NS(parallel_config=None))
        original_reduce = dist.all_reduce
        # The same cases must work whichever protocol each rank launches first.
        for step, (load_failure, graph_failure) in enumerate(
            [(None, None), (world - 1, None), (None, world - 1), (0, world - 1)] * 2
        ):
            launched = Event()
            src = step % tp_size
            marker = {"phase": "storage_source_import", "ordinal": step}

            def enqueue(
                tensor: torch.Tensor,
                *,
                op: Any,
                group: Any,
                launched: Event = launched,
            ) -> None:
                work = original_reduce(tensor, op=op, group=group, async_op=True)
                launched.set()
                work.wait(timeout=timedelta(seconds=20))

            def loading(
                marker: dict[str, Any] = marker,
                src: int = src,
                load_failure: int | None = load_failure,
            ) -> bool:
                received = control.broadcast_object(
                    marker if rank == ranks[src] else None, src
                )
                assert received == marker
                return control.collective_all_true(rank != load_failure)

            def preparing(graph_failure: int | None = graph_failure) -> bool:
                error = (
                    RuntimeError("injected graph failure")
                    if rank == graph_failure
                    else None
                )
                try:
                    prepare(runner, error, dummy_run=True)
                    return False
                except RuntimeError as result:
                    assert "source/metadata preparation failed" in str(result)
                    assert result.__cause__ is error
                    return True

            # Enqueue the first reduction before launching the other thread.
            # Opposite per-rank ordering reproduces the original shared-PG race.
            with (
                patch.object(dist, "all_reduce", enqueue),
                ThreadPoolExecutor(2) as pool,
            ):
                if (rank + step // 4) % 2:
                    load = pool.submit(loading)
                    assert launched.wait(15)
                    graph = pool.submit(preparing)
                else:
                    graph = pool.submit(preparing)
                    assert launched.wait(15)
                    load = pool.submit(loading)
                assert load.result(timeout=25) == (load_failure not in ranks)
                assert graph.result(timeout=25) == (graph_failure is not None)
            dist.barrier()

        control.close()
        control.close()
        # Destroying/recreating the cache transport must leave model groups usable.
        dist.barrier(group=model_tp)
        control = LMCacheControlGroup(ranks)
        assert control.collective_all_true(True)
        control.close()
        dist.barrier()
        if model_dp is not None:
            dist.destroy_process_group(model_dp)
        dist.destroy_process_group(model_tp)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size", [2, 4])
def test_real_gloo_loading_and_sfa_are_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
) -> None:
    """TP4 and disjoint TP2 groups preserve both protocols under reversed ordering."""
    if not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    source = (
        Path(__file__).resolve().parents[3]
        / "vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    )
    if not source.is_file():
        pytest.skip("Requires the sibling vllm-ascend checkout for real SFA code")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_gloo_worker,
            args=(rank, tp_size, (tmp_path / "store").as_uri(), str(source)),
        )
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 90
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert [process.exitcode for process in processes] == [0] * 4
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)
