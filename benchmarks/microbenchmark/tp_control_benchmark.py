# SPDX-License-Identifier: Apache-2.0
"""Time a complete metadata broadcast plus readiness vote, excluding startup.

Run on Linux with the matching vLLM installed, without a model or NPU:
  python -m torch.distributed.run --rdzv-backend=static \
      --rdzv-endpoint=127.0.0.1:29671 --nnodes=1 --node-rank=0 \
      --nproc-per-node=4 benchmarks/microbenchmark/tp_control_benchmark.py

The legacy arm is deliberately single-threaded: sharing its Gloo group with
concurrent model collectives is unsafe. This measures CPU control latency, not
serving TPOT, under an identical payload and operation count. All arms retain
the same process groups, so resource-contention costs need an end-to-end test.
"""

# Standard
from typing import Any, Callable
import argparse
import json
import statistics
import time

# Third Party
import torch
import torch.distributed as dist

# First Party
from lmcache.v1.control_group import LMCacheControlGroup


def legacy_ready(ready: bool) -> bool:
    """Run the original single-int CPU reduction on the model/default group."""
    flag = torch.tensor([int(ready)], dtype=torch.int32, device="cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def main() -> None:
    """Compare legacy MQ, isolated MQ, and the all-Gloo reference path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--payload-bytes", type=int, default=4096)
    args = parser.parse_args()
    if min(args.iterations, args.trials, args.payload_bytes) < 1 or args.warmup < 0:
        parser.error(
            "positive iterations/trials/payload and nonnegative warmup required"
        )

    # Import after argument parsing so --help also works without vLLM installed.
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    control = None
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        queue = MessageQueue.create_from_process_group(dist.group.WORLD, 1 << 22, 6)
        control = LMCacheControlGroup(list(range(world)))
        payload = {
            "phase": "storage_source_import",
            "metadata": bytes(args.payload_bytes),
        }
        arms = {
            "legacy_mq": (queue.broadcast_object, legacy_ready),
            "isolated_mq": (queue.broadcast_object, control.collective_all_true),
            "isolated_gloo_objects": (
                control.broadcast_object,
                control.collective_all_true,
            ),
        }
        samples: dict[str, list[float]] = {name: [] for name in arms}

        def run(
            broadcast: Callable[..., Any], vote: Callable[[bool], bool], count: int
        ) -> None:
            for iteration in range(count):
                received = broadcast(payload if rank == 0 else None)
                # Include a failed local vote to verify MIN, with the same
                # validation work in every measured arm.
                ready = iteration % 7 != 0 or rank != world - 1
                assert received == payload
                assert vote(ready) == (iteration % 7 != 0)

        for trial in range(args.trials):
            order = list(arms)
            order = order[trial % len(order) :] + order[: trial % len(order)]
            for name in order:
                run(*arms[name], args.warmup)
                dist.barrier()
                started = time.perf_counter_ns()
                run(*arms[name], args.iterations)
                elapsed = torch.tensor(
                    [(time.perf_counter_ns() - started) / (1000 * args.iterations)],
                    dtype=torch.float64,
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                samples[name].append(elapsed.item())
        if rank == 0:
            print(
                json.dumps(
                    {
                        "ranks": world,
                        "payload_bytes": args.payload_bytes,
                        "iterations": args.iterations,
                        "trials": args.trials,
                        "unit": "slowest_rank_us_per_broadcast_and_vote",
                        "median": {
                            name: statistics.median(values)
                            for name, values in samples.items()
                        },
                        "samples": samples,
                    }
                ),
                flush=True,
            )
        dist.barrier()
        del queue
    finally:
        if control is not None:
            control.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
