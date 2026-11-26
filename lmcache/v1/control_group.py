# SPDX-License-Identifier: Apache-2.0
"""Private CPU collectives for an LMCache TP cohort."""

# Standard
from collections.abc import Sequence
from typing import Any

# Third Party
import torch
import torch.distributed as dist


class LMCacheControlGroup:
    """Own one Gloo group, created before loading starts and closed after drain.

    All members of ``ranks`` must construct this group in the same startup order.
    The existing LMCache mailbox still orders its control messages; this group
    only separates their collective sequence from model execution. Object
    broadcast is a fallback for integrations without a message queue.
    """

    def __init__(self, ranks: Sequence[int]) -> None:
        if not ranks or len(set(ranks)) != len(ranks):
            raise ValueError("LMCache control ranks must be nonempty and unique")
        self._ranks = tuple(ranks)
        self._closed = False
        self._group = (
            dist.new_group(
                ranks=list(ranks),
                backend="gloo",
                use_local_synchronization=True,
                group_desc="lmcache_control",
            )
            if len(ranks) > 1
            else None
        )

    def collective_all_true(self, local_ready: bool) -> bool:
        """Return MIN(readiness) across the cohort; reject use after close."""
        if self._closed:
            raise RuntimeError("LMCache control group is closed")
        if self._group is None:
            return bool(local_ready)
        # Keep one CPU int32 reduction, as before. Do not share a mutable flag:
        # distinct callers may hold their input while waiting for the mailbox.
        flag = torch.tensor([int(local_ready)], dtype=torch.int32, device="cpu")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._group)
        return bool(flag.item())

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast from TP-local ``src`` using Gloo when no queue is available.

        Return the source object; raise ValueError for an invalid source or
        RuntimeError after close. Normal vLLM queue traffic bypasses this method.
        """
        if self._closed:
            raise RuntimeError("LMCache control group is closed")
        if not 0 <= src < len(self._ranks):
            raise ValueError(f"Invalid LMCache control source rank: {src}")
        if self._group is None:
            return obj
        objects = [obj]
        dist.broadcast_object_list(objects, src=self._ranks[src], group=self._group)
        return objects[0]

    def close(self) -> None:
        """Release only the owned group after users drain; safe to call again."""
        if self._closed:
            return
        if self._group is not None and dist.is_initialized():
            dist.destroy_process_group(self._group)
        self._group = None
        self._closed = True
