# SPDX-License-Identifier: Apache-2.0
"""CPU control collectives owned by one LMCache worker engine."""

# Standard
from collections.abc import Sequence
from typing import Any, Optional

# Third Party
import torch
import torch.distributed as dist


class LMCacheControlGroup:
    """Isolate LMCache metadata and readiness from model collectives.

    Args:
        ranks (Sequence[int]): Global ranks of the TP cohort, in TP rank order.
            Every member must construct this group during worker startup,
            before background loading starts. A singleton needs no distributed
            initialization.

    Raises:
        ValueError: If the rank list is empty.

    LMCache's ordered mailbox must still give all members the same order of
    control operations. Separate groups allow that order to differ from the
    model thread's collective order. Disjoint TP cohorts can initialize without
    a world-wide barrier.
    """

    def __init__(self, ranks: Sequence[int]) -> None:
        if not ranks:
            raise ValueError("LMCache control group needs at least one rank")
        self._ranks = list(ranks)
        self._closed = False
        self._group: Optional[dist.ProcessGroup] = None
        if len(self._ranks) > 1:
            self._group = dist.new_group(
                ranks=self._ranks,
                backend="gloo",
                use_local_synchronization=True,
                group_desc="lmcache_control",
            )

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Return the object broadcast by TP-local ``src`` on the control group.

        Args:
            obj (Any): Payload on the source rank; ignored on other ranks.
            src (int): Source rank within this TP cohort, defaulting to zero.

        Returns:
            Any: The source rank's payload on every member.

        Raises:
            ValueError: If the source rank is outside this cohort.
            RuntimeError: If the group has been closed.
        """
        self._ensure_open()
        if not 0 <= src < len(self._ranks):
            raise ValueError(f"Invalid LMCache control source rank: {src}")
        if self._group is None:
            return obj
        objects = [obj]
        dist.broadcast_object_list(objects, src=self._ranks[src], group=self._group)
        return objects[0]

    def collective_all_true(self, local_ready: bool) -> bool:
        """Agree on readiness without joining the model's collective sequence.

        Args:
            local_ready (bool): Whether this worker completed the control phase.

        Returns:
            bool: True only if every member reports readiness.

        Raises:
            RuntimeError: If the group has been closed.
        """
        self._ensure_open()
        if self._group is None:
            return bool(local_ready)
        flag = torch.tensor([int(local_ready)], dtype=torch.int32, device="cpu")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._group)
        return bool(flag.item())

    def close(self) -> None:
        """Release the owned group after control users stop; repeated calls are safe.

        The model's process groups are never destroyed here. If distributed
        teardown already destroyed all groups, only local ownership is cleared.
        """
        if self._closed:
            return
        if self._group is not None and dist.is_initialized():
            dist.destroy_process_group(self._group)
        self._group = None
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LMCache control group is closed")
