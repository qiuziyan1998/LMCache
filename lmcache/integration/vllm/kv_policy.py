# SPDX-License-Identifier: Apache-2.0
"""KV residency and retrieve decisions shared by scheduler and worker metadata."""

# Standard
from typing import TYPE_CHECKING, Optional, TypeGuard

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LoadSpec,
        ReqMeta,
        RequestTracker,
    )


def load_spec_requires_retrieve(spec: Optional["LoadSpec"]) -> TypeGuard["LoadSpec"]:
    """Return whether this step permits retrieve-side work.

    ``spec`` is the optional scheduler load spec. A full-resident marker takes
    precedence over ``can_load``; older producers without the marker keep the
    original load contract.
    """
    return (
        spec is not None
        and spec.can_load
        and not getattr(spec, "dsa_full_resident", False)
    )


def sparse_kv_managed(tracker: "RequestTracker", threshold: int) -> bool:
    """Return whether the tracked KV prefix needs sparse storage management.

    ``threshold`` is inclusive for resident contexts. A nonresident frontier
    on the tracker takes precedence even if logical token progress rolls back.
    """
    return tracker.dsa_nonresident_frontier > 0 or len(tracker.token_ids) > threshold


def is_prefill_commit(request: "ReqMeta") -> bool:
    """Whether this row can acknowledge the initial persisted prefill frontier."""
    save = getattr(request, "save_spec", None)
    return (
        request.is_last_prefill
        and not getattr(request, "is_decode_window_save", False)
        and not getattr(save, "ordinary_decode_save", False)
        and (
            not request.is_sparse_decode
            or getattr(request.load_spec, "dsa_full_resident", False)
        )
    )
