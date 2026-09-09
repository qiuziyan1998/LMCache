# SPDX-License-Identifier: Apache-2.0
"""Cold-load records without per-request wrappers or polling allocations.

Plans remain ordinary dictionaries; entries remain the existing six-item tuple.
This module imports no adapter, Torch or vLLM implementation at runtime.
"""

# Standard
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, TypeAlias, TypedDict

if TYPE_CHECKING:
    # Third Party
    import torch

    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import ReqMeta, WorkerRetrieveState


class _ColdLoadOptionalFields(TypedDict, total=False):
    indexer_perf: dict[str, float]
    indexer_source_owners: tuple[Any, ...]


class ColdLoadPlan(_ColdLoadOptionalFields):
    """Existing request plan shared by paired background jobs.

    The submitter supplies required fields. The indexer worker optionally fills
    timing fields; live-split completion and cleanup retain/clear source owners.
    Buffers, cache lists and readiness gates are borrowed without copying.
    """

    request: "ReqMeta"
    tokens: list[int]
    token_mask: "torch.Tensor"
    token_count: int
    indexer_slots_cpu: "torch.Tensor"
    latent_kvcaches: "list[torch.Tensor]"
    indexer_kvcaches: "list[torch.Tensor]"
    planned_at: float
    plan_started: float
    latent_shared_ready: Future[None]


ColdIndexerResult: TypeAlias = tuple["torch.Tensor | None", Any, float, float]

# Tuple fields, in their existing positional order:
# 0 generation: request-attempt generation checked before publication.
# 1 latent_future: completed state, including final readiness and retained owners.
# 2 request: the original request metadata, retained through terminal cleanup.
# 3 indexer_block_ids: destination blocks held until safe completion/failure.
# 4 submitted_at: diagnostic submission timestamp (zero when disabled).
# 5 indexer_future: sibling result, drained before cleanup or publication.
ColdLoadEntry: TypeAlias = tuple[
    int,
    Future["WorkerRetrieveState"],
    "ReqMeta",
    set[int],
    float,
    Future[ColdIndexerResult],
]
