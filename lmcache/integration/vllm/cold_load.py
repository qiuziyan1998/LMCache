# SPDX-License-Identifier: Apache-2.0
"""Cold-load records without per-request wrappers or polling allocations.

Plans remain ordinary dictionaries; entries remain the existing six-item tuple.
This module imports no adapter, Torch or vLLM implementation at runtime.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeAlias, TypedDict
from weakref import WeakMethod

# First Party
from lmcache.v1.serving_perf import serving_perf_enabled

logger = logging.getLogger("lmcache.integration.vllm.vllm_v1_adapter")

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


class ColdLoadCoordinator:
    """Own pending cold jobs and their terminal polling lifecycle.

    Admission and polling run on the adapter's worker thread. The original
    futures retain background work. Terminal effects use weak method hooks;
    unfinished polls do not resolve callbacks or allocate bound method objects.
    """

    def __init__(
        self,
        publish: Callable[[str, ColdLoadEntry, Any, bool, bool], None],
        fail: Callable[[str, ColdLoadEntry, Any, BaseException, bool], bool],
        requires_restart: Callable[[], bool],
    ) -> None:
        """Borrow terminal bound methods weakly and initialize empty job state.

        The publication hook commits a ready state; the failure hook returns
        whether owners are safe to retire. Neither hook is resolved by an
        empty or unfinished poll. Missing terminal owners fail closed.
        """
        self.futures: dict[str, ColdLoadEntry] = {}
        self.last_latent_future: Future | None = None
        self.executor: ThreadPoolExecutor | None = None
        self.aborted: set[str] | None = None
        self.retirements: dict[int, Any] | None = None
        self._publish = WeakMethod(publish)
        self._fail = WeakMethod(fail)
        self._requires_restart = WeakMethod(requires_restart)

    def get_executor(self) -> ThreadPoolExecutor:
        """Return the existing lazily created pair-worker pool."""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="lmcache-dsa-cold"
            )
        return self.executor

    def poll(self) -> Optional[set[str]]:
        """Publish ready generations and retire only safely completed entries.

        Return newly finished request IDs, or None while nothing is terminal.
        Native-unknown failures and missing terminal hooks propagate without
        removing entries. Serving readiness checks never become host waits.
        """
        futures = self.futures
        if not futures:
            return None
        perf_enabled = serving_perf_enabled()
        finished: set[str] = set()
        for req_id, entry in list(futures.items()):
            (
                generation,
                future,
                request,
                indexer_block_ids,
                submitted_at,
                indexer_future,
            ) = entry
            if not future.done() or not indexer_future.done():
                continue
            state = None
            aborted_ids = self.aborted
            was_aborted = bool(aborted_ids is not None and req_id in aborted_ids)
            try:
                assert request.load_spec is not None
                state = future.result()
                readiness = state.dense_load_readiness
                query = getattr(readiness, "query", None)
                # Keep the request parked without synchronizing the model
                # thread while its small metadata copies are still pending.
                if callable(query) and not query():
                    continue
                actual_generation = getattr(
                    request.load_spec, "dsa_cold_load_generation", None
                )
                if generation != actual_generation:
                    raise RuntimeError(
                        "Cold compact completion generation mismatch: "
                        f"req_id={req_id}, expected={generation}, "
                        f"actual={actual_generation}"
                    )
                publish = self._publish()
                if publish is None:
                    raise RuntimeError("Cold-load publication owner is unavailable")
                publish(req_id, entry, state, was_aborted, perf_enabled)
            except BaseException as exc:
                fail = self._fail()
                if fail is None:
                    raise RuntimeError(
                        "Cold-load cleanup owner is unavailable"
                    ) from exc
                if not fail(req_id, entry, state, exc, perf_enabled):
                    continue
            futures.pop(req_id, None)
            if was_aborted:
                assert aborted_ids is not None
                aborted_ids.discard(req_id)
                if not aborted_ids:
                    self.aborted = None
            finished.add(req_id)
        if not futures:
            self.last_latent_future = None
            if self.executor is not None:
                self.executor.shutdown(wait=False, cancel_futures=False)
                self.executor = None
        return finished or None

    def drain_retirements(
        self,
        engine: Any,
        release_state: Callable[..., None],
        *,
        block: bool = False,
        req_id: Optional[str] = None,
    ) -> None:
        """Poll exact fences during serving; blocking is reserved for shutdown.

        A missing or failed query retains ownership rather than falling back to
        a host wait on the model thread.
        """
        retirements = self.retirements
        if not retirements:
            return
        query = getattr(
            getattr(engine, "gpu_connector", None),
            "query_dense_load_readiness",
            None,
        )
        for identity, state in tuple(retirements.items()):
            if req_id is not None and state.req_id != req_id:
                continue
            if not block:
                if getattr(state, "_dense_retirement_query_failed", False):
                    continue
                try:
                    if not callable(query):
                        raise RuntimeError(
                            "NPU connector has no nonblocking dense-load "
                            "readiness query API"
                        )
                    if not query(state.dense_load_readiness):
                        continue
                except Exception:
                    logger.exception(
                        "Dense-load retirement query failed; retaining "
                        "owners: req_id=%s",
                        state.req_id,
                    )
                    state._dense_retirement_query_failed = True
                    continue
            release_state(state, engine, dense_ready=not block)
            if retirements.get(identity) is state:
                retirements.pop(identity)
        if not retirements:
            self.retirements = None

    def submit_pair(
        self,
        plan: ColdLoadPlan,
        generation: int,
        indexer_block_ids: set[int],
        submitted_at: float,
        npu_device_id: Optional[int],
        executor: ThreadPoolExecutor,
        indexer_job: Callable[[ColdLoadPlan, Optional[int]], ColdIndexerResult],
        latent_job: Callable[..., Any],
    ) -> None:
        """Submit the existing job pair, opening the sibling gate on failure.

        Job callables and the executor are borrowed for submission only. The
        exact original futures, request, block set and timestamp form the entry.
        """
        indexer_future = executor.submit(
            indexer_job,
            plan,
            npu_device_id,
        )
        previous_latent_future = self.last_latent_future
        try:
            latent_future = executor.submit(
                latent_job,
                plan,
                npu_device_id,
                indexer_future,
                previous_latent_future,
            )
        except BaseException as submit_error:
            # The staged Group-1 path waits on this gate before it can finish.
            # Resolve it before draining the already-submitted sibling, or a
            # failed second submit deadlocks this exception path forever.
            latent_shared_ready = plan["latent_shared_ready"]
            if not latent_shared_ready.done():
                latent_shared_ready.set_exception(submit_error)
            # Indexer work may already target the allocated blocks. Fence it
            # before unwinding so those blocks cannot be reused.
            try:
                indexer_future.result()
            except BaseException as indexer_error:
                requires_restart = self._requires_restart()
                if requires_restart is None:
                    raise RuntimeError(
                        "Cold-load failure owner is unavailable"
                    ) from indexer_error
                if requires_restart():
                    raise indexer_error
            raise submit_error
        self.last_latent_future = latent_future
        request = plan["request"]
        self.futures[request.req_id] = (
            generation,
            latent_future,
            request,
            indexer_block_ids,
            submitted_at,
            indexer_future,
        )
