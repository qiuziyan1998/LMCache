# SPDX-License-Identifier: Apache-2.0
"""Native direct-push execution with borrowed engine and registration.

Owns the executor, operation limit and in-flight futures. The calling event loop
mutates lifecycle state; worker threads execute producer fences and native calls.
The connector owns registration and the borrowed engine. Cancellation drains the
original native future; ambiguous completion retains it until terminal evidence.
Importing this module neither imports the persistent connector nor acquires an
engine. Safety deadlines and completion telemetry retain their existing clocks.
"""

# Standard
import asyncio
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any, Callable

# Third Party
import torch

# First Party
from lmcache.v1.remote_fill.protocol import DestinationPageDescriptor
from lmcache.v1.remote_fill.native import (
    DIRECT_PUSH_H0_QUALIFICATION_V1,
    DirectPushSourcePlan,
    NativeDirectPushActivation,
    NativeDirectPushAmbiguousError,
    NativeDirectPushPreSubmitError,
    NativeDirectPushResult,
    NativeDirectPushTerminalError,
    PreparedDirectPushSource,
)


async def drain_native_task(task: asyncio.Future[Any]) -> None:
    """Keep native transfer buffers alive through coroutine cancellation."""
    try:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
    finally:
        del task


def validate_direct_push_source_ranges(
    source_ptrs: tuple[int, ...],
    lengths: tuple[int, ...],
    owner_ranges: list[tuple[int, int]],
) -> None:
    """Reject source extents not covered by sorted retained owner ranges."""
    starts = [ptr for ptr, _ in owner_ranges]
    prefix_ends: list[int] = []
    for ptr, size in owner_ranges:
        prefix_ends.append(max(prefix_ends[-1] if prefix_ends else 0, ptr + size))
    for ptr, length in zip(source_ptrs, lengths, strict=True):
        index = bisect_right(starts, ptr) - 1
        if index < 0 or ptr + length > prefix_ends[index]:
            raise ValueError("Native direct push source lies outside retained owners")


def direct_push_owner_ranges(owners: tuple[Any, ...]) -> list[tuple[int, int]]:
    """Return unique source-storage registrations for retained owners."""
    ranges = {
        (
            int(owner.untyped_storage().data_ptr()),
            int(owner.untyped_storage().nbytes()),
        )
        for owner in owners
    }
    if not ranges or any(ptr <= 0 or size <= 0 for ptr, size in ranges):
        raise ValueError("Direct push requires valid retained source owners")
    return sorted(ranges)


def validate_direct_push_activation(
    activation: NativeDirectPushActivation | None,
) -> NativeDirectPushActivation:
    """Fail closed unless the compatibility contract and ARM proof are valid."""
    if activation is None:
        raise RuntimeError("Native direct push requires explicit activation")
    if activation.h0_qualification != DIRECT_PUSH_H0_QUALIFICATION_V1:
        raise RuntimeError("Native direct push compatibility contract is invalid")
    if not activation.arm_acknowledged:
        raise RuntimeError("Native direct push requires ARM_WINDOW acknowledgement")
    if not activation.native_transfer_attempt_id:
        raise ValueError("Native direct push requires a transfer attempt identifier")
    return activation


class MooncakeDirectPushTransport:
    """Bounded asynchronous wrapper over Mooncake's native sync-write API.

    The wrapper owns a worker pool and operation semaphore that are independent
    from persistent Mooncake puts. Source owners remain reachable until the
    original native call reaches a terminal return, even after a timeout.
    """

    def __init__(
        self,
        register_source_owners: Callable[[tuple[Any, ...]], None],
        transfer_engine: Any,
        *,
        worker_count: int,
        max_operations: int,
        timeout_seconds: float,
    ) -> None:
        """Create a bounded, initially idle direct-push transport.

        Args:
            register_source_owners: Connector-owned idempotent registration
                callback shared with persistent Mooncake transfers.
            transfer_engine: Borrowed native Mooncake TransferEngine.
            worker_count: Number of dedicated direct-push worker threads.
            max_operations: Maximum native calls that may remain in flight.
            timeout_seconds: Deadline before completion becomes ambiguous.

        Raises:
            ValueError: If a resource bound is not positive.
        """
        if worker_count <= 0 or max_operations <= 0 or timeout_seconds <= 0:
            raise ValueError("Direct push resource bounds must be positive")
        self._register_source_owners = register_source_owners
        self._transfer_engine = transfer_engine
        self._timeout_seconds = timeout_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="lmcache-remote-fill",
        )
        self._operation_limit = asyncio.Semaphore(max_operations)
        self._task_lock = asyncio.Lock()
        self._inflight: set[asyncio.Future[NativeDirectPushResult]] = set()
        self._closed = False

    async def push_external_pages(
        self,
        *,
        remote_session: str,
        source_plan: DirectPushSourcePlan | PreparedDirectPushSource,
        destination_descriptors: tuple[DestinationPageDescriptor, ...],
        activation: NativeDirectPushActivation | None,
    ) -> NativeDirectPushResult:
        """Push exact source page runs into trusted remote destinations.

        Args:
            remote_session: Mooncake destination session ``host:rpc_port``.
            source_plan: P-local pointers, sizes, owners, and producer fences.
            destination_descriptors: HMAC-verified and unexpired descriptors.
            activation: Compatibility contract and ``ARM_WINDOW`` proof.

        Returns:
            Aggregate terminal native transfer result with exact byte counts.

        Raises:
            RuntimeError: If the helper is unqualified, unarmed, or closed.
            ValueError: If source and destination coverage or bytes differ.
            NativeDirectPushTerminalError: If native code is nonzero.
            NativeDirectPushAmbiguousError: If the native call misses its
                deadline; the exception retains the original terminal future.
        """
        activation = validate_direct_push_activation(activation)
        if not remote_session:
            raise ValueError("Native direct push requires a remote session")
        prepared_source = (
            source_plan if isinstance(source_plan, PreparedDirectPushSource) else None
        )
        raw_source_plan = (
            prepared_source.source_plan if prepared_source is not None else source_plan
        )
        if not raw_source_plan.producer_events or any(
            not callable(getattr(event, "synchronize", None))
            for event in raw_source_plan.producer_events
        ):
            raise ValueError("Native direct push requires real producer events")
        vectors = self._build_vectors(
            remote_session,
            raw_source_plan,
            destination_descriptors,
            activation.native_transfer_attempt_id,
        )
        owner_ranges = direct_push_owner_ranges(raw_source_plan.owners)
        validate_direct_push_source_ranges(vectors[0], vectors[2], owner_ranges)

        loop = asyncio.get_running_loop()

        source_event_wait_ms = (
            prepared_source.source_event_wait_ms if prepared_source is not None else 0.0
        )
        source_fences_ready_monotonic = (
            prepared_source.source_fences_ready_monotonic
            if prepared_source is not None
            else 0.0
        )
        source_registration_ms = (
            prepared_source.source_registration_ms
            if prepared_source is not None
            else 0.0
        )

        def prepare_source() -> None:
            nonlocal source_event_wait_ms, source_fences_ready_monotonic
            nonlocal source_registration_ms
            try:
                source_device = next(
                    (
                        owner.device
                        for owner in raw_source_plan.owners
                        if getattr(getattr(owner, "device", None), "type", None)
                        == "npu"
                    ),
                    None,
                )
                if source_device is not None:
                    torch.npu.set_device(source_device)
                event_wait_started = perf_counter()
                seen_events: set[int] = set()
                for event in raw_source_plan.producer_events:
                    if id(event) in seen_events:
                        continue
                    seen_events.add(id(event))
                    event.synchronize()
                source_fences_ready_monotonic = perf_counter()
                source_event_wait_ms = (perf_counter() - event_wait_started) * 1000
                registration_started = perf_counter()
                self._register_source_owners(raw_source_plan.owners)
                source_registration_ms = (perf_counter() - registration_started) * 1000
            except Exception as exc:
                raise NativeDirectPushPreSubmitError(
                    "Native direct push failed before submission"
                ) from exc

        if prepared_source is None:
            await asyncio.to_thread(prepare_source)
        native_slot_wait_ms = 0.0

        def run_native() -> NativeDirectPushResult:
            started = perf_counter()
            native_return = self._transfer_engine.batch_transfer_sync_write(
                remote_session,
                list(vectors[0]),
                list(vectors[1]),
                list(vectors[2]),
            )
            ended = perf_counter()
            if not isinstance(native_return, int) or isinstance(native_return, bool):
                raise RuntimeError(
                    "Mooncake native direct push returned an ambiguous status"
                )
            return NativeDirectPushResult(
                native_transfer_attempt_id=activation.native_transfer_attempt_id,
                return_code=native_return,
                vector_count=len(vectors[0]),
                transferred_bytes=sum(vectors[2]),
                elapsed_ms=(ended - started) * 1000,
                source_event_wait_ms=source_event_wait_ms,
                source_fences_ready_monotonic=source_fences_ready_monotonic,
                source_registration_ms=source_registration_ms,
                native_slot_wait_ms=native_slot_wait_ms,
                native_started_monotonic=started,
                native_ended_monotonic=ended,
            )

        # Producer fences can remain incomplete while chunked prefill is still
        # running. Waiting and idempotent source registration happen before
        # admission to the scarce native-transfer slots, so an unrelated ready
        # request is not serialized behind model computation.
        native_slot_started = perf_counter()
        async with self._task_lock:
            if self._closed:
                raise RuntimeError("Native direct push transport is closed")
            await self._operation_limit.acquire()
            if self._closed:
                self._operation_limit.release()
                raise RuntimeError("Native direct push transport is closed")
            native_slot_wait_ms = (perf_counter() - native_slot_started) * 1000
            try:
                future = loop.run_in_executor(self._executor, run_native)
            except BaseException:
                self._operation_limit.release()
                raise
            self._inflight.add(future)

        def release_operation(done: asyncio.Future[NativeDirectPushResult]) -> None:
            self._inflight.discard(done)
            self._operation_limit.release()

        future.add_done_callback(release_operation)
        try:
            result = await asyncio.wait_for(
                asyncio.shield(future), timeout=self._timeout_seconds
            )
        except asyncio.CancelledError:
            await drain_native_task(future)
            raise
        except asyncio.TimeoutError as exc:
            raise NativeDirectPushAmbiguousError(
                activation.native_transfer_attempt_id,
                future,
            ) from exc
        except NativeDirectPushPreSubmitError:
            raise
        except Exception as exc:
            raise NativeDirectPushAmbiguousError(
                activation.native_transfer_attempt_id,
                future,
            ) from exc
        if result.return_code != 0:
            raise NativeDirectPushTerminalError(result)
        return result

    async def close(self) -> None:
        """Drain native calls and close the dedicated worker pool."""
        self._closed = True
        async with self._task_lock:
            inflight = tuple(self._inflight)
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        self._executor.shutdown(wait=True, cancel_futures=False)

    @staticmethod
    def _build_vectors(
        remote_session: str,
        source_plan: DirectPushSourcePlan,
        destination_descriptors: tuple[DestinationPageDescriptor, ...],
        native_transfer_attempt_id: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if not source_plan.pages or not destination_descriptors:
            raise ValueError("Native direct push requires nonempty pages")
        destinations: dict[tuple[str, int], DestinationPageDescriptor] = {}
        for descriptor in destination_descriptors:
            identity = (descriptor.canonical_key, descriptor.kv_group)
            if identity in destinations:
                raise ValueError("Native direct push destination is duplicated")
            if descriptor.remote_session != remote_session:
                raise ValueError("Native direct push remote session changed")
            if descriptor.native_transfer_attempt_id != native_transfer_attempt_id:
                raise ValueError("Native direct push attempt identifier changed")
            if descriptor.destination_ptr <= 0 or descriptor.destination_length <= 0:
                raise ValueError("Native direct push destination is invalid")
            destinations[identity] = descriptor

        sources: set[tuple[str, int]] = set()
        source_ptrs: list[int] = []
        destination_ptrs: list[int] = []
        lengths: list[int] = []
        for page in source_plan.pages:
            identity = (page.canonical_key, page.kv_group)
            if identity in sources:
                raise ValueError("Native direct push source page is duplicated")
            sources.add(identity)
            descriptor = destinations.get(identity)
            if descriptor is None:
                raise ValueError("Native direct push destination page is missing")
            if not page.source_ptrs or len(page.source_ptrs) != len(
                page.source_lengths
            ):
                raise ValueError("Native direct push source vectors are invalid")
            offset = 0
            for source_ptr, length in zip(
                page.source_ptrs, page.source_lengths, strict=True
            ):
                if source_ptr <= 0 or length <= 0:
                    raise ValueError("Native direct push source extent is invalid")
                source_ptrs.append(source_ptr)
                destination_ptrs.append(descriptor.destination_ptr + offset)
                lengths.append(length)
                offset += length
            if offset != descriptor.destination_length:
                raise ValueError("Native direct push page byte count differs")
        if sources != set(destinations):
            raise ValueError("Native direct push page coverage differs")
        return tuple(source_ptrs), tuple(destination_ptrs), tuple(lengths)

    _validate_source_ranges = staticmethod(validate_direct_push_source_ranges)
