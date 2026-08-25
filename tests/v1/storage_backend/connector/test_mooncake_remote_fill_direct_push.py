# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.remote_fill.protocol import DestinationPageDescriptor
from lmcache.v1.remote_fill.native import (
    DIRECT_PUSH_H0_QUALIFICATION_V1,
    DirectPushPageSource,
    DirectPushSourcePlan,
    NativeDirectPushActivation,
    NativeDirectPushAmbiguousError,
    NativeDirectPushTerminalError,
)
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeDirectPushTransport,
    MooncakestoreConnector,
)


class _Owner:
    def __init__(self, size: int = 64) -> None:
        self.tensor = torch.empty(size, dtype=torch.uint8)
        self.device = self.tensor.device

    def untyped_storage(self):
        return self.tensor.untyped_storage()


class _NPUOwner(_Owner):
    def __init__(self, size: int = 64) -> None:
        super().__init__(size)
        self.device = type("_NPUDevice", (), {"type": "npu"})()


class _ReadyEvent:
    def __init__(self, calls: list[tuple]) -> None:
        self._calls = calls

    def synchronize(self) -> None:
        self._calls.append(("event",))


class _GlobalTE:
    def __init__(self, calls: list[tuple]) -> None:
        self._calls = calls

    def register_buffer(self, ptrs: list[int], sizes: list[int]) -> None:
        self._calls.append(("register", ptrs, sizes))


class _Engine:
    def __init__(
        self,
        calls: list[tuple],
        *,
        return_code: int = 0,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self._calls = calls
        self._return_code = return_code
        self._entered = entered
        self._release = release

    def batch_transfer_sync_write(
        self,
        remote_session: str,
        source_ptrs: list[int],
        destination_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        self._calls.append(
            (
                "native",
                remote_session,
                source_ptrs,
                destination_ptrs,
                lengths,
            )
        )
        if self._entered is not None:
            self._entered.set()
        if self._release is not None:
            assert self._release.wait(timeout=2)
        return self._return_code


def _activation(attempt: str = "attempt-1") -> NativeDirectPushActivation:
    return NativeDirectPushActivation(
        h0_qualification=DIRECT_PUSH_H0_QUALIFICATION_V1,
        native_transfer_attempt_id=attempt,
        arm_acknowledged=True,
    )


def _descriptor(
    *,
    key: str,
    group: int,
    ptr: int,
    length: int,
    attempt: str = "attempt-1",
) -> DestinationPageDescriptor:
    return DestinationPageDescriptor(
        canonical_key=key,
        chunk_index=group,
        remote_session="decoder:1234",
        destination_ptr=ptr,
        destination_length=length,
        reservation_id=f"reservation-{group}",
        window_id=0,
        kv_group=group,
        transfer_id="transfer-1",
        request_attempt=1,
        destination_engine_epoch=1,
        shared_cache_generation=1,
        manifest_digest="manifest",
        native_transfer_attempt_id=attempt,
        expires_at=1000.0,
        capability_mac="verified-before-helper",
    )


def _source_plan(
    owner: _Owner,
    calls: list[tuple],
) -> DirectPushSourcePlan:
    base = owner.untyped_storage().data_ptr()
    return DirectPushSourcePlan(
        pages=(
            DirectPushPageSource(
                canonical_key="page-0",
                kv_group=0,
                source_ptrs=(base, base + 8),
                source_lengths=(8, 12),
            ),
            DirectPushPageSource(
                canonical_key="page-1",
                kv_group=1,
                source_ptrs=(base + 32,),
                source_lengths=(16,),
            ),
        ),
        owners=(owner,),
        producer_events=(_ReadyEvent(calls),),
    )


def _owner_registrar(global_te: _GlobalTE):
    """Build the shared connector registration callback used by direct push."""

    registered: dict[int, int] = {}
    lock = threading.Lock()

    def register(owners: tuple[object, ...]) -> None:
        ranges = {
            int(owner.untyped_storage().data_ptr()): int(
                owner.untyped_storage().nbytes()
            )
            for owner in owners
        }
        with lock:
            pending = {
                ptr: size for ptr, size in ranges.items() if ptr not in registered
            }
            if pending:
                global_te.register_buffer(list(pending), list(pending.values()))
                registered.update(pending)

    return register


def test_instrumented_connector_forwards_remote_fill_direct_push() -> None:
    calls: list[dict[str, object]] = []

    class _Wrapped:
        async def push_external_pages(self, **kwargs):
            calls.append(kwargs)
            return "done"

        def get_remote_fill_destination_session(self):
            return "decoder:1234"

    connector = object.__new__(InstrumentedRemoteConnector)
    connector._connector = _Wrapped()
    kwargs = {
        "remote_session": "decoder:1234",
        "source_plan": object(),
        "destination_descriptors": (),
        "activation": object(),
    }

    result = asyncio.run(connector.push_external_pages(**kwargs))

    assert result == "done"
    assert calls == [kwargs]
    assert connector.get_remote_fill_destination_session() == "decoder:1234"

    mooncake = object.__new__(MooncakestoreConnector)
    mooncake._shared_local_segment = "decoder:5678"
    assert mooncake.get_remote_fill_destination_session() == "decoder:5678"


def test_native_direct_push_requires_explicit_activation() -> None:
    calls: list[tuple] = []
    owner = _Owner()
    transport = MooncakeDirectPushTransport(
        _owner_registrar(_GlobalTE(calls)),
        _Engine(calls),
        worker_count=1,
        max_operations=1,
        timeout_seconds=1,
    )

    async def run() -> None:
        with pytest.raises(RuntimeError, match="explicit activation"):
            await transport.push_external_pages(
                remote_session="decoder:1234",
                source_plan=_source_plan(owner, calls),
                destination_descriptors=(
                    _descriptor(key="page-0", group=0, ptr=0x5000, length=20),
                    _descriptor(key="page-1", group=1, ptr=0x6000, length=16),
                ),
                activation=None,
            )
        await transport.close()

    asyncio.run(run())
    assert calls == []


@pytest.mark.parametrize(
    (
        "qualification",
        "arm_acknowledged",
        "attempt",
        "exception",
        "message",
    ),
    [
        ("invalid", True, "attempt-1", RuntimeError, "compatibility contract"),
        (
            DIRECT_PUSH_H0_QUALIFICATION_V1,
            False,
            "attempt-1",
            RuntimeError,
            "ARM_WINDOW",
        ),
        (
            DIRECT_PUSH_H0_QUALIFICATION_V1,
            True,
            "",
            ValueError,
            "attempt identifier",
        ),
    ],
)
def test_native_direct_push_rejects_invalid_activation_proof(
    qualification: str,
    arm_acknowledged: bool,
    attempt: str,
    exception: type[Exception],
    message: str,
) -> None:
    calls: list[tuple] = []
    transport = MooncakeDirectPushTransport(
        _owner_registrar(_GlobalTE(calls)),
        _Engine(calls),
        worker_count=1,
        max_operations=1,
        timeout_seconds=1,
    )
    activation = NativeDirectPushActivation(
        h0_qualification=qualification,
        native_transfer_attempt_id=attempt,
        arm_acknowledged=arm_acknowledged,
    )

    async def run() -> None:
        with pytest.raises(exception, match=message):
            await transport.push_external_pages(
                remote_session="decoder:1234",
                source_plan=object(),
                destination_descriptors=(),
                activation=activation,
            )
        await transport.close()

    asyncio.run(run())
    assert calls == []


def test_connector_push_orders_event_registration_and_exact_native_vectors() -> None:
    calls: list[tuple] = []
    owner = _Owner()
    connector = object.__new__(MooncakestoreConnector)
    global_te = _GlobalTE(calls)
    connector._shared_global_te = global_te
    connector._ensure_shared_external_owners_registered = _owner_registrar(global_te)
    connector._shared_transfer_engine = _Engine(calls)
    connector._direct_push_transport = None
    connector._direct_push_transport_init_lock = threading.Lock()
    connector._direct_push_worker_count = 1
    connector._direct_push_max_operations = 1
    connector._direct_push_timeout_seconds = 1
    connector._external_put_lock = asyncio.Lock()
    connector._shared_external_buffers = {}
    connector._shared_external_registration_lock = threading.Lock()
    plan = _source_plan(owner, calls)
    plan = DirectPushSourcePlan(
        plan.pages,
        plan.owners,
        (_ReadyEvent(calls), _ReadyEvent(calls)),
    )
    base = owner.untyped_storage().data_ptr()

    async def run() -> None:
        await connector._external_put_lock.acquire()
        try:
            result = await asyncio.wait_for(
                connector.push_external_pages(
                    remote_session="decoder:1234",
                    source_plan=plan,
                    destination_descriptors=(
                        _descriptor(key="page-0", group=0, ptr=0x5000, length=20),
                        _descriptor(key="page-1", group=1, ptr=0x6000, length=16),
                    ),
                    activation=_activation(),
                ),
                timeout=1,
            )
        finally:
            connector._external_put_lock.release()
        assert result.return_code == 0
        assert result.transferred_bytes == 36
        assert result.vector_count == 3
        assert result.source_event_wait_ms >= 0
        assert result.source_registration_ms >= 0
        assert result.native_slot_wait_ms >= 0
        assert result.native_started_monotonic > 0
        assert result.native_ended_monotonic >= result.native_started_monotonic
        await connector._direct_push_transport.close()

    asyncio.run(run())
    assert [call[0] for call in calls] == [
        "event",
        "event",
        "register",
        "native",
    ]
    assert calls[2] == ("register", [base], [owner.untyped_storage().nbytes()])
    assert calls[3] == (
        "native",
        "decoder:1234",
        [base, base + 8, base + 32],
        [0x5000, 0x5008, 0x6000],
        [8, 12, 16],
    )


def test_incomplete_producer_fence_does_not_hold_native_operation_slot() -> None:
    calls: list[tuple] = []
    fence_entered = threading.Event()
    fence_release = threading.Event()
    owner = _Owner()

    class _BlockingEvent:
        @staticmethod
        def synchronize() -> None:
            fence_entered.set()
            assert fence_release.wait(timeout=2)

    slow_plan = _source_plan(owner, calls)
    slow_plan = DirectPushSourcePlan(
        slow_plan.pages,
        slow_plan.owners,
        (_BlockingEvent(),),
    )
    fast_plan = _source_plan(owner, calls)
    transport = MooncakeDirectPushTransport(
        _owner_registrar(_GlobalTE(calls)),
        _Engine(calls),
        worker_count=1,
        max_operations=1,
        timeout_seconds=1,
    )
    descriptors = (
        _descriptor(key="page-0", group=0, ptr=0x5000, length=20),
        _descriptor(key="page-1", group=1, ptr=0x6000, length=16),
    )

    async def run() -> None:
        slow = asyncio.create_task(
            transport.push_external_pages(
                remote_session="decoder:1234",
                source_plan=slow_plan,
                destination_descriptors=descriptors,
                activation=_activation(),
            )
        )
        assert await asyncio.to_thread(fence_entered.wait, 1)
        fast = await transport.push_external_pages(
            remote_session="decoder:1234",
            source_plan=fast_plan,
            destination_descriptors=descriptors,
            activation=_activation(),
        )
        assert fast.return_code == 0
        assert [call[0] for call in calls].count("native") == 1
        fence_release.set()
        assert (await slow).return_code == 0
        await transport.close()

    asyncio.run(run())
    assert [call[0] for call in calls].count("native") == 2


def test_persistent_and_direct_paths_share_one_global_te_registration() -> None:
    """Concurrent sinks register each stable NPU owner exactly once."""

    calls: list[tuple] = []
    owner = _NPUOwner()
    connector = object.__new__(MooncakestoreConnector)
    connector._shared_global_te = _GlobalTE(calls)
    connector._shared_external_buffers = {}
    connector._shared_external_registration_lock = threading.Lock()
    connector._external_buffers = {}
    connector.registered_buffer_ptr = None
    connector.registered_buffer_size = 0
    owners = (owner,)

    operations = [connector._ensure_shared_external_owners_registered] * 16
    operations += [connector._register_external_owners] * 16
    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(lambda operation: operation(owners), operations))

    registrations = [call for call in calls if call[0] == "register"]
    assert len(registrations) == 1
    assert registrations[0][1] == [owner.untyped_storage().data_ptr()]
    assert registrations[0][2] == [owner.untyped_storage().nbytes()]


def test_direct_push_allows_one_group_when_other_group_already_exists() -> None:
    calls: list[tuple] = []
    owner = _Owner()
    full_plan = _source_plan(owner, calls)
    plan = DirectPushSourcePlan(
        pages=(full_plan.pages[1],),
        owners=full_plan.owners,
        producer_events=full_plan.producer_events,
    )
    transport = MooncakeDirectPushTransport(
        _owner_registrar(_GlobalTE(calls)),
        _Engine(calls),
        worker_count=1,
        max_operations=1,
        timeout_seconds=1,
    )

    async def run() -> None:
        result = await transport.push_external_pages(
            remote_session="decoder:1234",
            source_plan=plan,
            destination_descriptors=(
                _descriptor(key="page-1", group=1, ptr=0x6000, length=16),
            ),
            activation=_activation(),
        )
        assert result.transferred_bytes == 16
        await transport.close()

    asyncio.run(run())
    assert calls[-1][0] == "native"


def test_direct_push_rejects_byte_mismatch_before_native_submission() -> None:
    calls: list[tuple] = []
    owner = _Owner()
    transport = MooncakeDirectPushTransport(
        _owner_registrar(_GlobalTE(calls)),
        _Engine(calls),
        worker_count=1,
        max_operations=1,
        timeout_seconds=1,
    )

    async def run() -> None:
        with pytest.raises(ValueError, match="byte count differs"):
            await transport.push_external_pages(
                remote_session="decoder:1234",
                source_plan=_source_plan(owner, calls),
                destination_descriptors=(
                    _descriptor(key="page-0", group=0, ptr=0x5000, length=19),
                    _descriptor(key="page-1", group=1, ptr=0x6000, length=16),
                ),
                activation=_activation(),
            )
        await transport.close()

    asyncio.run(run())
    assert calls == []


def test_direct_push_rejects_missing_producer_fence() -> None:
    calls: list[tuple] = []
    owner = _Owner()
    plan = _source_plan(owner, calls)
    plan = DirectPushSourcePlan(plan.pages, plan.owners, ())
    transport = MooncakeDirectPushTransport(
        _owner_registrar(_GlobalTE(calls)),
        _Engine(calls),
        worker_count=1,
        max_operations=1,
        timeout_seconds=1,
    )

    async def run() -> None:
        with pytest.raises(ValueError, match="real producer events"):
            await transport.push_external_pages(
                remote_session="decoder:1234",
                source_plan=plan,
                destination_descriptors=(
                    _descriptor(key="page-0", group=0, ptr=0x5000, length=20),
                    _descriptor(key="page-1", group=1, ptr=0x6000, length=16),
                ),
                activation=_activation(),
            )
        await transport.close()

    asyncio.run(run())
    assert calls == []


def test_direct_push_surfaces_nonzero_terminal_status() -> None:
    calls: list[tuple] = []
    owner = _Owner()
    transport = MooncakeDirectPushTransport(
        _owner_registrar(_GlobalTE(calls)),
        _Engine(calls, return_code=-7),
        worker_count=1,
        max_operations=1,
        timeout_seconds=1,
    )

    async def run() -> None:
        with pytest.raises(NativeDirectPushTerminalError) as raised:
            await transport.push_external_pages(
                remote_session="decoder:1234",
                source_plan=_source_plan(owner, calls),
                destination_descriptors=(
                    _descriptor(key="page-0", group=0, ptr=0x5000, length=20),
                    _descriptor(key="page-1", group=1, ptr=0x6000, length=16),
                ),
                activation=_activation(),
            )
        assert raised.value.result.return_code == -7
        await transport.close()

    asyncio.run(run())


def test_ambiguous_timeout_retains_source_until_original_call_finishes() -> None:
    calls: list[tuple] = []
    entered = threading.Event()
    release = threading.Event()
    owner = _Owner()
    owner_ref = weakref.ref(owner)
    plan = _source_plan(owner, calls)
    transport = MooncakeDirectPushTransport(
        _owner_registrar(_GlobalTE(calls)),
        _Engine(calls, entered=entered, release=release),
        worker_count=1,
        max_operations=1,
        timeout_seconds=0.01,
    )

    async def run() -> None:
        nonlocal owner, plan
        with pytest.raises(NativeDirectPushAmbiguousError) as raised:
            await transport.push_external_pages(
                remote_session="decoder:1234",
                source_plan=plan,
                destination_descriptors=(
                    _descriptor(key="page-0", group=0, ptr=0x5000, length=20),
                    _descriptor(key="page-1", group=1, ptr=0x6000, length=16),
                ),
                activation=_activation(),
            )
        assert await asyncio.to_thread(entered.wait, 1)
        owner = None  # type: ignore[assignment]
        plan = None  # type: ignore[assignment]
        gc.collect()
        assert owner_ref() is not None
        release.set()
        terminal = await raised.value.wait_for_terminal()
        assert terminal.return_code == 0
        await transport.close()

    asyncio.run(run())
