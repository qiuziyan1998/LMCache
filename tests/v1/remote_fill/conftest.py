# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the hardware-independent remote-fill protocol."""

# Standard
from dataclasses import dataclass, field
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.remote_fill import (
    PROTOCOL_VERSION,
    ArmWindowRequest,
    ControlPage,
    FinishRequest,
    InProcessRemoteFillTransport,
    NegotiationSpec,
    NegotiateRequest,
    OpenRequest,
    OperationIdentity,
    PageDisposition,
    PreparedPage,
    ProtocolLimits,
    RemoteFillClient,
    RemoteFillService,
    RemoteFillStateCore,
    ReportTransferCompleteRequest,
    ReserveWindowRequest,
    ReservedPageView,
    StatusRequest,
    content_digest,
    manifest_digest,
    transaction_manifest_digest,
)


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Advance the clock by ``seconds``."""

        self.now += seconds


@dataclass
class FakePageLifecycle:
    """Opaque, deliberately noncontiguous allocator used by public tests."""

    commit_result: bool = True
    fail_release: bool = False
    next_address: int = 0x100000
    prepared_override: tuple[PreparedPage, ...] | None = None
    existing_keys: set[str] = field(default_factory=set)
    missing_keys: set[str] = field(default_factory=set)
    released: list[tuple[tuple[Any, ...], str]] = field(default_factory=list)
    raw_released: list[tuple[tuple[Any, ...], str]] = field(default_factory=list)
    committed: list[tuple[tuple[str, ...], tuple[Any, ...]]] = field(
        default_factory=list
    )
    prepare_calls: int = 0
    commit_calls: int = 0

    def prepare_pages(
        self,
        transfer_id: str,
        window_id: int,
        pages: tuple[ControlPage, ...],
        reserve_missing: bool,
    ) -> tuple[PreparedPage, ...]:
        """Snapshot existing pages and allocate absent pages with gaps."""

        del transfer_id, window_id
        self.prepare_calls += 1
        if self.prepared_override is not None:
            return self.prepared_override
        allocations = []
        for index, page in enumerate(pages):
            if page.canonical_key in self.existing_keys:
                allocations.append(
                    PreparedPage(handle=None, disposition=PageDisposition.EXISTING)
                )
                continue
            if page.canonical_key in self.missing_keys or not reserve_missing:
                allocations.append(
                    PreparedPage(handle=None, disposition=PageDisposition.MISSING)
                )
                continue
            base = self.next_address + index * 0x100000
            ptr = base + 0x1000
            allocations.append(
                PreparedPage(
                    handle=(page.kv_group, page.chunk_index, ptr),
                    destination_ptr=ptr,
                    destination_length=page.expected_bytes,
                    reservation_base=base,
                    reservation_length=page.expected_bytes + 0x2000,
                )
            )
        self.next_address += len(pages) * 0x200000
        return tuple(allocations)

    def release_prepared_pages(
        self,
        pages: tuple[PreparedPage, ...],
        reason: str,
    ) -> None:
        """Release raw invalid or incomplete allocation results."""

        if self.fail_release:
            raise RuntimeError("injected release failure")
        self.raw_released.append((tuple(page.handle for page in pages), reason))

    def release_pages(
        self,
        pages: tuple[ReservedPageView, ...],
        reason: str,
    ) -> None:
        """Record exactly-once release by opaque handle."""

        if self.fail_release:
            raise RuntimeError("injected release failure")
        self.released.append((tuple(page.prepared.handle for page in pages), reason))

    def commit_pages(
        self,
        transfer_id: str,
        required_pages: tuple[ControlPage, ...],
        pages: tuple[ReservedPageView, ...],
        finish: FinishRequest,
    ) -> bool:
        """Record a mock admission without pretending to own LocalCPU."""

        del transfer_id, finish
        self.commit_calls += 1
        self.committed.append(
            (
                tuple(page.canonical_key for page in required_pages),
                tuple(page.prepared.handle for page in pages),
            )
        )
        return self.commit_result


class RequestFactory:
    """Construct consistent unsealed requests for client-side sealing."""

    def __init__(self, transfer_id: str = "transfer-1") -> None:
        self.transfer_id = transfer_id
        self.sequence = 0
        self.manifest_digest_seed = content_digest("seed")
        self.window_manifests: dict[int, str] = {}

    def common(self, operation_id: str | None = None) -> OperationIdentity:
        """Create the next common operation identity."""

        self.sequence += 1
        return OperationIdentity(
            protocol_version=PROTOCOL_VERSION,
            operation_id=operation_id or f"op-{self.sequence}",
            operation_sequence=self.sequence,
            payload_digest="",
            transfer_id=self.transfer_id,
            request_attempt=1,
            destination_engine_epoch=7,
            shared_cache_generation=11,
        )

    def open(self, operation_id: str | None = None) -> OpenRequest:
        """Create an OPEN request."""

        return OpenRequest(
            common=self.common(operation_id),
            request_id="request-1",
            source_engine_id="prefiller",
            destination_engine_id="decoder",
            destination_dp_rank=0,
            required_store_end_hint=8192,
            planned_window_count_hint=2,
            cache_namespace_tag="namespace-v3",
            layout_tag="layout-v3",
            model_artifact_id="model-build-1",
            manifest_digest_seed=self.manifest_digest_seed,
        )

    def negotiate(
        self,
        operation_id: str | None = None,
        *,
        token_hash_algorithm: str = "builtin",
        python_hash_seed: str = "0",
    ) -> NegotiateRequest:
        """Create a static layout negotiation request."""

        return NegotiateRequest(
            common=self.common(operation_id),
            cache_namespace_tag="namespace-v3",
            layout_tag="layout-v3",
            model_artifact_id="model-build-1",
            chunk_size=1024,
            model_layout="mla-dsa",
            group_dimensions=(576, 128),
            layer_count=79,
            save_only_first_rank=True,
            shared_group1=True,
            tp_size=8,
            dp_size=2,
            global_te_push=True,
            token_hash_algorithm=token_hash_algorithm,
            python_hash_seed=python_hash_seed,
        )

    def pages(
        self,
        chunk_count: int = 1,
        *,
        start_chunk: int = 0,
    ) -> tuple[ControlPage, ...]:
        """Create complete two-group pages for ``chunk_count`` chunks."""

        return tuple(
            ControlPage(
                canonical_key=f"key-g{group}-c{chunk}",
                kv_group=group,
                chunk_index=chunk,
                chunk_start=chunk * 1024,
                chunk_end=(chunk + 1) * 1024,
                valid_tokens=1024,
                destination_tp_rank=0,
                expected_bytes=4096 + group * 1024,
                layer_count=79,
                layout_tag="layout-v3",
            )
            for chunk in range(start_chunk, start_chunk + chunk_count)
            for group in (0, 1)
        )

    def reserve(
        self,
        pages: tuple[ControlPage, ...] | None = None,
        *,
        window_id: int = 0,
        operation_id: str | None = None,
        source_generation: int = 3,
        reserve_missing: bool = True,
    ) -> ReserveWindowRequest:
        """Create a RESERVE_WINDOW request."""

        selected = pages or self.pages()
        digest = manifest_digest(selected)
        self.window_manifests[window_id] = digest
        return ReserveWindowRequest(
            common=self.common(operation_id),
            window_id=window_id,
            source_generation=source_generation,
            manifest_digest=digest,
            control_pages=selected,
            reserve_missing=reserve_missing,
        )

    def arm(
        self,
        reserve: ReserveWindowRequest,
        reserve_response: Any,
        operation_id: str | None = None,
    ) -> ArmWindowRequest:
        """Create an ARM_WINDOW request from the reservation evidence."""

        return ArmWindowRequest(
            common=self.common(operation_id),
            window_id=reserve.window_id,
            native_transfer_attempt_id=(reserve_response.native_transfer_attempt_id),
            manifest_digest=reserve.manifest_digest,
            destination_descriptor_digest=(
                reserve_response.destination_descriptor_digest
            ),
        )

    def report(
        self,
        reserve: ReserveWindowRequest,
        reserve_response: Any,
        *,
        return_code: int = 0,
        completed_bytes: int | None = None,
        operation_id: str | None = None,
    ) -> ReportTransferCompleteRequest:
        """Create a terminal native completion report."""

        expected = sum(
            descriptor.destination_length for descriptor in reserve_response.descriptors
        )
        return ReportTransferCompleteRequest(
            common=self.common(operation_id),
            window_id=reserve.window_id,
            native_transfer_attempt_id=(reserve_response.native_transfer_attempt_id),
            native_return_code=return_code,
            completed_bytes=expected if completed_bytes is None else completed_bytes,
            manifest_digest=reserve.manifest_digest,
        )

    def finish(
        self,
        *,
        required_store_end: int = 1024,
        persistent_common_end: int | None = None,
        final_partial_valid_tokens: int = 0,
        operation_id: str | None = None,
    ) -> FinishRequest:
        """Create a FINISH request."""

        return FinishRequest(
            common=self.common(operation_id),
            required_store_end=required_store_end,
            persistent_common_end=(
                required_store_end
                if persistent_common_end is None
                else persistent_common_end
            ),
            final_manifest_digest=transaction_manifest_digest(
                self.manifest_digest_seed,
                tuple(self.window_manifests.items()),
                required_store_end,
                final_partial_valid_tokens,
            ),
            final_partial_valid_tokens=final_partial_valid_tokens,
        )

    def status(
        self,
        window_id: int = -1,
        operation_id: str | None = None,
    ) -> StatusRequest:
        """Create a STATUS request."""

        return StatusRequest(
            common=self.common(operation_id),
            window_id=window_id,
        )


@dataclass
class Harness:
    """Connected mock client, service, state, and page lifecycle."""

    client: RemoteFillClient
    transport: InProcessRemoteFillTransport
    service: RemoteFillService
    state: RemoteFillStateCore
    lifecycle: FakePageLifecycle
    clock: FakeClock
    requests: RequestFactory
    negotiation: NegotiationSpec


@pytest.fixture
def harness() -> Harness:
    """Create a fully connected deterministic remote-fill harness."""

    limits = ProtocolLimits()
    clock = FakeClock()
    lifecycle = FakePageLifecycle()
    negotiation = NegotiationSpec(
        cache_namespace_tag="namespace-v3",
        layout_tag="layout-v3",
        model_artifact_id="model-build-1",
        chunk_size=1024,
        model_layout="mla-dsa",
        group_dimensions=(576, 128),
        layer_count=79,
        save_only_first_rank=True,
        shared_group1=True,
        tp_size=8,
        dp_size=2,
        global_te_push=True,
        destination_engine_id="decoder",
        destination_dp_rank=0,
        destination_remote_session="decoder-host:19001",
        token_hash_algorithm="builtin",
        python_hash_seed="0",
    )
    state = RemoteFillStateCore(
        destination_engine_epoch=7,
        shared_cache_generation=11,
        descriptor_verification_key=b"test descriptor verification key",
        negotiation=negotiation,
        page_lifecycle=lifecycle,
        limits=limits,
        reservation_ttl_sec=5.0,
        descriptor_ttl_sec=2.0,
        native_hard_timeout_sec=20.0,
        terminal_record_ttl_sec=10.0,
        clock=clock,
    )
    service = RemoteFillService(state)
    transport = InProcessRemoteFillTransport(service)
    return Harness(
        client=RemoteFillClient(transport),
        transport=transport,
        service=service,
        state=state,
        lifecycle=lifecycle,
        clock=clock,
        requests=RequestFactory(),
        negotiation=negotiation,
    )
