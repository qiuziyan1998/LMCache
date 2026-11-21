# SPDX-License-Identifier: Apache-2.0
"""Fixed page pools preserve payloads and share capacity across all purposes."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from random import Random
from types import SimpleNamespace
import logging

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.indexer_c8 import IndexerC8Layout
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.page_slot_pool import PageSlotPool, PageSlotSpec
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.shared_cpu_cache import PassiveSharedViewAllocator, SharedChunkHandle
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


def make_allocator(slots: int = 8) -> TensorMemoryAllocator:
    specs = (
        PageSlotSpec((6, 6), 1024, slots),
        PageSlotSpec((256, 130, 256), 1024, slots),
    )
    return TensorMemoryAllocator(
        torch.zeros(sum(s.slots * s.slot_bytes for s in specs), dtype=torch.uint8),
        page_slot_specs=specs,
    )


def allocate(
    owner: TensorMemoryAllocator | LocalCPUBackend,
    group: int,
    counts: list[int],
    purpose: str = "cache",
) -> list | None:
    widths = (6, 6) if group == 0 else (256, 130, 256)
    dtype = torch.bfloat16 if group == 0 else torch.uint8
    return owner.batched_allocate_layer_pages(
        [torch.Size([1024 * width // dtype.itemsize]) for width in widths],
        [dtype] * len(widths),
        len(counts),
        len(widths),
        MemoryFormat.KV_MLA_LATENT_FMT if group == 0 else MemoryFormat.KV_DSA_INDEX_FMT,
        valid_tokens=counts,
        full_tokens=1024,
        allocation_purpose=purpose,
        **({"eviction": False} if isinstance(owner, LocalCPUBackend) else {}),
    )


@pytest.mark.parametrize("group", [0, 1])
def test_partial_pages_keep_exact_layer_bytes_and_use_one_full_slot(group: int) -> None:
    owner = make_allocator()
    counts = [1, 128, 512, 960, 1024]
    pages = allocate(owner, group, counts)
    assert pages is not None
    spec = owner.page_slot_pool.specs[group]
    for page, tokens in zip(pages, counts, strict=True):
        assert page.get_physical_size() == spec.slot_bytes
        assert page.valid_tokens == tokens
        assert page.get_size() == sum(spec.layer_token_bytes) * tokens
        assert page.raw_data.numel() == page.get_size()  # Padding is never a payload.
        offset = 0
        for layer, width in enumerate(spec.layer_token_bytes):
            assert page.layer_size_bytes(layer) == tokens * width
            assert page.layer_data_ptr(layer) == page.data_ptr + offset
            page.layer_tensor(layer).fill_(layer + 1)
            offset += width * tokens
        for layer in range(page.num_layers):
            assert torch.all(page.layer_tensor(layer) == layer + 1)
    snapshot = owner.page_slot_pool.snapshot()[group]
    assert snapshot["partial_slots"] == 4
    assert snapshot["padding_bytes"] == len(pages) * spec.slot_bytes - sum(
        p.get_size() for p in pages
    )
    owner.batched_free(pages)
    assert owner.memcheck()
    assert owner.total_allocated_size == 0


def test_batch_failure_rolls_back_earlier_partial_pages() -> None:
    owner = make_allocator(slots=2)
    assert allocate(owner, 0, [1, 128, 1024]) is None
    assert owner.total_allocated_size == 0
    assert owner.page_slot_pool.snapshot()[0]["partial_slots"] == 0
    assert owner.memcheck()


def test_page_construction_failure_returns_all_reserved_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = make_allocator()

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("object construction failed")

    monkeypatch.setattr("lmcache.v1.memory_management.LayerPageMemoryObj", fail)
    with pytest.raises(RuntimeError, match="construction failed"):
        allocate(owner, 0, [1024, 1024])
    assert owner.total_allocated_size == owner.num_active_allocations == 0
    assert owner.memcheck()


def test_pinned_owner_cannot_be_reused_before_final_unpin() -> None:
    owner = make_allocator(slots=1)
    PinMonitor.GetOrCreate(LMCacheEngineConfig.from_defaults())
    page = allocate(owner, 0, [1])[0]
    page.pin()
    page.ref_count_down()
    assert page.is_valid()
    assert allocate(owner, 0, [1024]) is None
    page.unpin()
    assert not page.is_valid()
    replacement = allocate(owner, 0, [1024])[0]
    replacement.ref_count_down()
    assert owner.memcheck()


def test_shared_capacity_estimate_charges_full_slot_for_short_tail() -> None:
    owner = make_allocator()
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=1024,
        use_layerwise=True,
        remote_url="mooncakestore://test",
        extra_config=dict(
            enable_shared_cpu_cache=True,
            save_only_first_rank=True,
            mooncake_page_first_multi_buffer=True,
            mooncake_layer_merged_page_objects=True,
        ),
    )
    local = LocalCPUBackend(config, memory_allocator=owner)
    engine = object.__new__(LMCacheEngine)
    engine.config = config
    engine.metadata = SimpleNamespace(indexer_c8_layout=None)
    engine.num_layers_for_group = lambda group: 2
    engine._estimate_shared_cpu_bytes_per_layer = lambda group, tokens: tokens * 6
    engine._shared_local_cpu_backend = lambda: local
    assert engine._shared_cpu_estimated_physical_page_bytes(0, 1) == 12288
    assert engine._shared_cpu_estimated_physical_page_bytes(0, 1024) == 12288
    engine._shared_cpu_request_leases = {}
    keys = [
        CacheEngineKey("test", 1, 0, i, torch.bfloat16, kv_group=0) for i in range(8)
    ]
    details = engine._shared_cpu_runtime_capacity_details(
        req_id="r",
        phase="cold",
        kv_group=0,
        keys_layer_major=[
            [key.split_layers(2)[layer] for key in keys] for layer in range(2)
        ],
        chunk_locations_layer_major=[["RemoteBackend"] * 8 for _ in range(2)],
        chunk_token_lengths=[1024] * 7 + [1],
    )
    assert details["required_bytes"] == details["free_bytes"] == 8 * 12288
    assert details["fits"] and details["capacity_scan_skipped"]


def test_group_identity_is_not_guessed_from_small_payload_size() -> None:
    owner = make_allocator()
    large = allocate(owner, 0, [1])[0]
    small = allocate(owner, 1, [1])[0]
    assert large.get_physical_size() == 12288
    assert small.get_physical_size() == 659456
    with pytest.raises(ValueError, match="all-layer"):
        owner.batched_allocate(
            torch.Size([6]), torch.bfloat16, 1, MemoryFormat.KV_MLA_LATENT_FMT
        )
    with pytest.raises(ValueError, match="runtime layout"):
        owner.batched_allocate_layer_pages(
            [torch.Size([3])],
            [torch.bfloat16],
            1,
            1,
            MemoryFormat.KV_MLA_LATENT_FMT,
            valid_tokens=1,
            full_tokens=1,
        )
    large.ref_count_down()
    small.ref_count_down()
    assert owner.memcheck()


def test_concurrent_purposes_share_full_quota_without_overallocating() -> None:
    spec = PageSlotSpec((12,), 1024, 40)
    pool = PageSlotPool(2 * 40 * spec.slot_bytes, (spec, spec))

    def attempt(index: int) -> list:
        try:
            return pool.allocate_pages(
                0, 12, 1, ("cache", "checkpoint", "workspace")[index % 3]
            )
        except MemoryError:
            return []

    with ThreadPoolExecutor(max_workers=12) as executor:
        allocations = [
            block for result in executor.map(attempt, range(100)) for block in result
        ]
    assert len(allocations) == len({address for address, _ in allocations}) == 40
    for purpose in ("cache", "checkpoint", "workspace"):
        with pytest.raises(MemoryError):
            pool.allocate_pages(0, 12, 1, purpose)
        assert pool.shortages({0: 1, 1: 1}, purpose) == {0: 1, 1: 0}
    for block in allocations:
        pool.free(*block)
    assert pool.check_consistency()


@pytest.mark.parametrize("purpose", ["cache", "checkpoint", "workspace"])
def test_every_purpose_can_use_the_last_slot_and_reuse_another_purposes_slot(
    purpose: str,
) -> None:
    owner = make_allocator(slots=3)
    pages = allocate(owner, 0, [1, 512])
    last = allocate(owner, 0, [1024], purpose)[0]
    assert owner.page_slot_pool.snapshot()[0]["free_slots"] == 0
    for label in ("cache", "checkpoint", "workspace"):
        assert allocate(owner, 0, [1], label) is None
    pages[0].ref_count_down()
    reused = allocate(owner, 0, [1], purpose)[0]
    assert reused.metadata.address == pages[0].metadata.address
    owner.batched_free([pages[1], last, reused])
    assert owner.total_allocated_size == 0 and owner.memcheck()


def test_random_release_order_cannot_fragment_either_group() -> None:
    specs = (PageSlotSpec((91_008,), 1024, 20), PageSlotSpec((3616,), 1024, 20))
    pool = PageSlotPool(sum(s.slots * s.slot_bytes for s in specs), specs)
    rng = Random(713)
    live: list[tuple[int, int]] = []
    for _ in range(1500):
        if live and rng.random() < 0.5:
            pool.free(*live.pop(rng.randrange(len(live))))
        else:
            group = rng.randrange(2)
            if pool.snapshot()[group]["free_slots"]:
                live.extend(
                    pool.allocate_pages(
                        group,
                        specs[group].layer_token_bytes[0] * rng.randint(1, 1024),
                        1,
                        "cache",
                    )
                )
    for address, size in live:
        pool.free(address, size)
    assert pool.check_consistency()
    assert pool.total_allocated_size == 0


def test_reclaim_targets_missing_group_and_preserves_borrowed_and_pinned_owners() -> (
    None
):
    owner = make_allocator(slots=3)
    config = LMCacheEngineConfig.from_defaults(local_cpu=True)
    PinMonitor.GetOrCreate(config)
    backend = LocalCPUBackend(config, memory_allocator=owner)
    pages = allocate(owner, 0, [1, 512, 1024])
    index = allocate(owner, 1, [1])[0]
    keys = [
        CacheEngineKey("test", 1, 0, i, torch.bfloat16, kv_group=0) for i in range(3)
    ]
    index_key = CacheEngineKey("test", 1, 0, 9, torch.uint8, kv_group=1)
    backend.batched_submit_layer_pages(keys, pages)
    backend.batched_submit_layer_pages([index_key], [index])
    index.ref_count_down()
    pages[0].ref_count_down()
    pages[1].pin()
    pages[1].ref_count_down()
    # pages[2] keeps a borrow; only page[0] can be reclaimed.
    assert not backend.reclaim_layer_page_slots({0: 2})
    assert all(backend.contains(key) for key in keys)
    assert backend.reclaim_layer_page_slots({0: 1})
    assert not backend.contains(keys[0]) and backend.contains(index_key)
    assert pages[1].is_valid() and pages[2].is_valid()
    pages[1].unpin()
    pages[2].ref_count_down()
    backend.clear()
    assert owner.total_allocated_size == 0 and owner.memcheck()


def test_checkpoint_reclaim_reports_entries_checked_and_stops_at_first_match(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = make_allocator(slots=2)
    backend = LocalCPUBackend(
        LMCacheEngineConfig.from_defaults(local_cpu=True), memory_allocator=owner
    )
    logger = logging.getLogger("lmcache.v1.storage_backend.local_cpu_backend")
    monkeypatch.setattr(logger, "handlers", [caplog.handler])
    caplog.set_level(logging.INFO, logger=logger.name)
    keys = [
        CacheEngineKey("test", 1, 0, 0, torch.bfloat16, kv_group=0),
        CacheEngineKey("test", 1, 0, 1, torch.uint8, kv_group=1),
        CacheEngineKey("test", 1, 0, 2, torch.uint8, kv_group=1),
    ]
    latent = allocate(owner, 0, [1])
    index = allocate(owner, 1, [1, 1])
    assert latent is not None and index is not None
    pages = latent + index
    backend.batched_submit_layer_pages(keys, pages)
    try:
        for page in pages:
            page.ref_count_down()
        assert not backend.reclaim_layer_page_slots(
            {1: 1}, allocation_purpose="checkpoint", max_scan_entries=1
        )
        assert all(backend.contains(key) for key in keys)
        assert backend.reclaim_layer_page_slots(
            {1: 1}, allocation_purpose="checkpoint", max_scan_entries=3
        )
        assert [backend.contains(key) for key in keys] == [True, False, True]
        logs = [
            line for line in caplog.messages if "reclaim purpose=checkpoint" in line
        ]
        assert len(logs) == 2
        assert "scanned=1 scan_ms=" in logs[0]
        assert "scanned=2 scan_ms=" in logs[1]
    finally:
        backend.clear()
    assert owner.total_allocated_size == 0 and owner.memcheck()


def test_partial_page_passive_handles_map_only_effective_layer_data() -> None:
    owner = make_allocator()
    page = allocate(owner, 1, [7])[0]
    passive = PassiveSharedViewAllocator(
        slab_tensor=owner.buffer, shm_name="fixed-slots", generation=1
    )
    for layer in range(3):
        page.layer_tensor(layer).fill_(layer + 1)
        key = CacheEngineKey("test", 1, 0, 1, torch.uint8, kv_group=1).split_layers(3)[
            layer
        ]
        handle = SharedChunkHandle.from_memory_obj(
            memory_obj=page,
            request_id="r",
            phase="retrieve",
            key=key,
            layer_id=layer,
            kv_group=1,
            chunk_index=0,
            shm_name="fixed-slots",
            generation=1,
            producer_rank=0,
        )
        assert (
            handle.physical_size == handle.logical_size == page.layer_size_bytes(layer)
        )
        view = passive.create_view(
            handle,
            expected_request_id="r",
            expected_phase="retrieve",
            expected_layer_id=layer,
            expected_kv_group=1,
            expected_chunk_index=0,
            expected_key=key,
            expected_shape=page.get_shapes()[layer],
            expected_dtype=torch.uint8,
            expected_fmt=MemoryFormat.KV_DSA_INDEX_FMT,
        )
        assert torch.all(view.tensor == layer + 1)
        view.ref_count_down()
    page.ref_count_down()
    assert owner.memcheck()


@pytest.fixture
def runtime_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LMCacheEngineConfig, LMCacheMetadata, list[tuple[int, dict]]]:
    # Replace only the pinned allocator constructor; execute real backend setup.
    recorded = []
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.local_cpu_backend.MixedMemoryAllocator",
        lambda size, **kw: recorded.append((size, kw)) or SimpleNamespace(),
    )
    metadata = LMCacheMetadata(
        "test",
        1,
        1,
        0,
        0,
        torch.bfloat16,
        (79, 1, 1024, 1, 576),
        use_mla=True,
        chunk_size=1024,
        runtime_kv_group_layer_counts=(79, 22),
        indexer_c8_layout=IndexerC8Layout(
            c8_layers=(False,) * 3 + (True,) * 16 + (False,) * 3
        ),
    )
    extra = dict(
        enable_shared_cpu_cache=True,
        save_only_first_rank=True,
        mooncake_page_first_multi_buffer=True,
        mooncake_layer_merged_page_objects=True,
    )
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=1024,
        max_local_cpu_size=250,
        reserve_local_cpu_size=0,
        use_layerwise=True,
        dsa_two_groups=True,
        decode_preemption_checkpoint=True,
        remote_url="mooncakestore://test",
        extra_config=extra,
    )
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.local_cpu_backend.SystemMemoryDetector.get_available_memory_gb",
        lambda: 300,
    )
    return config, metadata, recorded


@pytest.mark.parametrize("retired_reserves", [False, True])
def test_runtime_config_derives_incident_sizes_and_checks_slab_before_allocation(
    runtime_backend: tuple[LMCacheEngineConfig, LMCacheMetadata, list],
    retired_reserves: bool,
) -> None:
    config, metadata, recorded = runtime_backend
    config.extra_config["local_cpu.fixed_slot_counts"] = [2760, 2760]
    if retired_reserves:
        config.extra_config.update(
            {
                "local_cpu.checkpoint_slot_reserve": [89, 89],
                "local_cpu.workspace_slot_reserve": [12, 12],
            }
        )
    LocalCPUBackend(config, metadata)
    specs = recorded[0][1]["page_slot_specs"]
    assert [s.slot_bytes for s in specs] == [93_192_192, 3_702_784]
    assert [s.slots for s in specs] == [2760, 2760]
    pool = PageSlotPool(recorded[0][0], specs)
    for purpose in ("cache", "checkpoint", "workspace"):
        assert pool.shortages({0: 2760, 1: 2760}, purpose) == {0: 0, 1: 0}
    assert sum(s.slots * s.slot_bytes for s in specs) < 250 * 2**30
    config.extra_config["local_cpu.fixed_slot_counts"] = [3000, 3000]
    with pytest.raises(ValueError, match="effective CPU slab"):
        LocalCPUBackend(config, metadata)
    assert len(recorded) == 1


@pytest.mark.parametrize(
    "configured,available,reserve,first_rank,expected_gib,expected_slots",
    [
        (250, 300, 0, None, 250, 2770),
        (128, 300, 0, None, 128, 1418),
        (250, 120, 20, None, 100, 1108),
        (250, 300, 0, 120, 120, 1329),
    ],
)
def test_automatic_counts_use_effective_slab_capacity(
    runtime_backend: tuple[LMCacheEngineConfig, LMCacheMetadata, list],
    monkeypatch: pytest.MonkeyPatch,
    configured: int,
    available: int,
    reserve: int,
    first_rank: int | None,
    expected_gib: int,
    expected_slots: int,
) -> None:
    config, metadata, recorded = runtime_backend
    config.max_local_cpu_size = configured
    config.reserve_local_cpu_size = reserve
    if first_rank is not None:
        config.extra_config["first_rank_max_local_cpu_size"] = first_rank
    monkeypatch.setattr(
        "lmcache.v1.storage_backend.local_cpu_backend.SystemMemoryDetector.get_available_memory_gb",
        lambda: available,
    )
    LocalCPUBackend(config, metadata)
    slab_bytes, kwargs = recorded[0]
    assert slab_bytes == expected_gib * 2**30
    specs = kwargs["page_slot_specs"]
    assert [s.slots for s in specs] == [expected_slots, expected_slots]
    pool = PageSlotPool(slab_bytes, specs)
    assert 0 <= pool.unassigned_bytes < sum(s.slot_bytes for s in specs)
    for purpose in ("cache", "checkpoint", "workspace"):
        assert pool.shortages({0: expected_slots, 1: expected_slots}, purpose) == {
            0: 0,
            1: 0,
        }
    assert "local_cpu.fixed_slot_counts" not in config.extra_config


@pytest.mark.parametrize(
    "chunk_tokens,alignment,expected_sizes,expected_slots",
    [
        (512, 4096, [46_596_096, 1_851_392], 5540),
        (1024, 2**21, [94_371_840, 4_194_304], 2723),
    ],
)
def test_automatic_counts_follow_runtime_chunk_and_physical_alignment(
    runtime_backend: tuple[LMCacheEngineConfig, LMCacheMetadata, list],
    chunk_tokens: int,
    alignment: int,
    expected_sizes: list[int],
    expected_slots: int,
) -> None:
    config, metadata, recorded = runtime_backend
    config.chunk_size = chunk_tokens
    config.extra_config["local_cpu.pinned_align_bytes"] = alignment
    LocalCPUBackend(config, metadata)
    specs = recorded[0][1]["page_slot_specs"]
    assert [s.slot_bytes for s in specs] == expected_sizes
    assert [s.slots for s in specs] == [expected_slots, expected_slots]


@pytest.mark.parametrize(
    "slab_bytes,expected_slots",
    [
        (96_894_975, 0),
        (96_894_976, 1),
        (193_789_951, 1),
    ],
)
def test_automatic_counts_require_a_complete_group_pair_before_allocating_slab(
    runtime_backend: tuple[LMCacheEngineConfig, LMCacheMetadata, list],
    slab_bytes: int,
    expected_slots: int,
) -> None:
    config, metadata, recorded = runtime_backend
    config.max_local_cpu_size = slab_bytes / 2**30
    if not expected_slots:
        with pytest.raises(
            ValueError, match="cannot fit one fixed page slot per group"
        ):
            LocalCPUBackend(config, metadata)
        assert not recorded
    else:
        LocalCPUBackend(config, metadata)
        assert [s.slots for s in recorded[0][1]["page_slot_specs"]] == [1, 1]


@pytest.mark.parametrize("compatible", [False, True])
def test_automatic_mode_preserves_legacy_allocator_when_disabled_or_inapplicable(
    runtime_backend: tuple[LMCacheEngineConfig, LMCacheMetadata, list],
    compatible: bool,
) -> None:
    config, metadata, recorded = runtime_backend
    config.use_layerwise = compatible
    if compatible:
        config.extra_config["local_cpu.fixed_slots_enabled"] = False
    LocalCPUBackend(config, metadata)
    assert recorded[0][1]["page_slot_specs"] is None


@pytest.mark.parametrize(
    "extra",
    [
        {"local_cpu.fixed_slots_enabled": "false"},
        {"local_cpu.fixed_slots_enabled": False, "local_cpu.fixed_slot_counts": [1, 1]},
        {"local_cpu.fixed_slot_counts": [1]},
        {"local_cpu.fixed_slot_counts": [0, 1]},
        {"local_cpu.fixed_slot_counts": [True, 1]},
        {"local_cpu.fixed_slot_counts": [1.5, 1]},
    ],
)
def test_invalid_slot_settings_fail_before_slab_allocation(
    runtime_backend: tuple[LMCacheEngineConfig, LMCacheMetadata, list],
    extra: dict,
) -> None:
    config, metadata, recorded = runtime_backend
    config.extra_config.update(extra)
    with pytest.raises(ValueError):
        LocalCPUBackend(config, metadata)
    assert not recorded


def test_explicit_enable_requires_a_compatible_layer_page_backend(
    runtime_backend: tuple[LMCacheEngineConfig, LMCacheMetadata, list],
) -> None:
    config, metadata, recorded = runtime_backend
    config.use_layerwise = False
    config.extra_config["local_cpu.fixed_slots_enabled"] = True
    with pytest.raises(ValueError, match="shared MLA/DSA Mooncake layer pages"):
        LocalCPUBackend(config, metadata)
    assert not recorded
