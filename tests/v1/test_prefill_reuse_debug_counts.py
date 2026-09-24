# SPDX-License-Identifier: Apache-2.0
"""Public layerwise loads report actual reuse with bounded scalar summaries."""

# Standard
from collections import deque
from types import SimpleNamespace
from typing import Any, Iterator

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1 import cache_engine as engine_module
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import LayerPageSource, MemoryFormat
from lmcache.v1.shared_cpu_cache import (
    PassiveSharedViewAllocator,
    SharedChunkHandle,
    SharedHandleBatch,
    SharedHandleEnvelope,
)


def key_for(chunk: int, group: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="model",
        world_size=8,
        worker_id=0,
        chunk_hash=100 + chunk,
        dtype=torch.float16,
        kv_group=group,
    )


class TokenPlan:
    """Observe whether the real plan adapter supplies tokens or shared hashes."""

    def __init__(self) -> None:
        self.sources: list[str] = []

    def process_tokens(self, **kwargs: Any) -> Iterator:
        group = kwargs["kv_group"]
        if "hashes" in kwargs:
            self.sources.append("g0")
            offsets = kwargs["offsets"]
        else:
            self.sources.append("tok")
            total = len(kwargs["tokens"])
            offsets = [min(4, total - start) for start in range(0, total, 4)]
        start = 0
        for chunk, size in enumerate(offsets):
            yield start, start + size, key_for(chunk, group)
            start += size


class HostConsumer:
    """Append only supplied host sources without device work or synchronization."""

    def supports_dense_sparse_cache_retention(self) -> bool:
        return True

    def prepare_layerwise_prefill_source_pointers(
        self,
        sources: list,
        host_rows: list,
        device_rows: list,
        **kwargs: Any,
    ) -> None:
        if not host_rows:
            host_rows.extend([] for _ in sources)
        for layer, source in enumerate(sources):
            if isinstance(source, LayerPageSource):
                host_rows[layer].extend(
                    page.layer_data_ptr(layer) for page in source.pages
                )
                host_rows[layer].extend(obj.data_ptr for obj in source.suffix)
            else:
                host_rows[layer].extend(obj.data_ptr for obj in source)
        device_rows[:] = [None] * len(sources)

    def batched_to_gpu(self, *args: Any, **kwargs: Any) -> Iterator:
        while True:
            yield None

    def synchronize_dense_load_stream(self) -> None:
        pytest.fail("Successful diagnostics must not synchronize")


class PassiveEngine(LMCacheEngine):
    """Use real public retrieval, allocator, reuse logic and request ownership."""

    def __init__(self) -> None:
        self.config = SimpleNamespace()
        self.num_layers = 2
        self.metadata = SimpleNamespace(
            first_rank=0, worker_id=1, is_first_rank=lambda: False
        )
        self.gpu_connector = HostConsumer()
        self.token_database = TokenPlan()
        self.shared_cpu_cache_generation = 7
        self.shared_cpu_cache_passive_allocator = PassiveSharedViewAllocator(
            reuse_prefill=True,
            slab_tensor=torch.zeros(4096, dtype=torch.uint8),
            shm_name="/reuse-debug",
            generation=7,
        )
        self.stats_monitor = SimpleNamespace(
            on_retrieve_request=lambda _: 1,
            on_retrieve_finished=lambda *_: None,
        )
        self._shared_cpu_request_leases = {}
        self.incoming: deque[SharedHandleEnvelope] = deque()

    def is_healthy(self) -> bool:
        return True

    def _is_passive(self) -> bool:
        return True

    def _should_use_shared_layerwise_retrieve(self, kv_group: int) -> bool:
        return True

    def _remote_fill_pair_lookup_enabled(self) -> bool:
        return False

    def _expected_shared_cpu_chunk_metadata(
        self,
        *,
        kv_group: int,
        num_tokens: int,
    ) -> tuple[torch.Size, torch.dtype, MemoryFormat]:
        return torch.Size([num_tokens]), torch.float16, MemoryFormat.KV_MLA_LATENT_FMT

    def _receive_shared_envelope(self) -> SharedHandleEnvelope:
        return self.incoming.popleft()

    def queue_prefix(self, total: int, group: int, *, compact: bool) -> None:
        chunks = (total + 3) // 4
        base = group * 1024
        batch = SharedHandleBatch(
            shm_name="/reuse-debug",
            producer_rank=0,
            num_layers=2,
            num_chunks=chunks,
            physical_sizes=[8] + [16] * (chunks - 1),
            chunk_hashes=[100 + index for index in range(chunks)],
            offsets=[
                base + 128 + layer * 128 + chunk * 16
                for layer in range(2)
                for chunk in range(chunks - 1)
            ],
            page_offsets=[base],
            page_physical_sizes=[16],
        )
        for layer in range(1 if compact else 2):
            handles = []
            if not compact:
                for chunk in range(chunks):
                    start, end = chunk * 4, min(total, chunk * 4 + 4)
                    handles.append(
                        SharedChunkHandle(
                            request_id="request-for-reuse-debug",
                            phase="dense_prefix",
                            key=key_for(chunk, group).get_layer(layer),
                            layer_id=layer,
                            kv_group=group,
                            chunk_index=chunk,
                            shm_name="/reuse-debug",
                            offset=base + layer * 128 + chunk * 16,
                            physical_size=16,
                            logical_size=(end - start) * 2,
                            shape=torch.Size([end - start]),
                            dtype=torch.float16,
                            fmt=MemoryFormat.KV_MLA_LATENT_FMT,
                            generation=7,
                            producer_rank=0,
                            cached_positions=list(range(start, end)),
                        )
                    )
            self.incoming.append(
                SharedHandleEnvelope(
                    request_id="request-for-reuse-debug",
                    phase="dense_prefix",
                    request_ordinal=0,
                    layer_id=layer,
                    kv_group=group,
                    status="ok",
                    generation=7,
                    handles=handles,
                    batch=batch if compact else None,
                )
            )


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_summary_counts_actual_prefix_reuse_and_replaced_tail(
    monkeypatch: pytest.MonkeyPatch,
    compact: bool,
    enabled: bool,
) -> None:
    monkeypatch.setattr(engine_module, "assert_layerwise_gpu_connector", lambda _: None)
    # These concise diagnostics work without either verbose timing mode.
    monkeypatch.setattr(engine_module, "prefill_start_timing_enabled", lambda: False)
    monkeypatch.setattr(engine_module, "serving_perf_enabled", lambda: False)
    monkeypatch.setattr(
        engine_module, "prefill_reuse_debug_enabled", lambda rank: enabled and rank == 1
    )
    records: list[tuple[int, str, dict[str, Any]]] = []

    def record(rank: int, stage: str, **fields: Any) -> None:
        records.append((rank, stage, fields))

    monkeypatch.setattr(engine_module, "prefill_reuse_debug_log", record)
    engine = PassiveEngine()
    caches = [
        {
            name: []
            for name in (
                "cached_keys",
                "cached_starts",
                "cached_ends",
                "cached_memory_objs",
                "cached_chunk_dev_ptrs",
                "cached_chunk_ptrs_npu",
                "cached_shared_handles",
            )
        }
        for _ in range(2)
    ]
    all_sources = []
    for step, total in enumerate((6, 8, 10, 10)):
        shared_plan: dict = {}
        for group in (0, 1):
            engine.queue_prefix(total, group, compact=compact)
            results = list(
                engine.retrieve_layer(
                    list(range(total)),
                    req_id="request-for-reuse-debug",
                    kv_group=group,
                    _retain_shared_dense_cache=True,
                    _reuse_shared_dense_prefix=True,
                    deferred_layerwise_get=True,
                    prefill_dma_block_ids_by_bank=((0,), (1,)),
                    shared_cpu_request_preflight_state=shared_plan,
                    **caches[group],
                )
            )
            assert results[-1].tolist() == [True] * total
            assert not engine.incoming
            all_sources.extend(
                obj for row in caches[group]["cached_memory_objs"] for obj in row
            )
            if not enabled:
                assert records == []
                continue
            assert len(records) == (step * 2 + group + 1) * 2
            rank, stage, plan = records[-2]
            assert (rank, stage, plan["p"], plan["g"]) == (1, "plan", total, group)
            assert plan["h"] == ("tok" if group == 0 else "g0")
            assert plan["k"] == ((total + 3) // 4) * 2
            assert isinstance(plan["ms"], float) and plan["ms"] >= 0
            rank, stage, sources = records[-1]
            assert (rank, stage, sources["p"], sources["g"]) == (1, "src", total, group)
            assert sources["c"] == int(compact)
            assert sources["v"] == (
                ("0/1" if step == 0 else "1/0") if compact else "0/0"
            )
            assert (
                sources["t"]
                == (
                    ("0/2", "0/2", "2/2", "4/0")
                    if compact
                    else ("0/4", "2/2", "4/2", "6/0")
                )[step]
            )
            assert sources["x"] == ("0/2", "1/1", "2/1", "3/0")[step]
            assert isinstance(sources["w"], float) and sources["w"] >= 0
            assert isinstance(sources["ms"], float) and sources["ms"] >= 0
    assert engine.token_database.sources == ["tok", "g0"] * 4
    engine.release_shared_cpu_sparse_request("request-for-reuse-debug")
    assert all(not obj.is_valid() for obj in all_sources)
