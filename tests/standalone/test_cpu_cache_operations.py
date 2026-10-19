# SPDX-License-Identifier: Apache-2.0
"""Actual CPU source operations without constructing a device cache engine."""

# Standard
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import gc
import subprocess
import sys

# Third Party
import pytest
import torch
from msgspec.structs import replace

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.rpc.storage_protocol import GroupLayout, LayoutSpec
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


def load_operations():
    """Load the production module directly, avoiding engine bootstrap imports."""
    source = Path(__file__).resolve().parents[2] / "lmcache/v1/cpu_cache_operations.py"
    spec = spec_from_file_location("cpu_cache_operations_test", source)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


operations = load_operations()


class CPUAllocator:
    """CPU tensor storage with the same owner metadata as a registered slab."""

    def __init__(self):
        self.buffer = torch.zeros(65536, dtype=torch.uint8)
        self.pin_allocator = TensorMemoryAllocator(self.buffer)
        self.shm_name = "cpu-test-slab"

    def __getattr__(self, name):
        return getattr(self.pin_allocator, name)


class CPUStorage(operations.CPUCacheOperations):
    """Independent CPU owner using the production LocalCPU allocator paths."""

    def __init__(self):
        self.config = LMCacheEngineConfig.from_defaults(chunk_size=4)
        self.config.local_cpu = True
        self.config.dsa_two_groups = True
        PinMonitor.GetOrCreate(self.config)
        self.layout = LayoutSpec(
            "model",
            "cache",
            "builtin",
            "int",
            64,
            4,
            4,
            (
                GroupLayout(
                    0,
                    ("latent0", "latent1"),
                    ("bfloat16",) * 2,
                    ((4,), (4,)),
                    int(MemoryFormat.KV_MLA_LATENT_FMT.value),
                ),
                GroupLayout(
                    1,
                    ("index0", "index1", "index2"),
                    ("uint8",) * 3,
                    ((2,), (1, 1), (4,)),
                    int(MemoryFormat.KV_DSA_INDEX_FMT.value),
                ),
            ),
        )
        self.layout.validate()
        self.metadata = SimpleNamespace(
            worker_id=0,
            use_mla=True,
            get_dtypes=lambda: [torch.bfloat16, torch.uint8],
        )
        self.num_layers = 2
        self.dsa_two_groups = True
        self._transfer_layer_counts = (2, 3)
        self.lookup_pins = defaultdict(lambda: defaultdict(list))
        self._remote_fill_lookup_plans = {}
        self._shared_cpu_request_leases = {}
        self.allocator = CPUAllocator()
        self.shared_cpu_cache_name = self.allocator.shm_name
        self.local = LocalCPUBackend(self.config, memory_allocator=self.allocator)
        self.storage_manager = SimpleNamespace(
            local_cpu_backend=self.local,
            storage_backends={"LocalCPUBackend": self.local},
            batched_unpin=lambda keys, locations: self.local.batched_unpin(keys),
        )

    def key(self, index, group, tokens):
        return CacheEngineKey(
            "model",
            4,
            0,
            index,
            self._shared_cpu_dtype_for_kv_group(group),
            {"lmcache.tag.internal.valid_tokens": str(tokens)} if tokens < 4 else None,
            kv_group=group,
        )

    def page(self, group, tokens):
        geometry = [
            self._expected_shared_cpu_chunk_metadata(
                kv_group=group,
                num_tokens=tokens,
                layer_id=layer,
            )
            for layer in range(self.num_layers_for_group(group))
        ]
        pages = self.allocator.batched_allocate_layer_pages(
            [shape for shape, _, _ in geometry],
            [dtype for _, dtype, _ in geometry],
            1,
            len(geometry),
            geometry[0][2],
            valid_tokens=tokens,
            full_tokens=tokens,
        )
        assert pages is not None
        page = pages[0]
        for layer in range(page.num_layers):
            page.layer_tensor(layer).fill_(group * 10 + layer + 1)
        return page


def test_source_module_imports_without_an_engine_or_npu():
    source = Path(__file__).resolve().parents[2] / "lmcache/v1/cpu_cache_operations.py"
    script = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('cpu_operations', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'lmcache.v1.cache_engine' not in sys.modules
assert 'lmcache_ascend.v1.cache_engine' not in sys.modules
assert 'torch_npu' not in sys.modules
assert 'vllm' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(source)], check=True)


def test_registered_geometry_covers_unequal_groups_and_mixed_storage():
    owner = CPUStorage()
    try:
        assert [owner.num_layers_for_group(i) for i in (0, 1)] == [2, 3]
        assert owner._expected_shared_cpu_chunk_metadata(
            kv_group=0,
            num_tokens=3,
        ) == (torch.Size([6]), torch.bfloat16, MemoryFormat.KV_MLA_LATENT_FMT)
        assert [
            owner._expected_shared_cpu_chunk_metadata(
                kv_group=1,
                num_tokens=3,
                layer_id=layer,
            )[0].numel()
            for layer in range(3)
        ] == [6, 6, 12]
        owner.layout = None
        with pytest.raises(ValueError, match="not been registered"):
            owner._expected_shared_cpu_chunk_metadata(kv_group=0, num_tokens=4)
    finally:
        owner.local.close()


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("missing_remote", [False, True])
def test_exact_partial_page_resolution_and_gc_off_cleanup(group, missing_remote):
    owner = CPUStorage()
    was_enabled = gc.isenabled()
    gc.disable()
    first, last = owner.page(group, 4), owner.page(group, 1)
    keys = [owner.key(7, group, 4), owner.key(8, group, 1)]
    owner.local.batched_submit_layer_pages(keys[:1], [first])
    first.ref_count_down()  # The cache now owns the first allocation.

    def retrieve(requested):
        assert requested == keys[1:]
        if missing_remote:
            return []
        last.ref_count_up()  # Reproduce a backend's returned caller ownership.
        return [last]

    owner.storage_manager.storage_backends["RemoteBackend"] = SimpleNamespace(
        batched_get_layer_pages=retrieve,
    )
    arguments = dict(
        req_id="request",
        phase="cpu-prepare",
        kv_group=group,
        keys_layer_major=[
            [key.split_layers(owner.num_layers_for_group(group))[layer] for key in keys]
            for layer in range(owner.num_layers_for_group(group))
        ],
        page_chunks=2,
        base_page_keys=keys,
        exact_chunk_locations=["LocalCPUBackend", "RemoteBackend"],
    )
    try:
        if missing_remote:
            with pytest.raises(ValueError, match="result count"):
                owner._resolve_shared_rank0_layer_pages(**arguments)
            assert first.get_ref_count() == 1
            assert not first.is_pinned
        else:
            rows, page_count = owner._resolve_shared_rank0_layer_pages(**arguments)
            assert page_count == 2
            assert len(rows) == owner.num_layers_for_group(group)
            assert all(row[0] is first and row[1] is last for row in rows)
            assert last.valid_tokens == 1
            assert last.layer_tensor(0).numel() == 2
            assert last.layer_tensor(len(rows) - 1).flatten()[0] == group * 10 + len(
                rows
            )
            owner._release_shared_retrieve_objs(list(rows[0]), unpin=True)
            assert first.get_ref_count() == 1
        assert last.get_ref_count() == 1
        assert not last.is_pinned
    finally:
        last.ref_count_down()
        owner.local.clear()
        assert owner.allocator.total_allocated_size == 0
        owner.local.close()
        if was_enabled:
            gc.enable()


def test_source_layout_rejects_fractional_dtype_bytes():
    owner = CPUStorage()
    try:
        first = owner.layout.groups[0]
        owner.layout = replace(
            owner.layout,
            groups=(replace(first, plane_token_bytes=((3,), (4,))),)
            + owner.layout.groups[1:],
        )
        with pytest.raises(ValueError, match="not dtype-aligned"):
            owner._expected_shared_cpu_chunk_metadata(kv_group=0, num_tokens=1)
    finally:
        owner.local.close()
