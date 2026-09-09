# SPDX-License-Identifier: Apache-2.0
"""Publication call counts and live validation using real slab-backed objects."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import MemoryFormat, TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.shared_cpu_cache import SharedChunkHandle


@pytest.fixture
def publication(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    monitor = PinMonitor(
        SimpleNamespace(pin_check_interval_sec=3600, pin_timeout_sec=3600)
    )
    monitor.stop_monitoring()
    monkeypatch.setattr(PinMonitor, "GetOrCreate", lambda config=None: monitor)
    allocator = TensorMemoryAllocator(torch.empty(4096, dtype=torch.uint8), 64)
    engine = object.__new__(LMCacheEngine)
    engine.shared_cpu_cache_name = "publication-slab"
    engine.shared_cpu_cache_generation = 11
    engine.config = SimpleNamespace(dsa_two_groups=True)
    engine.dsa_two_groups = True
    engine.fmt = MemoryFormat.KV_2LTD
    engine.metadata = LMCacheMetadata(
        "model", 2, 2, 0, 0, torch.bfloat16, (3, 1, 8, 1, 1), use_mla=True
    )
    slab = SimpleNamespace(
        shm_name=engine.shared_cpu_cache_name,
        pin_allocator=allocator,
        buffer=allocator.buffer,
    )
    engine.storage_manager = SimpleNamespace(
        local_cpu_backend=SimpleNamespace(memory_allocator=slab)
    )
    objects = []
    keys = []
    with monitor.protect_pins() as pins:
        for index in range(17):
            tokens = 3 if index == 16 else 8
            obj = allocator.allocate(
                torch.Size([tokens]), torch.bfloat16, MemoryFormat.KV_MLA_LATENT_FMT
            )
            assert obj is not None
            obj.metadata.cached_positions = torch.arange(index * 8, index * 8 + tokens)
            obj.pin()
            pins.append(obj)
            objects.append(obj)
            keys.append(
                CacheEngineKey("model", 2, 0, index, torch.bfloat16).get_layer(2)
            )
    yield SimpleNamespace(
        engine=engine, slab=slab, objects=objects, keys=keys, monitor=monitor
    )
    for obj in objects:
        monitor.release_pin_lease(obj)
        obj.ref_count_down()
    assert allocator.num_active_allocations == 0
    assert monitor.get_monitored_count() == 0


def _build(publication: SimpleNamespace, **kwargs: Any) -> list[SharedChunkHandle]:
    arguments = dict(
        req_id="request",
        phase="load",
        keys_layer=publication.keys,
        mem_objs_layer=publication.objects,
        layer_id=2,
        kv_group=0,
        chunk_index_base=5,
    )
    arguments.update(kwargs)
    # Private entry point is necessary to measure the publication hot path.
    return publication.engine._make_shared_handles_for_layer(**arguments)


@pytest.mark.parametrize("group", [0, 1])
def test_publication_context_once_per_batch_keeps_every_validation(
    publication: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, group: int
) -> None:
    engine = publication.engine
    spies = {}
    for name in (
        "_shared_rank0_object_context",
        "_shared_local_cpu_backend",
        "_shared_cpu_dtype_for_kv_group",
        "_memory_format_for_kv_group",
        "_validate_rank0_shared_mem_obj",
    ):
        spies[name] = Mock(wraps=getattr(engine, name))
        monkeypatch.setattr(engine, name, spies[name])
    with monkeypatch.context() as mutation:
        keys = [replace(key, kv_group=group) for key in publication.keys]
        if group == 1:
            for obj in publication.objects:
                mutation.setattr(obj.metadata, "fmt", MemoryFormat.KV_DSA_INDEX_FMT)
        expected = [
            SharedChunkHandle.from_memory_obj(
                request_id="request",
                phase="load",
                key=key,
                layer_id=2,
                kv_group=group,
                chunk_index=5 + index,
                shm_name=engine.shared_cpu_cache_name,
                memory_obj=obj,
                generation=11,
                producer_rank=0,
            )
            for index, (key, obj) in enumerate(
                zip(keys, publication.objects, strict=True)
            )
        ]
        for _ in range(2):
            assert _build(publication, kv_group=group, keys_layer=keys) == expected
    for name, spy in spies.items():
        assert spy.call_count == (34 if name == "_validate_rank0_shared_mem_obj" else 2)
    assert all(
        (obj.get_ref_count(), obj.metadata.pin_count) == (1, 1)
        for obj in publication.objects
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("parent_allocator", None, "does not belong"),
        ("dtype", None, "dtype-less"),
        ("dtype", torch.float32, "dtype does not match"),
        ("fmt", MemoryFormat.UNDEFINED, "undefined format"),
        ("fmt", MemoryFormat.KV_2LTD, "format does not match"),
        ("address", -1, "invalid slab bounds"),
        ("address", 4096, "invalid slab bounds"),
        ("phy_size", 0, "invalid slab bounds"),
        ("phy_size", 1, "invalid slab bounds"),
        ("get_size", lambda: 0, "invalid slab bounds"),
        ("pin_count", 0, "must be pinned"),
    ],
)
def test_later_object_mutation_is_validated_not_cached(
    publication: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    match: str,
) -> None:
    engine = publication.engine
    validate = engine._validate_rank0_shared_mem_obj
    checked = []
    with monkeypatch.context() as mutation:

        def validate_then_mutate(obj: Any, **kwargs: Any) -> None:
            checked.append(obj)
            validate(obj, **kwargs)
            if obj is publication.objects[0]:
                target = publication.objects[1]
                if field not in ("parent_allocator", "get_size"):
                    target = target.metadata
                mutation.setattr(target, field, value)

        mutation.setattr(engine, "_validate_rank0_shared_mem_obj", validate_then_mutate)
        with pytest.raises(ValueError, match=match) as error:
            _build(publication)
        assert "chunk_index=6" in str(error.value)
    assert checked == publication.objects[:2]
    assert all(obj.get_ref_count() == 1 for obj in publication.objects)


@pytest.mark.parametrize("changed", ["name", "parent", "size", "dtype", "format"])
def test_context_is_refreshed_after_slab_or_group_mutation(
    publication: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    _build(publication)
    with monkeypatch.context() as mutation:
        if changed == "name":
            mutation.setattr(publication.slab, "shm_name", "another-slab")
        elif changed == "parent":
            mutation.setattr(publication.slab, "pin_allocator", object())
        elif changed == "size":
            mutation.setattr(publication.slab, "buffer", torch.empty(1))
        elif changed == "dtype":
            mutation.setattr(publication.engine.metadata, "kv_dtype", torch.float32)
        else:
            mutation.setattr(publication.engine.metadata, "use_mla", False)
        with pytest.raises(ValueError, match="Shared CPU cache"):
            _build(publication)
    assert len(_build(publication)) == 17


def test_empty_partial_and_prevalidated_batches_do_not_resolve_context(
    publication: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = Mock(side_effect=AssertionError("unexpected context resolution"))
    monkeypatch.setattr(publication.engine, "_shared_rank0_object_context", context)
    assert _build(publication, keys_layer=[], mem_objs_layer=[]) == []
    with pytest.raises(ValueError, match="partial layer handles"):
        _build(publication, keys_layer=[])
    assert len(_build(publication, validate_memory_objs=False)) == 17
    context.assert_not_called()
