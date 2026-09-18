# SPDX-License-Identifier: Apache-2.0
"""Execute both shared-retrieve generators with CPU storage/stream doubles."""

# Standard
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import ast
import logging

# Third Party
import pytest
import torch


PATH = Path(__file__).resolve().parents[2] / "lmcache/v1/cache_engine.py"


class Page:
    pass


@pytest.fixture(scope="module")
def engine_type():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheEngine"
    )
    names = {
        "_retrieve_layer_shared_rank0",
        "_retrieve_layer_shared_passive",
        "_prepare_shared_prefill_sources",
        "_submit_prepared_shared_prefill_layers",
    }
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    cls.bases, cls.decorator_list = [], []
    ns = dict(
        torch=torch,
        logger=logging.getLogger(__name__),
        assert_layerwise_gpu_connector=lambda _: None,
        serving_perf_enabled=lambda: False,
        mooncake_page_layout_enabled=lambda config: config.pages,
        mooncake_layer_pages_enabled=lambda config: config.pages,
        LayerPageMemoryObj=Page,
        LayerCacheEngineKey=type("LayerKey", (), {}),
        SharedHandleEnvelope=NS,
        LayerPageSource=lambda pages, layer, suffix: NS(
            pages=pages, layer_id=layer, suffix=suffix
        ),
    )
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(unit), str(PATH), "exec"), ns)
    return ns[cls.name]


def build(engine_type, layers, group, *, pages=False):
    engine = engine_type()
    events, wire, copies = [], [], []
    sources = [[object(), object()] for _ in range(layers)]
    if pages:
        sources = [[Page(), Page()]] * layers
    engine.config = NS(pages=pages)
    engine.metadata = NS(first_rank=0, worker_id=0)
    engine.shared_cpu_cache_generation = 1
    engine.num_layers_for_group = lambda g: layers
    engine._remote_fill_pair_lookup_enabled = lambda: False
    engine._is_shared_page_first_location_plan = lambda _: False
    engine.storage_manager = object()
    engine.stats_monitor = Mock()
    engine._adopt_dense_shared_retrieve_cache = Mock(return_value=False)
    engine._release_shared_retrieve_objs = lambda objs, **kw: (
        events.append(("release", len(objs))),
        objs.clear(),
    )
    engine._release_retained_dense_retrieve_objs = engine._release_shared_retrieve_objs
    engine._retain_unsafe_layerwise_retrieve_objs = lambda objs, **kw: events.append(
        ("retain_unsafe", len(objs))
    )
    engine._close_shared_retrieve_consumer = lambda consumer: consumer.close()
    engine._shared_layerwise_error_envelope = lambda **kw: NS(status="error", **kw)

    def resolve(**kw):
        events.append(("resolve", kw["layer_id"]))
        return sources[kw["layer_id"]]

    engine._resolve_shared_rank0_layer_mem_objs = resolve
    engine._resolve_shared_rank0_layer_pages = lambda **kw: (sources, 2)
    engine._make_shared_handle_batch = lambda *a, **kw: NS(num_chunks=2)
    engine._make_shared_handles_for_layer = lambda **kw: kw["mem_objs_layer"]
    engine._broadcast_shared_envelope = lambda envelope: (
        events.append(("broadcast", envelope.layer_id)),
        wire.append(envelope),
    )
    engine._receive_matching_shared_envelope = lambda **kw: (
        events.append(("receive", kw["layer_id"])),
        wire.pop(0),
    )[1]
    engine._validate_shared_layerwise_envelope = lambda envelope, **kw: None
    engine._make_passive_layer_page_views = lambda *a, **kw: tuple(sources[0])
    engine._expected_shared_cpu_chunk_metadata = lambda **kw: (
        (kw["num_tokens"],),
        "bf16",
        "latent",
    )
    engine.shared_cpu_cache_passive_allocator = NS(
        create_view=lambda obj, **kw: (
            events.append(("view", kw["expected_layer_id"])),
            obj,
        )[1]
    )

    def prepare(rows, host, device, *, kv_group):
        assert len(rows) == layers and kv_group == group
        events.append(("pointer_table", len(rows)))
        host[:] = [[100 * layer, 100 * layer + 1] for layer in range(layers)]
        device[:] = [torch.tensor(row) for row in host]

    def consume(starts, ends, **kwargs):
        assert starts == [0, 4] and ends == [4, 5]  # include a partial tail
        try:
            command = yield None
            for layer in range(layers):
                if kwargs["deferred_layerwise_get"]:
                    assert len(kwargs["cached_chunk_ptrs_npu"]) == layers
                copies.append((layer, command))
                events.append(("copy", layer))
                command = yield None
            events.append(("sync_load", group))
            yield None
        finally:
            events.append(("close_load", group))

    engine.gpu_connector = NS(
        append_sparse_chunk_ptr_cache_for_layers=prepare, batched_to_gpu=consume
    )
    return engine, events, wire, copies


def retrieve(
    engine, group, layers, *, passive=False, pages=False, deferred=True, kwargs=None
):
    kwargs = {} if kwargs is None else kwargs
    kwargs["deferred_layerwise_get"] = deferred
    common = dict(
        keys_layer_major=[[object(), object()] for _ in range(layers)],
        ret_mask=torch.zeros(5, dtype=torch.bool)
        if passive
        else torch.ones(5, dtype=torch.bool),
        req_id="request",
        monitor_req_id=1,
        kv_group=group,
        kwargs=kwargs,
    )
    if passive:
        return engine._retrieve_layer_shared_passive(
            starts_all=[0, 4], ends_all=[4, 5], **common
        )
    return engine._retrieve_layer_shared_rank0(
        starts=[0, 4],
        ends=[4, 5],
        chunk_locations_layer_major=[["LocalCPUBackend"] * 2 for _ in range(layers)],
        location="LocalCPUBackend",
        planned_page_chunks=2 if pages else 0,
        **common,
    )


@pytest.mark.parametrize("group,layers", [(0, 1), (0, 4), (0, 79), (1, 22)])
@pytest.mark.parametrize("pages", [False, True])
def test_all_metadata_prepared_before_first_yield_bank_commands_stay_lazy(
    engine_type, group, layers, pages
):
    rank0, events0, wire, copies0 = build(engine_type, layers, group, pages=pages)
    passive, events1, _, copies1 = build(engine_type, layers, group, pages=pages)
    passive._receive_matching_shared_envelope = lambda **kw: (
        events1.append(("receive", kw["layer_id"])),
        wire.pop(0),
    )[1]
    host, device = [], []
    gen0 = retrieve(
        rank0,
        group,
        layers,
        pages=pages,
        kwargs=dict(cached_chunk_dev_ptrs=host, cached_chunk_ptrs_npu=device),
    )
    gen1 = retrieve(passive, group, layers, passive=True, pages=pages)
    assert next(gen0).item() == next(gen1).item() == 5
    assert copies0 == copies1 == []
    assert len(host) == len(device) == layers  # caller's lists, not replacements
    assert not wire
    for events in (events0, events1):
        assert events[-1] == ("pointer_table", layers)
        events.clear()
    for layer in range(layers):
        command = {"slot_mapping": torch.arange(5) + 5 * (layer % 2)}
        assert gen0.send(command) is gen1.send(command) is None
        assert copies0[-1][1]["layer_request"] is command
        assert copies1[-1][1]["layer_request"] is command
        assert events0 == events1 == [("copy", row) for row in range(layer + 1)]
    # No sources released and no CPU wait until the existing final drain.
    assert next(gen0).all() and next(gen1).all()
    assert ("sync_load", group) in events0 and ("sync_load", group) in events1
    gen0.close()
    gen1.close()


def test_legacy_non_p_path_still_resolves_one_layer_at_a_time(engine_type):
    engine, events, _, copies = build(engine_type, 4, 0)
    gen = retrieve(engine, 0, 4, deferred=False)
    next(gen)
    assert [item for item in events if item[0] == "resolve"] == [("resolve", 0)]
    assert copies == []
    next(gen)
    assert len(copies) == 1
    assert [item for item in events if item[0] == "resolve"] == [
        ("resolve", 0),
        ("resolve", 1),
    ]
    gen.close()


@pytest.mark.parametrize("fail_stage", ["resolve", "pointer_table"])
def test_preparation_error_releases_objects_without_launching_payload(
    engine_type, fail_stage
):
    engine, events, wire, copies = build(engine_type, 4, 0)
    if fail_stage == "resolve":
        original = engine._resolve_shared_rank0_layer_mem_objs

        def resolve(**kw):
            if kw["layer_id"] == 2:
                raise ValueError("injected failure")
            return original(**kw)

        engine._resolve_shared_rank0_layer_mem_objs = resolve
    else:
        engine.gpu_connector.append_sparse_chunk_ptr_cache_for_layers = Mock(
            side_effect=ValueError("injected failure")
        )
    with pytest.raises(ValueError, match="injected failure"):
        next(retrieve(engine, 0, 4))
    assert copies == []
    assert ("release", 4 if fail_stage == "resolve" else 8) in events
    assert not any(event[0] == "retain_unsafe" for event in events)
    if fail_stage == "resolve":
        assert wire[-1].status == "error"


def test_closing_after_submission_fences_before_releasing(engine_type):
    engine, events, _, copies = build(engine_type, 4, 0)
    gen = retrieve(engine, 0, 4)
    next(gen)
    gen.send({"slot_mapping": torch.arange(5)})
    assert len(copies) == 1
    events.clear()
    gen.close()
    assert events == [("close_load", 0), ("release", 8)]


def test_empty_group_broadcasts_all_skips_before_first_yield(engine_type):
    rank0, events0, wire, copies0 = build(engine_type, 4, 0)
    passive, events1, _, copies1 = build(engine_type, 4, 0)
    passive._receive_matching_shared_envelope = lambda **kw: wire.pop(0)
    common = dict(
        keys_layer_major=[],
        ret_mask=torch.zeros(5, dtype=torch.bool),
        req_id="empty",
        monitor_req_id=1,
        kv_group=0,
        kwargs={"deferred_layerwise_get": True},
    )
    gen0 = rank0._retrieve_layer_shared_rank0(
        starts=[], ends=[], chunk_locations_layer_major=[], location=None, **common
    )
    gen1 = passive._retrieve_layer_shared_passive(starts_all=[], ends_all=[], **common)
    assert next(gen0) is None
    assert len(wire) == 4  # the passive rank must not wait for forward callbacks
    assert next(gen1).item() == 0
    assert wire == []
    assert len(list(gen0)) == len(list(gen1)) == 5
    assert copies0 == copies1 == []
    assert not any(event[0] == "pointer_table" for event in events0 + events1)
