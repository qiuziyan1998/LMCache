# SPDX-License-Identifier: Apache-2.0
"""CPU regressions through the full production retrieve_layer entry point.

Only storage/device services are replaced. In particular, do not bypass the
shared-retrieve wrapper: it must relay bank commands to either TP rank path.
"""

# Standard
from collections.abc import Callable, Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import ast
import logging

# Third Party
import pytest
import torch


ENGINE_PATH = Path(__file__).resolve().parents[2] / "lmcache/v1/cache_engine.py"


class MaterializationError(RuntimeError):
    pass


class Key:
    def split_layers(self, count: int) -> list[int]:
        return list(range(count))


@pytest.fixture(scope="module")
def engine_type() -> type:
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LMCacheEngine"
    )
    names = {"retrieve_layer", "_remote_fill_recompute_result"}
    cls.body = [node for node in cls.body if getattr(node, "name", None) in names]
    assert {node.name for node in cls.body} == names
    cls.bases = []
    cls.decorator_list = []
    namespace = dict(
        torch=torch,
        logger=logging.getLogger(__name__),
        CacheEngineKey=Key,
        _RemoteFillMaterializationError=MaterializationError,
        _lmcache_nvtx_annotate=lambda fn: fn,
        serving_perf_enabled=lambda: False,
        mooncake_page_layout_enabled=lambda config: False,
        log_remote_fill_diagnostic=lambda *args, **kwargs: None,
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(ENGINE_PATH), "exec"), namespace
    )
    return namespace[cls.name]


def make_engine(
    engine_type: type,
    passive: bool,
    retrieve: Callable[..., Generator],
    *,
    layers: int = 4,
    planned: bool = True,
) -> Any:
    engine = engine_type()
    engine.is_healthy = lambda: True
    engine._num_transfer_layers_for_call = lambda group, kwargs: layers
    engine.num_layers_for_group = lambda group: layers
    engine._should_use_shared_layerwise_retrieve = lambda group: True
    engine._is_passive = lambda: passive
    engine.storage_manager = object()
    engine.gpu_connector = object()
    engine.config = SimpleNamespace(chunk_size=4)
    engine.shared_cpu_cache_strict = True
    engine.stats_monitor = Mock()
    engine._get_req_id = lambda kwargs: kwargs["req_id"]
    engine._dense_retrieve_token_results = lambda *args: iter([(0, 4, Key())])
    engine._remote_fill_retrieve_plan = lambda *args: (
        [("LocalCPUBackend", False)] if planned else None
    )
    engine._find_shared_rank0_chunk_location = lambda key: "LocalCPUBackend"
    engine._remote_fill_pair_lookup_enabled = lambda: planned
    engine._remote_fill_local_full_hint = lambda configs: 4 if planned else None
    engine.lookup_unpin = Mock()
    method = (
        "_retrieve_layer_shared_passive" if passive else "_retrieve_layer_shared_rank0"
    )
    setattr(engine, method, retrieve)
    return engine


def start(engine: Any, group: int = 0) -> Generator:
    return engine.retrieve_layer(
        list(range(4)),
        req_id="bank-relay",
        kv_group=group,
        slot_mapping=torch.arange(4),
        deferred_layerwise_get=True,
    )


@pytest.mark.parametrize("passive", [False, True])
@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("layers", [3, 4, 78])
def test_shared_retrieve_relays_banks_without_changing_load_timing(
    engine_type: type, passive: bool, group: int, planned: bool, layers: int
) -> None:
    """A 78-layer run must not leave odd layers reading stale layer-77 history."""
    banks = torch.empty(8, dtype=torch.int64)
    for bank in range(2):
        banks[bank * 4 : (bank + 1) * 4] = max(range(bank, layers, 2))
    received, submitted, closed, released = [], [], [], []

    def retrieve(**kwargs: Any) -> Generator:
        try:
            mask = kwargs["ret_mask"]
            mask.fill_(True)
            command = yield mask.sum()
            for layer in range(layers):
                received.append(command)
                mapping = (
                    kwargs["kwargs"]["slot_mapping"]
                    if command is None
                    else command["slot_mapping"]
                )
                banks[mapping] = layer
                submitted.append(layer)
                command = yield None
            released.append(group)
            yield mask
        finally:
            closed.append(group)

    engine = make_engine(engine_type, passive, retrieve, layers=layers, planned=planned)
    gen = start(engine, group)
    assert next(gen).item() == 4
    assert submitted == []  # Prime only; no eager layer submissions.
    commands = []
    wrong_layers = []
    for layer in range(layers):
        command = (
            None if layer == 0 else {"slot_mapping": torch.arange(4) + (layer % 2) * 4}
        )
        commands.append(command)
        assert gen.send(command) is None
        assert submitted == list(range(layer + 1))
        if not torch.all(banks[(layer % 2) * 4 : (layer % 2 + 1) * 4] == layer):
            wrong_layers.append(layer)
        assert released == []  # Preserve the deferred final-source lifetime.
    assert next(gen).all()
    assert released == [group]
    with pytest.raises(StopIteration):
        next(gen)
    assert closed == [group]
    assert wrong_layers == [], f"History loaded into wrong banks: {wrong_layers}"
    assert all(
        actual is expected for actual, expected in zip(received, commands, strict=True)
    )


@pytest.mark.parametrize("passive", [False, True])
def test_plain_next_preserves_result_count_and_values(
    engine_type: type, passive: bool
) -> None:
    received = []
    values = [4, None, None, None, None, torch.ones(4, dtype=torch.bool)]

    def retrieve(**kwargs: Any) -> Generator:
        for value in values:
            received.append((yield value))

    actual = list(start(make_engine(engine_type, passive, retrieve)))
    assert len(actual) == len(values)
    assert all(left is right for left, right in zip(actual, values, strict=True))
    assert received == [None] * len(values)


@pytest.mark.parametrize("passive", [False, True])
@pytest.mark.parametrize("steps", [1, 3, 6])
def test_close_immediately_closes_inner_even_if_another_reference_exists(
    engine_type: type, passive: bool, steps: int
) -> None:
    closed = []

    def inner() -> Generator:
        try:
            for _ in range(6):
                yield None
        finally:
            closed.append(True)

    held_reference = inner()
    gen = start(make_engine(engine_type, passive, lambda **kwargs: held_reference))
    for _ in range(steps):
        next(gen)
    gen.close()
    gen.close()
    assert closed == [True]
    assert held_reference.gi_frame is None


@pytest.mark.parametrize("passive", [False, True])
def test_throw_reaches_inner_and_next_send_is_preserved(
    engine_type: type, passive: bool
) -> None:
    closed, received = [], []
    error = ValueError("injected consumer failure")

    def retrieve(**kwargs: Any) -> Generator:
        try:
            try:
                yield "ready"
            except ValueError as exc:
                assert exc is error
                received.append((yield "handled"))
        finally:
            closed.append(True)

    gen = start(make_engine(engine_type, passive, retrieve))
    assert next(gen) == "ready"
    assert gen.throw(error) == "handled"
    command = {"slot_mapping": torch.arange(4) + 4}
    with pytest.raises(StopIteration):
        gen.send(command)
    assert received[0] is command
    assert closed == [True]


@pytest.mark.parametrize("passive", [False, True])
@pytest.mark.parametrize("yielded_before_error", [0, 1, 3, 5])
def test_remote_fill_failure_keeps_protocol_count_and_recompute_mask(
    engine_type: type, passive: bool, yielded_before_error: int
) -> None:
    closed = []

    def retrieve(**kwargs: Any) -> Generator:
        try:
            for _ in range(yielded_before_error):
                yield None
            raise MaterializationError("simulated materialization failure")
        finally:
            closed.append(True)

    engine = make_engine(engine_type, passive, retrieve)
    results = list(start(engine, 1))
    assert len(results) == 6  # Four layers, completion gate, final mask.
    assert results[:-1] == [None] * 5
    assert torch.equal(results[-1], torch.zeros(4, dtype=torch.bool))
    assert closed == [True]
    engine.stats_monitor.on_retrieve_finished.assert_called_once()


@pytest.mark.parametrize("passive", [False, True])
@pytest.mark.parametrize("planned,yielded_before_error", [(False, 2), (True, 6)])
def test_unrecoverable_remote_fill_failure_is_not_swallowed(
    engine_type: type, passive: bool, planned: bool, yielded_before_error: int
) -> None:
    closed = []
    error = MaterializationError("must propagate")

    def retrieve(**kwargs: Any) -> Generator:
        try:
            for _ in range(yielded_before_error):
                yield None
            raise error
        finally:
            closed.append(True)

    gen = start(make_engine(engine_type, passive, retrieve, planned=planned))
    for _ in range(yielded_before_error):
        next(gen)
    with pytest.raises(MaterializationError) as raised:
        next(gen)
    assert raised.value is error
    assert closed == [True]
