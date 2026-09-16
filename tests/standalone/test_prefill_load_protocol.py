# SPDX-License-Identifier: Apache-2.0
"""CPU execution of the real prefill callbacks without vLLM/NPU imports.

The transfer generators model asynchronous bank writes and source ownership.
This checks callback/row/bank ordering, not device kernel or HCCL correctness.
Run with --confcutdir=tests/standalone to avoid GPU conftest initialization.
"""

# Standard
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import ast
import logging

# Third Party
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "lmcache/integration/vllm/vllm_v1_adapter.py"
ENGINE_PATH = ROOT / "lmcache/v1/cache_engine.py"


def load_callbacks():
    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LMCacheConnectorV1Impl"
    )
    names = {
        "_is_dsa_two_groups",
        "_is_indexer_layer_wait",
        "_layerwise_wait_group",
        "_layerwise_layer_id_from_name",
        "_layerwise_has_indexer_model_layer",
        "_layerwise_required_wait_groups",
        "_layerwise_wait_should_advance",
        "supports_layerwise_prefill_transfer_window",
        "_layerwise_prefill_transfer_layer_id",
        "_layerwise_prefill_load_position",
        "_layerwise_prefill_slot_mapping",
        "_wait_for_layerwise_prefill_bank",
        "_advance_deferred_layerwise_prefill_load",
        "_advance_dense_layerwise_retriever",
        "_complete_layerwise_retrieve_group",
        "_sparse_retrieve_state_guard",
        "_abort_layerwise_retrieve_step",
        "wait_for_layer_load",
        "submit_layerwise_prefill_load",
        "_prime_dense_prefix_retrievers",
        "_drain_layerwise_retrievers",
        "_close_layerwise_retriever",
    }
    methods = [n for n in cls.body if getattr(n, "name", None) in names]
    assert len(methods) == len(names)
    cls.bases = []
    cls.decorator_list = []
    cls.body = methods
    namespace = dict(
        torch=torch,
        logger=logging.getLogger(__name__),
        contextmanager=contextmanager,
        _layerwise_prefill_p_node_enabled=lambda: True,
        _lmcache_nvtx_annotate=lambda fn: fn,
        LMCacheConnectorMetadata=SimpleNamespace,
        serving_perf_now=lambda: 0.0,
        serving_perf_enabled=lambda: False,
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
        compile(ast.fix_missing_locations(module), str(ADAPTER_PATH), "exec"), namespace
    )
    return namespace[cls.name]


def make_step(adapter, latent_layers, indexer_layers, *, step=0, requests=2):
    """Prime finite generators, with different contents in every request/row."""
    adapter.current_layer = 0
    adapter.num_layers = len(latent_layers)
    adapter._latent_layer_names = [
        f"model.layers.{i}.self_attn.attn" for i in latent_layers
    ]
    adapter._indexer_layer_names = [
        f"model.layers.{i}.self_attn.indexer.k_cache" for i in indexer_layers
    ]
    adapter._indexer_model_layers = set(indexer_layers)
    adapter.config = SimpleNamespace(dsa_two_groups=True)
    adapter.use_layerwise = True
    adapter.kv_role = "kv_producer"
    adapter._deferred_layerwise_prefill_load_active = True
    adapter._layerwise_waited_groups = set()
    adapter._cold_perf_load_started = {}
    adapter._cold_perf_dense_load_started = {}
    adapter._cold_perf_dense_load_completed = {}
    adapter._layerwise_sparse_shared_ordered = []
    adapter._layerwise_retriever_is_sparse = [False] * requests
    adapter._layerwise_requests = []
    adapter.layerwise_retrievers = []
    adapter._finalize_worker_retrieve_state_from_metadata = Mock()
    adapter._validate_dense_retrieve_result = Mock()
    adapter._abort_layerwise_retrieve_step = Mock()
    fences, submitted, released, closed = [], [], [], []
    banks = {}
    gpu = SimpleNamespace(
        supports_layerwise_prefill_transfer_window=True,
        wait_for_layerwise_prefill_load=lambda **kw: fences.append(kw),
    )
    adapter.lmcache_engine = SimpleNamespace(gpu_connector=gpu)

    def retrieve(req, group, count):
        try:
            command = yield torch.tensor(3)
            for row in range(count):
                mapping = req._layerwise_prefill_device_slot_mappings[row % 2][group]
                if row:
                    assert command is not None, "bank delta was dropped"
                    assert torch.equal(command["slot_mapping"], mapping)
                else:
                    assert command is None
                submitted.append((req.req_id, group, row))
                banks[req.req_id, group, row % 2] = (step, row, tuple(mapping.tolist()))
                command = yield None
            released.append((req.req_id, group))
            yield torch.ones(3, dtype=torch.bool)
        finally:
            closed.append((req.req_id, group))

    for req_index in range(requests):
        req = SimpleNamespace(
            req_id=f"req-{step}-{req_index}",
            is_sparse_decode=False,
            block_allocation_mode="prefill_child",
            _layerwise_prefill_device_slot_mappings=tuple(
                tuple(
                    torch.arange(3)
                    + step * 1000
                    + req_index * 100
                    + bank * 10
                    + group * 4
                    for group in (0, 1)
                )
                for bank in (0, 1)
            ),
        )
        adapter._layerwise_requests.append(req)
        latent = retrieve(req, 0, len(latent_layers))
        indexer = retrieve(req, 1, len(indexer_layers)) if indexer_layers else None
        adapter.layerwise_retrievers.append((latent, indexer))
        adapter._prime_dense_prefix_retrievers(latent, indexer)
    metadata = SimpleNamespace(requests=list(adapter._layerwise_requests))
    adapter._parent = SimpleNamespace(_get_connector_metadata=lambda: metadata)
    return SimpleNamespace(
        fences=fences,
        submitted=submitted,
        released=released,
        closed=closed,
        banks=banks,
        requests=metadata.requests,
    )


@pytest.mark.parametrize(
    "latent_layers,indexer_layers",
    [
        (list(range(79)), [0, 1, 2] + list(range(6, 79, 4))),
        (list(range(80)), [0, 1, 2] + list(range(6, 79, 4))),  # MTP without own indexer
        (list(range(80)), [0, 1, 2] + list(range(6, 79, 4)) + [79]),
        (list(range(8)), [0, 1, 2, 6]),  # indexer completes before LATENT
        ([4, 5], [4, 5]),  # execution ordinal != model layer number
        ([0], [0]),
        (list(range(4)), []),
    ],
)
@pytest.mark.parametrize("indexer_first", [False, True])
def test_two_steps_read_the_right_history_bank_and_drain_each_group_once(
    latent_layers, indexer_layers, indexer_first
):
    adapter = load_callbacks()()
    for step in range(2):
        state = make_step(adapter, latent_layers, indexer_layers, step=step)
        for execution, model_layer in enumerate(latent_layers):
            names = [(0, execution, adapter._latent_layer_names[execution])]
            if model_layer in indexer_layers:
                row = indexer_layers.index(model_layer)
                names.append((1, row, adapter._indexer_layer_names[row]))
            for group, row, name in reversed(names) if indexer_first else names:
                adapter.wait_for_layer_load(name)
                assert adapter.current_layer == execution
                for req in state.requests:
                    mapping = req._layerwise_prefill_device_slot_mappings[row % 2][
                        group
                    ]
                    assert state.banks[req.req_id, group, row % 2] == (
                        step,
                        row,
                        tuple(mapping.tolist()),
                    )
                # Entry waits may be repeated by preprocessing. No second drain.
                before = list(state.submitted), list(state.released)
                adapter.wait_for_layer_load(name)
                assert (state.submitted, state.released) == before
            for _, _, name in names:
                adapter.submit_layerwise_prefill_load(name)
            assert adapter.current_layer == execution + 1
        validations = adapter._validate_dense_retrieve_result.call_args_list
        expected_groups = {0, 1} if indexer_layers else {0}
        assert len(validations) == len(state.requests) * len(expected_groups)
        assert {
            (call.args[0].req_id, call.kwargs["kv_group"]) for call in validations
        } == {
            (req.req_id, group) for req in state.requests for group in expected_groups
        }
        expected = len(state.requests) * (len(latent_layers) + len(indexer_layers))
        assert len(state.submitted) == len(set(state.submitted)) == expected
        groups = 2 if indexer_layers else 1
        assert (
            len(state.released)
            == len(set(state.released))
            == len(state.requests) * groups
        )
        assert sorted(state.closed) == sorted(state.released)
        assert not adapter.layerwise_retrievers
        assert not adapter._deferred_layerwise_prefill_load_active
        assert not adapter._deferred_layerwise_prefill_drained_groups
        adapter._finalize_worker_retrieve_state_from_metadata.assert_called_once()
        adapter._abort_layerwise_retrieve_step.assert_not_called()


def test_submit_still_rejects_wrong_execution_layer_and_group_order():
    adapter = load_callbacks()()
    make_step(adapter, list(range(8)), [0, 1, 2, 6], requests=1)
    with pytest.raises(RuntimeError, match="submitted out of order"):
        adapter.submit_layerwise_prefill_load(adapter._indexer_layer_names[0])
    with pytest.raises(RuntimeError, match="does not match callback"):
        adapter.submit_layerwise_prefill_load(adapter._latent_layer_names[6])
    adapter._drain_layerwise_retrievers(finish_dense=False)


def test_cache_miss_still_fences_banks_without_advancing_generators():
    adapter = load_callbacks()()
    state = make_step(adapter, list(range(8)), [0, 1, 2, 6], requests=0)
    adapter._deferred_layerwise_prefill_load_active = False
    for name in adapter._latent_layer_names + adapter._indexer_layer_names:
        adapter.wait_for_layer_load(name)
        adapter.submit_layerwise_prefill_load(name)
    assert len(state.fences) == 12
    assert not state.submitted


def test_failed_group_load_resets_banks_and_closes_all_request_generators():
    adapter = load_callbacks()()
    state = make_step(adapter, list(range(8)), [0, 1, 2, 6])
    # Use real abort cleanup for this failure path, not the success-path spy.
    del adapter._abort_layerwise_retrieve_step
    adapter._worker_retrieve_state = {}
    adapter._drop_worker_retrieve_state = Mock()
    reset = Mock()
    adapter.lmcache_engine.gpu_connector.reset_layerwise_prefill_transfer_state = reset
    original = adapter._advance_dense_layerwise_retriever

    def fail_second_request(request, pair, group, row):
        if request.req_id == state.requests[1].req_id:
            raise RuntimeError("simulated H2D failure")
        return original(request, pair, group, row)

    adapter._advance_dense_layerwise_retriever = fail_second_request
    with pytest.raises(RuntimeError, match="simulated H2D failure"):
        adapter.submit_layerwise_prefill_load(adapter._latent_layer_names[0])
    reset.assert_called_once_with(synchronize=True)
    assert len(state.closed) == 4
    assert not adapter.layerwise_retrievers
    assert not adapter._deferred_layerwise_prefill_drained_groups
    assert not adapter._deferred_layerwise_prefill_load_active
    adapter._finalize_worker_retrieve_state_from_metadata.assert_not_called()


@pytest.mark.parametrize(
    "method", ["_retrieve_layer_shared_rank0", "_retrieve_layer_shared_passive"]
)
@pytest.mark.parametrize("page_source", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("layer_id", [0, 1])
def test_shared_cpu_generators_forward_the_bank_command(
    method, page_source, dynamic, layer_id
):
    """Execute the actual shared-engine yield/send block for both rank paths."""
    tree = ast.parse(ENGINE_PATH.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method
    )
    loop = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "layer_id"
        and any(
            isinstance(part, ast.If) and ast.unparse(part.test) == "layer_id == 0"
            for part in n.body
        )
    )
    # Isolate the transfer hand-off from allocator/broadcast dependencies.
    first = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "layer_id == 0"
    )
    send = next(
        i for i in range(first, len(loop.body)) if isinstance(loop.body[i], ast.Try)
    )
    body = loop.body[first : send + 1]
    source = object()
    consumer = Mock()
    ns = dict(
        torch=torch,
        ret_mask=torch.ones(3, dtype=torch.bool),
        layer_id=layer_id,
        mem_obj_consumer=consumer,
        perf_enabled=False,
        mem_objs_layer=[source],
        deferred_layerwise_get=True,
        layer_page_chunks=int(page_source),
        page_chunks=int(page_source),
        layer_pages=("page",),
        passive_page_tuple=("page",),
        passive_pages=("page",) if page_source else (),
        LayerPageSource=lambda *args: ("page-source", args),
    )
    generated = ast.FunctionDef(
        name="transfer",
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=body,
        decorator_list=[],
    )
    module = ast.Module(body=[generated], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ENGINE_PATH), "exec"), ns)
    gen = ns["transfer"]()
    next(gen)
    command = {"slot_mapping": torch.tensor([10, 11, 12])} if dynamic else None
    with pytest.raises(StopIteration):
        gen.send(command)
    payload = consumer.send.call_args.args[0]
    if dynamic:
        assert payload["layer_request"] is command
        payload = payload["memory_objs"]
    assert payload[0] == "page-source" if page_source else payload == [source]
