# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete banked history before submitting any layer's H2D."""

# Standard
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator

# Third Party
import pytest
import torch


SOURCE = (
    Path(__file__).resolve().parents[2] / "lmcache/integration/vllm/vllm_v1_adapter.py"
)


def adapter_class(timed: bool = False) -> Any:
    """Use the production priming boundary without an accelerator runtime."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LMCacheConnectorV1Impl"
    )
    names = {
        "_prime_dense_prefix_retrievers",
        "_full_hit_recalc_last_token",
        "_trim_prefill_for_recalc_last",
        "_load_token_mask_for_retrieve",
    }
    module = ast.parse("from __future__ import annotations\nclass Adapter: pass")
    module.body[1].body = [
        node for node in cls.body if getattr(node, "name", None) in names
    ]
    namespace = {
        "prefill_start_timing_enabled": lambda: timed,
        "prefill_start_timing_log": lambda *args, **kwargs: None,
        "time": SimpleNamespace(perf_counter=lambda: 1.0),
        "logger": None,
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["Adapter"]


def prime_method(timed: bool = False) -> Any:
    return adapter_class(timed)._prime_dense_prefix_retrievers


def retriever(
    actions: list[str],
    label: str,
    count: int | None,
) -> Generator[torch.Tensor | None, None, None]:
    """Model the all-layer preparation yield followed by first-bank upload."""
    actions.append(label + ":prepared")
    yield None if count is None else torch.tensor(count)
    actions.append(label + ":h2d")
    yield None


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("missing_count", [None, 0, 16383])
@pytest.mark.parametrize("timed", [False, True])
def test_short_history_fails_after_both_preparations_before_h2d(
    group: int,
    missing_count: int | None,
    timed: bool,
) -> None:
    actions: list[str] = []
    counts: list[int | None] = [16384, 16384]
    counts[group] = missing_count
    with pytest.raises(RuntimeError, match="incomplete before model execution") as exc:
        prime_method(timed)(
            retriever(actions, "latent", counts[0]),
            retriever(actions, "indexer", counts[1]),
            req_id="chunked-prefill",
            prefix_tokens=16384,
            required_mask=torch.ones(16384, dtype=torch.bool),
        )
        actions.append("model")
    assert f"kv_group={group}" in str(exc.value)
    assert "req_id=chunked-prefill" in str(exc.value)
    assert "required_tokens=16384" in str(exc.value)
    assert actions == ["latent:prepared", "indexer:prepared"]


@pytest.mark.parametrize(
    "mask,recalc,count",
    [
        ([True] * 8, False, 8),
        ([True] * 8, True, 7),
        ([True] * 8, True, 8),
        ([False] * 4 + [True] * 4, False, 4),
        ([False] * 4 + [True] * 4, True, 3),
        ([False] * 8, True, None),
    ],
)
def test_required_coverage_uses_mask_and_only_exempts_recomputed_tail(
    mask: list[bool],
    recalc: bool,
    count: int | None,
) -> None:
    actions: list[str] = []
    required_mask = torch.tensor(mask, dtype=torch.bool)
    original = required_mask.clone()
    prime_method()(
        retriever(actions, "latent", count),
        retriever(actions, "indexer", count),
        req_id="full-hit",
        prefix_tokens=len(mask),
        required_mask=required_mask,
        recalc_last_token=recalc,
    )
    assert actions == [
        "latent:prepared",
        "indexer:prepared",
        "latent:h2d",
        "indexer:h2d",
    ]
    assert torch.equal(required_mask, original)


def test_recalc_exemption_is_at_most_one_token() -> None:
    actions: list[str] = []
    with pytest.raises(RuntimeError, match="required_tokens=7"):
        prime_method()(
            retriever(actions, "latent", 6),
            None,
            required_mask=torch.ones(8, dtype=torch.bool),
            recalc_last_token=True,
        )
    assert actions == ["latent:prepared"]


@pytest.mark.parametrize(
    "available,success", [(18879, True), (18878, True), (18877, False)]
)
def test_full_hit_uses_actual_trim_and_mask_helpers(
    available: int,
    success: bool,
) -> None:
    adapter = adapter_class()
    tokens = list(range(18879))
    request = SimpleNamespace(
        is_sparse_decode=False,
        load_spec=SimpleNamespace(lmcache_cached_tokens=18879, vllm_cached_tokens=0),
    )
    recalc = adapter._full_hit_recalc_last_token(
        request.load_spec,
        len(tokens),
        is_sparse_decode=False,
    )
    tokens, slots = adapter._trim_prefill_for_recalc_last(
        request,
        tokens,
        torch.arange(len(tokens)),
    )
    # The production helper deliberately keeps the full partial-chunk key.
    assert len(tokens) == slots.numel() == 18879
    mask = adapter._load_token_mask_for_retrieve(request, len(tokens), 1024)
    actions: list[str] = []

    def run() -> None:
        adapter._prime_dense_prefix_retrievers(
            retriever(actions, "latent", available),
            None,
            prefix_tokens=len(tokens),
            required_mask=mask,
            recalc_last_token=recalc,
        )

    if success:
        run()
        assert actions[-1] == "latent:h2d"
    else:
        with pytest.raises(RuntimeError, match="required_tokens=18878"):
            run()
        assert actions == ["latent:prepared"]


@pytest.mark.parametrize("timed", [False, True])
def test_ordinary_priming_keeps_original_order_and_no_new_coverage_gate(
    timed: bool,
) -> None:
    actions: list[str] = []
    prime_method(timed)(
        retriever(actions, "latent", None),
        retriever(actions, "indexer", None),
        req_id="ordinary",
    )
    assert actions == [
        "latent:prepared",
        "indexer:prepared",
        "latent:h2d",
        "indexer:h2d",
    ]


def test_later_request_miss_prevents_model_for_entire_mixed_batch() -> None:
    actions: list[str] = []
    prime = prime_method()
    with pytest.raises(RuntimeError, match="req_id=second"):
        for req_id, counts in (("first", (8, 8)), ("second", (8, 0))):
            prime(
                retriever(actions, req_id + ":latent", counts[0]),
                retriever(actions, req_id + ":indexer", counts[1]),
                req_id=req_id,
                required_mask=torch.ones(8, dtype=torch.bool),
            )
        actions.append("model")
    assert "model" not in actions
    assert not any(
        action.startswith("second") and action.endswith("h2d") for action in actions
    )


@pytest.mark.parametrize(
    "p_node,deferred", [(False, False), (False, True), (True, False), (True, True)]
)
def test_start_load_only_enables_gate_for_deferred_p_nodes(
    p_node: bool,
    deferred: bool,
) -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_prime_dense_prefix_retrievers"
    )
    expression = next(
        value.value for value in call.keywords if value.arg == "required_mask"
    )
    mask = object()
    selected = eval(
        compile(ast.Expression(expression), str(SOURCE), "eval"),
        {
            "self": SimpleNamespace(
                _layerwise_prefill_p_node=p_node,
                _deferred_layerwise_prefill_load_active=deferred,
            ),
            "token_mask": mask,
        },
    )
    assert selected is (mask if p_node and deferred else None)
