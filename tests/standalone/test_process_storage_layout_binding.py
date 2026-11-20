# SPDX-License-Identifier: Apache-2.0
"""Execute the Ascend engine binding method with CPU layout stand-ins.

No NPU detection or native allocation is qualified by this extracted-method test.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_registered_planar_or_interleaved_and_mixed_precision_geometry():
    path = (
        Path(__file__).resolve().parents[3]
        / "LMCache-Ascend/lmcache_ascend/v1/cache_engine.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(child, ast.FunctionDef)
            and child.name == "prepare_storage_layout"
            for child in node.body
        )
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_storage_layout"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    calls = []
    widths = {0: (512, 64), 1: ((128, 2), (256,))}
    owner = SimpleNamespace(
        _ensure_layerwise_connector_layout=lambda **kwargs: calls.append(kwargs),
        gpu_connector=SimpleNamespace(checkpoint_plane_widths=widths.__getitem__),
        _shared_cpu_dtype_for_kv_group=lambda group: torch.bfloat16
        if group == 0
        else torch.uint8,
    )
    bind = namespace["prepare_storage_layout"]
    bind(owner, {0: ["latent"], 1: ["indexer"]})
    assert owner.storage_plane_token_bytes == {0: (1024, 128), 1: ((128, 2), (256,))}
    assert calls == [
        {"kvcaches": ["latent"], "kv_group": 0},
        {"kvcaches": ["indexer"], "kv_group": 1},
    ]
    widths[0] = (576,)
    bind(owner, {0: ["interleaved"], 1: ["indexer"]})
    assert owner.storage_plane_token_bytes[0] == (1152,)
