# SPDX-License-Identifier: Apache-2.0
"""Fresh-process regression for Ascend config patching before singleton creation."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["local", "process"])
def test_early_bootstrap_keeps_the_ascend_config_class(mode):
    root = Path(__file__).resolve().parents[2]
    script = r'''
import ast
import builtins
import logging
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
from typing import Optional

import lmcache.v1.config as config_module
from lmcache.v1.rpc.storage_bootstrap import early_bootstrap_decoder_storage

root = Path(sys.argv[1])
def extract(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(
        ast.Module(body=[prefix, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]

utils = ModuleType("lmcache.integration.vllm.utils")
utils.__dict__.update(
    os=__import__("os"), logger=logging.getLogger("config-probe"),
    LMCacheEngineConfig=config_module.LMCacheEngineConfig,
    _config_instance=None, _config_lock=threading.Lock(),
    apply_vllm_config_overrides=lambda *a: None,
    create_lmcache_metadata=lambda *a: None,
)
extract(root / "lmcache/integration/vllm/utils.py",
        "lmcache_get_or_create_config", utils.__dict__)
sys.modules[utils.__name__] = utils
patch = extract(root.parent / "LMCache-Ascend/lmcache_ascend/__init__.py",
                "_patch_config", dict(sys=sys, Optional=Optional))
original_import = builtins.__import__
ascend = ModuleType("lmcache_ascend")
patched = False
def importing(name, *args, **kwargs):
    global patched
    if name == "lmcache_ascend":
        if not patched:
            patch()
            patched = True
        return ascend
    return original_import(name, *args, **kwargs)
builtins.__import__ = importing
worker = NS(
    vllm_config=NS(kv_transfer_config=NS(kv_connector="LMCacheAscendConnectorV1Dynamic")),
    parallel_config=NS(pipeline_parallel_size=2),
)
if sys.argv[2] == "local":
    assert early_bootstrap_decoder_storage(worker) is None
else:
    # Stop at topology validation before any native registration or service launch.
    try:
        early_bootstrap_decoder_storage(worker)
    except ValueError as error:
        assert "local TP/shared-CPU receiver topology" in str(error)
    else:
        raise AssertionError("Expected topology validation")
# Connector initialization imports Ascend and enforces this exact class identity.
import lmcache_ascend
config = utils.lmcache_get_or_create_config()
assert isinstance(config, config_module.LMCacheEngineConfig), "Cached pre-Ascend config"
assert hasattr(config, "p2p_use_npu")
assert config.decoder_storage_mode == sys.argv[2]
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith("LMCACHE_")}
    env["LMCACHE_DECODER_STORAGE_MODE"] = mode
    result = subprocess.run(
        [sys.executable, "-c", script, str(root), mode],
        cwd=root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
