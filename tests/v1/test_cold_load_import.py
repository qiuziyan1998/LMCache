# SPDX-License-Identifier: Apache-2.0
"""Cold-load coordination imports preserve caller logging configuration."""

# Standard
from pathlib import Path
import subprocess
import sys


def test_cold_load_import_preserves_adapter_logger_handlers() -> None:
    """A fresh import borrows the adapter logger without resetting handlers."""
    script = """
import logging
import sys

logger = logging.getLogger("lmcache.integration.vllm.vllm_v1_adapter")
handler = logging.NullHandler()
logger.handlers[:] = [handler]
logger.propagate = True

from lmcache.integration.vllm import cold_load

assert cold_load.logger is logger
assert logger.handlers == [handler]
assert logger.propagate is True
assert "torch" not in sys.modules
assert "vllm" not in sys.modules
assert "lmcache.integration.vllm.vllm_v1_adapter" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
