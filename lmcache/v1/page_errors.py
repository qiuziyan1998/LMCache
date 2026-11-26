# SPDX-License-Identifier: Apache-2.0
"""Recoverable failures to acquire immutable KV pages before publication."""


class LayerPageUnavailable(RuntimeError):
    """A planned physical page was not acquired; no caller may use its data."""
