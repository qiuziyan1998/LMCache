# SPDX-License-Identifier: Apache-2.0
"""The two-plane indexer packet stored as one atomic byte-valued cache object."""

# Standard
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndexerC8Layout:
    """A2/A3 INT8 keys with one FP16 scale per token (no numeric conversion)."""

    head_dim: int = 128

    def __post_init__(self) -> None:
        if type(self.head_dim) is not int or self.head_dim != 128:
            raise ValueError("Indexer C8 supports a 128-element key and one FP16 scale")

    @property
    def token_bytes(self) -> int:
        return self.head_dim + 2

    def layer_bytes(self, tokens: int) -> int:
        """Logical bytes, excluding allocator padding; reject negative counts."""
        if type(tokens) is not int or tokens < 0:
            raise ValueError("Indexer C8 token count must be a nonnegative integer")
        return tokens * self.token_bytes

    def plane_ranges(self, tokens: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return (byte offset, byte length) for keys then scales in a layer."""
        self.layer_bytes(tokens)
        keys = tokens * self.head_dim
        return (0, keys), (keys, tokens * 2)

    def descriptor(self) -> dict[str, str | int]:
        """Immutable execution identity, applied to keys of BOTH KV groups."""
        return {
            "layout": "dsa-index-c8-planar-v1",
            "quantization": "hadamard128-dynamic-int8-fp16-scale-v1",
            "head_dim": self.head_dim,
            "key_dtype": "int8",
            "scale_dtype": "float16",
            "storage_dtype": "uint8",
        }
