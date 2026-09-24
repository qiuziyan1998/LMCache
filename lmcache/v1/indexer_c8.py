# SPDX-License-Identifier: Apache-2.0
"""The two-plane indexer packet stored as one atomic byte-valued cache object."""

# Standard
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndexerC8Layout:
    """A2/A3 INT8 keys with one FP16 scale per token (no numeric conversion)."""

    head_dim: int = 128
    c8_layers: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if type(self.head_dim) is not int or self.head_dim != 128:
            raise ValueError("Indexer C8 supports a 128-element key and one FP16 scale")
        if not isinstance(self.c8_layers, tuple) or any(
            type(enabled) is not bool for enabled in self.c8_layers
        ):
            raise ValueError("Indexer precision policy must be an immutable boolean tuple")

    @property
    def mixed(self) -> bool:
        return bool(self.c8_layers) and not all(self.c8_layers)

    def is_c8(self, layer_id: int) -> bool:
        if layer_id < 0 or (self.c8_layers and layer_id >= len(self.c8_layers)):
            raise IndexError(f"Invalid physical indexer layer: {layer_id}")
        return self.c8_layers[layer_id] if self.c8_layers else True

    def token_bytes_for(self, layer_id: int) -> int:
        """Stored bytes per token of a physical indexer owner."""
        return self.head_dim + 2 if self.is_c8(layer_id) else self.head_dim * 2

    @property
    def token_bytes(self) -> int:
        if self.mixed:
            raise ValueError("Mixed indexer storage requires a physical layer ID")
        return self.head_dim + 2

    def layer_bytes(self, tokens: int, layer_id: int | None = None) -> int:
        """Logical bytes, excluding allocator padding; reject negative counts."""
        if type(tokens) is not int or tokens < 0:
            raise ValueError("Indexer C8 token count must be a nonnegative integer")
        return tokens * (
            self.token_bytes if layer_id is None else self.token_bytes_for(layer_id)
        )

    def plane_ranges(
        self, tokens: int, layer_id: int | None = None
    ) -> tuple[tuple[int, int], ...]:
        """Return (byte offset, byte length) for keys then scales in a layer."""
        size = self.layer_bytes(tokens, layer_id)
        if layer_id is not None and not self.is_c8(layer_id):
            return ((0, size),)
        keys = tokens * self.head_dim
        return (0, keys), (keys, tokens * 2)

    def descriptor(self) -> dict[str, object]:
        """Immutable execution identity, applied to keys of BOTH KV groups."""
        descriptor: dict[str, object] = {
            "layout": "dsa-index-c8-planar-v1",
            "quantization": "hadamard128-dynamic-int8-fp16-scale-v1",
            "head_dim": self.head_dim,
            "key_dtype": "int8",
            "scale_dtype": "float16",
            "storage_dtype": "uint8",
        }
        if self.mixed:
            descriptor.update(
                layout="dsa-index-mixed-planar-v1",
                key_dtype="per-layer-int8-or-model-dtype",
                c8_layers=self.c8_layers,
            )
        return descriptor
