"""Key-value cache support for autoregressive decoding."""

from __future__ import annotations

from typing import Optional

import torch


class KVCache:
    """
    Key-Value cache for autoregressive generation.

    Modes:
      - full: stores all previous keys and values
      - sliding_window: keeps the most recent window_size positions
      - none: disables storage while preserving the same call interface
    """

    def __init__(self, mode: str = "full", window_size: int = 256, n_layers: int = 6):
        assert mode in ("full", "sliding_window", "none")
        self.mode = mode
        self.window_size = window_size
        self.n_layers = n_layers
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def get(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return cached (K, V) for this layer, or None if not yet cached."""
        if self.mode == "none":
            return None
        return self._cache.get(layer_idx)

    def update(
        self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new_k/new_v to the cache and return the full tensors to attend over.
        Tensors are expected in shape (batch, heads, seq, head_dim).
        """
        if self.mode == "none":
            self._cache_misses += 1
            return new_k, new_v

        cached = self._cache.get(layer_idx)
        if cached is None:
            self._cache_misses += 1
            k, v = new_k, new_v
        else:
            self._cache_hits += 1
            old_k, old_v = cached
            k = torch.cat([old_k, new_k], dim=2)
            v = torch.cat([old_v, new_v], dim=2)

        if self.mode == "sliding_window":
            k = k[:, :, -self.window_size :, :]
            v = v[:, :, -self.window_size :, :]

        self._cache[layer_idx] = (k.detach(), v.detach())
        return k, v

    def clear(self) -> None:
        """Reset cache."""
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0
