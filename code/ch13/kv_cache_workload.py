"""Shared workload configuration for ch13 KV cache benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from common.python.smoke import is_smoke_mode
from typing import Sequence, Tuple

import torch


@dataclass(frozen=True)
class KVCacheWorkload:
    """Canonical KV cache benchmark settings used by baseline & optimized paths."""

    batch_size: int = 4
    num_layers: int = 6
    num_heads: int = 16
    head_dim: int = 64
    sequence_lengths: Tuple[int, ...] = (1024, 1536, 2048)
    dtype: torch.dtype = torch.float16
    page_size: int = 256
    block_size: int = 128

    @property
    def hidden_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def max_seq_len(self) -> int:
        return max(self.sequence_lengths)

    def lengths(self) -> Tuple[int, ...]:
        """Return the canonical sequence lengths."""
        return self.sequence_lengths


def get_workload() -> KVCacheWorkload:
    """Return the canonical workload settings."""
    if is_smoke_mode():
        # Leaner workload for quick, low-memory sweeps.
        return KVCacheWorkload(
            batch_size=1,
            num_layers=2,
            num_heads=8,
            head_dim=64,
            sequence_lengths=(256, 384, 512),
            dtype=torch.float16,
            page_size=128,
            block_size=64,
        )
    return KVCacheWorkload()
