"""Multi-GPU wrapper for optimized continuous batching; skips when GPUs < 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch04.optimized_continuous_batching import OptimizedContinuousBatchingBenchmark


def get_benchmark() -> OptimizedContinuousBatchingBenchmark:
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: optimized_continuous_batching_multigpu requires >=2 GPUs")
    return OptimizedContinuousBatchingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
