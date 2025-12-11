"""Multi-GPU wrapper for the optimized NVLink benchmark that skips on single-GPU hosts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch04.optimized_nvlink import OptimizedNVLinkBenchmark


class MultiGPUOptimizedNVLinkBenchmark(OptimizedNVLinkBenchmark):
    def setup(self) -> None:  # type: ignore[override]
        if torch.cuda.device_count() < 2:
            raise RuntimeError("SKIPPED: requires >=2 GPUs")
        super().setup()


def get_benchmark() -> OptimizedNVLinkBenchmark:
    return MultiGPUOptimizedNVLinkBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
