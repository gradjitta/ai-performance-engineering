"""Baseline: CUDA decode kernel (global-load) for the TMA double-buffered comparison.

This benchmark reuses the `labs/moe_cuda` baseline kernel wrapper so that
`optimized_decode_double_buffer_tma.py` has a valid, equivalent baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.harness.benchmark_harness import BaseBenchmark  # noqa: E402
from labs.moe_cuda.baseline_decode_kernel import BaselineDecodeKernelBenchmark  # noqa: E402


def get_benchmark() -> BaseBenchmark:
    return BaselineDecodeKernelBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
