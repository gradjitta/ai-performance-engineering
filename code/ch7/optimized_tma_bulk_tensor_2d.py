"""Python harness wrapper for optimized_tma_bulk_tensor_2d.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedTMABulkTensor2D(CudaBinaryBenchmark):
    """Wraps the cp.async.bulk.tensor 2D path (TMA-backed)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_tma_bulk_tensor_2d",
            friendly_name="Optimized 2D tensor copy (TMA bulk)",
            iterations=3,
            warmup=1,
            timeout_seconds=120,
            require_tma_instructions=True,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific memory_access metrics."""
        base_metrics = super().get_custom_metrics() or {}
        base_metrics.update({
            "memory_access.is_coalesced": 1.0,
            "memory_access.expected_efficiency_pct": 100.0,
        })
        return base_metrics

def get_benchmark() -> OptimizedTMABulkTensor2D:
    """Factory for discover_benchmarks()."""
    return OptimizedTMABulkTensor2D()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nOptimized 2D tensor copy (TMA bulk): {timing:.3f} ms")
