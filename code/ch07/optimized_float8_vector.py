"""Python harness wrapper for optimized_float8_vector.cu - 32-byte Vectorized Loads."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
class OptimizedFloat8VectorBenchmark(CudaBinaryBenchmark):
    """Wraps the 32-byte vectorized load benchmark for Blackwell."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_float8_vector",
            friendly_name="32-byte Vectorized Loads",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
            workload_params={"type": "float8_vector"},
        )
        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return None
def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedFloat8VectorBenchmark()
if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
