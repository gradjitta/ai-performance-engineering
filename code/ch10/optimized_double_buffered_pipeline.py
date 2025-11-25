"""Python harness wrapper for optimized_double_buffered_pipeline.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedDoubleBufferedPipelineBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized double-buffered pipeline kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_double_buffered_pipeline",
            friendly_name="Optimized Double-buffered Pipeline",
            iterations=3,
            warmup=1,
            timeout_seconds=180,
            requires_pipeline_api=True,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific pipeline metrics."""
        base_metrics = super().get_custom_metrics() or {}
        base_metrics.update({
            "pipeline.uses_clusters": 1.0,
            "pipeline.uses_pipeline": 1.0,
        })
        return base_metrics

def get_benchmark() -> OptimizedDoubleBufferedPipelineBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDoubleBufferedPipelineBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Double-buffered Pipeline: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
