"""Ch7 baseline memory access benchmark (uncoalesced)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineMemoryAccessBenchmark(CudaBinaryBenchmark):
    """Wraps the uncoalesced CUDA kernel baseline."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_memory_access",
            friendly_name="Ch7 Baseline Memory Access",
            iterations=3,
            warmup=1,
            timeout_seconds=90,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access pattern metrics for ch7 analysis."""
        base_metrics = super().get_custom_metrics() or {}
        base_metrics.update({
            # Memory access pattern characteristics (baseline = uncoalesced)
            "memory_access.is_coalesced": 0.0,  # 0 = baseline uncoalesced
            "memory_access.expected_efficiency_pct": 3.125,  # 1/32 for stride-32 access
        })
        return base_metrics


def get_benchmark() -> BaselineMemoryAccessBenchmark:
    return BaselineMemoryAccessBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nCh7 Baseline Memory Access: "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
