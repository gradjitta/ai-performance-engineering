"""Expose the double-buffered pipeline baseline under the TMEM triple-overlap alias.

This lets the benchmarking harness invoke the existing baseline kernel using
the documentation-friendly target `ch10:tmem_triple_overlap_baseline`.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch10.baseline_double_buffered_pipeline import (  # noqa: E402
    BaselineDoubleBufferedPipelineBenchmark,
)


def get_benchmark() -> BaselineDoubleBufferedPipelineBenchmark:
    """Reuse the existing double-buffered GEMM baseline."""
    return BaselineDoubleBufferedPipelineBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
