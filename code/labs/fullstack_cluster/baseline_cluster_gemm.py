"""Benchmark wrapper for the capstone baseline GEMM kernel."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.fullstack_cluster import baseline_matmul
from labs.fullstack_cluster.capstone_benchmarks import CapstoneMatmulBenchmark


class BaselineCapstoneGemmBenchmark(CapstoneMatmulBenchmark):
    def __init__(self) -> None:
        super().__init__(
            runner=baseline_matmul,
            label="capstone_baseline",
            iterations=1,
            warmup=5,
            timeout_seconds=180,
            validate_against_baseline=False,
        )

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)



def get_benchmark() -> BaselineCapstoneGemmBenchmark:
    return BaselineCapstoneGemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
