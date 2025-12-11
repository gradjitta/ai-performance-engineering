"""Template for creating new benchmarks with full verification.

Copy this file into your chapter directory and fill in the benchmark-specific
details. The mixin enforces the required verification methods so new benchmarks
fail fast if verification payloads are missing.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class MyBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Description of what this benchmark measures.

    This benchmark demonstrates [technique/optimization/pattern].

    Key aspects:
    - What it measures (e.g., "Matrix multiplication performance")
    - What technique it uses (e.g., "Naive implementation" or "Optimized with tensor cores")
    - Expected performance characteristics
    """

    def __init__(self):
        """Initialize benchmark with device resolution."""
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input_data: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        """Setup phase: initialize models, data, etc."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.model = nn.Linear(256, 256).to(self.device)
        self.input_data = torch.randn(32, 256, device=self.device)
        self._verify_input = self.input_data.detach().clone()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Function to benchmark. Must be callable with no args."""
        if self.model is None or self.input_data is None or self._verify_input is None:
            raise RuntimeError("setup() must initialize model and inputs before benchmarking")

        with self._nvtx_range("my_benchmark_operation"):
            output = self.model(self.input_data)
            self._synchronize()

        # Capture verification payload from the timed run (no fallbacks)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=output.detach().clone(),
            batch_size=self._verify_input.shape[0],
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            output_tolerance=(1e-4, 1e-4),
        )

    def teardown(self) -> None:
        """Cleanup phase."""
        self.model = None
        self.input_data = None
        self._verify_input = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> Optional[BenchmarkConfig]:
        """Optional: return benchmark-specific config overrides."""
        return None

    def validate_result(self) -> Optional[str]:
        """Optional: validate benchmark result, return error message if invalid."""
        return None

    # Verification methods (fail-fast; provided by mixin)
    def get_verify_output(self) -> torch.Tensor:
        """Return benchmark output captured during the timed run."""
        return super().get_verify_output()

    def get_input_signature(self):
        """Return workload description for verification."""
        return super().get_input_signature()

    def get_output_tolerance(self):
        """Return (rtol, atol) tolerance for output comparison."""
        return super().get_output_tolerance()


def get_benchmark():
    """Factory function that returns a benchmark instance.
    
    This function is required for benchmark discovery.
    It should return an instance of your benchmark class.
    
    Returns:
        Benchmark instance
    """
    return MyBenchmark()


# Optional: Add a main function for standalone testing
if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness
    
    # Create benchmark instance
    benchmark = get_benchmark()
    
    # Create harness with default config
    harness = BenchmarkHarness()
    
    # Run benchmark
    result = harness.benchmark(benchmark)
    
    # Print results
    if result.timing:
        print(f"Mean: {result.timing.mean_ms:.2f} ms")
        print(f"Median: {result.timing.median_ms:.2f} ms")
        print(f"Std: {result.timing.std_ms:.2f} ms")
    
    if result.memory:
        print(f"Peak memory: {result.memory.peak_mb:.2f} MB")
