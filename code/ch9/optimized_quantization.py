"""optimized_quantization.py - Optimized quantization in kernel efficiency/arithmetic intensity context.

Demonstrates quantization for reduced precision operations.
Quantization: Uses quantization to reduce numerical precision.
Reduces memory usage and improves performance.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import fp8_autocast
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class FP8MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        assert TE_AVAILABLE
        self.fc1 = te.Linear(
            input_dim,
            hidden_dim,
            bias=False,
            params_dtype=torch.float16,
            device="cuda",
        )
        self.fc2 = te.Linear(
            hidden_dim,
            input_dim,
            bias=False,
            params_dtype=torch.float16,
            device="cuda",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with fp8_autocast(enabled=True):
            out = self.fc1(x)
            out = F.gelu(out)
            out = self.fc2(out)
        return out


class OptimizedQuantizationBenchmark(Benchmark):
    """Optimized: Quantization for reduced precision operations.
    
    Quantization: Uses quantization to reduce numerical precision.
    Reduces memory usage and improves performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.use_te = TE_AVAILABLE
    
    def setup(self) -> None:
        """Setup: Initialize quantized model."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)

        if self.use_te:
            self.model = FP8MLP(1024, 2048).to(self.device).eval()
            self.input = torch.randn(32, 1024, device=self.device, dtype=torch.float16)
        else:
            from common.python.quantization_utils import quantize_model_to_fp8, get_quantization_dtype
            self.model = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
            )
            self.model = quantize_model_to_fp8(self.model, self.device, precision='fp8')
            input_dtype = get_quantization_dtype('fp8')
            self.input = torch.randn(32, 1024, device=self.device, dtype=input_dtype)

        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Quantized operations."""
        # Use conditional NVTX ranges - only enabled when profiling
    
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
    
        config = self.get_config()
    
        enable_nvtx = get_nvtx_enabled(config) if config else False
    
    
        with nvtx_range("optimized_quantization", enable=enable_nvtx):
            with torch.no_grad():
                if self.use_te:
                    assert self.model is not None
                    output = self.model(self.input)  # type: ignore[arg-type]
                else:
                    output = self.model(self.input)  # type: ignore[operator]
                
                # Optimization: FP8 Quantization benefits
                # - Reduced memory usage (FP8: 2x vs FP16, 4x vs FP32)
                # - Improved performance (faster FP8 computation on Tensor Cores)
                # - Better memory bandwidth utilization (less data to transfer)
                # - Native CUDA support (not CPU-only like qint8)
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedQuantizationBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedQuantizationBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Quantization")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()

