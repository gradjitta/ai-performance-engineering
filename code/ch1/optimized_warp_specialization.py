"""optimized_performance.py - Ch1-appropriate optimizations.

This file demonstrates ONLY Ch1-appropriate techniques:
- Proper warmup iterations
- Accurate timing with torch.cuda.synchronize()
- NVTX range marking for profiling
- Basic TF32 enable for Tensor Cores
- cuDNN benchmark mode

NOTE: This file was refactored from "warp_specialization" which incorrectly
used Ch10 (warp specialization) and Ch12 (CUDA graphs) techniques.

WHAT'S APPROPRIATE FOR CH1:
  ✅ torch.cuda.synchronize() for timing
  ✅ Warmup iterations
  ✅ NVTX markers
  ✅ TF32 enable
  ✅ cuDNN benchmark mode
  
WHAT'S NOT APPROPRIATE FOR CH1 (moved to correct chapters):
  ❌ CUDA Graphs → Ch12
  ❌ Warp specialization → Ch10
  ❌ Streams → Ch11
  ❌ torch.compile → Ch14
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from ch1.workload_config import WORKLOAD


class OptimizedPerformanceBenchmark(BaseBenchmark):
    """Ch1 Optimized: Basic performance techniques.
    
    Ch1-APPROPRIATE Optimizations Applied:
    1. TF32 for Tensor Cores (reduces FP32 matmul time)
    2. cuDNN benchmark mode (auto-selects optimal kernels)
    3. Proper warmup before timing
    4. torch.cuda.synchronize() for accurate measurements
    5. NVTX markers for profiler visibility
    
    BEFORE (baseline):
        - No TF32
        - No cuDNN benchmarking
        - May include cold-start in timing
        
    AFTER (optimized):
        - TF32 enabled: ~2x faster FP32 matmuls on Ampere+
        - cuDNN finds optimal convolution algorithms
        - Warmup ensures kernels are cached
    
    WHY these are Ch1 techniques:
        These are foundational settings that EVERY GPU program should use.
        They require no code changes to the algorithm itself.
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs = None
        self.batch_size = 64
        self.seq_len = 1024
        self.hidden_dim = 512
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(WORKLOAD.performance_microbatches),
            tokens_per_iteration=float(WORKLOAD.performance_microbatches * 64 * 1024),
        )
    
    def setup(self) -> None:
        """Setup with Ch1-appropriate optimizations."""
        
        if torch.cuda.is_available():
            # ============================================================
            # Ch1 Optimization 1: cuDNN Benchmark Mode
            # ============================================================
            # Allows cuDNN to auto-tune and select the fastest algorithm
            # for the given input shapes. One-time cost during warmup.
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # ============================================================
            # Ch1 Optimization 2: TF32 for Tensor Cores
            # ============================================================
            # TF32 uses Tensor Cores for FP32 matmuls with minimal
            # precision loss. ~2x speedup on Ampere/Hopper/Blackwell.
            enable_tf32()
        
        torch.manual_seed(42)
        
        # Simple model - no advanced optimizations
        self.model = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 256, bias=True),
        ).to(self.device).eval()
        self.model.requires_grad_(False)
        
        # Create input batches (no graph capture - that's Ch12!)
        self.inputs = [
            torch.randn(self.batch_size, self.seq_len, device=self.device)
            for _ in range(WORKLOAD.performance_microbatches)
        ]
        
        # ============================================================
        # Ch1 Optimization 3: Proper Warmup
        # ============================================================
        # Run a few iterations to:
        # - JIT compile any lazy kernels
        # - Let cuDNN find optimal algorithms
        # - Warm up GPU caches
        with torch.no_grad():
            for inp in self.inputs[:2]:
                _ = self.model(inp)
        
        if self.device.type == "cuda":
            # ============================================================
            # Ch1 Optimization 4: Synchronize After Warmup
            # ============================================================
            torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark with Ch1-appropriate techniques."""
        # ============================================================
        # Ch1 Optimization 5: NVTX Range for Profiling
        # ============================================================
        # Makes this region visible in Nsight Systems
        with self._nvtx_range("ch1_optimized_performance"):
            total = 0.0
            with torch.no_grad():
                for inp in self.inputs:
                    output = self.model(inp)
                    total += output.sum()
            
            # Proper synchronization for timing accuracy
            self._synchronize()
            self._checksum = total
    
    def teardown(self) -> None:
        """Clean up resources."""
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
            use_subprocess=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Ch1-appropriate metrics: basic throughput and efficiency."""
        return {
            "ch1.batch_size": float(self.batch_size),
            "ch1.sequence_length": float(self.seq_len),
            "ch1.tf32_enabled": float(torch.backends.cuda.matmul.allow_tf32),
            "ch1.cudnn_benchmark": float(torch.backends.cudnn.benchmark),
        }

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.inputs is None:
            return "Input not initialized"
        return None


# Keep old class name for backwards compatibility
OptimizedWarpSpecializationBenchmark = OptimizedPerformanceBenchmark


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedPerformanceBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    config = benchmark.get_config()
    config.use_subprocess = False
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=config
    )
    result = harness.benchmark(benchmark)
    print(f"\nCh1 Optimized Performance: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("\nApplied Ch1 Techniques:")
    print("  ✅ TF32 enabled for Tensor Cores")
    print("  ✅ cuDNN benchmark mode")
    print("  ✅ Proper warmup iterations")
    print("  ✅ NVTX markers for profiling")
