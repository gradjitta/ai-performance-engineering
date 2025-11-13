"""optimized_streams.py - Optimized CUDA streams for parallel execution in GEMM context.

Demonstrates CUDA streams for parallel execution of independent operations.
Streams: Uses CUDA streams to overlap computation and memory transfers.
Improves GPU utilization through parallel execution.
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

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")

class OptimizedStreamsBenchmark(Benchmark):
    """Optimized: CUDA streams for parallel execution.
    
    Streams: Uses CUDA streams to overlap computation and memory transfers.
    Improves GPU utilization through parallel execution.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.num_chunks = 6
        self.num_streams = 3
        self.chunk_size = 256
        self.input_dim = 4096
        self.hidden_dim = 4096
        self.host_batches = []
        self.device_buffers = []
        self.streams = []
    
    def setup(self) -> None:
        """Setup: Initialize model and CUDA streams."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        torch.manual_seed(42)
        # Optimization: CUDA streams for parallel execution
        # Streams allow independent operations to execute concurrently
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.input_dim, bias=False),
        ).to(self.device).half().eval()
        
        self.host_batches = [
            torch.randn(self.chunk_size, self.input_dim, dtype=torch.float16, pin_memory=True)
            for _ in range(self.num_chunks)
        ]
        
        # Create CUDA streams and reusable device buffers for parallel execution
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        self.device_buffers = [
            torch.empty(self.chunk_size, self.input_dim, device=self.device, dtype=torch.float16)
            for _ in range(self.num_streams)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Parallel execution with CUDA streams."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_streams", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: CUDA streams for parallel execution
                # Independent operations execute concurrently on different streams
                # Streams: parallel execution improves GPU utilization
                reductions = []
                for idx, host_batch in enumerate(self.host_batches):
                    stream = self.streams[idx % self.num_streams]
                    device_buffer = self.device_buffers[idx % self.num_streams]
                    with torch.cuda.stream(stream):
                        device_buffer.copy_(host_batch, non_blocking=True)
                        output = self.model(device_buffer)
                        reductions.append(output.sum(dtype=torch.float32))
                
                # Synchronize streams (streams: wait for completion)
                for stream in self.streams:
                    stream.synchronize()
                torch.cuda.synchronize()
                
                # Optimization: CUDA streams benefits
                # - Parallel execution of independent operations
                # - Overlaps computation and memory transfers
                # - Better GPU utilization
                # - Improved throughput through parallelism
                _ = torch.stack(reductions).sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.host_batches = []
        self.device_buffers = []
        self.streams = []
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if not self.host_batches:
            return "Inputs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedStreamsBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedStreamsBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Streams")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
