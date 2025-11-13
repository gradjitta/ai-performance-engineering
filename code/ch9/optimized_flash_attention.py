"""optimized_flash_attention.py - Optimized FlashAttention in kernel efficiency/arithmetic intensity context.

Demonstrates FlashAttention for memory-efficient attention computation.
Flash attention: Uses FlashAttention for reduced memory and improved efficiency.
Provides O(N) memory complexity instead of O(N^2).
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
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class OptimizedFlashAttentionBenchmark(Benchmark):
    """Optimized: FlashAttention for memory-efficient attention.
    
    Flash attention: Uses FlashAttention for reduced memory and improved efficiency.
    Provides O(N) memory complexity instead of O(N^2).
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.qkv_proj: Optional[nn.Linear] = None
        self.output_proj: Optional[nn.Linear] = None
        self.num_heads: Optional[int] = None
        self.head_dim: Optional[int] = None
    
    def setup(self) -> None:
        """Setup: Initialize FlashAttention model."""

        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: FlashAttention
        # Flash attention reduces memory usage and improves efficiency
        
        hidden_dim = 256
        num_heads = 8
        head_dim = hidden_dim // num_heads
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False).to(self.device).to(torch.float16).eval()
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False).to(self.device).to(torch.float16).eval()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # FlashAttention-optimized input
        self.input = torch.randn(4, 128, hidden_dim, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: FlashAttention computation."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_flash_attention", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: FlashAttention
                # Uses FlashAttention for memory-efficient attention
                # Flash attention: reduced memory (O(N) vs O(N^2))
                batch, seqlen, _ = self.input.shape
                qkv = self.qkv_proj(self.input)
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.view(batch, seqlen, self.num_heads, self.head_dim)
                k = k.view(batch, seqlen, self.num_heads, self.head_dim)
                v = v.view(batch, seqlen, self.num_heads, self.head_dim)
                output = flash_attn_func(q, k, v, causal=True)
                output = output.reshape(batch, seqlen, -1)
                output = self.output_proj(output)
                
                # Optimization: FlashAttention benefits
                # - Reduced memory usage (O(N) vs O(N^2))
                # - Improved efficiency
                # - Better kernel efficiency
                # - Memory-efficient attention computation
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
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
        if self.qkv_proj is None or self.output_proj is None:
            return "Projection layers not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedFlashAttentionBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
    mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedFlashAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Flash Attention")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()

