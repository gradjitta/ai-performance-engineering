#!/usr/bin/env python3
"""Optimized: vLLM v1 with CUDA graphs and prefix caching.

Demonstrates optimized vLLM v1 usage with:
- Bucketed CUDA graphs for common shapes
- Prefix caching for repeated prompts
- Optimized KV cache management
- Chunked prefill for long contexts
"""

import torch
import time
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)

# Check for vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available, using simulation mode")


class OptimizedVLLMV1Integration:
    """Optimized vLLM v1 with CUDA graphs and prefix caching."""
    
    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        max_tokens: int = 128,
        batch_size: int = 8,
        use_vllm: bool = True,
        enable_chunked_prefill: bool = True,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.enable_chunked_prefill = enable_chunked_prefill
        
        if not self.use_vllm:
            logger.info("Running in simulation mode (vLLM not available)")
    
    def setup(self):
        """Initialize optimized vLLM model."""
        if self.use_vllm:
            # Optimized: CUDA graphs enabled, prefix caching, chunked prefill
            self.llm = LLM(
                model=self.model_name,
                enforce_eager=False,  # Enable CUDA graphs
                enable_prefix_caching=True,  # Enable prefix caching
                enable_chunked_prefill=self.enable_chunked_prefill,
                max_num_batched_tokens=8192,  # Optimize for throughput
                max_num_seqs=256,  # Higher concurrent sequences
                gpu_memory_utilization=0.9,  # Use more GPU memory
                dtype="bfloat16",
                # Blackwell-specific optimizations
                kv_cache_dtype="fp8_e4m3" if torch.cuda.is_available() else "auto",
            )
            
            logger.info(f"Loaded model: {self.model_name}")
            logger.info("Optimized config: CUDA graphs, prefix caching, chunked prefill")
            if torch.cuda.is_available():
                logger.info("Using FP8 KV cache for Blackwell")
        
        # Create prompts with common prefix for caching
        common_prefix = "Once upon a time in a land far away, "
        self.prompts = [
            f"{common_prefix}there was a {i}" 
            for i in range(self.batch_size)
        ]
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0.8,
            top_p=0.95,
        )
    
    def run(self) -> Dict[str, float]:
        """Execute optimized vLLM inference."""
        if not self.use_vllm:
            # vLLM required for this benchmark
            logger.warning("vLLM not available - benchmark requires vLLM installation")
            raise RuntimeError("vLLM required for this benchmark. Install with: pip install vllm")
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Generate (CUDA graphs will be used after warmup)
        outputs = self.llm.generate(self.prompts, self.sampling_params)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = end - start
        
        # Calculate metrics
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput = total_tokens / elapsed
        mean_latency_ms = (elapsed / len(self.prompts)) * 1000
        
        logger.info(f"Throughput: {throughput:.2f} tokens/sec")
        logger.info(f"Mean latency: {mean_latency_ms:.2f} ms")
        
        return {
            "mean_latency_ms": mean_latency_ms,
            "throughput_tokens_per_sec": throughput,
            "total_tokens": total_tokens,
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.use_vllm and hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()


def run_benchmark(
    model_name: str = "facebook/opt-125m",
    max_tokens: int = 128,
    batch_size: int = 8,
    use_vllm: bool = True,
    enable_chunked_prefill: bool = True,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run optimized vLLM v1 benchmark."""
    
    benchmark = OptimizedVLLMV1Integration(
        model_name=model_name,
        max_tokens=max_tokens,
        batch_size=batch_size,
        use_vllm=use_vllm,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(
        iterations=5,  # More iterations to benefit from CUDA graph warmup
        warmup=5,  # Warmup for CUDA graph capture
        profile_mode=profile,
    )
    
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)
    
    result = harness.benchmark(
        benchmark.run,
        name="optimized_vllm_v1_integration"
    )
    
    metrics = benchmark.run()
    benchmark.cleanup()
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        **metrics,
        "model": model_name,
        "optimizations": "cuda_graphs+prefix_caching+chunked_prefill+fp8_kv",
    }


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
