"""
Self-contained tcgen05 kernel loader for the Matching cuBLAS lab.

This module JIT-compiles the tcgen05 GEMM kernel without depending on
any other chapter or common code.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_LAB_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LAB_DIR.parents[1]  # labs/matching_cublas -> labs -> code

# CUTLASS include paths - check multiple possible locations
_CUTLASS_CANDIDATES = [
    _REPO_ROOT / "third_party" / "cutlass" / "include",  # Standalone CUTLASS with SM100 support
    _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
    _REPO_ROOT / "third_party" / "pytorch-src" / "third_party" / "fbgemm" / "external" / "cutlass" / "include",
]


def _find_cutlass_include() -> Path | None:
    """Find CUTLASS include directory."""
    for cand in _CUTLASS_CANDIDATES:
        if cand.exists():
            return cand
    return None


def _get_cuda_flags() -> list[str]:
    """Get CUDA compiler flags for tcgen05."""
    flags = ["-std=c++20"]
    
    # Add CUTLASS include
    cutlass_inc = _find_cutlass_include()
    if cutlass_inc:
        flags.append(f"-I{cutlass_inc}")
    else:
        raise RuntimeError(
            "CUTLASS include directory not found. "
            "Please ensure third_party/cutlass is available."
        )
    
    # SM100a for Blackwell (enables TMEM/tcgen05)
    major, minor = torch.cuda.get_device_capability()
    if major >= 10:
        flags.append("-gencode=arch=compute_100a,code=sm_100a")
    else:
        raise RuntimeError(
            f"tcgen05 requires SM 10.0+ (Blackwell). "
            f"Current GPU is SM {major}.{minor}"
        )
    
    return flags


def _load_kernel(source_file: Path, name_prefix: str):
    """Generic kernel loader."""
    if not source_file.exists():
        raise FileNotFoundError(f"{source_file.name} not found in {_LAB_DIR}")
    
    cuda_flags = _get_cuda_flags()
    src_hash = hashlib.md5(source_file.read_bytes()).hexdigest()[:8]
    build_name = f"{name_prefix}_{src_hash}"
    
    print(f"  [Compiling {source_file.name} (first time only)...]")
    module = load(
        name=build_name,
        sources=[str(source_file)],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-std=c++20"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return module


@lru_cache(maxsize=1)
def load_tcgen05_module():
    """JIT-compile and load the basic tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_gemm.cu", "lab_tcgen05")


@lru_cache(maxsize=1)
def load_tcgen05_pipelined_module():
    """JIT-compile and load the 2-stage pipelined tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_pipelined.cu", "lab_tcgen05_pipelined")


@lru_cache(maxsize=1)
def load_tcgen05_3stage_module():
    """JIT-compile and load the 3-stage pipelined tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_3stage.cu", "lab_tcgen05_3stage")


@lru_cache(maxsize=1)
def load_tcgen05_swizzled_module():
    """JIT-compile and load the swizzled tile scheduling tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_swizzled.cu", "lab_tcgen05_swizzled")


@lru_cache(maxsize=1)
def load_tcgen05_cluster_module():
    """JIT-compile and load the thread block cluster tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_cluster.cu", "lab_tcgen05_cluster")


@lru_cache(maxsize=1)
def load_tcgen05_persistent_module():
    """JIT-compile and load the persistent kernel tcgen05 GEMM."""
    return _load_kernel(_LAB_DIR / "tcgen05_persistent.cu", "lab_tcgen05_persistent")


@lru_cache(maxsize=1)
def load_tcgen05_warp_spec_module():
    """JIT-compile and load the warp-specialized tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_warp_spec.cu", "lab_tcgen05_warp_spec")





def matmul_tcgen05(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM: C = A @ B^T
    
    Args:
        a: MxK FP16 tensor
        b: NxK FP16 tensor (transposed layout)
    
    Returns:
        MxN FP16 tensor
    """
    module = load_tcgen05_module()
    return module.matmul_tcgen05(a, b)


def matmul_tcgen05_bias_silu(
    a: torch.Tensor, 
    b: torch.Tensor, 
    bias: torch.Tensor
) -> torch.Tensor:
    """Execute tcgen05 GEMM with fused bias+SiLU epilogue.
    
    Args:
        a: MxK FP16 tensor
        b: NxK FP16 tensor (transposed layout)
        bias: N-element FP16 or FP32 bias vector
    
    Returns:
        MxN FP16 tensor with bias+SiLU applied
    """
    module = load_tcgen05_module()
    return module.matmul_tcgen05_bias_silu(a, b, bias)


def matmul_tcgen05_pipelined(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 2-stage pipelined tcgen05 GEMM: C = A @ B^T
    
    Overlaps TMA loads of tile K+1 with compute of tile K.
    
    Args:
        a: MxK FP16 tensor
        b: NxK FP16 tensor (transposed layout)
    
    Returns:
        MxN FP16 tensor
    """
    module = load_tcgen05_pipelined_module()
    return module.matmul_tcgen05_pipelined(a, b)


def matmul_tcgen05_3stage(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 3-stage pipelined tcgen05 GEMM: C = A @ B^T
    
    Deeper pipelining with 3 shared memory buffers.
    Prefetches 2 tiles ahead while computing current tile.
    
    Args:
        a: MxK FP16 tensor
        b: NxK FP16 tensor (transposed layout)
    
    Returns:
        MxN FP16 tensor
    """
    module = load_tcgen05_3stage_module()
    return module.matmul_tcgen05_3stage(a, b)


def matmul_tcgen05_swizzled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute swizzled tcgen05 GEMM: C = A @ B^T
    
    3-stage pipeline with swizzled tile scheduling for L2 optimization.
    Tiles are processed in cache-friendly order using XOR swizzle.
    """
    module = load_tcgen05_swizzled_module()
    return module.matmul_tcgen05_swizzled(a, b)


def matmul_tcgen05_cluster(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with 2x1 thread block clusters: C = A @ B^T
    
    Uses thread block clusters for better L2 cache utilization.
    2 CTAs along M dimension share a cluster, improving A matrix reuse.
    """
    module = load_tcgen05_cluster_module()
    return module.matmul_tcgen05_cluster(a, b)


def matmul_tcgen05_persistent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute persistent kernel tcgen05 GEMM: C = A @ B^T
    
    CTAs stay resident and process multiple output tiles:
    - Launch one CTA per SM
    - Work-stealing for load balancing
    - Better L2 locality between consecutive tiles
    """
    module = load_tcgen05_persistent_module()
    return module.matmul_tcgen05_persistent(a, b)


def matmul_tcgen05_warp_spec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute warp-specialized tcgen05 GEMM: C = A @ B^T
    
    True warp specialization based on CUTLASS patterns:
    - Producer warp: ONLY issues TMA loads (runs ahead)
    - Consumer warp: ONLY does MMA compute
    - 4-stage pipeline for deep latency hiding
    - Producer and consumer run IN PARALLEL
    """
    module = load_tcgen05_warp_spec_module()
    return module.matmul_tcgen05_warp_spec(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_persistent_module():
    """JIT-compile and load the persistent tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_persistent.cu", "lab_tcgen05_persistent")


def matmul_tcgen05_persistent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute persistent tcgen05 GEMM: C = A @ B^T
    
    PERSISTENT kernel - CTAs stay resident and process multiple tiles:
    - Launch exactly num_sms CTAs (one per SM)
    - Each CTA processes multiple tiles in stride
    - Amortizes launch overhead
    - Better L2 cache utilization between consecutive tiles
    """
    module = load_tcgen05_persistent_module()
    return module.matmul_tcgen05_persistent(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_persistent_pipelined_module():
    """JIT-compile and load the persistent+pipelined tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_persistent_pipelined.cu", "lab_tcgen05_persistent_pipelined")


def matmul_tcgen05_persistent_pipelined(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute persistent + 2-stage pipelined tcgen05 GEMM: C = A @ B^T
    
    BEST OF BOTH:
    - Persistent: CTAs stay resident, process multiple tiles
    - Pipelined: 2-stage double buffering within each tile
    - Barriers init ONCE, phase bits continue flipping
    """
    module = load_tcgen05_persistent_pipelined_module()
    return module.matmul_tcgen05_persistent_pipelined(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_true_warp_spec_module():
    """JIT-compile and load the TRUE warp-specialized tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_true_warp_spec.cu", "lab_tcgen05_true_warp_spec")


def matmul_tcgen05_true_warp_spec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute TRUE warp-specialized tcgen05 GEMM: C = A @ B^T
    
    CUTLASS-style warp specialization:
    - Warp 0: PRODUCER - issues TMA loads, runs AHEAD
    - Warp 1: CONSUMER - waits for data, executes MMA
    - 4-stage pipeline with true parallelism
    - Producer can be 3 tiles ahead of consumer!
    """
    module = load_tcgen05_true_warp_spec_module()
    return module.matmul_tcgen05_true_warp_spec(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_no_mma_barrier_module():
    """JIT-compile and load the no-MMA-barrier tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_no_mma_barrier.cu", "lab_tcgen05_no_mma_barrier")


def matmul_tcgen05_no_mma_barrier(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM WITHOUT MMA barrier: C = A @ B^T
    
    Removes the MMA barrier to reduce pipeline bubbles.
    Only TMA barriers are used for synchronization.
    """
    module = load_tcgen05_no_mma_barrier_module()
    return module.matmul_tcgen05_no_mma_barrier(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_large_k_module():
    """JIT-compile and load the large K-tile tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_large_k.cu", "lab_tcgen05_large_k")


def matmul_tcgen05_large_k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with larger K-tiles (128 vs 64): C = A @ B^T
    
    2x larger K-tiles = 2x more MMA work per loop iteration.
    Reduces loop overhead and barrier wait frequency.
    Requires K % 128 == 0.
    """
    module = load_tcgen05_large_k_module()
    return module.matmul_tcgen05_large_k(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_unrolled_module():
    """JIT-compile and load the unrolled tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_unrolled.cu", "lab_tcgen05_unrolled")


def matmul_tcgen05_unrolled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with explicit loop unrolling: C = A @ B^T
    
    Explicitly unrolls the inner K-block loop for better instruction scheduling.
    4-stage pipeline with unrolled MMA.
    """
    module = load_tcgen05_unrolled_module()
    return module.matmul_tcgen05_unrolled(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_6stage_module():
    """JIT-compile and load the 6-stage tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_6stage.cu", "lab_tcgen05_6stage")


def matmul_tcgen05_6stage(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with 6-stage deep pipeline: C = A @ B^T
    
    Very deep pipeline with 6 stages.
    Prologue fills 5 stages, mainloop keeps 5 ahead.
    """
    module = load_tcgen05_6stage_module()
    return module.matmul_tcgen05_6stage(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_cutlass_style_module():
    """JIT-compile and load the CUTLASS-style tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_cutlass_style.cu", "lab_tcgen05_cutlass_style")


def matmul_tcgen05_cutlass_style(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute CUTLASS-style tcgen05 GEMM: C = A @ B^T
    
    Proper producer/consumer state machine pattern:
    - PipelineState with index, phase, count
    - Producer and consumer advance independently
    - Full/empty barrier pairs for proper sync
    """
    module = load_tcgen05_cutlass_style_module()
    return module.matmul_tcgen05_cutlass_style(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_multicast_module():
    """JIT-compile and load the TMA multicast tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_multicast.cu", "lab_tcgen05_multicast")


def matmul_tcgen05_multicast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with TMA multicast: C = A @ B^T
    
    Uses TMA multicast to broadcast B tiles to multiple CTAs.
    Requires cluster launch for full benefit.
    """
    module = load_tcgen05_multicast_module()
    return module.matmul_tcgen05_multicast(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_no_wait_module():
    """JIT-compile and load the no-wait tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_no_wait.cu", "lab_tcgen05_no_wait")


def matmul_tcgen05_no_wait(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM without MMA barrier wait in mainloop: C = A @ B^T
    
    Key CUTLASS pattern: No MMA barrier wait after each k_tile.
    Only wait before epilogue to ensure all MMA complete.
    """
    module = load_tcgen05_no_wait_module()
    return module.matmul_tcgen05_no_wait(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_no_wait_swizzle_module():
    """JIT-compile and load the no-wait swizzled tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_no_wait_swizzle.cu", "lab_tcgen05_no_wait_swizzle")


def matmul_tcgen05_no_wait_swizzle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with no-wait pattern + swizzled tiles: C = A @ B^T
    
    Combines no-wait CUTLASS pattern with swizzled tile scheduling
    for improved L2 cache utilization.
    """
    module = load_tcgen05_no_wait_swizzle_module()
    return module.matmul_tcgen05_no_wait_swizzle(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_no_wait_5stage_module():
    """JIT-compile and load the 5-stage no-wait tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_no_wait_5stage.cu", "lab_tcgen05_no_wait_5stage")


def matmul_tcgen05_no_wait_5stage(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with 5-stage no-wait pattern + swizzle: C = A @ B^T
    
    5-stage pipeline with no MMA barrier wait and swizzled tiles.
    """
    module = load_tcgen05_no_wait_5stage_module()
    return module.matmul_tcgen05_no_wait_5stage(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_warp_parallel_module():
    """JIT-compile and load the warp-parallel tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_warp_parallel.cu", "lab_tcgen05_warp_parallel")


def matmul_tcgen05_warp_parallel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with true warp parallelism: C = A @ B^T
    
    Different warps for different roles:
    - Warp 0: MMA (consumer)
    - Warp 1: Mainloop Load (producer)
    - Warp 2: Epilogue
    
    Producer and consumer run IN PARALLEL!
    """
    module = load_tcgen05_warp_parallel_module()
    return module.matmul_tcgen05_warp_parallel(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_cluster_module():
    """JIT-compile and load the cluster launch tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_cluster.cu", "lab_tcgen05_cluster")


def matmul_tcgen05_cluster(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with cluster launch: C = A @ B^T
    
    Uses cudaLaunchKernelEx with 2x1 cluster.
    CTAs in cluster share B tiles.
    """
    module = load_tcgen05_cluster_module()
    return module.matmul_tcgen05_cluster(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_tma_multicast_module():
    """JIT-compile and load the TMA multicast tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_tma_multicast.cu", "lab_tcgen05_tma_multicast")


def matmul_tcgen05_tma_multicast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with TMA multicast: C = A @ B^T
    
    Uses SM90_TMA_LOAD_MULTICAST for B tiles.
    Only cluster leader loads B, other CTAs receive via multicast.
    Reduces memory bandwidth by 2x for B.
    """
    module = load_tcgen05_tma_multicast_module()
    return module.matmul_tcgen05_tma_multicast(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_sm100_warp_spec_module():
    """JIT-compile and load the SM100 warp-specialized tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_sm100_warp_spec.cu", "lab_tcgen05_sm100_warp_spec")


def matmul_tcgen05_sm100_warp_spec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute SM100 tcgen05 GEMM with true warp specialization: C = A @ B^T
    
    - SM100_TMA_2SM_LOAD for A
    - SM100_TMA_2SM_LOAD_MULTICAST for B
    - Dedicated producer warp (TMA)
    - Dedicated consumer warp (MMA)
    - Cluster 2x1 with B multicast
    """
    module = load_tcgen05_sm100_warp_spec_module()
    return module.matmul_tcgen05_sm100_warp_spec(a, b)


@lru_cache(maxsize=1)
def load_tcgen05_true_warp_spec2_module():
    """JIT-compile and load the true warp-specialized tcgen05 GEMM kernel v2."""
    return _load_kernel(_LAB_DIR / "tcgen05_true_warp_spec2.cu", "lab_tcgen05_true_warp_spec2")


def matmul_tcgen05_true_warp_spec2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with true warp specialization v2: C = A @ B^T
    
    - Warp 0: Producer (issues TMA loads)
    - Warp 1: Consumer (executes MMA)
    - Full producer/consumer pipeline
    """
    module = load_tcgen05_true_warp_spec2_module()
    return module.matmul_tcgen05_true_warp_spec2(a, b)

