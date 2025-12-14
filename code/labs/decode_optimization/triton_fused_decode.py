"""Warp-specialized Triton fused MLP for decode step (LayerNorm + Linear -> GELU -> Linear).

Designed for small batch (e.g., 8) and hidden sizes around 2-4K on Blackwell (SM100/121).
Uses tensor descriptors (TMA) and warp specialization to overlap copies/compute.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import triton
import triton.language as tl
try:
    import triton.runtime._allocation as _triton_alloc  # type: ignore
except Exception:
    _triton_alloc = None

from core.benchmark.triton_compat import ensure_triton_compat

def _install_triton_allocator() -> None:
    if _triton_alloc is None:
        return
    
    # Check if allocator is already set
    try:
        current = _triton_alloc._allocator.get()
        if not isinstance(current, _triton_alloc.NullAllocator):
            return  # Already configured
    except Exception:
        pass
    
    class _TorchCudaBuffer:
        """Wrapper for PyTorch's caching allocator pointers."""
        __slots__ = ("_ptr",)

        def __init__(self, ptr: int):
            self._ptr = ptr

        def data_ptr(self) -> int:
            return self._ptr

        def __del__(self):
            if self._ptr:
                try:
                    torch.cuda.caching_allocator_delete(self._ptr)
                except Exception:
                    pass
                self._ptr = 0

    class _TorchCudaAllocator:
        """Allocator that reuses PyTorch's caching allocator for Triton."""

        def __call__(self, size: int, alignment: int, stream):
            if size == 0:
                return _TorchCudaBuffer(0)
            if stream is None:
                current_stream = torch.cuda.current_stream()
                stream = current_stream.cuda_stream
                device_idx = current_stream.device.index
            else:
                device_idx = torch.cuda.current_device()
            if device_idx is None:
                device_idx = torch.cuda.current_device()
            ptr = torch.cuda.caching_allocator_alloc(size, device_idx, stream=stream)
            return _TorchCudaBuffer(ptr)

    import triton
    triton.set_allocator(_TorchCudaAllocator())


def _triton_allocator_ready() -> bool:
    if _triton_alloc is None:
        return False
    try:
        current = _triton_alloc._allocator.get()
        return not isinstance(current, _triton_alloc.NullAllocator)
    except Exception:
        return False


def _ensure_triton_runtime_ready() -> None:
    """Ensure Triton is ready in the *current* execution context.

    Triton's allocator is stored in a ``contextvars.ContextVar``. The harness may
    run warmup and measurement in different threads (timeout enforcement), which
    means a one-time global guard is insufficient: the allocator must be set for
    each thread/context that launches Triton kernels.
    """
    if not _triton_allocator_ready():
        # Ensure CUDA is initialized before configuring Triton allocator.
        torch.cuda.current_stream()
        _install_triton_allocator()
        if not _triton_allocator_ready():
            raise RuntimeError("Triton allocator is not configured (triton.set_allocator() failed)")
    ensure_triton_compat()


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 1,
                "BLOCK_N": 128,
                "GROUP_N": 4,
                "BLOCK_K": 128,
                "BLOCK_H1": 128,
                "NUM_STAGES": 3,
            },
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 1,
                "BLOCK_N": 128,
                "GROUP_N": 2,
                "BLOCK_K": 128,
                "BLOCK_H1": 128,
                "NUM_STAGES": 3,
            },
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 1,
                "BLOCK_N": 128,
                "GROUP_N": 8,
                "BLOCK_K": 128,
                "BLOCK_H1": 128,
                "NUM_STAGES": 3,
            },
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["M", "H"],
)
@triton.jit
def fused_decode_mlp_kernel(
    X_ptr,
    W1_ptr,
    B1_ptr,
    W2_ptr,
    B2_ptr,
    Out_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_w1k,
    stride_w1n,
    stride_w2k,
    stride_w2n,
    stride_outm,
    stride_outn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H1: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * (GROUP_N * BLOCK_N)

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # TMA descriptors
    X_desc = tl.make_tensor_descriptor(
        X_ptr,
        shape=[M, H],
        strides=[stride_xm, stride_xk],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    W1_desc = tl.make_tensor_descriptor(
        W1_ptr,
        shape=[H, H],
        strides=[stride_w1k, stride_w1n],
        block_shape=[BLOCK_K, BLOCK_H1],
    )
    W2_desc = tl.make_tensor_descriptor(
        W2_ptr,
        shape=[H, H],
        strides=[stride_w2k, stride_w2n],
        block_shape=[BLOCK_H1, BLOCK_N],
    )

    # Triton does not support list comprehensions in kernels; unroll up to GROUP_N=8.
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over hidden (K) chunks for the first matmul, fusing projection 2 inside.
    # Note: warp_specialize=True causes MLIR PassManager failures on some Blackwell configs.
    # Use software pipelining (num_stages) instead for stable performance.
    H_tiles = (H + BLOCK_H1 - 1) // BLOCK_H1
    K_tiles = (H + BLOCK_K - 1) // BLOCK_K
    for ht in tl.range(0, H_tiles, num_stages=NUM_STAGES):
        h_start = ht * BLOCK_H1

        # Compute hidden1 chunk = X @ W1_chunk
        hidden_chunk = tl.zeros((BLOCK_M, BLOCK_H1), dtype=tl.float32)
        for kt in tl.range(0, K_tiles):
            k_start = kt * BLOCK_K
            x_block = tl.load_tensor_descriptor(X_desc, [m0, k_start])
            w1_block = tl.load_tensor_descriptor(W1_desc, [k_start, h_start])
            hidden_chunk += tl.dot(x_block, w1_block, out_dtype=tl.float32)

        # Add bias1 slice and GELU
        bias1 = tl.load(B1_ptr + h_start + tl.arange(0, BLOCK_H1))
        bias1 = bias1[None, :]
        hidden_chunk += bias1
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        hidden_chunk = 0.5 * hidden_chunk * (1.0 + tl.math.erf(hidden_chunk * 0.7071067811865475))

        # Projection 2 for this hidden slice
        if GROUP_N >= 1:
            g0_n0 = n0
            w2_0 = tl.load_tensor_descriptor(W2_desc, [h_start, g0_n0])
            acc0 += tl.dot(hidden_chunk, w2_0.to(tl.float32), out_dtype=tl.float32)
        if GROUP_N >= 2:
            g1_n0 = n0 + 1 * BLOCK_N
            w2_1 = tl.load_tensor_descriptor(W2_desc, [h_start, g1_n0])
            acc1 += tl.dot(hidden_chunk, w2_1.to(tl.float32), out_dtype=tl.float32)
        if GROUP_N >= 3:
            g2_n0 = n0 + 2 * BLOCK_N
            w2_2 = tl.load_tensor_descriptor(W2_desc, [h_start, g2_n0])
            acc2 += tl.dot(hidden_chunk, w2_2.to(tl.float32), out_dtype=tl.float32)
        if GROUP_N >= 4:
            g3_n0 = n0 + 3 * BLOCK_N
            w2_3 = tl.load_tensor_descriptor(W2_desc, [h_start, g3_n0])
            acc3 += tl.dot(hidden_chunk, w2_3.to(tl.float32), out_dtype=tl.float32)
        if GROUP_N >= 5:
            g4_n0 = n0 + 4 * BLOCK_N
            w2_4 = tl.load_tensor_descriptor(W2_desc, [h_start, g4_n0])
            acc4 += tl.dot(hidden_chunk, w2_4.to(tl.float32), out_dtype=tl.float32)
        if GROUP_N >= 6:
            g5_n0 = n0 + 5 * BLOCK_N
            w2_5 = tl.load_tensor_descriptor(W2_desc, [h_start, g5_n0])
            acc5 += tl.dot(hidden_chunk, w2_5.to(tl.float32), out_dtype=tl.float32)
        if GROUP_N >= 7:
            g6_n0 = n0 + 6 * BLOCK_N
            w2_6 = tl.load_tensor_descriptor(W2_desc, [h_start, g6_n0])
            acc6 += tl.dot(hidden_chunk, w2_6.to(tl.float32), out_dtype=tl.float32)
        if GROUP_N >= 8:
            g7_n0 = n0 + 7 * BLOCK_N
            w2_7 = tl.load_tensor_descriptor(W2_desc, [h_start, g7_n0])
            acc7 += tl.dot(hidden_chunk, w2_7.to(tl.float32), out_dtype=tl.float32)

    if GROUP_N >= 1:
        offs_n0 = n0 + tl.arange(0, BLOCK_N)
        bias2_0 = tl.load(B2_ptr + offs_n0)
        out_ptrs0 = Out_ptr + (offs_m[:, None] * stride_outm + offs_n0[None, :] * stride_outn)
        tl.store(out_ptrs0, (acc0 + bias2_0[None, :]).to(OUT_DTYPE))
    if GROUP_N >= 2:
        offs_n1 = n0 + 1 * BLOCK_N + tl.arange(0, BLOCK_N)
        bias2_1 = tl.load(B2_ptr + offs_n1)
        out_ptrs1 = Out_ptr + (offs_m[:, None] * stride_outm + offs_n1[None, :] * stride_outn)
        tl.store(out_ptrs1, (acc1 + bias2_1[None, :]).to(OUT_DTYPE))
    if GROUP_N >= 3:
        offs_n2 = n0 + 2 * BLOCK_N + tl.arange(0, BLOCK_N)
        bias2_2 = tl.load(B2_ptr + offs_n2)
        out_ptrs2 = Out_ptr + (offs_m[:, None] * stride_outm + offs_n2[None, :] * stride_outn)
        tl.store(out_ptrs2, (acc2 + bias2_2[None, :]).to(OUT_DTYPE))
    if GROUP_N >= 4:
        offs_n3 = n0 + 3 * BLOCK_N + tl.arange(0, BLOCK_N)
        bias2_3 = tl.load(B2_ptr + offs_n3)
        out_ptrs3 = Out_ptr + (offs_m[:, None] * stride_outm + offs_n3[None, :] * stride_outn)
        tl.store(out_ptrs3, (acc3 + bias2_3[None, :]).to(OUT_DTYPE))
    if GROUP_N >= 5:
        offs_n4 = n0 + 4 * BLOCK_N + tl.arange(0, BLOCK_N)
        bias2_4 = tl.load(B2_ptr + offs_n4)
        out_ptrs4 = Out_ptr + (offs_m[:, None] * stride_outm + offs_n4[None, :] * stride_outn)
        tl.store(out_ptrs4, (acc4 + bias2_4[None, :]).to(OUT_DTYPE))
    if GROUP_N >= 6:
        offs_n5 = n0 + 5 * BLOCK_N + tl.arange(0, BLOCK_N)
        bias2_5 = tl.load(B2_ptr + offs_n5)
        out_ptrs5 = Out_ptr + (offs_m[:, None] * stride_outm + offs_n5[None, :] * stride_outn)
        tl.store(out_ptrs5, (acc5 + bias2_5[None, :]).to(OUT_DTYPE))
    if GROUP_N >= 7:
        offs_n6 = n0 + 6 * BLOCK_N + tl.arange(0, BLOCK_N)
        bias2_6 = tl.load(B2_ptr + offs_n6)
        out_ptrs6 = Out_ptr + (offs_m[:, None] * stride_outm + offs_n6[None, :] * stride_outn)
        tl.store(out_ptrs6, (acc6 + bias2_6[None, :]).to(OUT_DTYPE))
    if GROUP_N >= 8:
        offs_n7 = n0 + 7 * BLOCK_N + tl.arange(0, BLOCK_N)
        bias2_7 = tl.load(B2_ptr + offs_n7)
        out_ptrs7 = Out_ptr + (offs_m[:, None] * stride_outm + offs_n7[None, :] * stride_outn)
        tl.store(out_ptrs7, (acc7 + bias2_7[None, :]).to(OUT_DTYPE))


def fused_decode_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Fused decode MLP: ``y = GELU(x @ w1 + b1) @ w2 + b2``.

    Args:
        x: [batch, hidden] input
        w1: [hidden, hidden] weight in *[in_features, out_features]* layout (i.e. ``nn.Linear.weight.T``)
        b1: [hidden] bias
        w2: [hidden, hidden] weight in *[in_features, out_features]* layout (i.e. ``nn.Linear.weight.T``)
        b2: [hidden] bias
    
    Uses Triton TMA-backed kernel for optimal performance on Blackwell GPUs.
    """
    _ensure_triton_runtime_ready()
    
    assert x.is_cuda and w1.is_cuda and w2.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert w1.dtype == x.dtype and w2.dtype == x.dtype

    B, H = x.shape
    # This kernel is specialized for TMA-friendly, fully-tiled shapes and does not
    # mask out-of-bounds accesses. Keep it fail-fast.
    if H % 1024 != 0:
        raise ValueError(f"fused_decode_mlp requires hidden_size divisible by 1024, got {H}")
    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(B, meta["BLOCK_M"]), triton.cdiv(H, meta["BLOCK_N"] * meta["GROUP_N"]))
    
    out_dtype = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
    fused_decode_mlp_kernel[grid](
        x,
        w1,
        b1,
        w2,
        b2,
        out,
        B,
        H,
        x.stride(0),
        x.stride(1),
        w1.stride(0),
        w1.stride(1),
        w2.stride(0),
        w2.stride(1),
        out.stride(0),
        out.stride(1),
        OUT_DTYPE=out_dtype,
    )
    return out
