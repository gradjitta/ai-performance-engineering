# Chapter-Scoped Performance Optimization Analysis

This document provides optimization suggestions that are **properly scoped to each chapter's content**. The rule is simple: each `chXX/optimized_*.py` should demonstrate techniques from **that chapter only**, not jump ahead to future chapters. Labs can use **any technique** as they are end-to-end exercises.

---

## Chapter Content Reference

| Chapter | Topic | Key Techniques |
|---------|-------|----------------|
| Ch1 | Introduction | Overview, benchmarking basics |
| Ch2 | Hardware | Grace CPU, Blackwell GPU, NVLink-C2C, memory hierarchy |
| Ch3 | OS/Docker/K8s | NUMA, CPU pinning, drivers, containers |
| Ch4 | Distributed | NCCL, streams for comm/compute overlap, multi-GPU |
| Ch5 | Storage I/O | GDS, sequential access, NVMe tuning, prefetching |
| Ch6 | GPU Architecture | Threads/warps/blocks, SIMT, memory hierarchy basics |
| Ch7 | Memory Access | Coalescing, vectorization, shared memory, bank conflicts |
| Ch8 | Occupancy | Warp efficiency, ILP, profiling, launch bounds |
| Ch9 | Arithmetic Intensity | Tiling, kernel fusion, Tensor Cores, roofline |
| Ch10 | Intra-Kernel Pipelining | Double buffering, clusters, DSMEM, warp specialization, TMA |
| Ch11 | Streams | Inter-kernel pipelining, async execution |
| Ch12 | CUDA Graphs | Graph capture/replay, dynamic scheduling, atomics |
| Ch13 | PyTorch Profiling | PyTorch profiler, FP8, data pipelines, DDP/FSDP |
| Ch14 | torch.compile & Triton | TorchDynamo, TorchInductor, Triton kernels |
| Ch15-20 | Advanced | MoE, paged attention, vLLM, speculative decoding, FP4 |

---

## Ch1: Introduction

### Current Issues
- `optimized_warp_specialization.py` uses **CUDA Graphs** (Ch12 technique) - this is out of scope
- Warp specialization itself is a Ch10 topic, not Ch1

### What Ch1 Examples Should Demonstrate
Ch1 is introductory. Examples should show:
1. **Basic benchmarking patterns** - timing, warmup, iterations
2. **Simple baseline vs. optimized comparison** without advanced techniques
3. **Profiling setup** basics (NVTX markers)

### Recommended Fix
```python
# ch1/optimized_performance.py should show:
# - Proper warmup iterations
# - torch.cuda.synchronize() for accurate timing
# - NVTX range marking
# - Basic TF32 enable (torch.backends.cuda.matmul.allow_tf32 = True)
# 
# NOT: CUDA graphs, warp specialization, streams, etc.
```

**Action**: Rename or refactor `optimized_warp_specialization.py` to demonstrate only Ch1-appropriate techniques.

---

## Ch2: AI System Hardware Overview

### Current State
✅ `optimized_grace_coherent_memory.py` is **appropriately scoped**:
- Zero-copy for small buffers (Grace-Blackwell NVLink-C2C)
- NUMA-aware allocation
- Strategy selection based on buffer size

### Possible Enhancements (Ch2-appropriate)
1. **NVLink bandwidth measurement** - Show bidirectional throughput
2. **Memory tier awareness** - Demonstrate HBM vs. LPDDR5X access patterns
3. **Grace-Blackwell detection** - Proper compute capability checks

### Example Ch2-Appropriate Optimization
```python
# Ch2 is about HARDWARE. Optimizations should focus on:
# - Memory placement (HBM vs. CPU DRAM)
# - NVLink-C2C coherent access
# - Buffer sizing thresholds for different transfer strategies
# 
# NOT: CUDA streams (Ch11), graphs (Ch12), torch.compile (Ch14)
```

---

## Ch3: OS, Docker, Kubernetes Tuning

### Current State
Ch3 examples focus on NUMA awareness and container configuration.

### Ch3-Appropriate Optimizations
1. **NUMA binding** - `numactl`, CPU affinity
2. **GPU persistence mode** - Reduce driver latency
3. **Huge pages** - Memory allocation efficiency
4. **Container device passthrough** - NVIDIA Container Toolkit

### Example Code
```python
# Ch3 optimizations:
import os
os.sched_setaffinity(0, {0, 1, 2, 3})  # Pin to NUMA node 0 CPUs

# NOT: CUDA kernels, torch.compile, streams, etc.
```

---

## Ch4: Tuning Distributed Training and Inference

### Current State
Ch4 covers NCCL, GPUDirect, and overlapping communication with computation.

### Ch4-Appropriate Optimizations
1. **NCCL environment tuning** - `NCCL_ALGO`, `NCCL_PROTO`
2. **Async all-reduce** - Using separate streams
3. **Gradient bucketing** - DDP bucket size tuning
4. **GPUDirect RDMA** - Direct NIC-to-GPU transfers

### Important Note
Ch4 can use **basic streams** for overlapping, but **NOT** CUDA graphs or torch.compile.

```python
# Ch4: Overlap gradient all-reduce with backward pass
# Uses separate streams for communication
comm_stream = torch.cuda.Stream()
with torch.cuda.stream(comm_stream):
    dist.all_reduce(gradients, async_op=True)
```

---

## Ch5: GPU-Based Storage I/O Optimizations

### Current State
Ch5 covers GDS, sequential access, and data loading.

### Ch5-Appropriate Optimizations
1. **GDS (GPUDirect Storage)** - Direct SSD-to-GPU DMA
2. **Sequential read patterns** - Large contiguous reads
3. **DataLoader tuning** - `num_workers`, `pin_memory`, `prefetch_factor`
4. **io_uring** - Async I/O for high IOPS

### Example
```python
# Ch5: DataLoader optimization
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)
```

---

## Ch6: GPU Architecture, CUDA Programming, Memory Hierarchy

### Current State
Ch6 introduces threads, warps, blocks, grids, and memory hierarchy basics.

### Ch6-Appropriate Optimizations
1. **Thread block sizing** - Multiples of 32 (warp size)
2. **Memory hierarchy awareness** - Register → Shared → L1 → L2 → HBM
3. **Basic shared memory usage** - Manual tiling
4. **Warp divergence avoidance** - Uniform control flow

### NOT Ch6 Topics
- TMA (introduced in Ch6 but advanced usage is Ch10)
- Warp specialization (Ch10)
- Streams/Graphs (Ch11/12)

```cuda
// Ch6: Basic tiled shared memory
__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
tile[ty][tx] = input[row * N + col];
__syncthreads();
```

---

## Ch7: Memory Coalescing, Vectorization, Shared Memory

### Current State
Ch7 has excellent examples:
- `baseline_copy_scalar.cu` vs. `optimized_copy_coalesced.cu`
- `baseline_transpose.cu` vs. `optimized_transpose_padded.cu`
- TMA examples

### Ch7-Appropriate Optimizations
1. **Coalesced memory access** - Sequential thread → sequential memory
2. **Vectorized loads** - `float4`, `float2`, 128-bit loads
3. **Shared memory tiling** - Load once, use many times
4. **Bank conflict avoidance** - Padding, swizzling

### Example Enhancement
```cuda
// Ch7: Vectorized load (128-bit = 4 floats)
float4 vec = *reinterpret_cast<const float4*>(&input[idx * 4]);
// Coalesced: threads 0-31 load consecutive float4s
```

### Missing Optimization Opportunity
The transpose examples could add **swizzling** for bank conflict elimination:
```cuda
// Instead of just padding, show swizzle pattern:
// tile[ty][tx ^ ty] = input[...];  // XOR swizzle
```

---

## Ch8: Occupancy Tuning, Warp Efficiency, Pipeline Scheduling

### Current State
Ch8 focuses on profiling, occupancy, and ILP.

### Ch8-Appropriate Optimizations
1. **Launch bounds** - `__launch_bounds__(256, 4)` for occupancy hints
2. **Register pressure management** - Avoid spilling
3. **ILP within threads** - Unroll loops, process multiple elements
4. **Profiler-guided optimization** - Nsight Compute metrics

### Example
```cuda
// Ch8: Launch bounds for occupancy
__global__ __launch_bounds__(256, 4)  // 256 threads/block, min 4 blocks/SM
void kernel(...) {
    // Process multiple elements per thread (ILP)
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc0 += input[idx + i * stride];
    }
}
```

---

## Ch9: Increasing Arithmetic Intensity

### Current State
Ch9 covers tiling, kernel fusion, and Tensor Cores.

### Ch9-Appropriate Optimizations
1. **Multi-level tiling** - Register → shared → global
2. **Kernel fusion** - Combine elementwise ops
3. **Tensor Core usage** - `wmma` API, `mma` instructions
4. **Roofline analysis** - FLOPS/byte improvement

### Example
```cuda
// Ch9: Fused elementwise operation (sin + sqrt)
// Instead of 2 kernels, one kernel with higher arithmetic intensity
z[i] = sqrt(sin(x[i]));  // 2 ops per load, 1 store
```

### NOT Ch9 Topics
- torch.compile fusion (that's Ch14)
- Async pipelining (that's Ch10)

---

## Ch10: Intra-Kernel Pipelining and Thread Block Clusters

### Current State
✅ `optimized_flash_attention.py` now properly demonstrates Ch10 concepts:
- Tiled attention (SDPA FlashAttention backend)
- O(seq_len) memory via tiling
- Producer-consumer pipelining analogy

### Ch10-Appropriate Optimizations
1. **Double buffering** - `cuda::pipeline`, 2-stage/3-stage
2. **TMA (Tensor Memory Accelerator)** - Hardware async copy
3. **Thread block clusters** - Multi-SM cooperation via DSMEM
4. **Warp specialization** - Loader/compute/storer warps

### Example Double Buffering
```cuda
// Ch10: 2-stage double buffering with cuda::pipeline
cuda::pipeline<2> pipe;
pipe.producer_acquire();
cuda::memcpy_async(smem_buffer[stage], global_ptr, bytes, pipe);
pipe.producer_commit();
// ... compute on other buffer ...
pipe.consumer_wait();
```

### Missing Optimization Opportunity
Add a **3-stage pipeline** example to show deeper latency hiding:
```cuda
// 3-stage: prefetch(i+2), compute(i+1), consume(i)
cuda::pipeline<3> pipe;
```

---

## Ch11: Inter-Kernel Pipelining with CUDA Streams

### Current State
✅ `optimized_streams.py` properly demonstrates streams.

### Ch11-Appropriate Optimizations
1. **Multiple CUDA streams** - Concurrent kernel execution
2. **Stream-ordered memory** - `cudaMallocAsync`/`cudaFreeAsync`
3. **Event synchronization** - `cudaEventRecord`, `cudaStreamWaitEvent`
4. **H2D/D2H overlap** - Copy while computing

### Example
```python
# Ch11: Stream-based overlap
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

with torch.cuda.stream(stream1):
    tensor1.matmul_(weight1)  # Compute on stream1

with torch.cuda.stream(stream2):
    tensor2.copy_(cpu_data, non_blocking=True)  # Copy on stream2

# Both operations can overlap
```

---

## Ch12: CUDA Graphs and Dynamic Scheduling

### Current State
✅ `optimized_cuda_graphs.py` properly demonstrates graph capture/replay.

### Ch12-Appropriate Optimizations
1. **CUDA graph capture** - Eliminate launch overhead
2. **Graph replay** - Reuse captured work
3. **Atomic work queues** - Dynamic load balancing
4. **Batch atomics** - Reduce contention

### Example
```python
# Ch12: CUDA graph capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(static_input)  # Capture
g.replay()  # Fast replay
```

---

## Ch13: Profiling and Tuning PyTorch Workloads

### Current State
✅ `optimized_precisionfp8_te.py` demonstrates FP8 with Transformer Engine.

### Ch13-Appropriate Optimizations
1. **PyTorch profiler** - `torch.profiler.profile()`
2. **FP8 quantization** - Transformer Engine, `fp8_autocast`
3. **Data pipeline tuning** - DataLoader workers, prefetching
4. **DDP/FSDP** - Distributed training patterns
5. **Fused optimizers** - `fused=True` in AdamW

### NOT Ch13 Topics
- torch.compile (that's Ch14)
- Triton kernels (that's Ch14)

### Enhancement Opportunity
Add `DelayedScaling` recipe tuning:
```python
# Ch13: Optimized FP8 recipe
fp8_recipe = te_recipe.DelayedScaling(
    margin=8,               # More headroom for outliers
    interval=1,             # Update every step
    amax_history_len=1024,  # Longer history for stability
    hysteresis=True,        # Prevent scale oscillation
)
```

---

## Ch14: PyTorch Compiler and OpenAI Triton

### Current State
✅ `optimized_model_eager.py` demonstrates `torch.compile`.

### Ch14-Appropriate Optimizations
1. **torch.compile modes** - `reduce-overhead`, `max-autotune`
2. **Triton kernels** - Custom fused operations
3. **Graph breaks** - Minimize with `fullgraph=True`
4. **Dynamic shapes** - `torch._dynamo.mark_dynamic`
5. **Compiled autograd** - `torch.compile(backend="cudagraphs")`

### Example
```python
# Ch14: max-autotune with cudagraphs
compiled_model = torch.compile(
    model,
    mode="max-autotune",
    fullgraph=True,
)
```

### Missing Optimization Opportunity
Add **Triton autotuning** example:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

---

## Labs: End-to-End Optimizations

Labs can use **ALL techniques** from any chapter. They should demonstrate real-world, full-stack optimization.

### Lab Optimization Checklist

| Technique | Labs That Should Use It |
|-----------|------------------------|
| TF32/BF16/FP8/FP4 | fast_nanochat, kv_cache_compression, distributed_training |
| torch.compile | fast_nanochat_compile, flexattention |
| CUDA Graphs | fast_nanochat_graph, persistent_decode |
| Streams | fast_nanochat_streams, async_input_pipeline |
| Double Buffering | fast_nanochat_double_buffer_tma |
| Warp Specialization | fast_nanochat_warp_specialized |
| TMA | blackwell_matmul_tma |
| Thread Block Clusters | blackwell_matmul_cluster |
| FSDP2 FP8 | distributed_training |
| Speculative Decoding | speculative_decode |

### Example: `labs/fast_nanochat/` Should Combine
```python
# Labs can use EVERYTHING:
# 1. FP8 (Ch13)
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    ...

# 2. torch.compile (Ch14)
compiled_model = torch.compile(model, mode="max-autotune")

# 3. CUDA Graphs (Ch12)
with torch.cuda.graph(g):
    output = compiled_model(static_input)

# 4. Streams for prefill/decode overlap (Ch11)
prefill_stream = torch.cuda.Stream()
decode_stream = torch.cuda.Stream()

# 5. Double buffering (Ch10)
# ... pipeline KV cache updates ...
```

---

## Summary: Chapter-Appropriate vs. Cross-Chapter

### Chapter Examples: Use ONLY That Chapter's Techniques

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| Ch1 using CUDA graphs | Ch1 using basic timing/warmup |
| Ch6 using torch.compile | Ch6 using shared memory basics |
| Ch9 using async pipeline | Ch9 using kernel fusion |
| Ch10 using torch.compile | Ch10 using double buffering |

### Labs: Use ALL Techniques

Labs are **end-to-end** exercises where readers apply everything learned:
- FP8/FP4 quantization
- torch.compile
- CUDA graphs
- Streams
- Pipelining
- TMA
- Clusters
- Warp specialization

---

## Specific Files Requiring Updates

### High Priority

1. **`ch1/optimized_warp_specialization.py`**
   - Issue: Uses CUDA graphs (Ch12) and claims warp specialization (Ch10)
   - Fix: Rename to `optimized_basic_benchmark.py` and remove CUDA graphs

2. **`ch1/baseline_warp_specialization.py`**
   - Issue: Warp specialization is not a Ch1 topic
   - Fix: Rename to `baseline_basic_benchmark.py`

### Medium Priority

3. **`ch7/optimized_transpose.py`**
   - Enhancement: Add swizzle pattern example for bank conflicts

4. **`ch8/optimized_double_buffering_pipelined.py`**
   - Issue: Double buffering is Ch10, not Ch8
   - Fix: Move to ch10 or rename to focus on occupancy tuning

5. **`ch14/triton_examples.py`**
   - Enhancement: Add complete autotuning example with multiple configs

### Labs (Can Use Everything)

6. **`labs/fast_nanochat/optimized_fast_nanochat_fp8.py`**
   - Enhancement: Add CUDA graph capture for decode loop

7. **`labs/flexattention/optimized_flex_attention.py`**
   - Enhancement: Already uses torch.compile, could add CUDA graphs

---

---

## Ch15: Multinode Inference, Parallelism, KV Management

### What Ch15 Teaches
- Disaggregated prefill/decode architecture
- Tensor/Pipeline/Expert/Data/Context parallelism
- KV cache management and transfer (NIXL)
- MoE routing and load balancing

### Current State
✅ Files like `optimized_moe_shared_expert_overlap.py` appropriately demonstrate:
- Stream-overlapped MoE dispatch
- All-to-all communication patterns
- Expert parallelism with top-k routing

### Ch15-Appropriate Optimizations
```python
# Ch15: Stream-overlapped MoE dispatch
for s_idx, stream in enumerate(streams):
    with torch.cuda.stream(stream):
        for eid in unique_expert_ids:
            out[mask] += self.experts[eid](tokens[mask]) * weights[mask]
```

### Missing Optimization Opportunities
1. **NIXL KV transfer** - Show GPUDirect RDMA for KV cache handoff
2. **Disaggregated scheduler** - Route prefill vs decode to different pools
3. **Prefix caching** - Hash-based KV reuse across requests

---

## Ch16: Profiling, Debugging, Production Inference

### What Ch16 Teaches
- Prometheus/Grafana monitoring
- DCGM metrics collection
- Nsight Systems/Compute profiling
- GPTQ/AWQ quantization for inference
- PagedAttention

### Current State
✅ Files like `optimized_paged_attention_blackwell.py` demonstrate PagedAttention
✅ `dcgm_prometheus_exporter.py` shows metrics collection

### Ch16-Appropriate Optimizations
```python
# Ch16: DCGM metrics for monitoring
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
util = pynvml.nvmlDeviceGetUtilizationRates(handle)
mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
```

### Enhancement Opportunities
1. **AWQ quantization example** - 4-bit weight-only quantization
2. **Prefix cache hit rate monitoring** - Track cache efficiency
3. **Latency percentile tracking** - P95/P99 SLO monitoring

---

## Ch17: vLLM/SGLang Deployment and Router Optimization

### What Ch17 Teaches
- vLLM/SGLang deployment patterns
- Dynamic routing and scheduling
- Pipeline parallelism for inference
- Prefill-decode disaggregation at scale

### Current State
✅ `optimized_prefill_decode_disagg_multigpu.py` shows disaggregated deployment
✅ `optimized_dynamic_routing.py` shows request routing

### Ch17-Appropriate Optimizations
```python
# Ch17: Dynamic routing based on sequence length
def route_request(seq_len, gpu_util):
    if seq_len > 4096 or gpu_util > 0.8:
        return "pipeline_parallel_pool"
    return "tensor_parallel_pool"
```

---

## Ch18: Advanced Prefill-Decode and Attention Optimization

### What Ch18 Teaches
- FlashMLA, ThunderMLA decode kernels
- FlexDecoding (PyTorch)
- Speculative decoding (Medusa, EAGLE)
- PagedAttention optimizations
- CUDA graph bucketing for decode

### Current State
✅ `optimized_speculative_decoding.py` properly demonstrates:
- Draft/target model architecture
- Parallel token prediction and verification

✅ `optimized_flexdecoding.py` shows PyTorch FlexAttention decode backend

### Ch18-Appropriate Optimizations
```python
# Ch18: Speculative decoding with draft model
draft_tokens = draft_model.generate(input_ids, num_tokens=K)
target_probs = target_model(draft_tokens)
accepted = verify_tokens(draft_tokens, target_probs)
```

### Enhancement Opportunities
1. **CUDA graph bucketing** - Capture graphs for different sequence lengths
2. **Multi-draft speculative decoding** - Use multiple draft models

---

## Ch19: Dynamic and Adaptive Inference

### What Ch19 Teaches
- Dynamic parallelism switching (TP vs PP)
- Dynamic precision (FP8/FP4 based on confidence)
- Real-time cache management
- RL-based tuning

### Current State
✅ `optimized_nvfp4_training.py` shows NVFP4 precision
✅ `dynamic_precision_switching.py` shows confidence-based precision selection

### Ch19-Appropriate Optimizations
```python
# Ch19: Dynamic precision based on entropy
entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
if entropy.mean() < CONFIDENCE_THRESHOLD:
    # Use FP4 for high-confidence predictions
    use_fp4 = True
else:
    # Use FP8 for uncertain predictions
    use_fp4 = False
```

---

## Ch20: AI-Assisted Performance Optimizations

### What Ch20 Teaches
- AlphaTensor-style algorithm discovery
- AI-generated CUDA/Triton kernels
- RL-based kernel optimization
- Future of AI systems engineering

### Current State
✅ `ai_kernel_generator.py` demonstrates AI-assisted kernel generation concept
✅ `optimized_autotuning.py` shows autotuning patterns

### Ch20-Appropriate Optimizations
```python
# Ch20: AI-assisted kernel generation loop
for iteration in range(max_iters):
    code = llm.generate_kernel(prompt)
    valid, runtime = verifier.test(code)
    if valid and runtime < target:
        return code
    prompt = refine_with_feedback(prompt, verifier.feedback)
```

---

## Labs: End-to-End Integration

Labs can use **ALL techniques** from any chapter. Here's the complete lab analysis:

### `labs/fast_nanochat/`

| File | Techniques Used | Status |
|------|-----------------|--------|
| `optimized_fast_nanochat_fp8.py` | FP8, streams, pinned memory | ✅ Good |
| `optimized_fast_nanochat_graph.py` | CUDA graphs | ✅ Good |
| `optimized_fast_nanochat_compile.py` | torch.compile | ✅ Good |
| `optimized_fast_nanochat_streams.py` | CUDA streams | ✅ Good |
| `optimized_fast_nanochat_warp_specialized.py` | Warp specialization | ✅ Good |

**Enhancement**: Combine ALL techniques in one "ultimate" version:
```python
# labs/fast_nanochat/optimized_fast_nanochat_ultimate.py
# Should use: FP8 + torch.compile + CUDA graphs + streams + warp spec
```

### `labs/flexattention/`

| File | Techniques Used | Status |
|------|-----------------|--------|
| `optimized_flex_attention.py` | torch.compile, block sparsity, relative bias | ✅ Good |

**Enhancement**: Add CUDA graph capture for the compiled attention

### `labs/kv_cache_compression/`

| File | Techniques Used | Status |
|------|-----------------|--------|
| `optimized_kv_cache_nvfp4.py` | NVFP4, Transformer Engine | ✅ Good |

**Enhancement**: Add paged attention integration

### `labs/blackwell_matmul/`

| File | Techniques Used | Status |
|------|-----------------|--------|
| `optimized_blackwell_matmul_tma.py` | TMA | ✅ Good |
| `optimized_blackwell_matmul_cluster.py` | Thread block clusters | ✅ Good |
| `optimized_blackwell_matmul_tcgen05.py` | TCGEN05/TMEM | ✅ Good |

**Status**: Excellent coverage of Blackwell-specific optimizations!

### `labs/speculative_decode/`

Should combine:
- Draft model generation
- Target model verification
- CUDA graphs for decode loop
- FP8 for draft model (smaller, faster)

### `labs/distributed_training/`

Should combine:
- FSDP2 sharding
- FP8 communication compression
- Gradient accumulation
- Overlapped all-reduce

---

## Complete Chapter-Technique Mapping

| Ch | Topic | Appropriate Techniques |
|----|-------|----------------------|
| 1 | Intro | Basic timing, warmup, NVTX |
| 2 | Hardware | NVLink-C2C, NUMA, memory tiers |
| 3 | OS/K8s | CPU pinning, containers, drivers |
| 4 | Distributed | NCCL, basic streams for comm overlap |
| 5 | Storage | GDS, DataLoader, sequential I/O |
| 6 | CUDA Basics | Warps, blocks, shared memory basics |
| 7 | Memory | Coalescing, vectorization, bank conflicts |
| 8 | Occupancy | ILP, launch bounds, profiling |
| 9 | Arithmetic | Tiling, fusion, Tensor Cores |
| 10 | Pipelining | Double buffer, TMA, clusters, warp spec |
| 11 | Streams | Inter-kernel overlap |
| 12 | Graphs | CUDA graphs, atomics, dynamic sched |
| 13 | PyTorch | Profiler, FP8, DDP/FSDP |
| 14 | Compiler | torch.compile, Triton |
| 15 | MoE/PD | Expert parallel, KV mgmt, disagg |
| 16 | Production | Monitoring, quantization, PagedAttn |
| 17 | vLLM/SGLang | Routing, scheduling, deployment |
| 18 | Decode | FlashMLA, FlexDecode, speculative |
| 19 | Adaptive | Dynamic precision, RL tuning |
| 20 | AI-Assisted | Kernel generation, autotuning |
| **Labs** | **All** | **Everything combined!** |

---

## Summary of Issues Found

### High Priority Fixes

| File | Issue | Fix |
|------|-------|-----|
| `ch1/optimized_warp_specialization.py` | Uses CUDA graphs (Ch12) | Remove graphs, use basic timing |
| `ch1/baseline_warp_specialization.py` | Warp spec is Ch10 | Rename to basic benchmark |
| `ch8/optimized_double_buffering_pipelined.py` | Double buffering is Ch10 | Move to ch10 or focus on occupancy |

### Medium Priority Enhancements

| File | Enhancement |
|------|-------------|
| `ch7/optimized_transpose.py` | Add swizzle pattern |
| `ch14/triton_examples.py` | Add complete autotuning |
| `ch15/optimized_disaggregated_inference.py` | Add NIXL KV transfer |
| `ch18/optimized_speculative_decoding.py` | Add CUDA graph bucketing |

### Labs Should Add

| Lab | Missing Technique |
|-----|-------------------|
| `fast_nanochat` | Ultimate version combining all |
| `flexattention` | CUDA graph capture |
| `speculative_decode` | FP8 draft model |
| `distributed_training` | FP8 communication |

---

## Conclusion

The key insight is that **chapter examples are teaching tools** - they should demonstrate ONE concept clearly. **Labs are integration exercises** - they should combine multiple concepts for maximum performance.

When reviewing or adding examples:
1. Check what chapter the file is in
2. Verify techniques match that chapter's content
3. Move cross-chapter examples to labs or later chapters
4. Labs can (and should) use everything!

