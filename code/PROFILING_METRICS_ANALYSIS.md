# Profiling Metrics Analysis: Chapter-by-Chapter Review

## Executive Summary

This analysis reviews each chapter's profiling approach to ensure we're collecting the **most appropriate metrics** to understand:
1. **WHY something is faster** (root cause analysis)
2. **HOW it can be made faster** (optimization opportunities)

## TWO-LEVEL METRICS SYSTEM

Understanding the distinction between these two mechanisms is critical:

### Level 1: Hardware Profiler Metrics (ncu/nsys)
**What**: Low-level GPU hardware counters collected by NVIDIA profilers
**When collected**: Only when running with `ncu` or `nsys` profiler
**Where defined**: `common/python/profiler_config.py` → `CH6_KERNEL_METRICS`, `CH7_MEMORY_METRICS`, etc.
**Purpose**: Understand hardware-level bottlenecks (stalls, bank conflicts, cache misses)

Example metrics:
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (bank conflicts)
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` (memory stalls)
- `sm__inst_executed_pipe_tensor.sum` (tensor core usage)

### Level 2: Python-Computed Metrics (get_custom_metrics)
**What**: Domain-specific metrics computed in Python during benchmark execution
**When collected**: EVERY benchmark run (no profiler needed)
**Where defined**: Each benchmark's `get_custom_metrics()` method
**Purpose**: Compute derived metrics like arithmetic intensity, speedup ratios, efficiency %

Example usage:
```python
from common.python.benchmark_metrics import compute_roofline_metrics

def get_custom_metrics(self) -> Optional[dict]:
    return compute_roofline_metrics(
        total_flops=self.flops,
        total_bytes=self.bytes,
        elapsed_ms=self._elapsed_ms,
        precision="fp16",
    )
```

### Helper Module: `common/python/benchmark_metrics.py`
Pre-built metric calculators for each chapter's domain:
- `compute_memory_transfer_metrics()` - Chapter 2
- `compute_kernel_fundamentals_metrics()` - Chapter 6
- `compute_memory_access_metrics()` - Chapter 7
- `compute_optimization_metrics()` - Chapter 8
- `compute_roofline_metrics()` - Chapter 9
- `compute_stream_metrics()` - Chapter 11
- `compute_graph_metrics()` - Chapter 12
- `compute_precision_metrics()` - Chapters 13, 19
- `compute_inference_metrics()` - Chapters 15-18
- `compute_speculative_decoding_metrics()` - Chapter 18

## Current Infrastructure (SOLID)

The codebase has a solid profiling infrastructure including:
- `BenchmarkHarness` with nsys/ncu/Proton integration
- `ProfilerConfig` with ROOFLINE_METRICS, DEEP_DIVE_METRICS, and MINIMAL_METRICS
- `NcuMetrics`/`NsysMetrics` Pydantic models for structured data
- Deep profiling report generation (`tools/analysis/deep_profiling_report.py`)
- NVTX range instrumentation for timeline analysis
- **NEW**: `common/python/benchmark_metrics.py` - Helper functions for domain metrics
- **NEW**: Chapter-specific ncu metrics in `profiler_config.py`

### Key Gaps Identified
1. **Missing `get_custom_metrics()`** in most benchmarks
2. **Inconsistent use** of chapter-specific ncu metrics
3. **`blackwell_profiling_guide.py`** was buried in ch17 (now copied to common/python)

---

## Chapter-by-Chapter Analysis

### Chapter 1: Performance Basics
**Topic**: Goodput measurement, baseline performance concepts

**Current Metrics**: Basic timing (mean_ms, median_ms, std_ms)

**Missing Metrics for WHY/HOW**:
- [ ] **GPU utilization %** - to show baseline GPU efficiency
- [ ] **Kernel launch overhead** - critical for showing cost of small operations
- [ ] **Instructions per cycle (IPC)** - basic efficiency metric
- [ ] **Memory transactions** - data movement baseline

**Recommended Additions**:
```python
# In ch1/baseline_performance.py and optimized_performance.py
def get_config(self) -> BenchmarkConfig:
    return BenchmarkConfig(
        iterations=5,
        warmup=1,
        ncu_metrics=["gpu__time_duration.avg", "sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    )
```

**Recommendation**: Add `WorkloadMetadata.bytes_per_iteration` to compute achieved bandwidth vs peak.

---

### Chapter 2: Memory Architecture & Transfers
**Topic**: PCIe vs NVLink, memory transfer optimization

**Current Metrics**: `WorkloadMetadata.bytes_per_iteration` ✓

**Missing Metrics for WHY/HOW**:
- [ ] **Achieved bandwidth (GB/s)** - vs theoretical peak
- [ ] **Transfer efficiency %** - actual vs expected
- [ ] **PCIe link utilization** - for baseline comparisons
- [ ] **NVLink utilization** - for optimized comparisons

**Recommended Fix**:
```python
# Add to baseline_memory_transfer.py
def get_custom_metrics(self) -> Optional[Dict[str, float]]:
    elapsed_s = self._last_elapsed_ms / 1000.0 if hasattr(self, '_last_elapsed_ms') else 0.001
    bytes_transferred = self.N * 4  # float32
    achieved_gbps = (bytes_transferred / 1e9) / elapsed_s
    # PCIe Gen5 x16: ~64 GB/s theoretical
    pcie_theoretical_gbps = 64.0
    return {
        "achieved_bandwidth_gbps": achieved_gbps,
        "pcie_efficiency_pct": (achieved_gbps / pcie_theoretical_gbps) * 100,
    }
```

---

### Chapter 3: System Configuration & NUMA
**Topic**: NUMA awareness, topology optimization

**Current Metrics**: Basic timing

**Missing Metrics for WHY/HOW**:
- [ ] **NUMA node affinity** - which node executed the work
- [ ] **Cross-NUMA traffic** - bandwidth lost to remote memory
- [ ] **CPU utilization** - for CPU-bound operations
- [ ] **Memory locality %** - percentage of local vs remote accesses

**Recommended Addition**: Add nsys trace with CPU sampling enabled:
```python
# Use nsys_trace_types="cuda,nvtx,osrt,numa" for NUMA visibility
```

---

### Chapter 4: Multi-GPU & Communication
**Topic**: DataParallel, DDP, NCCL, NVLink

**Current Metrics**: `WorkloadMetadata` with tokens/requests

**Missing Metrics for WHY/HOW**:
- [ ] **NCCL bandwidth (GB/s)** - actual vs peak NVLink
- [ ] **Communication/Computation overlap %** - critical for DDP efficiency
- [ ] **Gradient bucket sizes** - for understanding bucketing efficiency
- [ ] **AllReduce latency breakdown** - per-operation timing

**Recommended nsys Configuration**:
```python
# Enable NCCL tracing
nsys_trace_types="cuda,nvtx,osrt,nccl"
```

**Recommended ncu Metrics**:
```python
# Add these for multi-GPU:
["nvlink_throughput", "gpu__compute_memory_throughput_coalesced"]
```

---

### Chapter 5: Storage & I/O
**Topic**: GPUDirect Storage, data loading

**Current Metrics**: Basic timing + tokens/iteration

**Missing Metrics for WHY/HOW**:
- [ ] **I/O throughput (MB/s)** - storage read/write speed
- [ ] **CPU-GPU transfer time** - data loading overhead
- [ ] **GDS bypass rate** - when using GPUDirect Storage
- [ ] **Queue depth utilization** - I/O parallelism

**Recommended Addition**:
```python
# Add GDS-specific metrics when available
if self._gds_enabled:
    return {"gds_throughput_mbps": measured_throughput}
```

---

### Chapter 6: CUDA Kernel Fundamentals
**Topic**: Bank conflicts, launch bounds, ILP

**Current Metrics**: Extension-based timing + NVTX ranges

**Missing Metrics for WHY/HOW**:
- [x] **Bank conflicts** - tracked via CUDA extension
- [ ] **Warp divergence %** - critical for understanding control flow
- [ ] **Achieved occupancy vs theoretical** - key optimization target
- [ ] **Instruction mix** - FMA vs separate ops

**Recommended ncu Metrics** (add to `get_config()`):
```python
ncu_metrics=[
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",  # bank conflicts
    "smsp__sass_average_branch_targets_threads_uniform.pct",     # warp divergence
    "sm__warps_active.avg.pct_of_peak_sustained_active",         # occupancy
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active", # FMA utilization
]
```

---

### Chapter 7: Memory Access Patterns
**Topic**: Coalescing, vectorization, TMA

**Current Metrics**: CUDA binary wrapper with timing regex

**Missing Metrics for WHY/HOW**:
- [ ] **Memory efficiency %** - bytes requested vs transferred
- [ ] **L1/L2 cache hit rates** - critical for tiling analysis
- [ ] **Global memory load efficiency** - coalescing effectiveness
- [ ] **Sectors per request** - vectorization effectiveness

**Recommended ncu Metrics**:
```python
MEMORY_ACCESS_METRICS = [
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit_rate.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_ld.pct",  # coalescing
    "smsp__sass_average_data_bytes_per_sector_mem_global_st.pct",
    "lts__t_sectors_op_read_hit_rate.pct",   # L2 read hit
    "lts__t_sectors_op_write_hit_rate.pct",  # L2 write hit
]
```

---

### Chapter 8: Kernel Optimization Techniques
**Topic**: Double buffering, loop unrolling, occupancy tuning

**Current Metrics**: CUDA binary benchmarks with specialized kernels

**Missing Metrics for WHY/HOW**:
- [ ] **Pipeline utilization %** - for double buffering
- [ ] **Instruction-level parallelism** - ILP achieved
- [ ] **Register spilling** - negative occupancy impact
- [ ] **Stall reasons breakdown** - what's blocking execution

**Recommended ncu Metrics** (critical for this chapter):
```python
OPTIMIZATION_METRICS = [
    # Stall analysis - THE KEY to understanding "why faster"
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_memory_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    # Register pressure
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
]
```

---

### Chapter 9: Compute-Bound Optimization
**Topic**: Tensor cores, warp specialization, CUTLASS

**Current Metrics**: Basic timing + NVTX

**Missing Metrics for WHY/HOW**:
- [ ] **Tensor core utilization %** - critical for compute-bound
- [ ] **FP16/FP8/TF32 TFLOPS achieved** - vs peak
- [ ] **Arithmetic intensity** - FLOPS/byte
- [ ] **Roofline position** - memory vs compute bound classification

**Recommended ncu Metrics**:
```python
TENSOR_CORE_METRICS = [
    "sm__inst_executed_pipe_tensor.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",  # FP64 FMA
    "smsp__sass_thread_inst_executed_op_dmma_pred_on.sum",  # Tensor ops
]
```

**Add roofline calculation**:
```python
def get_custom_metrics(self) -> Optional[Dict[str, float]]:
    flops = 2 * M * N * K  # for GEMM
    bytes_moved = (M*K + K*N + M*N) * element_size
    arithmetic_intensity = flops / bytes_moved
    return {
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "is_compute_bound": arithmetic_intensity > RIDGE_POINT,
    }
```

---

### Chapter 10: Advanced Pipelines & Clusters
**Topic**: Thread block clusters, distributed shared memory, TMA pipelines

**Current Metrics**: CUDA binary benchmarks + NVTX

**Missing Metrics for WHY/HOW**:
- [ ] **DSM utilization** - distributed shared memory usage
- [ ] **Cluster efficiency** - multi-CTA coordination overhead
- [ ] **TMA throughput** - tensor memory accelerator performance
- [ ] **Pipeline stage latency** - per-stage timing

**Recommended Blackwell-specific metrics**:
```python
CLUSTER_METRICS = [
    "sm__ctas_active.avg",
    "sm__maximum_warps_per_active_cu.avg",
    "gpu__time_duration.sum",
    # Note: DSM metrics may require specific ncu sections
]
```

---

### Chapter 11: CUDA Streams
**Topic**: Stream overlap, multi-stream execution

**Current Metrics**: Basic timing with `_synchronize()` calls

**Missing Metrics for WHY/HOW**:
- [ ] **Overlap efficiency %** - concurrent kernel execution time vs sequential
- [ ] **Stream utilization** - how many streams are active
- [ ] **Copy-compute overlap** - H2D/D2H overlapping with compute
- [ ] **Stream dependencies** - implicit/explicit synchronization points

**Recommended nsys Analysis**:
```python
# Enable detailed CUDA event tracing
nsys_trace_types="cuda-hw,nvtx,osrt"  # cuda-hw gives hardware events
```

**Add stream-specific metrics**:
```python
def get_custom_metrics(self) -> Optional[Dict[str, float]]:
    # Compare sequential vs overlapped execution
    sequential_time_ms = sum(stream_times)
    overlapped_time_ms = max(stream_times)  # actual parallel execution
    return {
        "overlap_efficiency_pct": (1 - overlapped_time_ms/sequential_time_ms) * 100,
        "streams_active": len(self.streams),
    }
```

---

### Chapter 12: CUDA Graphs
**Topic**: Graph capture, dynamic parallelism

**Current Metrics**: Extension-based timing

**Missing Metrics for WHY/HOW**:
- [ ] **Launch overhead reduction** - graph replay vs individual launches
- [ ] **Graph capture time** - one-time cost
- [ ] **Graph replay time** - per-iteration cost
- [ ] **Node count** - graph complexity

**Recommended Additions**:
```python
def get_custom_metrics(self) -> Optional[Dict[str, float]]:
    return {
        "graph_nodes": self.graph.num_nodes() if self.graph else 0,
        "launch_overhead_eliminated_us": self._baseline_launch_time_us - self._graph_launch_time_us,
    }
```

---

### Chapter 13: PyTorch Optimization
**Topic**: torch.compile, FP8, mixed precision

**Current Metrics**: `WorkloadMetadata` with tokens + NVTX ✓

**Missing Metrics for WHY/HOW**:
- [ ] **Compilation time** - torch.compile overhead
- [ ] **Dynamo graph breaks** - compilation barriers
- [ ] **FP8 scaling factor stability** - for precision analysis
- [ ] **Operator fusion rate** - how many ops were fused

**Recommended torch.compile metrics**:
```python
# Enable dynamo logging for compilation analysis
import torch._dynamo
torch._dynamo.config.log_level = "INFO"

def get_custom_metrics(self) -> Optional[Dict[str, float]]:
    return {
        "graph_breaks": torch._dynamo.utils.counters["graph_breaks"],
        "ops_fused": self._fused_op_count,
    }
```

---

### Chapter 14: Triton & Compilers
**Topic**: Triton kernels, torch.compile integration

**Current Metrics**: Basic timing + NVTX

**Missing Metrics for WHY/HOW**:
- [ ] **Triton kernel compile time** - JIT overhead
- [ ] **Proton metrics** - Triton-native profiling
- [ ] **Block scheduling efficiency** - grid/block utilization
- [ ] **Auto-tuning convergence** - selected config vs alternatives

**Recommended Proton integration** (already available):
```python
# Enable Proton profiling for Triton
from common.python.profiling_runner import run_proton_profiling
```

---

### Chapter 15: Inference Architecture
**Topic**: KV cache, continuous batching, disaggregated inference

**Current Metrics**: Basic timing + tokens/requests

**Missing Metrics for WHY/HOW**:
- [ ] **KV cache hit rate** - cache reuse efficiency
- [ ] **Batch utilization %** - actual vs max batch
- [ ] **Prefill vs decode time breakdown** - phase-specific timing
- [ ] **Memory fragmentation** - cache management overhead

**Recommended Inference metrics**:
```python
def get_custom_metrics(self) -> Optional[Dict[str, float]]:
    return {
        "kv_cache_hit_rate_pct": (cache_hits / total_accesses) * 100,
        "batch_utilization_pct": (actual_batch / max_batch) * 100,
        "prefill_time_ms": self._prefill_time,
        "decode_time_ms": self._decode_time,
    }
```

---

### Chapter 16: Production Inference
**Topic**: Flash attention, paged attention, regional compilation

**Current Metrics**: `InferenceTimingStats` (TTFT/TPOT) ✓

**Missing Metrics for WHY/HOW**:
- [ ] **Attention kernel breakdown** - flash vs naive timing
- [ ] **Page fault rate** - for paged attention
- [ ] **Compilation region coverage** - % of model compiled
- [ ] **SLA compliance %** - requests meeting latency target

**Good existing support** - `InferenceTimingStats` is well-designed.

**Add**:
```python
def get_custom_metrics(self) -> Optional[Dict[str, float]]:
    return {
        "p99_sla_met_pct": (requests_under_sla / total_requests) * 100,
        "flash_attention_efficiency_pct": flash_time / naive_time * 100,
    }
```

---

### Chapter 17: Dynamic Routing & Profiling Guide
**Topic**: MoE routing, comprehensive profiling

**Current State**: ✓ **EXCELLENT** - This chapter has the most comprehensive profiling with:
- `blackwell_profiling_guide.py` with NsightSystemsProfiler, NsightComputeProfiler
- `BlackwellMetricsGuide.get_essential_blackwell_metrics()`
- `HBMMemoryAnalyzer` for HBM3e analysis

**Use as Template**: Other chapters should adopt patterns from ch17.

---

### Chapter 18: Speculative Decoding & FlexDecoding
**Topic**: Draft-verify loops, FlexAttention

**Current Metrics**: Manual timing with `perf_counter()`

**Missing Metrics for WHY/HOW**:
- [ ] **Acceptance rate** ✓ (already tracked)
- [ ] **Draft model efficiency** - draft time vs verify time ratio
- [ ] **Token speculation depth** - avg tokens accepted per round
- [ ] **Verification overhead** - cost of rejected tokens

**Recommended Integration** - Use harness instead of manual timing:
```python
# Replace manual timing with:
class BaselineSpeculativeDecodingBenchmark(BaseBenchmark):
    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return {
            "acceptance_rate_pct": self.acceptance_rate,
            "draft_verify_ratio": self.draft_time / self.verify_time,
            "avg_tokens_accepted": self.total_accepted / self.num_rounds,
        }
```

---

### Chapter 19: Precision & Quantization
**Topic**: FP8, NVFP4, dynamic quantization

**Current Metrics**: Training timing + `WorkloadMetadata`

**Missing Metrics for WHY/HOW**:
- [ ] **Quantization error** - precision loss measurement
- [ ] **Dynamic range utilization** - scaling factor effectiveness
- [ ] **FP8 vs FP16 speedup ratio** - actual precision tradeoff
- [ ] **Memory savings** - bytes reduced by quantization

**Recommended Additions**:
```python
def get_custom_metrics(self) -> Optional[Dict[str, float]]:
    return {
        "fp8_speedup_ratio": fp16_time / fp8_time,
        "memory_reduction_pct": (1 - fp8_mem / fp16_mem) * 100,
        "quantization_error_mse": mse_loss.item(),
    }
```

---

### Chapter 20: End-to-End Optimization
**Topic**: Full pipeline optimization, autotuning

**Current Metrics**: Basic timing

**Missing Metrics for WHY/HOW**:
- [ ] **End-to-end throughput** - tokens/second for full pipeline
- [ ] **Bottleneck identification** - which stage is limiting
- [ ] **Autotuning improvement %** - tuned vs default config
- [ ] **Resource utilization balance** - CPU/GPU/memory utilization

**Recommended Full-Pipeline metrics**:
```python
def get_custom_metrics(self) -> Optional[Dict[str, float]]:
    return {
        "e2e_throughput_tokens_per_sec": total_tokens / elapsed_s,
        "bottleneck_stage": self._identify_bottleneck(),
        "autotuning_improvement_pct": (default_time - tuned_time) / default_time * 100,
    }
```

---

## Labs Analysis

### `labs/blackwell_matmul/`
**Current State**: ✓ Good - Uses `GraceBlackwellMatmulBenchmark` with feature descriptors

**Missing**:
- [ ] Tensor core utilization metrics
- [ ] TMA throughput measurement

### `labs/flexattention/`
**Current State**: ✓ Good - Uses `WorkloadMetadata` + NVTX

**Missing**:
- [ ] Compiled vs eager timing comparison
- [ ] Block mask efficiency

### `labs/occupancy_tuning/`
**Current State**: ✓ Excellent - Uses Proton for Triton profiling

**This is a model lab** - uses sweep schedules with Proton metrics.

### `labs/kv_cache_compression/`
**Missing**:
- [ ] Compression ratio metrics
- [ ] Decompression overhead

### `labs/speculative_decode/`
**Missing**:
- [ ] Per-draft-model timing
- [ ] Tree speculation metrics

---

## Global Recommendations

### 1. Add Standard Metric Sets to Each Chapter
Create chapter-specific metric configurations in `profiler_config.py`:

```python
CHAPTER_METRICS = {
    "ch6_kernel_fundamentals": [
        *ROOFLINE_METRICS,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "smsp__sass_average_branch_targets_threads_uniform.pct",
    ],
    "ch7_memory_access": [
        *ROOFLINE_METRICS,
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_rate.pct",
        "smsp__sass_average_data_bytes_per_sector_mem_global_ld.pct",
    ],
    "ch8_optimization": [
        *DEEP_DIVE_METRICS,
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "smsp__warp_issue_stalled_dependency_per_warp_active.pct",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    ],
    # ... etc
}
```

### 2. Implement `get_custom_metrics()` Consistently
Every benchmark should implement `get_custom_metrics()` returning domain-specific KPIs.

### 3. Add Throughput Calculations
Use `WorkloadMetadata.bytes_per_iteration` to automatically compute achieved bandwidth.

### 4. Enable nsys Hardware Traces
Use `cuda-hw` trace for more detailed GPU timing:
```python
nsys_trace_types="cuda-hw,nvtx,osrt,cublas,cudnn"
```

### 5. Create Chapter-Specific Analysis Reports
Extend `deep_profiling_report.py` to generate chapter-specific insights.

---

## Implementation Priority

| Priority | Chapter | Change Required |
|----------|---------|-----------------|
| HIGH | ch6 | Add bank conflict + warp divergence metrics |
| HIGH | ch7 | Add memory efficiency metrics |
| HIGH | ch8 | Add stall reason breakdown |
| HIGH | ch9 | Add tensor core utilization |
| MEDIUM | ch1-5 | Add basic throughput calculations |
| MEDIUM | ch11 | Add stream overlap efficiency |
| MEDIUM | ch18 | Integrate with harness properly |
| LOW | ch17 | Already excellent - use as template |

---

## Summary

The profiling infrastructure is solid, but **chapter-specific metric collection is inconsistent**. 

### What Was Created/Updated

| File | Purpose |
|------|---------|
| `common/python/benchmark_metrics.py` | **NEW** - Helper functions for domain-specific metrics |
| `common/python/profiler_config.py` | **UPDATED** - Added chapter-specific ncu metric sets |
| `common/python/blackwell_profiling_guide.py` | **COPIED** - Moved from ch17 to common location |
| `ch2/baseline_memory_transfer.py` | Example with `get_custom_metrics()` |
| `ch6/baseline_bank_conflicts.py` | Example with `get_custom_metrics()` |
| `ch9/baseline_compute_bound.py` | Example using `compute_roofline_metrics()` |
| `ch11/baseline_streams.py` | Example with stream metrics |
| `ch13/optimized_precisionfp8_te.py` | Example with precision metrics |

### What Each Chapter Should Add

For every benchmark file (379 files total), add:

```python
def get_custom_metrics(self) -> Optional[dict]:
    """Return domain-specific metrics for [chapter topic].
    
    These metrics help understand WHY optimizations work and HOW to improve.
    """
    from common.python.benchmark_metrics import compute_XXX_metrics
    return compute_XXX_metrics(...)
```

### Priority Order

1. **HIGH PRIORITY** - Chapters where metrics matter most for understanding:
   - Ch6 (bank conflicts) - Add `compute_kernel_fundamentals_metrics()`
   - Ch7 (memory access) - Add `compute_memory_access_metrics()`
   - Ch8 (optimization) - Add `compute_optimization_metrics()` + stall ncu metrics
   - Ch9 (compute-bound) - Add `compute_roofline_metrics()`

2. **MEDIUM PRIORITY** - Performance characterization:
   - Ch11 (streams) - Add `compute_stream_metrics()`
   - Ch12 (graphs) - Add `compute_graph_metrics()`
   - Ch13 (precision) - Add `compute_precision_metrics()`

3. **LOWER PRIORITY** - Already have good coverage:
   - Ch15-18 (inference) - Add `compute_inference_metrics()`

### The Key Insight

Without metrics, you can only say "it's 2x faster". WITH proper metrics:

**Before**: "The optimized kernel is 2x faster"
**After**: "The optimized kernel is 2x faster BECAUSE:
  - Bank conflicts reduced from 32-way to 0 (ncu: `l1tex__data_bank_conflicts`)
  - Arithmetic intensity increased from 2.5 to 64 FLOP/byte (roofline shift)
  - Memory efficiency improved from 25% to 98% (coalescing fixed)"

This tells the reader WHY and HOW, which is the whole point of the book.

