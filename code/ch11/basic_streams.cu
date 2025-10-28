// basic_streams.cu -- CUDA 13.0 stream overlap demo with error handling.

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t status = (call);                                             \
    if (status != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(status));                            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Optimized kernel with launch bounds, vectorized loads, and async copy support
__global__ void __launch_bounds__(256, 8) scale_kernel(float* data, int n, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = data[idx] * scale + 0.001f;
  }
}

// Vectorized version using float4 for better memory throughput
__global__ void __launch_bounds__(256, 8) scale_kernel_vectorized(float* data, int n, float scale) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  
  if (idx + 3 < n) {
    // Load 4 floats at once (128-bit transaction)
    float4 vec = *reinterpret_cast<float4*>(&data[idx]);
    
    // Process all 4 elements
    vec.x = vec.x * scale + 0.001f;
    vec.y = vec.y * scale + 0.001f;
    vec.z = vec.z * scale + 0.001f;
    vec.w = vec.w * scale + 0.001f;
    
    // Store 4 floats at once
    *reinterpret_cast<float4*>(&data[idx]) = vec;
  } else {
    // Handle remaining elements
    for (int i = idx; i < n; i++) {
      data[i] = data[i] * scale + 0.001f;
    }
  }
}

// Shared memory version with prefetching (simpler than cp.async for this demo)
// On Blackwell, compiler optimizes this with TMA automatically
__global__ void __launch_bounds__(256, 8) scale_kernel_async(float* __restrict__ data, int n, float scale) {
  // Shared memory for prefetching and better cache utilization
  __shared__ float smem[256 * 4];  // 4 elements per thread for float4
  
  int tid = threadIdx.x;
  int idx = (blockIdx.x * blockDim.x + tid) * 4;
  
  if (idx + 3 < n) {
    // Load to shared memory (compiler will optimize on Blackwell)
    float4 vec = *reinterpret_cast<const float4*>(&data[idx]);
    *reinterpret_cast<float4*>(&smem[tid * 4]) = vec;
    
    __syncthreads();
    
    // Process from shared memory
    vec = *reinterpret_cast<float4*>(&smem[tid * 4]);
    vec.x = vec.x * scale + 0.001f;
    vec.y = vec.y * scale + 0.001f;
    vec.z = vec.z * scale + 0.001f;
    vec.w = vec.w * scale + 0.001f;
    
    __syncthreads();
    
    // Write back to global memory
    *reinterpret_cast<float4*>(&data[idx]) = vec;
  } else {
    // Handle remaining elements
    for (int i = idx; i < n; i++) {
      data[i] = data[i] * scale + 0.001f;
    }
  }
}

int main() {
  constexpr int N = 1 << 21;
  constexpr size_t BYTES = N * sizeof(float);

  float *h_a = nullptr, *h_b = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_a, BYTES));
  CUDA_CHECK(cudaMallocHost(&h_b, BYTES));
  for (int i = 0; i < N; ++i) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  float *d_a = nullptr, *d_b = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, BYTES));
  CUDA_CHECK(cudaMalloc(&d_b, BYTES));

  cudaStream_t stream1 = nullptr, stream2 = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, 0));

  CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, BYTES, cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, BYTES, cudaMemcpyHostToDevice, stream2));

  // Benchmark: Compare original, vectorized, and async versions
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  
  constexpr int WARMUP = 5;
  constexpr int ITERS = 100;
  
  // Original kernel
  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  for (int i = 0; i < WARMUP; ++i) {
    scale_kernel<<<grid, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream1));
  
  CUDA_CHECK(cudaEventRecord(start, stream1));
  for (int i = 0; i < ITERS; ++i) {
    scale_kernel<<<grid, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream1));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float ms_original = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_original, start, stop));
  
  // Vectorized kernel (adjust grid size for float4)
  dim3 grid_vec((N / 4 + block.x - 1) / block.x);
  for (int i = 0; i < WARMUP; ++i) {
    scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream1));
  
  CUDA_CHECK(cudaEventRecord(start, stream1));
  for (int i = 0; i < ITERS; ++i) {
    scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream1));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float ms_vectorized = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_vectorized, start, stop));
  
  // Async kernel
  for (int i = 0; i < WARMUP; ++i) {
    scale_kernel_async<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream1));
  
  CUDA_CHECK(cudaEventRecord(start, stream1));
  for (int i = 0; i < ITERS; ++i) {
    scale_kernel_async<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.1f);
  }
  CUDA_CHECK(cudaEventRecord(stop, stream1));
  CUDA_CHECK(cudaEventSynchronize(stop));
  
  float ms_async = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_async, start, stop));
  
  // Test with dual streams
  scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.1f);
  scale_kernel_vectorized<<<grid_vec, block, 0, stream2>>>(d_b, N, 0.9f);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, BYTES, cudaMemcpyDeviceToHost, stream1));
  CUDA_CHECK(cudaMemcpyAsync(h_b, d_b, BYTES, cudaMemcpyDeviceToHost, stream2));

  CUDA_CHECK(cudaStreamSynchronize(stream1));
  CUDA_CHECK(cudaStreamSynchronize(stream2));

  // Measure sequential vs overlapped pipelines
  auto measure_pipeline = [&](bool overlap) -> double {
    constexpr int BATCHES = 8;

    CUDA_CHECK(cudaMemsetAsync(d_a, 0, BYTES, stream1));
    CUDA_CHECK(cudaMemsetAsync(d_b, 0, BYTES, stream2));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!overlap) {
      for (int i = 0; i < BATCHES; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, BYTES, cudaMemcpyHostToDevice, stream1));
        scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.05f);
        CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, BYTES, cudaMemcpyDeviceToHost, stream1));

        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, BYTES, cudaMemcpyHostToDevice, stream1));
        scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_b, N, 0.95f);
        CUDA_CHECK(cudaMemcpyAsync(h_b, d_b, BYTES, cudaMemcpyDeviceToHost, stream1));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream1));
    } else {
      for (int i = 0; i < BATCHES; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, BYTES, cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, BYTES, cudaMemcpyHostToDevice, stream2));

        scale_kernel_vectorized<<<grid_vec, block, 0, stream1>>>(d_a, N, 1.05f);
        scale_kernel_vectorized<<<grid_vec, block, 0, stream2>>>(d_b, N, 0.95f);

        CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, BYTES, cudaMemcpyDeviceToHost, stream1));
        CUDA_CHECK(cudaMemcpyAsync(h_b, d_b, BYTES, cudaMemcpyDeviceToHost, stream2));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream1));
      CUDA_CHECK(cudaStreamSynchronize(stream2));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  double sequential_ms = measure_pipeline(false);
  double overlap_ms = measure_pipeline(true);
  double overlap_speedup = sequential_ms / overlap_ms;
  double overlap_pct = (1.0 - overlap_ms / sequential_ms) * 100.0;

  std::printf("\n=== Kernel Performance Comparison ===\n");
  std::printf("Original kernel:    %.3f ms (%.2f GB/s)\n", 
              ms_original / ITERS, 2.0f * BYTES / (ms_original / ITERS * 1e6));
  std::printf("Vectorized kernel:  %.3f ms (%.2f GB/s) [%.2fx speedup]\n", 
              ms_vectorized / ITERS, 2.0f * BYTES / (ms_vectorized / ITERS * 1e6),
              ms_original / ms_vectorized);
  std::printf("Async copy kernel:  %.3f ms (%.2f GB/s) [%.2fx speedup]\n", 
              ms_async / ITERS, 2.0f * BYTES / (ms_async / ITERS * 1e6),
              ms_original / ms_async);
  std::printf("Stream speedup (vectorized vs original): %.2fx\n", ms_original / ms_vectorized);
  std::printf("\nstream1 result: %.3f\n", h_a[0]);
  std::printf("stream2 result: %.3f\n", h_b[0]);
  std::printf("\n=== Stream Overlap Benchmark ===\n");
  std::printf("Sequential pipeline: %.2f ms\n", sequential_ms);
  std::printf("Overlapped pipeline: %.2f ms\n", overlap_ms);
  std::printf("Stream overlap speedup: %.2fx (%.1f%% latency reduction)\n",
              overlap_speedup, overlap_pct);
  std::printf("Stream overlap percent: %.1f%%\n", overlap_pct);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  return 0;
}
