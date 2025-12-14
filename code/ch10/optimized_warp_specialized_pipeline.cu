// Chapter 10: Book-aligned warp-specialized pipeline using cuda::pipeline for loader/compute/storer handoff.
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>
#include <numeric>

#include "../core/common/headers/cuda_verify.cuh"

namespace cg = cooperative_groups;

namespace {
constexpr int TILE_SIZE = 64;
constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;
constexpr int WARPS_PER_BLOCK = 3;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

__device__ void compute_full_tile(const float* __restrict__ A_tile,
                                  const float* __restrict__ B_tile,
                                  float* __restrict__ C_tile,
                                  int lane_id) {
    for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
        int row = idx / TILE_SIZE;
        int col = idx % TILE_SIZE;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += A_tile[row * TILE_SIZE + k] * B_tile[k * TILE_SIZE + col];
        }
        C_tile[idx] = acc;
    }
}

__global__ void optimized_warp_specialized_kernel(const float* __restrict__ A_global,
                                                  const float* __restrict__ B_global,
                                                  float* __restrict__ C_global,
                                                  int num_tiles) {
    cg::thread_block cta = cg::this_thread_block();

    extern __shared__ float shared_mem[];
    float* A_tile = shared_mem;
    float* B_tile = A_tile + TILE_ELEMS;
    float* C_tile = B_tile + TILE_ELEMS;

    using ab_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, 1>;
    using c_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, 1>;
    __shared__ alignas(ab_state_t) unsigned char ab_state_bytes[sizeof(ab_state_t)];
    __shared__ alignas(c_state_t) unsigned char c_state_bytes[sizeof(c_state_t)];
    auto* ab_state = reinterpret_cast<ab_state_t*>(ab_state_bytes);
    auto* c_state = reinterpret_cast<c_state_t*>(c_state_bytes);
    if (threadIdx.x == 0) {
        new (ab_state) ab_state_t();
        new (c_state) c_state_t();
    }
    __syncthreads();
    auto pipe_ab = cuda::make_pipeline(cta, ab_state);
    auto pipe_c = cuda::make_pipeline(cta, c_state);

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    auto warp = cg::tiled_partition<32>(cta);

    // Block-strided tiling so warps cooperate on the same tile.
    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

        if (warp_id == 0) {
            pipe_ab.producer_acquire();
            cuda::memcpy_async(
                warp,
                A_tile,
                A_global + offset,
                cuda::aligned_size_t<16>(static_cast<size_t>(TILE_ELEMS) * sizeof(float)),
                pipe_ab);
            cuda::memcpy_async(
                warp,
                B_tile,
                B_global + offset,
                cuda::aligned_size_t<16>(static_cast<size_t>(TILE_ELEMS) * sizeof(float)),
                pipe_ab);
            pipe_ab.producer_commit();
        }

        if (warp_id == 1) {
            // Ensure the shared C tile is free (store finished previous tile).
            pipe_c.producer_acquire();

            pipe_ab.consumer_wait();
            compute_full_tile(A_tile, B_tile, C_tile, lane_id);

            pipe_ab.consumer_release();
            // Publish computed C tile for the storer warp.
            pipe_c.producer_commit();
        }

        if (warp_id == 2) {
            pipe_c.consumer_wait();
            for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
                C_global[offset + idx] = C_tile[idx];
            }
            pipe_c.consumer_release();
        }
    }
}

void run_optimized(int tiles) {
    const size_t bytes = static_cast<size_t>(tiles) * TILE_ELEMS * sizeof(float);
    std::vector<float> h_A(bytes / sizeof(float));
    std::vector<float> h_B(bytes / sizeof(float));
    std::vector<float> h_C(bytes / sizeof(float));
    std::iota(h_A.begin(), h_A.end(), 0.0f);
    std::iota(h_B.begin(), h_B.end(), 1.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(std::min(tiles, 64));
    size_t shared_bytes = 3 * TILE_ELEMS * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    optimized_warp_specialized_kernel<<<grid, block, shared_bytes>>>(d_A, d_B, d_C, tiles);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (float v : h_C) checksum += v;

    printf("optimized_warp_specialized_pipeline: %d tiles, %.3f ms, checksum %.3f\n",
           tiles, ms, checksum / h_C.size());

#ifdef VERIFY
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
}  // namespace

int main() {
    run_optimized(128);
    return 0;
}
