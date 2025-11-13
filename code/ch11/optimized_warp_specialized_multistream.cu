// Chapter 11: Book-aligned optimized version overlapping batches across CUDA streams.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <numeric>

namespace {
constexpr int TILE = 32;
constexpr int TILE_ELEMS = TILE * TILE;
constexpr int THREADS = 96;

__device__ void compute_tile(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int lane) {
    for (int idx = lane; idx < TILE_ELEMS; idx += warpSize) {
        int row = idx / TILE;
        int col = idx % TILE;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += A[row * TILE + k] * B[k * TILE + col];
        }
        C[idx] = acc;
    }
}

__global__ void simple_warp_specialized_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C) {
    extern __shared__ float shared[];
    float* A_tile = shared;
    float* B_tile = shared + TILE_ELEMS;
    float* C_tile = shared + 2 * TILE_ELEMS;

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    if (warp_id == 0) {
        for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
            A_tile[idx] = A[idx];
            B_tile[idx] = B[idx];
        }
    }

    __syncthreads();

    if (warp_id == 1) {
        compute_tile(A_tile, B_tile, C_tile, lane_id);
    }

    __syncthreads();

    if (warp_id == 2) {
        for (int idx = lane_id; idx < TILE_ELEMS; idx += warpSize) {
            C[idx] = C_tile[idx];
        }
    }
}

void run_optimized() {
    constexpr int batches = 8;
    constexpr int num_streams = 2;
    const size_t bytes = TILE_ELEMS * sizeof(float);

    std::vector<float> h_A(batches * TILE_ELEMS);
    std::vector<float> h_B(batches * TILE_ELEMS);
    std::vector<float> h_C(batches * TILE_ELEMS);
    std::iota(h_A.begin(), h_A.end(), 0.0f);
    std::iota(h_B.begin(), h_B.end(), 1.0f);

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int b = 0; b < batches; ++b) {
        cudaStream_t st = streams[b % num_streams];
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaMallocAsync(&dA, bytes, st);
        cudaMallocAsync(&dB, bytes, st);
        cudaMallocAsync(&dC, bytes, st);

        cudaMemcpyAsync(dA, h_A.data() + b * TILE_ELEMS, bytes, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dB, h_B.data() + b * TILE_ELEMS, bytes, cudaMemcpyHostToDevice, st);

        simple_warp_specialized_kernel<<<1, THREADS, 3 * bytes, st>>>(dA, dB, dC);

        cudaMemcpyAsync(h_C.data() + b * TILE_ELEMS, dC, bytes, cudaMemcpyDeviceToHost, st);

        cudaFreeAsync(dA, st);
        cudaFreeAsync(dB, st);
        cudaFreeAsync(dC, st);
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double checksum = 0.0;
    for (float v : h_C) checksum += v;

    printf("optimized_warp_specialized_multistream: %.3f ms, checksum %.3f\n",
           ms, checksum / h_C.size());

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < num_streams; ++i) cudaStreamDestroy(streams[i]);
}
}  // namespace

int main() {
    run_optimized();
    return 0;
}
