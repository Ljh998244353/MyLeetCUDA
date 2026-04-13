#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>

#include "gemm_kernels.h"

using namespace nvcuda;

namespace {

inline void check_cuda(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expr,
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 8;

__global__ void wmma_tf32_gemm_kernel(const float* a, const float* b, float* c, int m, int n, int k,
                                      int lda, int ldb, int ldc) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warps_per_grid = (gridDim.x * blockDim.x) / warpSize;
    const int total_tiles_n = n / kWmmaN;
    const int total_tiles = (m / kWmmaM) * total_tiles_n;

    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += warps_per_grid) {
        const int tile_m = tile_idx / total_tiles_n;
        const int tile_n = tile_idx % total_tiles_n;

        wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int kk = 0; kk < k; kk += kWmmaK) {
            wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32,
                           wmma::row_major>
                a_frag;
            wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32,
                           wmma::row_major>
                b_frag;

            const float* tile_a = a + tile_m * kWmmaM * lda + kk;
            const float* tile_b = b + kk * ldb + tile_n * kWmmaN;
            wmma::load_matrix_sync(a_frag, tile_a, lda);
            wmma::load_matrix_sync(b_frag, tile_b, ldb);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float* tile_c = c + tile_m * kWmmaM * ldc + tile_n * kWmmaN;
        wmma::store_matrix_sync(tile_c, c_frag, ldc, wmma::mem_row_major);
    }
}

}  // namespace

void launch_wmma_cuda_gemm(const float* d_a, const float* d_b, float* d_c, int padded_m,
                           int padded_n, int padded_k, int lda, int ldb, int ldc) {
    constexpr int kThreads = 128;
    const int tiles = (padded_m / kWmmaM) * (padded_n / kWmmaN);
    const int blocks = std::max(1, (tiles + (kThreads / 32) - 1) / (kThreads / 32));
    wmma_tf32_gemm_kernel<<<blocks, kThreads>>>(d_a, d_b, d_c, padded_m, padded_n, padded_k, lda,
                                                ldb, ldc);
    CHECK_CUDA(cudaGetLastError());
}
