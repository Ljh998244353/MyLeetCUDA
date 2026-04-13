#include "gemm_kernels.h"

#include <cuda_runtime.h>

#include <cstdio>

namespace {

inline void check_cuda(cudaError_t err, const char* expr, const char* file,
                       int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expr,
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

__global__ void naive_gemm_kernel(const float* a, const float* b, float* c, int m,
                                  int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += a[row * k + kk] * b[kk * n + col];
  }
  c[row * n + col] = acc;
}

}  // namespace

void launch_naive_cuda_gemm(const float* d_a, const float* d_b, float* d_c,
                            int m, int n, int k) {
  constexpr int kBlockX = 16;
  constexpr int kBlockY = 16;
  const dim3 block(kBlockX, kBlockY);
  const dim3 grid((n + kBlockX - 1) / kBlockX, (m + kBlockY - 1) / kBlockY);
  naive_gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CHECK_CUDA(cudaGetLastError());
}
