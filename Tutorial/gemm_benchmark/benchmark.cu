#include "gemm_kernels.h"

#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

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

struct RunStats {
  std::string name;
  double milliseconds;
  double gflops;
  float max_error;
};

double calc_gflops(int m, int n, int k, double milliseconds) {
  const double flops = 2.0 * static_cast<double>(m) * n * k;
  return flops / (milliseconds * 1.0e6);
}

int round_up(int value, int tile) {
  return ((value + tile - 1) / tile) * tile;
}

std::vector<float> make_matrix(std::size_t size, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> out(size);
  for (float& value : out) {
    value = dist(rng);
  }
  return out;
}

RunStats bench_cpu(const std::vector<float>& a, const std::vector<float>& b,
                   const std::vector<float>& ref, int m, int n, int k) {
  std::vector<float> out(static_cast<std::size_t>(m) * n, 0.0f);
  const auto start = std::chrono::high_resolution_clock::now();
  cpu_gemm(a.data(), b.data(), out.data(), m, n, k);
  const auto stop = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(stop - start).count();
  return {"CPU C++", ms, calc_gflops(m, n, k, ms), max_abs_diff(out, ref)};
}

RunStats bench_naive_cuda(const std::vector<float>& a, const std::vector<float>& b,
                          const std::vector<float>& ref, int m, int n, int k,
                          int iters) {
  const std::size_t size_a = static_cast<std::size_t>(m) * k;
  const std::size_t size_b = static_cast<std::size_t>(k) * n;
  const std::size_t size_c = static_cast<std::size_t>(m) * n;
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, size_b * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_c, size_c * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a.data(), size_a * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b.data(), size_b * sizeof(float),
                        cudaMemcpyHostToDevice));

  for (int i = 0; i < 5; ++i) {
    launch_naive_cuda_gemm(d_a, d_b, d_c, m, n, k);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch_naive_cuda_gemm(d_a, d_b, d_c, m, n, k);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

  std::vector<float> out(size_c);
  CHECK_CUDA(cudaMemcpy(out.data(), d_c, size_c * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));

  const double avg_ms = total_ms / iters;
  return {"Naive CUDA", avg_ms, calc_gflops(m, n, k, avg_ms),
          max_abs_diff(out, ref)};
}

RunStats bench_wmma_cuda(const std::vector<float>& a, const std::vector<float>& b,
                         const std::vector<float>& ref, int m, int n, int k,
                         int iters) {
  const int padded_m = round_up(m, 16);
  const int padded_n = round_up(n, 16);
  const int padded_k = round_up(k, 8);
  const std::size_t size_a = static_cast<std::size_t>(padded_m) * padded_k;
  const std::size_t size_b = static_cast<std::size_t>(padded_k) * padded_n;
  const std::size_t size_c = static_cast<std::size_t>(padded_m) * padded_n;

  std::vector<float> padded_a(size_a, 0.0f);
  std::vector<float> padded_b(size_b, 0.0f);
  for (int row = 0; row < m; ++row) {
    std::copy_n(a.data() + row * k, k, padded_a.data() + row * padded_k);
  }
  for (int row = 0; row < k; ++row) {
    std::copy_n(b.data() + row * n, n, padded_b.data() + row * padded_n);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, size_b * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_c, size_c * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, padded_a.data(), size_a * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, padded_b.data(), size_b * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<float> out(static_cast<std::size_t>(m) * n, 0.0f);

  for (int i = 0; i < 3; ++i) {
    launch_wmma_cuda_gemm(d_a, d_b, d_c, padded_m, padded_n, padded_k, padded_k,
                          padded_n, padded_n);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch_wmma_cuda_gemm(d_a, d_b, d_c, padded_m, padded_n, padded_k, padded_k,
                          padded_n, padded_n);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  std::vector<float> padded_c(size_c, 0.0f);
  CHECK_CUDA(cudaMemcpy(padded_c.data(), d_c, size_c * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int row = 0; row < m; ++row) {
    std::copy_n(padded_c.data() + row * padded_n, n, out.data() + row * n);
  }

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));

  const double avg_ms = total_ms / iters;
  return {"Optimized CUDA (WMMA TF32)", avg_ms, calc_gflops(m, n, k, avg_ms),
          max_abs_diff(out, ref)};
}

void print_stats(const RunStats& stats) {
  std::printf("%-28s | %10.3f ms | %10.2f GFLOPS | max error %.6f\n",
              stats.name.c_str(), stats.milliseconds, stats.gflops,
              stats.max_error);
}

}  // namespace

int main(int argc, char** argv) {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int gpu_iters = 50;
  if (argc == 5) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
    gpu_iters = std::atoi(argv[4]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [M N K GPU_ITERS]\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::mt19937 rng(42);
  auto a = make_matrix(static_cast<std::size_t>(m) * k, rng);
  auto b = make_matrix(static_cast<std::size_t>(k) * n, rng);
  std::vector<float> reference(static_cast<std::size_t>(m) * n, 0.0f);
  cpu_gemm(a.data(), b.data(), reference.data(), m, n, k);

  std::printf("Benchmark GEMM with M=%d, N=%d, K=%d, GPU iters=%d\n", m, n, k,
              gpu_iters);
  std::printf("---------------------------------------------------------------\n");

  const RunStats cpu_stats = bench_cpu(a, b, reference, m, n, k);
  const RunStats naive_stats = bench_naive_cuda(a, b, reference, m, n, k, gpu_iters);
  const RunStats wmma_stats = bench_wmma_cuda(a, b, reference, m, n, k, gpu_iters);

  print_stats(cpu_stats);
  print_stats(naive_stats);
  print_stats(wmma_stats);
  return EXIT_SUCCESS;
}
