#pragma once

#include <cstddef>
#include <vector>

void cpu_gemm(const float* a, const float* b, float* c, int m, int n, int k);

void launch_naive_cuda_gemm(const float* d_a, const float* d_b, float* d_c,
                            int m, int n, int k);

void launch_wmma_cuda_gemm(const float* d_a, const float* d_b, float* d_c,
                           int padded_m, int padded_n, int padded_k, int lda,
                           int ldb, int ldc);

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs);
