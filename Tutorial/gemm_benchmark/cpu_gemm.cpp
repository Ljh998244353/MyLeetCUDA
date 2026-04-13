#include "gemm_kernels.h"

#include <algorithm>
#include <cmath>

void cpu_gemm(const float* a, const float* b, float* c, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        acc += a[row * k + kk] * b[kk * n + col];
      }
      c[row * n + col] = acc;
    }
  }
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  float diff = 0.0f;
  const std::size_t size = std::min(lhs.size(), rhs.size());
  for (std::size_t i = 0; i < size; ++i) {
    diff = std::max(diff, std::fabs(lhs[i] - rhs[i]));
  }
  return diff;
}
