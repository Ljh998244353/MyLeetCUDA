#include <cuda_runtime.h>
#include <torch/extension.h>

// 1. 朴素版本
__global__ void elementwise_add_f32_kernel(float *a, float *b, float *c, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

void elementwise_add_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda(), "a must a cuda tensor");
    TORCH_CHECK(b.is_cuda(), "b must a cuda tensor");
    TORCH_CHECK(c.is_cuda(), "c must a cuda tensor");

}
