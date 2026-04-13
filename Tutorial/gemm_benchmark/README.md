# Tutorial: 3 种 GEMM 性能对比

本目录提供一个最小但可直接运行的通用矩阵乘法（GEMM）对比实验，包含 3 种实现：

- `cpu_gemm.cpp`: 普通 C++ 三重循环版本。
- `naive_cuda.cu`: 最基础的 CUDA 版本，一个线程计算 `C` 中一个输出元素。
- `wmma_cuda.cu`: 基于 `WMMA TF32 Tensor Core` 的优化 CUDA 版本。

## 文件说明

- `gemm_kernels.h`: 三种实现的统一接口声明。
- `benchmark.cu`: 统一测试入口，负责构造输入、校验结果、统计耗时与 GFLOPS。
- `Makefile`: 编译入口，默认针对本机 RTX 4060 Laptop GPU 使用 `sm_89`。

当前 GPU 计时口径是“只统计 kernel 执行时间”，不包含 Host/Device 数据拷贝和显存申请释放；这样更适合直接比较 2 个 CUDA 内核本身的性能。

## 编译

```bash
cd Tutorial/gemm_benchmark
make
```

如需切换架构：

```bash
make CUDA_ARCH=sm_80
```

## 运行

默认测试 `1024 x 1024 x 1024`，GPU 部分跑 50 次取平均：

```bash
./gemm_benchmark
```

也可以手动指定 `M N K GPU_ITERS`：

```bash
./gemm_benchmark 2048 2048 2048 20
```

## 预期现象

- `CPU C++` 最直观，但性能最低。
- `Naive CUDA` 利用 GPU 并行度后，通常会明显快于 CPU。
- `Optimized CUDA (WMMA TF32)` 使用 Tensor Core 后，会进一步拉开和 naive CUDA 的差距。

## 实现差异总结

- CPU 版本的核心特征是访存与计算全部在串行循环里完成，几乎没有并行优化。
- naive CUDA 版本虽然把输出元素分给不同线程，但没有做共享内存分块、寄存器 blocking 或 Tensor Core 利用，因此性能上限较低。
- WMMA 版本将计算映射到 `16x16x8` 的 Tensor Core 指令片段，吞吐量更高，但引入了 TF32 精度路径，因此结果会和 CPU FP32 参考值存在小量误差。

## 结果解读

程序会输出每种实现的：

- 平均耗时（ms）
- 计算吞吐（GFLOPS）
- 与 CPU 参考结果的最大绝对误差（`max error`）

如果你后续想继续扩展这个教程，最自然的下一步是：

1. 在 naive CUDA 和 WMMA 之间再插入一个 shared memory tiling 版本。
2. 对 WMMA 版本补上 shared memory staging 和多 warp block tiling。
3. 增加 cuBLAS 结果，作为库实现的上限参考。

```bash

$ ./gemm_benchmark
Benchmark GEMM with M=1024, N=1024, K=1024, GPU iters=50
---------------------------------------------------------------
CPU C++                      |   2240.175 ms |       0.96 GFLOPS | max error 0.000000
Naive CUDA                   |      2.417 ms |     888.66 GFLOPS | max error 0.000023
Optimized CUDA (WMMA TF32)   |      0.459 ms |    4677.65 GFLOPS | max error 0.036350
(base)
# ljh @ DESKTOP-AODOC7U in ~/AI_infra_learning/CUDA_Learning/MyLeetCUDA/Tutorial/gemm_benchmark on git:main x [19:45:28]
$ ./gemm_benchmark 2048 2048 2048 20
Benchmark GEMM with M=2048, N=2048, K=2048, GPU iters=20
---------------------------------------------------------------
CPU C++                      |  51200.195 ms |       0.34 GFLOPS | max error 0.000000
Naive CUDA                   |     17.880 ms |     960.84 GFLOPS | max error 0.000038
Optimized CUDA (WMMA TF32)   |      3.355 ms |    5120.94 GFLOPS | max error 0.053055
(base)

```
