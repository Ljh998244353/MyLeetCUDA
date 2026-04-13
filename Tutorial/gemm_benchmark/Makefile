CUDA_ARCH ?= sm_89
NVCC := nvcc
TARGET := gemm_benchmark

NVCCFLAGS := -O3 -std=c++17 -lineinfo -Xcompiler=-fopenmp -arch=$(CUDA_ARCH)

SRCS := benchmark.cu cpu_gemm.cpp naive_cuda.cu wmma_cuda.cu

all: $(TARGET)

$(TARGET): $(SRCS) gemm_kernels.h
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@

clean:
	rm -f $(TARGET)
