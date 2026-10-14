// common/include/cuda_timer.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include "cuda_check.cuh"

class CudaTimer 
{
public:
    CudaTimer() { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
    ~CudaTimer() { CUDA_CHECK(cudaEventDestroy(start_)); CUDA_CHECK(cudaEventDestroy(stop_)); }

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    void stop()  { CUDA_CHECK(cudaEventRecord(stop_)); CUDA_CHECK(cudaEventSynchronize(stop_)); }

    float elapsed_ms() const 
    {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

    // 预热 + 计时，返回平均耗时（毫秒）
    template <typename KernelFunc>
    float benchmark(KernelFunc&& func, int warmup = 10, int iter = 100) 
    {
        for (int i = 0; i < warmup; i++) func();
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start_));
        for (int i = 0; i < iter; i++) func();
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms / iter;
    }

private:
    cudaEvent_t start_, stop_;
};