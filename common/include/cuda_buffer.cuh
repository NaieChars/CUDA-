#pragma once

#include <cuda_runtime.h>
#include "cuda_check.cuh"

template <typename T>
class CudaBuffer
{
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) { allocate(count); }  // count = 元素个数

    ~CudaBuffer() { release(); }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();

            ptr_ = other.ptr_;
            count_ = other.count_;

            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        return *this;
    }

    void allocate(size_t count)
    {
        release();

        count_ = count;
        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    void release()
    {
        if (ptr_ != nullptr)
        {
            CUDA_CHECK(cudaFree(ptr_));
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }

    size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};