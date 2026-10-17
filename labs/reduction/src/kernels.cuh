#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cstddef>

struct ReductionVariant
{
    int mode;
    const char* name;
    const char* kernel_name;
    bool implemented;
    bool multiple_items;
};

const std::array<ReductionVariant, 3>& reduction_variants();
const ReductionVariant& reduction_variant(int mode);

// A pass reads n elements and writes exactly grid.x partial sums.
// Block b owns [b * block * items, min((b + 1) * block * items, n)).
// Input/output never alias. block is a power of two in [32, 1024].
// Every block must write its output, including a partially filled last block.
__global__ void reduce_shared(const float* __restrict__ in, float* __restrict__ partial, size_t n);
__global__ void reduce_multi_item(const float* in, float* partial,
                                  size_t n, int items_per_thread);
__global__ void reduce_shuffle(const float* in, float* partial,
                               size_t n, int items_per_thread);

// Shared scratch space at launch: block * sizeof(float) bytes for all variants.
// You may use all or part of that space in your implementation.
void launch_reduction_pass(int mode, const float* in, float* partial,
                           size_t n, unsigned int grid, int block,
                           int items_per_thread);
