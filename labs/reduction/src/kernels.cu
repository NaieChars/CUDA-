#include "kernels.cuh"

#include <stdexcept>
#include "cuda_check.cuh"

namespace
{
// Finish a kernel below, then enable only its matching entry here.
// main and the scripts refuse to benchmark unfinished kernels.
constexpr bool kSharedImplemented = true;
constexpr bool kMultiItemImplemented = true;
constexpr bool kShuffleImplemented = true;
}

const std::array<ReductionVariant, 3>& reduction_variants()
{
    static const std::array<ReductionVariant, 3> variants{{
        {0, "shared", "reduce_shared", kSharedImplemented, false},
        {1, "multi_item", "reduce_multi_item", kMultiItemImplemented, true},
        {2, "shuffle", "reduce_shuffle", kShuffleImplemented, true},
    }};
    return variants;
}

const ReductionVariant& reduction_variant(int mode)
{
    for (const auto& variant : reduction_variants())
        if (variant.mode == mode) return variant;
    throw std::invalid_argument("mode must be 0, 1, or 2");
}

__global__ void reduce_shared(const float* __restrict__ in, float* __restrict__ partial, size_t n)
{
    // TODO 1: implement a shared-memory block reduction.
    // Contract: items_per_thread = 1; block b handles at most blockDim.x inputs.
    // Decide how to handle the tail, combine values, synchronize, and write
    // exactly one sum to partial[blockIdx.x].
    // No algorithm is supplied here; enable kSharedImplemented when ready.

    extern __shared__ float s[];
    const unsigned int tid = threadIdx.x; // tid: 单个 block 里的线程索引
    const size_t Idx = static_cast<size_t>(blockDim.x) * blockIdx.x + tid;

    s[tid] = Idx < n ? in[Idx] : 0.0f;  // 一个线程读取一个元素，放到共享内存
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s[tid] += s[tid + stride];

        if (stride > 1)
            __syncthreads();
    }

    if (tid == 0)
    {
        partial[blockIdx.x] = s[0];
    }
}

__global__ void reduce_multi_item(const float* in, float* partial,
                                  size_t n, int items_per_thread)
{
    // TODO 2: implement the multi-item version.
    // Each block owns one contiguous chunk of blockDim.x * items_per_thread
    // elements. Thread-to-element assignment and accumulation are your choices.
    // The output contract is the same as TODO 1.

    extern __shared__ float s[];
    const unsigned int tid = threadIdx.x;
    const size_t base = static_cast<size_t>(blockIdx.x) * items_per_thread * blockDim.x;    // 一个 block 的起始全局索引

    float sum = 0.0f;
    for (int j = 0; j < items_per_thread; j++)
    {
        const size_t idx = base + tid + static_cast<size_t>(blockDim.x) * j;

        if (idx < n)
            sum += in[idx];
    }
    s[tid] = sum;
    __syncthreads();


    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            s[tid] += s[tid + stride];

        if (stride > 1)
            __syncthreads();
    }

    if (tid == 0)
    {
        partial[blockIdx.x] = s[0];
    }
}

// tools: 工具函数，计算一个 warp 内所有线程的和
__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void reduce_shuffle(const float* in, float* partial,
                               size_t n, int items_per_thread)
{
    // TODO 3: implement the warp-shuffle version.
    // Input ownership is identical to TODO 2. Decide how values move within
    // and between warps and which synchronization/masks are required.
    // The output contract is the same as TODO 1.
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const size_t base = static_cast<size_t>(blockIdx.x) * items_per_thread * blockDim.x;

    // 每一个线程内部串行求和
    float sum = 0.0f;
    for (int j = 0; j < items_per_thread; j++)
    {
        const size_t idx = base + tid + static_cast<size_t>(blockDim.x) * j;

        if (idx < n)
            sum += in[idx];
    }

    extern __shared__ float warp_sums[];
    
    sum = warp_sum(sum);
    if (lane == 0)
        warp_sums[warp_id] = sum;
    
    __syncthreads();
    
    // 再将每一个 warp 的 sum 从共享内存搬回线程进行计算，这里用第一个 warp 计算即可
    const unsigned int num_warps = blockDim.x / 32;
    if (warp_id == 0)
    {
        float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);

        if (lane == 0)
            partial[blockIdx.x] = block_sum;
    }
}

void launch_reduction_pass(int mode, const float* in, float* partial,
                           size_t n, unsigned int grid, int block,
                           int items_per_thread)
{
    if (!reduction_variant(mode).implemented)
        throw std::logic_error("TODO: selected reduction kernel is not implemented");

    const size_t shared_bytes = static_cast<size_t>(block) * sizeof(float);
    switch (mode)
    {
    case 0:
        reduce_shared<<<grid, block, shared_bytes>>>(in, partial, n);
        break;
    case 1:
        reduce_multi_item<<<grid, block, shared_bytes>>>(
            in, partial, n, items_per_thread);
        break;
    case 2:
        reduce_shuffle<<<grid, block, shared_bytes>>>(
            in, partial, n, items_per_thread);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}
