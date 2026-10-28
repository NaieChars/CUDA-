#include "kernels.cuh"

#include <limits>
#include <stdexcept>
#include "cuda_check.cuh"

namespace
{
// Enable only after implementing the matching kernel.
// main and the scripts refuse to measure unfinished versions.
constexpr bool kNaiveImplemented = true;
constexpr bool kSharedImplemented = true;
constexpr bool kRegisterImplemented = true;
}

const std::array<GemmVariant, 3>& gemm_variants()
{
    using namespace sgemm_config;
    static const std::array<GemmVariant, 3> variants{{
        {0, "naive", "sgemm_naive", kNaiveImplemented,
         kBlockX, kBlockY, kBlockY, kBlockX, 0, 1, 1, 0},
        {1, "shared", "sgemm_shared", kSharedImplemented,
         kBlockX, kBlockY, kSharedTile, kSharedTile, kSharedTile, 1, 1,
         2 * static_cast<size_t>(kSharedTile) * kSharedTile * sizeof(float)},
        {2, "register_tiled", "sgemm_register", kRegisterImplemented,
         kBlockX, kBlockY, kRegisterTileM, kRegisterTileN, kRegisterTileK,
         kThreadTileM, kThreadTileN,
         (static_cast<size_t>(kRegisterTileM) * kRegisterTileK +
          static_cast<size_t>(kRegisterTileK) * kRegisterTileN) * sizeof(float)},
    }};
    return variants;
}

const GemmVariant& gemm_variant(int mode)
{
    for (const auto& variant : gemm_variants())
        if (variant.mode == mode) return variant;
    throw std::invalid_argument("mode must be 0, 1, or 2");
}

GemmLaunch make_gemm_launch(int mode, size_t m, size_t n)
{
    const auto& variant = gemm_variant(mode);
    if (m == 0 || n == 0) throw std::invalid_argument("M and N must be positive");
    const size_t gx = n / variant.tile_n + (n % variant.tile_n != 0);
    const size_t gy = m / variant.tile_m + (m % variant.tile_m != 0);
    if (gx > std::numeric_limits<unsigned int>::max() ||
        gy > std::numeric_limits<unsigned int>::max())
        throw std::invalid_argument("grid dimensions do not fit dim3");
    return {dim3(static_cast<unsigned int>(gx), static_cast<unsigned int>(gy)),
            dim3(variant.block_x, variant.block_y), variant.shared_bytes};
}

__global__ void sgemm_naive(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c, size_t m, size_t n, size_t k)
{
    // TODO 1: one thread computes one output in this block's output tile.
    // Decide the row/column mapping, accumulation, bounds, and output store.
    // Enable kNaiveImplemented only after completing the kernel.
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (size_t p = 0; p < k; p++)
    {
        sum += a[k * row + p] * b[n * p + col];
    }
    c[row * n + col] = sum;
}

__global__ void sgemm_shared(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c, size_t m, size_t n, size_t k)
{
    // TODO 2: shared-memory tiling; one output per thread.
    // The block owns a kSharedTile x kSharedTile output tile.
    // Dynamic scratch holds two kSharedTile x kSharedTile float tiles.
    // Design cooperative loads, reuse, synchronization, and M/N/K tails.

    constexpr int T = sgemm_config::kSharedTile;
    const size_t tx = threadIdx.x, ty = threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * T + tx;
    const size_t row = static_cast<size_t>(blockIdx.y) * T + ty;
    extern __shared__ float smem[];
    float* as = smem;
    float* bs = smem + T * T;

    float sum = 0.0f;
    for (size_t k0 = 0; k0 < k; k0 += T)
    {
        as[T * ty + tx] = (row < m && k0 + tx < k) ? a[row * k + k0 + tx] : 0.0f;
        bs[T * ty + tx] = (col < n && k0 + ty < k) ? b[(k0 + ty) * n + col] : 0.0f;
        __syncthreads();

        for (int e = 0; e < T; e++)
            sum += as[T * ty + e] * bs[T * e + tx];
        __syncthreads();
    }
    
    if (row < m && col < n)
        c[row * n + col] = sum;

}

__global__ void sgemm_register(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c, size_t m, size_t n, size_t k)
{
    // TODO 3: shared-memory tiling plus per-thread register tiling.
    // The block owns kRegisterTileM x kRegisterTileN outputs; each thread
    // computes kThreadTileM x kThreadTileN outputs. K tile: kRegisterTileK.
    // Choose thread/output and load mappings; cover the output tile once.
    // Dynamic scratch can hold A[BM,BK] and B[BK,BN]. Handle all tails.

    constexpr int BM = sgemm_config::kRegisterTileM,
                  BN = sgemm_config::kRegisterTileN,
                  BK = sgemm_config::kRegisterTileK;
    constexpr int TM = sgemm_config::kThreadTileM, 
                  TN = sgemm_config::kThreadTileN;
    constexpr int BX = sgemm_config::kBlockX, 
                  BY = sgemm_config::kBlockY;
    constexpr int NT = BX * BY;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BX + tx;
    const size_t block_row = static_cast<size_t>(blockIdx.y) * BM;
    const size_t block_col = static_cast<size_t>(blockIdx.x) * BN;

    extern __shared__ float smem[];
    float* as = smem;
    float* bs = smem + BM * BK;
    float acc[TM][TN] = {}; // 归零

    for (size_t k0 = 0; k0 < k; k0 += BK)
    {
        for (int q = tid; q < BM * BK; q += NT)
        {
            const size_t row = block_row + q / BK;
            const size_t p = k0 + q % BK;
            as[q] = (row < m && p < k) ? a[row * k + p] : 0.0f;
        }
        for (int q = tid; q < BK * BN; q +=NT)
        {
            const size_t p = k0 + q /BN;
            const size_t col = block_col + q % BN;
            bs[q] = (p < k && col < n) ? b[p * n + col] : 0.0f;
        }
        __syncthreads();

        for (int e = 0; e < BK; e++)
        {
            float a_frag[TM], b_frag[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_frag[i] = as[(ty + i * BY) * BK + e];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_frag[j] = bs[e * BN + tx + j * BX];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_frag[i] * b_frag[j];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        const size_t row = block_row + ty + i * BY;
        #pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            const size_t col = block_col + tx + j * BX;
            if (row < m && col < n)
                c[row * n + col] = acc[i][j];
        }
    }
}

void launch_gemm(int mode, const float* a, const float* b, float* c,
                 size_t m, size_t n, size_t k, const GemmLaunch& launch)
{
    if (!gemm_variant(mode).implemented)
        throw std::logic_error("TODO: selected GEMM kernel is not implemented");
    switch (mode)
    {
    case 0:
        sgemm_naive<<<launch.grid, launch.block, launch.shared_bytes>>>(a, b, c, m, n, k);
        break;
    case 1:
        sgemm_shared<<<launch.grid, launch.block, launch.shared_bytes>>>(a, b, c, m, n, k);
        break;
    case 2:
        sgemm_register<<<launch.grid, launch.block, launch.shared_bytes>>>(a, b, c, m, n, k);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}
