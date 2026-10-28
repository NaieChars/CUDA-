#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cstddef>

namespace sgemm_config
{
inline constexpr int kBlockX = 16;
inline constexpr int kBlockY = 16;
inline constexpr int kSharedTile = 16;
inline constexpr int kRegisterTileM = 32;
inline constexpr int kRegisterTileN = 32;
inline constexpr int kRegisterTileK = 16;
inline constexpr int kThreadTileM = 2;
inline constexpr int kThreadTileN = 2;

static_assert(kBlockX > 0 && kBlockY > 0 && kBlockX * kBlockY <= 1024);
static_assert(kSharedTile > 0 && kRegisterTileK > 0);
static_assert(kThreadTileM > 0 && kThreadTileN > 0);
static_assert(kSharedTile == kBlockX && kSharedTile == kBlockY);
static_assert(kRegisterTileM == kBlockY * kThreadTileM);
static_assert(kRegisterTileN == kBlockX * kThreadTileN);
}

struct GemmVariant
{
    int mode;
    const char* name;
    const char* kernel_name;
    bool implemented;
    int block_x, block_y;
    int tile_m, tile_n, tile_k;
    int thread_m, thread_n;
    size_t shared_bytes;
};

struct GemmLaunch
{
    dim3 grid;
    dim3 block;
    size_t shared_bytes;
};

const std::array<GemmVariant, 3>& gemm_variants();
const GemmVariant& gemm_variant(int mode);
GemmLaunch make_gemm_launch(int mode, size_t m, size_t n);

// All matrices are contiguous row-major: A[M,K], B[K,N], C[M,N].
// Compute C = A * B, overwriting every output; no alpha/beta or old C.
// Input/output never alias. M, N, K are positive, possibly nonmultiples of tiles.
// grid.x covers N (columns); grid.y covers M (rows).
// Each block owns its tile_m x tile_n output region, clipped to M x N.
// Use size_t for flattened indices; cast before large multiplications.
__global__ void sgemm_naive(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c, size_t m, size_t n, size_t k);
__global__ void sgemm_shared(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c, size_t m, size_t n, size_t k);
__global__ void sgemm_register(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c, size_t m, size_t n, size_t k);

// Dynamic shared scratch: naive=0; shared=2*T*T floats;
// register=(BM*BK + BK*BN) floats. Declare/partition it inside your kernels.
void launch_gemm(int mode, const float* a, const float* b, float* c,
                 size_t m, size_t n, size_t k, const GemmLaunch& launch);
