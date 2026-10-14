#include "kernels.cuh"

__global__ void normal_copy(const float* __restrict__ in, float* __restrict__ out, int n, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        out[idx] = in[idx];
    }
}


__global__ void stride_read(const float* __restrict__ in, float* __restrict__ out, int n, int stride)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    out[idx] = in[(size_t)idx * stride];
}

__global__ void stride_write(const float* __restrict__ in, float* __restrict__ out, int n, int stride)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    out[(size_t)idx * stride] = in[idx];
}

__global__ void transpose_matrix(const float* __restrict__ in, float* __restrict__ out , int row, int col)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(row) * static_cast<size_t>(col);
    if (idx >= total) return;

    // transpose
    size_t outRow = idx / row;
    size_t outCol = idx % row;

    out[idx] = in[outCol * static_cast<size_t>(col) + outRow];
}

#define TILE 32 // blockDim 必须是32的倍数，才能保证第一次读取元素到 shared memory 时是合并访存
__global__ void transpose_matrix_tiling(const float* __restrict__ in, float* __restrict__ out, int row, int col)
{
    // +1 是为了避免 shared memory bank conflict
    __shared__ float tile[TILE][TILE + 1];

    // thr_col, thr_row：每一个小block里的线程索引
    const int thr_col = threadIdx.x;
    const int thr_row = threadIdx.y;

    // 输入坐标：in[r][c]
    // blockIdx.x 对应输入列方向，blockIdx.y 对应输入行方向
    const int glo_col = blockIdx.x * TILE + thr_col;
    const int glo_row = blockIdx.y * TILE + thr_row;

    // 读全局内存：同一个 warp 中 thr_col 连续，
    // r 固定，c 连续，因此 in[r * col + c] 连续 -> 合并读
    if (glo_row < row && glo_col < col) 
    {
        tile[thr_row][thr_col] = in[static_cast<size_t>(glo_row) * col + glo_col];
    }

    __syncthreads();

    // 输出坐标：out[out_row][out_col]
    // 注意这里线程坐标发生了“转置”：
    // thr_row 作为输出行局部坐标，thr_col 作为输出列局部坐标
    const int out_row = blockIdx.x * TILE + thr_row; // 输出行，范围 [0, col)
    const int out_col = blockIdx.y * TILE + thr_col; // 输出列，范围 [0, row)

    // 写全局内存：同一个 warp 中 thr_col 连续，
    // out_row 固定，out_r 连续，因此 out[out_row * row + out_r] 连续 -> 合并写
    if (out_row < col && out_col < row) 
    {
        out[static_cast<size_t>(out_row) * row + out_col] = tile[thr_col][thr_row];
    }
}

// 每一个线程循环处理4个个元素
#define BLOCK_ROWS 8
__global__ void transpose_matrix_tiling_32x8(
    const float* __restrict__ in,
    float* __restrict__ out,
    int row,
    int col)
{
    __shared__ float tile[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int c = blockIdx.x * TILE + tx;
    const int r = blockIdx.y * TILE + ty;

    // 连续读取：每个线程负责 4 行
    #pragma unroll
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
    {
        if (r + j < row && c < col)
        {
            tile[ty + j][tx] = in[static_cast<size_t>(r + j) * col + c];
        }
    }

    __syncthreads();

    const int out_c = blockIdx.x * TILE + ty;
    const int out_r = blockIdx.y * TILE + tx;

    // 转置后连续写入
    #pragma unroll
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
    {
        if (out_c + j < col && out_r < row)
        {
            out[static_cast<size_t>(out_c + j) * row + out_r] = tile[tx][ty + j];
        }
    }
}