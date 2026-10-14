# pragma once

__global__ void stride_read(const float* __restrict__ in, float* __restrict__ out, int n, int stride);

__global__ void stride_write(const float* __restrict__ in, float* __restrict__ out, int n, int stride);

__global__ void transpose_matrix(const float* __restrict__ in, float* __restrict__ out , int row, int col);

__global__ void transpose_matrix_tiling(const float* __restrict__ in, float* __restrict__ out, int row, int col);

__global__ void transpose_matrix_tiling_32x8(const float* __restrict__ in, float* __restrict__ out, int row, int col);
// tools
// 普通copy
__global__ void normal_copy(const float* __restrict__ in, float* __restrict__ out, int n, int stride);