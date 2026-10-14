#include <cstdio>
#include <cstdlib>
#include "cuda_check.cuh"
#include "cuda_timer.cuh"
#include "cuda_buffer.cuh"
#include "kernels.cuh"

int main(int argc, char** argv)
{
    int mode   = 0;
    int stride = 1;
    int row    = 1024;
    int col    = 1024;
    int block  = 256;

    if (argc >= 2) mode   = atoi(argv[1]);
    if (argc >= 3) stride = atoi(argv[2]);
    if (argc >= 4) row    = atoi(argv[3]);
    if (argc >= 5) col    = atoi(argv[4]);
    if (argc >= 6) block  = atoi(argv[5]);

    int n = row * col;

    if (mode == 0)
    {
        // 实验 A：读写合并
        CudaBuffer<float> d_in(n * stride);
        CudaBuffer<float> d_out(n);

        CUDA_CHECK(cudaMemset(d_in.get(), 0, d_in.size() * sizeof(float)));

        int grid = (n + block - 1) / block;

        CudaTimer timer;
        float ms = timer.benchmark([&]() {
            //stride_read<<<grid, block>>>(d_in.get(), d_out.get(), n, stride);
            normal_copy<<<grid, block>>>(d_in.get(), d_out.get(), n, 1);    // test：替换的核函数为简单读写合并访存
        });

        double bytes_moved = 2.0 * n * sizeof(float);
        double bw = bytes_moved / (ms / 1000.0) / 1e9;

        printf(
            "Mode A | stride=%d | n=%d | block=%d | time=%.4f ms | "
            "Effective read & write bandwidth=%.2f GB/s\n",
            stride, n, block, ms, bw
        );
    }
    else if (mode == 1)
    {
        // 实验 B：读合并，写非合并
        CudaBuffer<float> d_in(n);
        CudaBuffer<float> d_out(n * stride);

        CUDA_CHECK(cudaMemset(d_in.get(), 0, d_in.size() * sizeof(float)));

        int grid = (n + block - 1) / block;

        CudaTimer timer;
        float ms = timer.benchmark([&]() {
            stride_write<<<grid, block>>>(d_in.get(), d_out.get(), n, stride);
        });

        double bytes_read = static_cast<double>(n) * sizeof(float);
        double bw = bytes_read / (ms / 1000.0) / 1e9;

        printf(
            "Mode B | stride=%d | n=%d | block=%d | time=%.4f ms | "
            "Effective read bandwidth=%.2f GB/s\n",
            stride, n, block, ms, bw
        );
    }
    else if (mode == 2)
    {
        // 实验 C：转置，读列写行
        CudaBuffer<float> d_in(row * col);
        CudaBuffer<float> d_out(row * col);

        CUDA_CHECK(cudaMemset(d_in.get(), 0, d_in.size() * sizeof(float)));

        int grid = (row * col + block - 1) / block;

        CudaTimer timer;
        float ms = timer.benchmark([&]() {
            transpose_matrix<<<grid, block>>>(d_in.get(), d_out.get(), row, col);
        });

        // 转置的“有效数据量”：读一次 + 写一次，各 row*col*sizeof(float)
        double bytes_rw = 2.0 * static_cast<double>(row) * col * sizeof(float);
        double bw = bytes_rw / (ms / 1000.0) / 1e9;

        printf(
            "Mode C | row=%d | col=%d | block=%d | time=%.4f ms | "
            "Effective bandwidth=%.2f GB/s\n",
            row, col, block, ms, bw
        );
    }
    else if (mode == 3)
    {
        // 实验 D：Tiling 转置
        CudaBuffer<float> d_in(row * col);
        CudaBuffer<float> d_out(row * col);

        CUDA_CHECK(cudaMemset(d_in.get(), 0, d_in.size() * sizeof(float)));

        // Tiled 转置必须使用 2D 的 block 和 grid
        // TILE 通常设置为 32，这样刚好是一个 warp 的大小，完美合并访存
        const int TILE = 32; 

        dim3 block_dim(TILE, TILE); // 32 x 32 = 1024 个线程
        dim3 grid_dim((col + TILE - 1) / TILE, (row + TILE - 1) / TILE);

        CudaTimer timer;
        float ms = timer.benchmark([&]() {
            transpose_matrix_tiling<<<grid_dim, block_dim>>>(d_in.get(), d_out.get(), row, col);
        });

        // 转置的“有效数据量”：读一次 + 写一次，各 row*col*sizeof(float)
        double bytes_rw = 2.0 * static_cast<double>(row) * col * sizeof(float);
        double bw = bytes_rw / (ms / 1000.0) / 1e9;

        printf(
            "Mode D (Tiled) | row=%d | col=%d | TILE=%d | time=%.4f ms | "
            "Effective bandwidth=%.2f GB/s\n",
            row, col, TILE, ms, bw
        );
    }
    else if (mode == 4)
    {
        // 新增：实验 E：循环 Tiling 转置，32 * 8
        CudaBuffer<float> d_in(row * col);
        CudaBuffer<float> d_out(row * col);

        CUDA_CHECK(cudaMemset(d_in.get(), 0, d_in.size() * sizeof(float)));

        const int TILE = 32; 
        const int BLOCK_ROWS = 8;

        dim3 grid((col + TILE - 1) / TILE, (row + TILE - 1) / TILE);
        dim3 block_32x8(TILE, BLOCK_ROWS);

        CudaTimer timer;
        float ms = timer.benchmark([&]() {
            transpose_matrix_tiling_32x8<<<grid, block_32x8>>>(d_in.get(), d_out.get(), row, col);
        });

        double bytes_rw = 2.0 * static_cast<double>(row) * col * sizeof(float);
        double bw = bytes_rw / (ms / 1000.0) / 1e9;

        printf(
            "Mode E (Tiled) | row=%d | col=%d | TILE=%d | time=%.4f ms | "
            "Effective bandwidth=%.2f GB/s\n",
            row, col, TILE, ms, bw
        );
    }

    return 0;
}