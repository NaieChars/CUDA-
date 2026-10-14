# 矩阵转置性能分析

## 1. 朴素方案与 Tiling 方案性能指标对比与分析

**实验对象：**
- **朴素方法**：矩阵 4096 x 4096，block 256
- **Tiling方法**：矩阵 4096 x 4096，Tile 32

**实验设置说明：**
实验包含 10 次预热，ncu 捕获第 11 次的数据；同时为了避免 ncu 带来的时间拉伸，先不带 ncu 跑一次程序，获得准确的运行时间与有效带宽。

---

### 1.1 朴素方法

#### 运行上下文 (Context)
> **Mode C** | `row=4096` | `col=4096` | `block=256` | **time=0.6919 ms** | **Effective bandwidth=194.00 GB/s**

#### 性能指标详情 (Metrics)

| Metric Name | Metric Unit | Metric Value |
| :--- | :---: | ---: |
| `dram__bytes_op_read.sum` | Mbyte | 67.11 |
| `dram__bytes_op_write.sum` | Mbyte | 66.02 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | % | 36.66 |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | % | 70.31 |
| `gpu__time_duration.sum` | ms | 1.03 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | - | 524,288 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum` | - | 524,288 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | sector | 16,777,216 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | sector | 2,097,152 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | % | 18.70 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | % | 83.52 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | inst | 27.02 |

---

### 1.2 Tiling 方法

#### 运行上下文 (Context)
> **Mode D (Tiled)** | `row=4096` | `col=4096` | `TILE=32` | **time=0.6246 ms** | **Effective bandwidth=214.88 GB/s**

#### 性能指标详情 (Metrics)

| Metric Name | Metric Unit | Metric Value |
| :--- | :---: | ---: |
| `dram__bytes_op_read.sum` | Mbyte | 67.12 |
| `dram__bytes_op_write.sum` | Mbyte | 62.74 |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | % | 43.37 |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | % | 43.37 |
| `gpu__time_duration.sum` | us | 851.39 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | - | 524,288 |
| `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum` | - | 524,288 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | sector | 2,097,152 |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | sector | 2,097,152 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | % | 26.81 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | % | 63.95 |
| `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` | inst | 13.68 |

---

### 1.3 分析与结论

- 朴素方法的全局读取存在明显的非合并访存，Load Sectors/Request 达到 32；Tiling 方法将其降至 4，与理想的连续访问模式一致。两种方法的写入均为 4，说明它们的写入访问合并情况基本相同。
- 针对朴素方法的**DRAM吞吐量和内存吞吐量**，内存吞吐量70.31%远大于DRAM吞吐量36.66%，这说明朴素方法的读取合并效率很差，L1TEX 层的访问请求数量明显膨胀。这与请求在显存接口之前的某个环节消耗了更多资源的解释相符，但还需要进一步查看 L1TEX、L2 等单元的吞吐细分才能确认。针对Tiling方法，DRAM吞吐量和内存吞吐量相等，Long Scoreboard 下降了约50%，这说明Tiling 减少了低效率的全局内存访问，并且伴随着内存相关等待减少。
-  优化后 DRAM 仍未达到持续峰值。当前表现可能同时受到访存延迟、可并发在途请求数量、SM 执行效率以及额外指令开销的影响，尚不能确定唯一瓶颈。
- 平均耗时并无太大提升的原因可能在于：同步线程存在开销；然后可能存在SM内部的指令数量变多，额外的存入和读取共享内存，以及同步指令，同时kernel里的索引计算变复杂了，这也许也是sm吞吐量增加的原因
- Occupancy 下降，`sm__warps_active` 从 83.52% 到 63.95%。在朴素方法下，block = 256，5060 一个 SM 的最大线程数为 1536（48 warp），此时一个 SM 可容纳的最大 block = 6，该条件下一个 block 里有 8 个 warp，一共可容纳 48 warp，理论上的 occupancy 为 100%，这里是 80% 可能由于其他原因。Tiling = 32 方法下，一个 block 线程数为 1024（32 个 warp），一个 SM 最多只能容纳 1 个 block，实际的 occupancy 只有 32/48 = 66.67%，这应该就是占用率下降的主要原因。

---

> 以上就是我在加入几个新指标前写的一个对几个重要指标的一些对比与分析与猜测，已经在上一个对话让gpt分析过了。
>
> 用到的命令行指令：  
> ncu --import .\.\labs\day04_coalescing\result\transpose_4096_4096_b256.ncu-rep --page details
> powershell -ExecutionPolicy Bypass -File .\labs\day04_coalescing\scripts\run_ncu.ps1（对于朴素方法，对应命令行参数是2，tilinga 方法是3，在命令行脚本修改）