# SGEMM 实验：从朴素实现到数据复用

目标：完成行主序 FP32 矩阵乘法 C[M,N] = A[M,K] × B[K,N]，依次研究 shared-memory tiling 和 register tiling。目录、公共工具、校验/计时/采集流程沿用 reduction，关键 GPU 算法由自己编写。

**当前状态：三版 kernel 已由用户实现，并完成正确性检查及首轮统一性能采集（2026-10-19）。** 原始指标、运行条件和分析见 [性能指标汇总](docs/性能指标汇总.md)。未完成的版本仍由启用标志保护，不会被脚本记录为成功。

## 1. 文件与职责

- src/kernels.cuh：接口、tile 常量和启动配置约定。
- src/kernels.cu：三个用户实现的 kernel、启用标志、启动函数。
- src/main.cu：参数解析、输入生成、完整 CPU-double 参考、逐元素校验、计时和采集范围。
- scripts/experiment_common.ps1：构建、版本发现、结果目录、源码/程序哈希及统计。
- scripts/run_benchmark.ps1：正确性检查或普通计时，按轮次轮换配置顺序。
- scripts/run_ncu.ps1：校验、预热后，仅采集一次 GEMM kernel，显式保存 clock/cache/replay 策略。
- docs/性能指标汇总.md：后续粘贴原始样本与记录分析。
- result/：运行时生成，受仓库现有 Git 忽略规则管理。

继续使用 common/include 中的 CudaBuffer、CudaTimer 和 CUDA_CHECK；reduction 与转置实验无需修改。

## 2. 从哪里开始写

先完成 src/kernels.cu 的 sgemm_naive，再依次完成后两版。每完成一版，仅将对应 kNaiveImplemented / kSharedImplemented / kRegisterImplemented 改成 true，然后重建。

| mode | 名称 | block(x,y) | block 输出区域 BM×BN | K 分块 BK | 每线程输出 TM×TN | 动态 shared |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | naive | 16×16 | 16×16 | 无 | 1×1 | 0 |
| 1 | shared | 16×16 | 16×16 | 16 | 1×1 | 2,048 byte |
| 2 | register_tiled | 16×16 | 32×32 | 16 | 2×2 | 4,096 byte |

这里的大小是明确的起步约定，不预设它们是最优参数。常量位于 sgemm_config；修改后须重建。static_assert 检查输出 tile 与线程分块关系；启动元数据、脚本 CSV 和计划随常量更新。

框架最初仅提供职责与资源约定；当前索引映射、协作加载、同步和寄存器累加已由用户完成。

### 接口约定

- A、B、C 均连续存储、行主序；无转置、padding leading dimension 或输入输出重叠。
- 本阶段计算 C=AB，完全覆盖 C；无需读取旧 C，也无 alpha/beta 累加。
- M、N、K 都是正整数，可以是矩形，也可以不被 tile 整除。
- grid.x 沿 N 列方向，grid.y 沿 M 行方向；block 负责对应 BM×BN 输出区域。
- shared/register 两版按各自 BK 处理 K；线程到输出/输入元素的具体映射由自己决定。
- shared scratch 已由 launcher 分配；需要时在 kernel 内声明 extern __shared__ 并自行分区。若改成静态 shared 或调整 padding，须同步调整元数据/启动资源，避免重复预留。
- 索引和形状使用 size_t；先转换再做大乘法。输出边界、加载边界和同步都由自己处理。
- mode 2 的 2×2 是每线程输出数量；如何映射这四个输出属于练习内容。

## 3. 构建和命令行

从仓库根目录执行，需配置 CUDA 与 MSVC/C++ 工具链：

~~~powershell
cmake -S . -B build
cmake --build build --config Release --target sgemm --parallel 1

.\build\labs\sgemm\Release\sgemm.exe --help
.\build\labs\sgemm\Release\sgemm.exe --list

# 不启动 GPU，未完成算法时也可查看；顺序是 mode M N K。
.\build\labs\sgemm\Release\sgemm.exe 0 127 193 65 --plan
.\build\labs\sgemm\Release\sgemm.exe 2 127 193 65 --plan
~~~

参数：sgemm.exe [mode=0] [M=512] [N=512] [K=512]。提供部分位置参数时，未提供的维度保持 512；建议显式写出三个维度。维度是逻辑尺寸，不是元素总数。

| 选项 | 用途 |
| --- | --- |
| --help / --list | 帮助 / 版本与 tile 元数据，不启动 GPU |
| --plan | 查看 grid、block、动态 shared 请求和 2MNK FLOP，不启动 GPU |
| --check-only | 只执行一次 GEMM 并逐元素校验，不计时 |
| --warmup N | 预热次数，默认 10 |
| --iterations N | CUDA events 计时区间的启动次数，默认 100 |
| --profile | 校验、预热后，仅暴露一次 GEMM 给 ncu；采集后再次校验 |
| --atol X --rtol X | 数值容差，均默认 1e-4 |

退出码：0 成功、1 CUDA 错误、2 参数或环境错误、3 所选 kernel 尚未实现、4 数值校验失败。--plan 是主机端计划，不代替设备的 grid/shared 上限检查。真实启动前会检查设备限制。

## 4. 校验与时间口径

输入使用两种不同的确定性、正负混合二进制分数，CPU 用 double 计算完整参考矩阵。每个 C 元素都须满足 abs(actual-reference) <= atol + rtol×abs(reference)，且 actual 必须有限；失败时显示失败元素数及第一个错误的行列。

首次启动前 C 用 NaN 填充，未写出的位置会失败。普通计时前后均校验；分配、输入上传、CPU 参考和 C 回读均不在 CUDA events 计时区间内。

完整 CPU 校验的开销为 O(MNK)，默认 512 用于保证初期练习方便；更大尺寸仍做完整校验，重复运行会重复计算参考。改输入或累加策略后应重新检查误差，当前有限输入分布和容差不保证所有分布的数值精度。

性能使用一次 GEMM 的平均 time_ms 和 GFLOP/s = 2MNK / (time_ms×10^6)，FMA 约定计为 2 FLOP。没有把 A/B/C 各读写一遍当作实际 DRAM 流量；真实流量由 profiler 测量。小矩阵可能明显受启动/提交间隙影响。

## 5. 普通运行脚本

默认自动构建，跳过所有 TODO；若没有可运行的实现，不生成结果目录。

~~~powershell
# 完成 mode 0 后：先检查非方阵和 M/N/K 各种尾部。
.\labs\sgemm\scripts\run_benchmark.ps1 -Modes 0 -Shapes '1x1x1','3x5x7','15x17x19','16x16x16','17x19x23','31x33x29','33x35x17' -CheckOnly -Repeat 1

# 只测已完成的 mode 0。
.\labs\sgemm\scripts\run_benchmark.ps1 -Modes 0 -Shapes '512x512x512'

# 三版写完后统一比较；每配置默认 10 个样本。
.\labs\sgemm\scripts\run_benchmark.ps1 -Modes 0,1,2 -Shapes '512x512x512','1024x1024x1024'

# 已自行重建时可复用程序。
.\labs\sgemm\scripts\run_benchmark.ps1 -Modes 0 -SkipBuild
~~~

Shapes 每个字符串严格按 MxNxK 解释；M/N/K 均须为正整数。-Repeat、-Warmup、-Iterations 和 -BuildDir 可调整。各配置按 run 循环移动顺序，保存 execution_order.csv；统计保留均值、中位数、样本标准差与范围，单样本标准差留空，重复或非有限样本不会补零。

## 6. ncu 采集

~~~powershell
.\labs\sgemm\scripts\run_ncu.ps1 -Modes 0 -Shapes '512x512x512'

# 快速检查采集链路。
.\labs\sgemm\scripts\run_ncu.ps1 -Modes 0 -Repeat 1 -Metrics 'gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed'

# 对照 profiler 的频率条件，策略写入 manifest。
.\labs\sgemm\scripts\run_ncu.ps1 -Modes 0,1,2 -ClockControl base -CacheControl all
~~~

默认 Repeat=3、ClockControl=none、CacheControl=all、ReplayMode=kernel。这些参数显式传给 ncu，并记录在 manifest 中。默认指标涉及吞吐、时钟、global requests/sectors、shared wavefront/conflicts、寄存器与驻留限制、FFMA、issue/eligible 和依赖状态。可通过 -Metrics 覆盖，必须提供目标设备支持的完整指标名；不支持/缺失/重复/无效指标会报错，不补成 0。

Stage=kernel 表示仅一次 GEMM 启动，且位于 cudaProfilerStart/Stop 范围中。普通计时 Stage=complete 也是同一 GEMM 的重复启动平均，但无 profiler 控制；即使计算范围相同，也应核对 clocks/cache/replay 后再比较。CPU 校验和预热不进入采集范围。

如果 PowerShell 执行策略阻止脚本，可通过进程级 Bypass 执行；不需要修改系统执行策略：

~~~powershell
powershell -ExecutionPolicy Bypass -File .\labs\sgemm\scripts\run_benchmark.ps1 -Modes 0
~~~

## 7. 结果与框架验证

result/<时间戳>_<阶段>_<唯一标识>/ 保存 manifest.json、execution_order.csv、逐次 normal/ncu 日志、metrics.csv、ncu-rep，以及 measurements_samples.csv / measurements_average.csv。CSV 同时保留 M/N/K、block、tile 与线程输出块元数据。

manifest 包含 exe/源码哈希、CPU 校验口径和 profiler 策略。-SkipBuild 只复用 exe，源码可能已改变；结合哈希检查，不假定它们一致。

框架最初通过 Release 构建和 48 组命令行/调度/TODO/非法参数检查，另验证了脚本解析及统计。三版完成后又通过 24 组小尺寸/矩形/尾部检查，shared/register 在 65×67×33 上通过 racecheck/synccheck；首轮性能比较包含 120 次普通计时进程和 45 份 NCU 报告。完整范围、源码/程序哈希、采集异常与限制均保留在性能指标汇总中。
