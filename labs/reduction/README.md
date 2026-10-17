# 并行归约实验：Reduction

目标：将一个 `float` 数组求和为一个标量，逐步研究 shared memory、每线程工作量、warp 通信和同步如何影响性能。文件组织沿用 day04，GPU 归约算法由自己实现。

**当前状态：三个 kernel 均已实现，已完成正确性校验和首轮性能比较。** 原始样本、采集条件和分析见 [性能指标汇总.md](docs/性能指标汇总.md)。新增 `scripts/run_comparison.ps1` 支持轮换配置顺序的完整计时、显式 clock/cache 条件下的 NCU 采集与 `items=1` 控制组；下文保留原有逐步实现说明。

## 1. 文件与职责

```text
reduction/
├── CMakeLists.txt
├── src/
│   ├── main.cu                    参数、输入、验证、多轮调度、计时、采集范围
│   ├── kernels.cuh                单轮归约的接口约定
│   └── kernels.cu                 三个 TODO kernel 与各自启用标志
├── scripts/
│   ├── experiment_common.ps1      构建、发现版本、统计与结果记录
│   ├── run_benchmark.ps1          完整归约的普通计时或正确性检查
│   └── run_ncu.ps1                第一轮大数组 kernel 的指标采集
├── docs/
│   └── 性能指标汇总.md             自己填写实验记录
└── result/                        运行时生成；结果不提交到 Git
```

继续使用仓库 `common/include` 下的 `CudaBuffer`、`CudaTimer` 和 `CUDA_CHECK`。公共工具及 day04 的代码不需要改动。

## 2. 从哪里开始写算法

打开 [kernels.cu](src/kernels.cu)，依次完成：

| mode | 名称 | kernel | 每线程元素数 |
| ---: | :--- | :--- | :--- |
| 0 | shared | `reduce_shared` | 固定 1 |
| 1 | multi_item | `reduce_multi_item` | 可配置，默认 4 |
| 2 | shuffle | `reduce_shuffle` | 可配置，默认 4 |

建议先只做 mode 0。框架未提供归约循环、shuffle 实现或完整伪代码；函数体中的 TODO 仅描述目标。

完成某个 kernel 后，将本文件顶部对应的标志改为 `true`：

```cpp
constexpr bool kSharedImplemented = false;
constexpr bool kMultiItemImplemented = false;
constexpr bool kShuffleImplemented = false;
```

只启用已完成的版本，重新构建。`--list` 会显示状态，两个脚本据此选择可测量的实现。

### 单轮归约的接口约定

- 输入是 `in[0..n)`，输出是 `partial[0..grid.x)`；二者不重叠。
- 每个 block 必须写出一个部分和到 `partial[blockIdx.x]`，不能依赖输出缓冲区原有值，也不需要原子累加。
- `block` 是 `[32, 1024]` 中的 2 的幂，并受实际设备上限约束。
- block `b` 负责的输入区间是 `[b * block * items, min((b + 1) * block * items, n))`。线程到元素的具体分配由自己决定。
- mode 0 中 `items=1`；其他版本接受 `[1,64]` 的 `items`。
- 尾部 block 可能不足一个完整区间，需要自己处理；同步与越界处理也由自己设计。
- 三个版本均预分配 `block * sizeof(float)` byte 的动态 shared memory；需要时在 kernel 内自行声明，可以只使用其中一部分。以后调整资源配置时应同步修改启动函数。
- 索引与长度使用 `size_t`，较大乘法应在转换后进行，避免先在 `int` 中溢出。

同一 kernel 既处理原始输入，也处理后续的部分和数组，所以不能只支持整除、固定长度或大数组。

## 3. main 已经完成的工作

### 输入与结果验证

默认输入长度为 `16,777,216`，使用确定性的非全零、正负混合数据；CPU 用 `double` 累加得到参考结果。验证标准为：

```text
abs(GPU结果 - CPU参考结果) <= atol + rtol * abs(CPU参考结果)
默认 atol=1e-3，rtol=1e-6
```

输入不使用随机种子，便于复现。浮点归约顺序可能改变误差，容差不是正确性的数学证明；如调整输入分布或算法，需要检查误差是否合理。

完整归约先验证一次，通过后才预热和计时；普通计时结束后再次检查结果。验证失败返回非零退出码，脚本停止该测试。两个 scratch 缓冲区在首次验证前用 NaN 填充，以发现未写出的输出，填充不计时。

### 多轮调度

main 按每个 block 的处理容量计算下一轮长度，使用两个 scratch 缓冲区交替保存部分和，直到得到一个标量。每次完整归约都重新从原始输入开始，不会误用上一次的结果。

例如 `n=16777216, block=256, items=1`：

```text
第 0 轮：16,777,216 -> 65,536 个部分和
第 1 轮：    65,536 ->    256 个部分和
第 2 轮：       256 ->      1 个最终结果
```

`n=1` 也执行一轮 kernel，以保持统一的输出约定。无需在 main 中实现跨 block 的 GPU 同步。

## 4. 构建与命令行

以下命令从仓库根目录运行，需要配置 CUDA 和主机编译工具：

```powershell
cmake -S . -B build
cmake --build build --config Release --target reduction --parallel

.\build\labs\reduction\Release\reduction.exe --help
.\build\labs\reduction\Release\reduction.exe --list
.\build\labs\reduction\Release\reduction.exe 0 16777216 256 1 --plan
```

参数顺序：

```text
reduction.exe [mode=0] [n=16777216] [block=256] [items_per_thread]
```

| 选项 | 用途 |
| :--- | :--- |
| `--help` | 查看帮助 |
| `--list` | 查看版本名称、kernel 名和实现状态，不需要启动 GPU |
| `--plan` | 查看各轮输入规模和 block 数；算法未实现时也能使用，不启动 GPU |
| `--check-only` | 只执行一次完整归约并校验，不输出性能数字 |
| `--warmup N` | 完整归约的预热次数，默认 10 |
| `--iterations N` | 普通计时区间内的完整归约次数，默认 100 |
| `--atol X --rtol X` | 指定误差容限 |
| `--profile` | 校验、预热后，只暴露第一轮给 ncu；由采集脚本使用 |

启用 mode 0 后，可先检查尾部尺寸，再做大数组测量：

```powershell
.\build\labs\reduction\Release\reduction.exe 0 1003 256 1 --check-only
.\build\labs\reduction\Release\reduction.exe 0 16777216 256 1
```

退出码：`0` 成功，`1` CUDA 错误，`2` 参数或运行配置错误，`3` 算法 TODO，`4` 数值校验失败。

## 5. 普通测量脚本

```powershell
# 自动构建 Release；默认跳过所有尚未完成的版本。
.\labs\reduction\scripts\run_benchmark.ps1

# 只测基础版，10 次程序运行，每次 10 次预热、100 次完整归约计时。
.\labs\reduction\scripts\run_benchmark.ps1 -Modes 0

# 只检查若干小尺寸与尾部，不计时；这里每个配置只运行一次。
.\labs\reduction\scripts\run_benchmark.ps1 -Modes 0 -Sizes 1,31,32,33,255,256,257,1003 -CheckOnly -Repeat 1

# 完成多元素版后，扫描参数。
.\labs\reduction\scripts\run_benchmark.ps1 -Modes 1 -Blocks 128,256,512 -ItemsPerThread 2,4,8
```

脚本默认 `Modes=0,1,2`、`Sizes=16777216`、`Blocks=256`、`ItemsPerThread=4`，mode 0 自动使用 1。`-Repeat`、`-Warmup`、`-Iterations` 可配置；`-SkipBuild` 明确复用已有程序；`-BuildDir` 可切换构建目录。

如果系统执行策略不允许直接运行 `.ps1`，可在 PowerShell 中通过进程级 Bypass 启动，例如：

```powershell
powershell -ExecutionPolicy Bypass -File .\labs\reduction\scripts\run_benchmark.ps1 -Modes 0
```

### 时间与带宽口径

- 普通计时覆盖一次完整归约的所有 kernel 轮次及它们之间的 GPU 时间线间隔，不包含 CPU 参考、分配、上传、结果回读或校验。
- 一次程序运行先预热 10 次完整归约，再用 CUDA events 计时 100 次，得到每次完整归约的平均时间。
- 脚本默认运行程序 10 次，保留这 10 个结果，统计均值、中位数、样本标准差和范围；只有一个样本时标准差留空。
- 输入有效带宽 = `n * sizeof(float) / 完整归约时间`，采用十进制 GB/s，**不使用转置的两倍数据量**。
- 中间部分和读写、最终输出属于额外流量，没有计入有效带宽分子。该数字也不是 DRAM 实测流量。

## 6. ncu 指标脚本

```powershell
.\labs\reduction\scripts\run_ncu.ps1 -Modes 0

# 快速检查采集链路时可暂时减少指标和重复次数。
.\labs\reduction\scripts\run_ncu.ps1 -Modes 0 -Repeat 1 -Metrics 'gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed'
```

默认采集时间、DRAM 字节与吞吐、全局访存请求/sector、occupancy 及资源限制、指令、调度、stall 与 shared memory 指标。可用 `-Metrics` 覆盖，也可用 `-NcuPath` 指定 ncu。

**采集范围是第一轮处理原始大数组的 kernel，记录为 `Stage=first_pass`。** 程序先在范围外完成校验和预热，再用 `cudaProfilerStart/Stop` 包住一次首轮启动。脚本使用 `--profile-from-start off --launch-count 1`，不靠“跳过十个 kernel”猜测完整归约的预热结束位置。

因此，ncu 时间不是完整归约时间，指标不能与后续小数组轮次混合平均。普通计时与 profiler 数据应分别分析。[NVIDIA 采集控制选项](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

## 7. 结果文件

普通计时和 ncu 分别创建独立目录，保留算法、长度、block 和每线程元素数：

```text
result/<时间戳>_<测量阶段>_<唯一标识>/
├── manifest.json
├── shared_n16777216_b256_i1_run01_normal.log      普通运行
├── shared_n16777216_b256_i1_normal_samples.csv
├── shared_n16777216_b256_i1_normal_average.csv
└── ...

result/<另一次时间戳>_first_pass_<唯一标识>/
├── manifest.json
├── shared_n16777216_b256_i1_run01.ncu-rep
├── shared_n16777216_b256_i1_run01_ncu.log
├── shared_n16777216_b256_i1_run01_metrics.csv
├── shared_n16777216_b256_i1_ncu_samples.csv
└── shared_n16777216_b256_i1_ncu_average.csv
```

`manifest.json` 保存 exe 路径与哈希、源码哈希、测试配置和指标列表。使用 `-SkipBuild` 时源码不一定与 exe 同步，应结合哈希和构建状态判断。

CSV 保留 `Case/Mode/Kernel/Stage/N/Block/ItemsPerThread/Run/Metric/Unit/Value`；汇总包含样本数、均值、中位数、标准差、最小值和最大值。缺失、无效、重复或非有限数值不会被补成零。

当前脚本按配置分组运行，与 day04 风格一致。遇到小幅差异或异常样本时，先查看逐次结果，不能只凭平均指标作因果判断。

## 8. 实验记录与指标分析

在 [性能指标汇总.md](docs/性能指标汇总.md)

| 版本 | n | block | 每线程元素数 | 校验 | 完整平均时间 ms | 输入有效带宽 GB/s |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: |
| shared | 待测 | 待测 | 1 | 待验证 | 待测 | 待测 |
| multi_item | 待测 | 待测 | 待测 | 待验证 | 待测 | 待测 |
| shuffle | 待测 | 待测 | 待测 | 待验证 | 待测 | 待测 |

后续每次只改变一个主要因素：先完成基础算法与尾部校验，再研究每线程工作量，最后研究 warp 通信。暂不加入原子版或库函数基线。

## 9. 框架验证范围

已检查 Release 构建、帮助与版本列表、无效参数、TODO 退出码、多个输入长度的调度计划、脚本的 TODO 跳过行为，以及统计计算和重复样本拒绝。

2026-10-16：三版通过 232 组 GPU 数值检查，完成普通计时与 NCU 采集；具体输入范围、误差条件和证据边界以 [性能指标汇总.md](docs/性能指标汇总.md) 为准。
