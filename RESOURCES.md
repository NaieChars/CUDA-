# CUDA 算子开发与优化 Resources

## Knowledge
- [NVIDIA CUDA 12.8.1：CUDA Binary Utilities](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-binary-utilities/index.html)
  第 06 课主阅读：cuobjdump 与 nvdisasm 的输入范围、dump-sass/resource-usage、cubin 提取与指令族。2026-10-19 已用本机 12.8.55 验证 shared 函数导出；架构与指令体以 sm_120 实际产物为准。
- [NVIDIA PTX ISA 8.7：Introduction](https://docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html#introduction)
  第 06 课用于 CUDA C++、PTX 虚拟指令集、后端机器码生成与 SASS 的关系；PTX 虚拟寄存器数不能当硬件资源计数，不要求初学者通读 ISA。
- [NVIDIA Nsight Compute：Source Page](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#source-page)
  第 06 课用于源码/SASS 并排视图、View、Resolve、Raw 与缺失信息处理。在线文档会更新，备课已核查本机 Nsight Compute 2025.2.0/docs/NsightCompute/index.html 对应说明；GUI 操作留给课堂练习，未把手册核查记为实操通过。
- [NVIDIA：Optimizing Parallel Reduction in CUDA，Mark Harris](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
  用于树形归约、载入时先相加以及串行局部累加与并行树的组合。本节推荐幻灯片 18（First Add During Load）和 31（Algorithm Cascading）。这是历史教程；旧硬件成绩不能预测当前 GPU 表现，后续 warp 同步须以当前编程指南为准。
- [NVIDIA：CUDA C++ Best Practices Guide — Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
  用于判断同一 warp 在同一条访存指令中访问的地址；本节据此推导 base + tid + j * blockDim.x。
- [NVIDIA：CUDA C++ Programming Guide — Synchronization Functions](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#synchronization-functions)
  用于检查 shared memory 可见性、block 屏障位置及参与条件。
- [NVIDIA：CUDA C++ Best Practices Guide — Floating Point Math Is Not Associative](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#floating-point-math-is-not-associative)
  用于解释改变求和分组后可能产生的数值误差，以及为什么需要容差校验。

- [NVIDIA：Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
  用于 warp shuffle、lane、mask 的入门。练习 02 主要阅读 Listing 1；本项目通过完整 warp 和补零维持参与条件，不照搬部分 lane 的归约示例。
- [NVIDIA：CUDA C++ Programming Guide — Warp Shuffle Functions](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#warp-shuffle-functions)
  用于核对 shfl_down 的来源 lane、越过宽度时返回自己的值、mask 参与规则，以及 shuffle 不提供内存屏障。

- [NVIDIA：CUDA C++ Programming Guide §6.2.4 — Shared Memory（含两版 MatMulKernel）](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#shared-memory)
  练习 03 阅读第一个朴素实现；练习 04 阅读第二个 tiled MatMulKernel，重点是 block 输出归属、每线程加载 A/B 各一个元素，以及“加载→同步→计算→同步”。官方例子假设尺寸整除；本项目额外以独立边界判断和补零支持 M/N/K 尾部，沿用框架的动态 shared。
- [NVIDIA：Best Practices §10.2.3.2 — Matrix Multiplication C=AB](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)
  用于输入复用与流量分析，尤其注意避免重复传输与合并访存改善是不同诊断。不能直接照搬其 K=warpSize、block=32×32 的示例假设；本项目为通用 K、block=16×16。源码标量读取计数不能当作 DRAM 字节或加速比。

- [NVIDIA CUTLASS：Efficient GEMM — Thread-level GEMM](https://docs.nvidia.com/cutlass/4.5.2/media/docs/cpp/efficient_gemm.html#thread-level-gemm)
  练习 05 主阅读：线程内二维分块、寄存器操作数复用与累加外积。只借鉴原理；本项目使用普通 FP32 乘加及自定线程映射，不要求学习 CUTLASS API、Tensor Core 或流水线。
- [NVIDIA Best Practices：Local Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#local-memory) 与 [Register Pressure](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#register-pressure)
  用于区分源码局部数组和实际寄存器分配，理解固定小数组、循环展开及寄存器压力的证据边界。实际使用量和溢出留给编译报告与 profiler 核查。

- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
  用于计数单位、子管线 throughput、replay/cache/clock 策略与越界指标的诊断。2026-10-19 的 SGEMM 实验优先核对本机 2025.2 随附文档及报告元数据；L2 命中率异常保留原值，不修剪或补猜。分析见 labs/sgemm/docs/性能指标汇总.md。

## Wisdom (Communities)
暂未选择社区。当前以仓库练习及自动校验反馈为主；用户尚未表达社区参与偏好。

## Gaps
归约与 SGEMM 的实测及证据边界分别保存在各自 docs/性能指标汇总.md。SGEMM 三版均已实现并完成 2026-10-19 的数值、限定样例竞争/同步检查及统一性能采集；第三版不能隔离纯寄存器复用贡献，L2 精确命中率、严格同频对照及逐 PC 等待归因仍缺可靠证据。报告生成不等于用户已掌握这些分析，后续以其阅读反馈安排迁移练习。
