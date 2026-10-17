# CUDA 算子开发与优化 Resources

## Knowledge
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

## Wisdom (Communities)
暂未选择社区。当前以仓库练习及自动校验反馈为主；用户尚未表达社区参与偏好。

## Gaps
尚无本仓库 multi_item 版本的实测性能数据，优化收益须由用户完成实现后测量。
