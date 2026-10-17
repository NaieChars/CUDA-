add_executable(reduction src/main.cu src/kernels.cu)
target_link_libraries(reduction PRIVATE cuda_common)
set_target_properties(reduction PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# Keep source locations available when examining the kernels in Nsight Compute.
target_compile_options(reduction PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
