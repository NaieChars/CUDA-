add_executable(sgemm src/main.cu src/kernels.cu)
target_link_libraries(sgemm PRIVATE cuda_common)
set_target_properties(sgemm PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
target_compile_options(sgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
