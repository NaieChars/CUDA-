add_executable(day04_coalescing src/main.cu src/kernels.cu)
target_link_libraries(day04_coalescing PRIVATE cuda_common)