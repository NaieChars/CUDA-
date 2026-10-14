# 创建接口库（header-only），不需要编译 .cu 文件
add_library(cuda_common INTERFACE)
target_include_directories(cuda_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
