cmake_minimum_required(VERSION 3.24)

project(cuda_labs
    LANGUAGES CXX CUDA
)

# GPU architecture
set(CMAKE_CUDA_ARCHITECTURES native)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# MSVC runtime: Release/RelWithDebInfo -> /MD, Debug -> /MDd
set(CMAKE_MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)

# Source / execution encoding
if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
    )
endif()

add_subdirectory(common)
add_subdirectory(labs)