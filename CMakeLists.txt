cmake_minimum_required(VERSION 3.26)

set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")

project(flashinfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()

set(TVM_FFI "~/miniconda3/envs/flashinfer/lib/python3.10/site-packages/tvm_ffi")

file(GLOB_RECURSE FLASHINFER_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/include/*.h"
        "${CMAKE_SOURCE_DIR}/include/*.cuh"
        "${CMAKE_SOURCE_DIR}/csrc/*.cu"
        "${CMAKE_SOURCE_DIR}/csrc/*.cpp"
        "${CMAKE_SOURCE_DIR}/csrc/*.h"
        "${CMAKE_SOURCE_DIR}/csrc/*.cuh"
        "${CMAKE_SOURCE_DIR}/flashinfer/*.inc"
        "${CMAKE_SOURCE_DIR}/flashinfer/*.cu"
)

add_library(flashinfer STATIC ${FLASHINFER_SOURCES}
        csrc/estimate.cu
        csrc/flashinfer_estimate_binding.cu
        include/flashinfer/estimate.cuh
        csrc/flashinfer_topk_binding.cu
        csrc/topk.cu
)

target_include_directories(flashinfer PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include
        ${CMAKE_SOURCE_DIR}/3rdparty/spdlog/include
        ${TVM_FFI}/include
)

set_target_properties(flashinfer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_ARCHITECTURES "native"
        LINKER_LANGUAGE CUDA
)


add_library(3rdparty_libs INTERFACE)

file(GLOB_RECURSE THIRDPARTY_HEADERS
        "${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include/*.h"
        "${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include/*.cuh"
        "${CMAKE_SOURCE_DIR}/3rdparty/spdlog/include/*.h"
)

target_sources(3rdparty_libs INTERFACE ${THIRDPARTY_HEADERS})

target_include_directories(3rdparty_libs INTERFACE
        ${CMAKE_SOURCE_DIR}/3rdparty/cutlass/include
        ${CMAKE_SOURCE_DIR}/3rdparty/spdlog/include
)

target_link_libraries(flashinfer PUBLIC 3rdparty_libs)