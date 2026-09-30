cmake_minimum_required(VERSION 3.24)
project(rokoko LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

option(ROKOKO_FP16 "Build FP16 variant (pre-baked FP16 weights)" ON)
option(ROKOKO_BUILD_LIB_ONLY "Only build the library, skip the executable" OFF)

# --- Library sources ---
set(LIB_SOURCES
    src/weights.cpp
    src/verbose.cpp
    src/kernels.cu
    src/cutlass_conv.cu
    src/cutlass_gemm.cu
    src/cutlass_gemm_f16.cu
    src/cutlass_conv_f16.cu
)

if(ROKOKO_FP16)
    list(APPEND LIB_SOURCES src/rokoko_f16.cpp)
else()
    list(APPEND LIB_SOURCES src/rokoko.cpp)
endif()

add_library(rokoko_lib STATIC ${LIB_SOURCES})

target_include_directories(rokoko_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(rokoko_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
)

target_compile_options(rokoko_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
)

target_link_libraries(rokoko_lib PUBLIC CUDA::cudart)
target_link_libraries(rokoko_lib PRIVATE pthread)

# --- Executable (optional) ---
if(NOT ROKOKO_BUILD_LIB_ONLY)
    add_executable(rokoko src/main.cu)
    target_link_libraries(rokoko rokoko_lib)
    target_include_directories(rokoko PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp-httplib
    )
endif()
