cmake_minimum_required(VERSION 3.24)

# Set the default before project() enables CUDA and populates a compiler default.
# Respect explicit toolchain/cache settings and CUDAARCHS for cross compilation.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()
project(rokoko LANGUAGES C CXX ASM)
set(ROKOKO_BACKEND "CUDA" CACHE STRING "Inference backend: CUDA or CPU")
set_property(CACHE ROKOKO_BACKEND PROPERTY STRINGS CUDA CPU)
if(ROKOKO_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
elseif(NOT ROKOKO_BACKEND STREQUAL "CPU")
    message(FATAL_ERROR "ROKOKO_BACKEND must be CUDA or CPU")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

option(ROKOKO_BUILD_LIB_ONLY "Only build the library, skip the executable" OFF)

# --- Library sources ---
set(LIB_SOURCES
    src/rokoko.cpp
    src/weights.cpp
    src/embedded.cpp
    src/verbose.cpp
)
if(ROKOKO_BACKEND STREQUAL "CPU")
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        message(FATAL_ERROR "The CPU backend currently requires x86-64 with AVX2, FMA and F16C")
    endif()
    list(APPEND LIB_SOURCES src/cpu/kernels.cpp src/cpu/math.cpp)
else()
    list(APPEND LIB_SOURCES src/kernels.cu src/cutlass_gemm.cu src/cutlass_gemm_f16.cu src/cutlass_conv_f16.cu)
endif()

# Download at build time, never at configure time or runtime. Both build systems
# bundle the same FP16 assets using one manifest and preparer.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ROKOKO_ASSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/assets" CACHE PATH "Verified build asset directory")
set(ROKOKO_ASSET_SOURCE "" CACHE PATH "Optional local directory of approved model assets")
option(ROKOKO_OFFLINE "Require local assets/dependencies; prohibit build downloads" OFF)
if(ROKOKO_BACKEND STREQUAL "CPU")
    include(cmake/OpenBLAS.cmake)
endif()
set(ASSET_ARGS --directory "${ROKOKO_ASSET_DIR}")
if(ROKOKO_ASSET_SOURCE)
    list(APPEND ASSET_ARGS --source "${ROKOKO_ASSET_SOURCE}")
endif()
if(ROKOKO_OFFLINE)
    list(APPEND ASSET_ARGS --offline)
endif()
set(ASSET_ASM "${ROKOKO_ASSET_DIR}/embedded.S")
add_custom_target(rokoko_assets
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_assets.py" ${ASSET_ARGS}
    BYPRODUCTS "${ASSET_ASM}"
    VERBATIM)
set_source_files_properties("${ASSET_ASM}" PROPERTIES GENERATED TRUE
    OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/assets/manifest.json;${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_assets.py")
list(APPEND LIB_SOURCES "${ASSET_ASM}")

add_library(rokoko_lib STATIC ${LIB_SOURCES})
add_dependencies(rokoko_lib rokoko_assets)

target_include_directories(rokoko_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_include_directories(rokoko_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
)

target_compile_options(rokoko_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-mavx2;-mfma>
)

if(ROKOKO_BACKEND STREQUAL "CPU")
    target_compile_definitions(rokoko_lib PUBLIC ROKOKO_CPU=1)
    target_compile_options(rokoko_lib PRIVATE -mf16c -fno-math-errno)
    target_include_directories(rokoko_lib PRIVATE "${OPENBLAS_SOURCE}")
    target_link_libraries(rokoko_lib PRIVATE rokoko_openblas m)
    add_dependencies(rokoko_lib openblas_build)
    set_source_files_properties(src/main.cu tests/runtime.cu tests/frontend/g2p_check.cu PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++")
else()
    target_link_libraries(rokoko_lib PUBLIC CUDA::cudart)
endif()
target_link_libraries(rokoko_lib PRIVATE pthread)

# --- Executable (optional) ---
if(NOT ROKOKO_BUILD_LIB_ONLY)
    add_executable(rokoko src/main.cu)
    target_link_libraries(rokoko rokoko_lib)
    target_include_directories(rokoko PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp-httplib
    )
endif()

option(ROKOKO_BUILD_DIAGNOSTICS "Build inference comparison tools" OFF)
if(ROKOKO_BUILD_DIAGNOSTICS)
    add_executable(rokoko_runtime tests/runtime.cu)
    target_link_libraries(rokoko_runtime rokoko_lib)
endif()
if(ROKOKO_BUILD_DIAGNOSTICS AND ROKOKO_BACKEND STREQUAL "CPU")
    add_executable(rokoko_g2p_check tests/frontend/g2p_check.cu)
    target_link_libraries(rokoko_g2p_check rokoko_lib)
    add_executable(rokoko_cpu_ops tests/cpu_ops.cpp)
    target_link_libraries(rokoko_cpu_ops rokoko_lib)
    enable_testing()
    add_test(NAME cpu_operators COMMAND rokoko_cpu_ops)
endif()
