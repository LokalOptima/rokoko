cmake_minimum_required(VERSION 3.24)

# Set the default before project() enables CUDA and populates a compiler default.
# Respect explicit toolchain/cache settings and CUDAARCHS for cross compilation.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()
project(rokoko LANGUAGES C CXX CUDA ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)



find_package(CUDAToolkit REQUIRED)

option(ROKOKO_BUILD_LIB_ONLY "Only build the library, skip the executable" OFF)

# --- Library sources ---
set(LIB_SOURCES
    src/rokoko.cpp
    src/weights.cpp
    src/embedded.cpp
    src/verbose.cpp
    src/kernels.cu
    src/cutlass_gemm.cu
    src/cutlass_gemm_f16.cu
    src/cutlass_conv_f16.cu
)

# Download at build time, never at configure time or runtime. Both build systems
# bundle the same FP16 assets using one manifest and preparer.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ROKOKO_ASSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/assets" CACHE PATH "Verified build asset directory")
set(ROKOKO_ASSET_SOURCE "" CACHE PATH "Optional local directory of approved model assets")
option(ROKOKO_OFFLINE "Require preseeded/local model assets; prohibit build downloads" OFF)
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
