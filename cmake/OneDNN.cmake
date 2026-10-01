# Pinned static CPU convolutions; shares the OpenBLAS workers through threadpool interop.
include(ExternalProject)
set(ROKOKO_ONEDNN_URL "https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v3.10.2.tar.gz" CACHE STRING "Pinned oneDNN archive URL (or local archive for offline builds)")
set(ONEDNN_PREFIX "${CMAKE_BINARY_DIR}/onednn")
set(ONEDNN_SOURCE "${ONEDNN_PREFIX}/src/onednn_build")
set(ONEDNN_BINARY "${ONEDNN_PREFIX}/src/onednn_build-build")
set(ONEDNN_ARCHIVE "${ROKOKO_ONEDNN_URL}")
if(ROKOKO_OFFLINE AND ONEDNN_ARCHIVE MATCHES "^https?://")
    if(EXISTS "${ONEDNN_PREFIX}/src/oneDNN-3.10.2.tar.gz")
        set(ONEDNN_ARCHIVE "${ONEDNN_PREFIX}/src/oneDNN-3.10.2.tar.gz")
    else()
        message(FATAL_ERROR "Offline CPU build needs -DROKOKO_ONEDNN_URL=/path/to/oneDNN-3.10.2.tar.gz")
    endif()
endif()
ExternalProject_Add(onednn_build
    PREFIX "${ONEDNN_PREFIX}"
    URL "${ONEDNN_ARCHIVE}"
    DOWNLOAD_NAME oneDNN-3.10.2.tar.gz
    URL_HASH SHA256=58a7399c86789bf3756117072ed946d764ba59dd1480f0e42efd4f9b6b7b9a64
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DDNNL_LIBRARY_TYPE=STATIC -DDNNL_CPU_RUNTIME=THREADPOOL -DDNNL_GPU_RUNTIME=NONE
        -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_GRAPH=OFF
        -DDNNL_ENABLE_WORKLOAD=INFERENCE -DDNNL_ENABLE_PRIMITIVE=CONVOLUTION,REORDER
        -DDNNL_ENABLE_PRIMITIVE_CPU_ISA=AVX2 -DONEDNN_ENABLE_GEMM_KERNELS_ISA=AVX2
        -DDNNL_ENABLE_JIT_PROFILING=OFF -DDNNL_ENABLE_ITT_TASKS=OFF
    LIST_SEPARATOR ,
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel 4
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${ONEDNN_BINARY}/src/libdnnl.a")
add_library(rokoko_onednn STATIC IMPORTED GLOBAL)
set_target_properties(rokoko_onednn PROPERTIES IMPORTED_LOCATION "${ONEDNN_BINARY}/src/libdnnl.a")
add_dependencies(rokoko_onednn onednn_build)
