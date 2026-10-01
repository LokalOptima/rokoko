# A pinned static BLAS keeps the CPU executable free of a libopenblas runtime dependency.
include(ExternalProject)
find_program(ROKOKO_MAKE_EXECUTABLE NAMES gmake make REQUIRED)
set(ROKOKO_OPENBLAS_URL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30.tar.gz" CACHE STRING "Pinned OpenBLAS archive URL (or local archive for offline builds)")
set(OPENBLAS_PREFIX "${CMAKE_BINARY_DIR}/openblas")
set(OPENBLAS_SOURCE "${OPENBLAS_PREFIX}/src/openblas_build")
set(OPENBLAS_ARCHIVE "${ROKOKO_OPENBLAS_URL}")
if(ROKOKO_OFFLINE AND OPENBLAS_ARCHIVE MATCHES "^https?://")
    if(EXISTS "${OPENBLAS_PREFIX}/src/OpenBLAS-0.3.30.tar.gz")
        set(OPENBLAS_ARCHIVE "${OPENBLAS_PREFIX}/src/OpenBLAS-0.3.30.tar.gz")
    else()
        message(FATAL_ERROR "Offline CPU build needs -DROKOKO_OPENBLAS_URL=/path/to/OpenBLAS-0.3.30.tar.gz")
    endif()
endif()
ExternalProject_Add(openblas_build
    PREFIX "${OPENBLAS_PREFIX}"
    URL "${OPENBLAS_ARCHIVE}"
    URL_HASH SHA256=27342cff518646afb4c2b976d809102e368957974c250a25ccc965e53063c95d
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE TRUE
    BUILD_COMMAND ${ROKOKO_MAKE_EXECUTABLE} -j4 libs netlib
        CC=${CMAKE_C_COMPILER} TARGET=HASWELL NOFORTRAN=1 NO_LAPACK=1
        NO_SHARED=1 NO_AFFINITY=1 USE_THREAD=1 NUM_THREADS=64 MAKE_NB_JOBS=4
        LIBNAMESUFFIX=rokoko
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${OPENBLAS_SOURCE}/libopenblasrokoko_haswellp-r0.3.30.a")
add_library(rokoko_openblas STATIC IMPORTED GLOBAL)
set_target_properties(rokoko_openblas PROPERTIES IMPORTED_LOCATION "${OPENBLAS_SOURCE}/libopenblasrokoko_haswellp-r0.3.30.a")
add_dependencies(rokoko_openblas openblas_build)
