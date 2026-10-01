// Small allocation/copy interface shared by the CUDA and CPU forward passes.
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>

#ifndef ROKOKO_CPU
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace rokoko {
#ifdef ROKOKO_CPU
using Half = _Float16;
using Stream = void *;
#else
using Half = __half;
using Stream = cudaStream_t;
#endif
namespace device {
#ifdef ROKOKO_CPU
using Status = int;
constexpr Status success = 0;
enum CopyKind { host_to_device, device_to_host, device_to_device };
template <class T> Status allocate(T **p, size_t n) {
    void *memory = nullptr;
    int status = posix_memalign(&memory, 256, n ? n : 256);
    *p = static_cast<T *>(memory);
    return status;
}
inline Status release(void *p) {
    std::free(p);
    return success;
}
inline Status copy(void *dst, const void *src, size_t n, CopyKind) {
    std::memcpy(dst, src, n);
    return success;
}
inline Status copy_async(void *dst, const void *src, size_t n, CopyKind kind, Stream) {
    return copy(dst, src, n, kind);
}
inline Status copy_2d(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width,
                      size_t height, CopyKind, Stream) {
    for (size_t i = 0; i < height; ++i)
        std::memcpy(static_cast<char *>(dst) + i * dpitch,
                    static_cast<const char *>(src) + i * spitch, width);
    return success;
}
inline Status zero(void *dst, int value, size_t n, Stream) {
    std::memset(dst, value, n);
    return success;
}
inline Status create_stream(Stream *s) {
    *s = nullptr;
    return success;
}
inline Status destroy_stream(Stream) { return success; }
inline Status synchronize(Stream) { return success; }
inline const char *error_string(Status s) { return std::strerror(s); }
#else
using Status = cudaError_t;
constexpr Status success = cudaSuccess;
using CopyKind = cudaMemcpyKind;
constexpr auto host_to_device = cudaMemcpyHostToDevice;
constexpr auto device_to_host = cudaMemcpyDeviceToHost;
constexpr auto device_to_device = cudaMemcpyDeviceToDevice;
template <class T> Status allocate(T **p, size_t n) { return cudaMalloc(p, n); }
inline Status release(void *p) { return cudaFree(p); }
inline Status copy(void *d, const void *s, size_t n, CopyKind k) { return cudaMemcpy(d, s, n, k); }
inline Status copy_async(void *d, const void *s, size_t n, CopyKind k, Stream st) {
    return cudaMemcpyAsync(d, s, n, k, st);
}
inline Status copy_2d(void *d, size_t dp, const void *s, size_t sp, size_t w, size_t h, CopyKind k,
                      Stream st) {
    return cudaMemcpy2DAsync(d, dp, s, sp, w, h, k, st);
}
inline Status zero(void *d, int v, size_t n, Stream st) { return cudaMemsetAsync(d, v, n, st); }
inline Status create_stream(Stream *s) { return cudaStreamCreate(s); }
inline Status destroy_stream(Stream s) { return cudaStreamDestroy(s); }
inline Status synchronize(Stream s) { return cudaStreamSynchronize(s); }
inline const char *error_string(Status s) { return cudaGetErrorString(s); }
#endif
} // namespace device
} // namespace rokoko
