// OpenBLAS supplies AVX2/FMA GEMM; F16C unpacks the bundled half weights.
// Caches contain weights only and are released with the single live context.
#include "math.h"
#include "backend_ops.h"
#include <cblas.h>
#include <immintrin.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
namespace rokoko::cpu {
static std::unordered_map<const Half *, std::vector<float>> unpacked;
void configure() {
    int threads = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    if (const char *value = std::getenv("ROKOKO_CPU_THREADS")) {
        char *end = nullptr;
        long n = std::strtol(value, &end, 10);
        if (!*value || *end || n < 1 || n > 64)
            throw std::invalid_argument("ROKOKO_CPU_THREADS must be in 1..64");
        threads = int(n);
    }
    openblas_set_num_threads(threads);
}
void to_float(const Half *x, float *y, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(
            y + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i))));
    for (; i < n; ++i)
        y[i] = float(x[i]);
}
void to_half(const float *x, Half *y, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128(
            reinterpret_cast<__m128i *>(y + i),
            _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    for (; i < n; ++i)
        y[i] = Half(x[i]);
}
const float *unpack_weight(const Half *p, size_t n) {
    auto [it, inserted] = unpacked.try_emplace(p);
    if (inserted) {
        it->second.resize(n);
        to_float(p, it->second.data(), n);
    }
    if (it->second.size() != n)
        throw std::logic_error("CPU weight view changed size");
    return it->second.data();
}
void clear_weights() { unpacked.clear(); }
} // namespace rokoko::cpu
using rokoko::Half;
using rokoko::Stream;
// All matrix interfaces use BLAS column-major conventions, as the CUDA backend does.
#define GEMM_ARGS                                                                                  \
    int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc,      \
        float alpha, float beta, float *, size_t, Stream
extern "C" int backend_gemm_tn(GEMM_ARGS) {
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C,
                ldc);
    return 0;
}
extern "C" int backend_gemm_nn(GEMM_ARGS) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C,
                ldc);
    return 0;
}
extern "C" int backend_gemm_nt(GEMM_ARGS) {
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, C,
                ldc);
    return 0;
}
#undef GEMM_ARGS
#define BATCH_ARGS                                                                                 \
    int M, int N, int K, const float *A, int lda, long long sa, const float *B, int ldb,           \
        long long sb, float *C, int ldc, long long sc, int batches, float alpha, float beta,       \
        float *, size_t, Stream
extern "C" int backend_gemm_batched_tn(BATCH_ARGS) {
    for (int b = 0; b < batches; ++b)
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A + b * sa, lda,
                    B + b * sb, ldb, beta, C + b * sc, ldc);
    return 0;
}
extern "C" int backend_gemm_batched_nn(BATCH_ARGS) {
    for (int b = 0; b < batches; ++b)
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A + b * sa, lda,
                    B + b * sb, ldb, beta, C + b * sc, ldc);
    return 0;
}
#undef BATCH_ARGS
static int half_gemm(bool transpose, int M, int N, int K, const Half *A, int lda, const Half *B,
                     int ldb, float *C, int ldc, float alpha, float beta) {
    const float *weights = rokoko::cpu::unpack_weight(A, size_t(lda) * (transpose ? M : K));
    // Only unpack the active extent: leading dimensions may include padding.
    std::vector<float> activation(size_t(ldb) * (N - 1) + K);
    rokoko::cpu::to_float(B, activation.data(), activation.size());
    cblas_sgemm(CblasColMajor, transpose ? CblasTrans : CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                weights, lda, activation.data(), ldb, beta, C, ldc);
    return 0;
}
#define HALF_ARGS                                                                                  \
    int M, int N, int K, const Half *A, int lda, const Half *B, int ldb, float *C, int ldc,        \
        float alpha, float beta, float *, size_t, Stream
extern "C" int backend_gemm_tn_f16(HALF_ARGS) {
    return half_gemm(true, M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
}
extern "C" int backend_gemm_nn_f16(HALF_ARGS) {
    return half_gemm(false, M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
}
#undef HALF_ARGS
extern "C" int backend_gemm_tn_bias_f16(int M, int N, int K, const Half *A, int lda, const Half *B,
                                        int ldb, float *C, int ldc, const float *bias, float *,
                                        size_t, Stream) {
    half_gemm(true, M, N, K, A, lda, B, ldb, C, ldc, 1, 0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i)
            C[j * ldc + i] += bias[i];
    return 0;
}
extern "C" int backend_conv1d_fprop_f16(const Half *x, const Half *w, const float *bias, float *y,
                                        const float *residual, float *, size_t, int ci, int co,
                                        int ti, int kernel, int stride, int pad, int dilation,
                                        Stream) {
    int to = (ti + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1, k = ci * kernel;
    const float *weights = rokoko::cpu::unpack_weight(w, size_t(co) * k);
    // Bounded im2col tiles avoid allocating an utterance-sized convolution buffer.
    constexpr int tile = 128;
    std::vector<float> col(size_t(tile) * k);
    for (int start = 0; start < to; start += tile) {
        int count = std::min(tile, to - start);
        for (int t = 0; t < count; ++t)
            for (int j = 0; j < kernel; ++j) {
                int source = (start + t) * stride - pad + j * dilation;
                float *dst = col.data() + size_t(t) * k + j * ci;
                if (source >= 0 && source < ti)
                    rokoko::cpu::to_float(x + size_t(source) * ci, dst, ci);
                else
                    std::fill_n(dst, ci, 0.f);
            }
        float *out = y + size_t(start) * co;
        if (residual && residual != y)
            std::copy_n(residual + size_t(start) * co, size_t(count) * co, out);
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, co, count, k, 1, weights, k,
                    col.data(), k, residual ? 1 : 0, out, co);
        if (bias)
            for (int t = 0; t < count; ++t)
                for (int c = 0; c < co; ++c)
                    out[t * co + c] += bias[c];
    }
    return 0;
}
extern "C" void clear_backend_gemm_cache() { rokoko::cpu::clear_weights(); }
extern "C" void clear_backend_gemm_f16_cache() {}
extern "C" void clear_backend_conv_f16_cache() {}
