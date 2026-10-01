// OpenBLAS supplies AVX2/FMA GEMM; F16C unpacks the bundled half weights.
// Weight caches and reusable staging are released with the single live context.
#include "math.h"
#include "convolution.h"
#include "parallel.h"
#include "backend_ops.h"
#include <cblas.h>
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
// This synchronous pthread-compatible entry point belongs to our pinned static
// OpenBLAS 0.3.30 build. Keep this adapter isolated; no private queue layouts or
// extra worker pool are needed. Revalidate it when upgrading OpenBLAS.
extern "C" int gotoblas_pthread(int, void *, void *, int);
namespace rokoko::cpu {
static std::unordered_map<const Half *, std::vector<float>> unpacked;
static std::vector<float> activation;
static thread_local bool working = false;
bool in_parallel() { return working; }
void parallel_for(size_t count, size_t grain,
                  void (*function)(void *, size_t, size_t), void *data) {
    if (!grain)
        throw std::invalid_argument("CPU parallel grain must be positive");
    size_t threads = std::min(size_t(openblas_get_num_threads()), count / grain);
    if (threads < 2 || working) {
        function(data, 0, count);
        return;
    }
    threads = std::min(threads, size_t(64));
    struct Task {
        void (*function)(void *, size_t, size_t);
        void *data;
        size_t begin, end;
        std::exception_ptr error;
    };
    std::array<Task, 64> tasks{};
    size_t offset = 0;
    for (size_t i = 0; i < threads; ++i) {
        size_t end = offset + count / threads + (i < count % threads);
        tasks[i] = {function, data, offset, end, {}};
        offset = end;
    }
    auto run = +[](void *p) {
        auto &task = *static_cast<Task *>(p);
        working = true;
        try {
            task.function(task.data, task.begin, task.end);
        } catch (...) {
            task.error = std::current_exception();
        }
        working = false;
    };
    gotoblas_pthread(int(threads), reinterpret_cast<void *>(run), tasks.data(), sizeof(Task));
    for (size_t i = 0; i < threads; ++i)
        if (tasks[i].error)
            std::rethrow_exception(tasks[i].error);
}
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
    parallel(n, 32768, [&](size_t begin, size_t end) {
        size_t i = begin;
        for (; i + 8 <= end; i += 8)
            _mm256_storeu_ps(
                y + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i))));
        for (; i < end; ++i)
            y[i] = float(x[i]);
    });
}
void to_half(const float *x, Half *y, size_t n) {
    parallel(n, 32768, [&](size_t begin, size_t end) {
        size_t i = begin;
        for (; i + 8 <= end; i += 8)
            _mm_storeu_si128(
                reinterpret_cast<__m128i *>(y + i),
                _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        for (; i < end; ++i)
            y[i] = Half(x[i]);
    });
}
void round_to_half(const float *x, float *y, size_t n) {
    parallel(n, 32768, [&](size_t begin, size_t end) {
        size_t i = begin;
        for (; i + 8 <= end; i += 8) {
            __m128i half = _mm256_cvtps_ph(_mm256_loadu_ps(x + i),
                                         _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm256_storeu_ps(y + i, _mm256_cvtph_ps(half));
        }
        for (; i < end; ++i)
            y[i] = float(Half(x[i]));
    });
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
void gemm(bool transpose, int M, int N, int K, const Half *a, int lda,
          const float *b, int ldb, float *c, int ldc, float alpha, float beta,
          const float *bias) {
    const float *w = unpack_weight(a, size_t(lda) * (transpose ? M : K));
    activation.resize(size_t(ldb) * (N - 1) + K);
    round_to_half(b, activation.data(), activation.size());
    cblas_sgemm(CblasColMajor, transpose ? CblasTrans : CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, w, lda, activation.data(), ldb, beta, c, ldc);
    if (bias)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i)
                c[j * ldc + i] += bias[i];
}
void clear_weights() {
    clear_convolutions();
    unpacked.clear();
    std::vector<float>().swap(activation);
}
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
    auto &activation = rokoko::cpu::activation;
    activation.resize(size_t(ldb) * (N - 1) + K);
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
    std::vector<float> input(size_t(ti) * ci);
    rokoko::cpu::to_float(x, input.data(), input.size());
    rokoko::cpu::convolution(input.data(), w, bias, y, residual, ci, co, ti,
                             kernel, stride, pad, dilation, ci);
    return 0;
}
extern "C" void clear_backend_gemm_cache() { rokoko::cpu::clear_weights(); }
extern "C" void clear_backend_gemm_f16_cache() {}
extern "C" void clear_backend_conv_f16_cache() {}
