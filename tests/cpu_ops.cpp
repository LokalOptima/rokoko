// Independent scalar/double oracles for SIMD layout, padding and reduction contracts.
#include "backend_ops.h"
#include "kernels.h"
#include "cpu/math.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace rokoko;
static void require(bool v, const char *msg) {
    if (!v)
        throw std::runtime_error(msg);
}
static void close(float a, double b, double tol = 2e-5) {
    require(std::isfinite(a) && std::abs(a - b) <= tol * (1 + std::abs(b)), "numeric mismatch");
}
static float sample(int i) { return float((i * 37 % 101) - 50) / 53; }
static void conversions() {
    std::vector<Half> half(65536), roundtrip(65536);
    std::vector<float> actual(65536);
    for (unsigned i = 0; i < 65536; ++i) {
        uint16_t bits = i;
        std::memcpy(&half[i], &bits, 2);
    }
    cpu::to_float(half.data(), actual.data(), half.size());
    cpu::to_half(actual.data(), roundtrip.data(), half.size());
    for (unsigned i = 0; i < 65536; ++i) {
        int exponent = (i >> 10) & 31, mantissa = i & 1023;
        double sign = (i & 32768) ? -1 : 1;
        if (exponent == 31) {
            require(mantissa ? std::isnan(actual[i]) : std::isinf(actual[i]), "half special value");
            continue;
        }
        double expected = sign * std::ldexp(exponent ? 1024 + mantissa : mantissa,
                                            exponent ? exponent - 25 : -24);
        require(double(actual[i]) == expected, "half unpack");
        require(std::memcmp(&half[i], &roundtrip[i], 2) == 0, "half roundtrip");
    }
    // Tail sizes, unaligned addresses, and exact half ties (round-to-nearest-even).
    for (int n = 1; n < 24; ++n) {
        std::vector<Half> h(n + 2);
        std::vector<float> x(n + 2, 1.00048828125f), y(n + 2);
        cpu::to_half(x.data() + 1, h.data() + 1, n);
        cpu::to_float(h.data() + 1, y.data() + 1, n);
        for (int i = 1; i <= n; ++i)
            require(y[i] == 1.f, "F16C tail tie");
    }
}
static void gemms() {
    constexpr int M = 13, N = 9, K = 7;
    for (int mode = 0; mode < 3; ++mode)
        for (float beta : {0.f, .7f}) {
            bool ta = mode == 0, tb = mode == 2;
            int lda = (ta ? K : M) + 3, ldb = (tb ? N : K) + 2, ldc = M + 4;
            std::vector<float> a(lda * (ta ? M : K)), b(ldb * (tb ? K : N)), c(ldc * N, -999), old;
            for (size_t i = 0; i < a.size(); ++i)
                a[i] = sample(i);
            for (size_t i = 0; i < b.size(); ++i)
                b[i] = sample(i + 13);
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < M; ++i)
                    c[j * ldc + i] = beta ? sample(i + j) : std::numeric_limits<float>::quiet_NaN();
            old = c;
            auto fn = mode == 0 ? backend_gemm_tn : mode == 1 ? backend_gemm_nn : backend_gemm_nt;
            fn(M, N, K, a.data(), lda, b.data(), ldb, c.data(), ldc, .8f, beta, nullptr, 0,
               nullptr);
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < M; ++i) {
                    double sum = 0;
                    for (int k = 0; k < K; ++k)
                        sum += double(a[ta ? i * lda + k : k * lda + i]) *
                               b[tb ? k * ldb + j : j * ldb + k];
                    close(c[j * ldc + i], .8f * sum + (beta ? beta * old[j * ldc + i] : 0));
                }
            for (int j = 0; j < N; ++j)
                for (int i = M; i < ldc; ++i)
                    require(c[j * ldc + i] == -999, "GEMM wrote stride padding");
            if (!tb) {
                std::vector<Half> ah(a.size()), bh(b.size());
                cpu::to_half(a.data(), ah.data(), a.size());
                cpu::to_half(b.data(), bh.data(), b.size());
                cpu::clear_weights();
                auto hfn = ta ? backend_gemm_tn_f16 : backend_gemm_nn_f16;
                hfn(M, N, K, ah.data(), lda, bh.data(), ldb, c.data(), ldc, 1, 0, nullptr, 0,
                    nullptr);
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < M; ++i) {
                        double sum = 0;
                        for (int k = 0; k < K; ++k)
                            sum +=
                                double(ah[ta ? i * lda + k : k * lda + i]) * float(bh[j * ldb + k]);
                        close(c[j * ldc + i], sum);
                    }
                cpu::clear_weights();
            }
        }
    // Interleaved heads exercise strideA/B/C independently from leading dimensions.
    constexpr int heads = 3, dim = 5, T = 7, ld = heads * dim;
    std::vector<float> q(T * ld), k(T * ld), score(heads * T * T), v(T * ld), out(T * ld);
    for (int i = 0; i < T * ld; ++i) {
        q[i] = sample(i);
        k[i] = sample(i + 5);
        v[i] = sample(i + 7);
    }
    backend_gemm_batched_tn(T, T, dim, k.data(), ld, dim, q.data(), ld, dim, score.data(), T, T * T,
                            heads, .4f, 0, nullptr, 0, nullptr);
    backend_gemm_batched_nn(dim, T, T, v.data(), ld, dim, score.data(), T, T * T, out.data(), ld,
                            dim, heads, 1, 0, nullptr, 0, nullptr);
    for (int h = 0; h < heads; ++h)
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < dim; ++d) {
                double sum = 0;
                for (int j = 0; j < T; ++j) {
                    double dot = 0;
                    for (int z = 0; z < dim; ++z)
                        dot += double(q[t * ld + h * dim + z]) * k[j * ld + h * dim + z];
                    sum += .4f * dot * v[j * ld + h * dim + d];
                }
                close(out[t * ld + h * dim + d], sum);
            }
}
static void convolutions() {
    constexpr int ci = 3, co = 5, T = 11;
    for (int K : {1, 3, 5})
        for (int stride : {1, 2})
            for (int dilation : {1, 2}) {
                int pad = K / 2, to = (T + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
                std::vector<Half> x(T * ci), w(co * K * ci);
                std::vector<float> bias(co), y(to * co), res(to * co);
                for (size_t i = 0; i < x.size(); ++i)
                    x[i] = Half(sample(i));
                for (size_t i = 0; i < w.size(); ++i)
                    w[i] = Half(sample(i + 11));
                for (int i = 0; i < co; ++i)
                    bias[i] = sample(i);
                for (size_t i = 0; i < res.size(); ++i)
                    res[i] = sample(i + 9);
                for (int mode = 0; mode < 3; ++mode) {
                    y = res;
                    cpu::clear_weights();
                    backend_conv1d_fprop_f16(x.data(), w.data(), bias.data(), y.data(),
                                             mode == 0   ? nullptr
                                             : mode == 1 ? res.data()
                                                         : y.data(),
                                             nullptr, 0, ci, co, T, K, stride, pad, dilation,
                                             nullptr);
                    for (int t = 0; t < to; ++t)
                        for (int o = 0; o < co; ++o) {
                            double sum = bias[o] + (mode ? res[t * co + o] : 0);
                            for (int k = 0; k < K; ++k) {
                                int p = t * stride - pad + k * dilation;
                                if (p >= 0 && p < T)
                                    for (int c = 0; c < ci; ++c)
                                        sum +=
                                            double(x[p * ci + c]) * float(w[(o * K + k) * ci + c]);
                            }
                            close(y[t * co + o], sum);
                        }
                }
            }
    cpu::clear_weights();
    std::vector<float> x(3 * 4), w(4 * 3), y(6 * 4);
    for (int i = 0; i < 12; ++i) {
        x[i] = sample(i);
        w[i] = sample(i + 4);
    }
    conv_transpose1d_depthwise_f32(x.data(), w.data(), nullptr, y.data(), 4, 3, 3, 2, 1, 1,
                                   nullptr);
    std::vector<double> ref(24, 0);
    for (int t = 0; t < 3; ++t)
        for (int c = 0; c < 4; ++c)
            for (int k = 0; k < 3; ++k) {
                int p = t * 2 - 1 + k;
                if (p >= 0 && p < 6)
                    ref[p * 4 + c] += double(x[t * 4 + c]) * w[c * 3 + k];
            }
    for (int i = 0; i < 24; ++i)
        close(y[i], ref[i]);
}
static void reductions_and_signal() {
    std::vector<float> x = {1000, 1001, 999, -1000, -999, -1001}, y(6), g = {1, 2, 3},
                       b = {.1f, .2f, .3f};
    softmax_f32(x.data(), y.data(), 2, 3, nullptr);
    for (int n = 0; n < 2; ++n) {
        double sum = 0;
        for (int i = 0; i < 3; ++i)
            sum += std::exp(double(x[n * 3 + i]) - x[n * 3]);
        for (int i = 0; i < 3; ++i)
            close(y[n * 3 + i], std::exp(double(x[n * 3 + i]) - x[n * 3]) / sum);
    }
    layer_norm_f32(x.data(), g.data(), b.data(), x.data(), 2, 3, 1e-5f, nullptr);
    for (int i = 0; i < 6; ++i) {
        int diffs[] = {0, 1, -1, 0, 1, -1};
        close(x[i], g[i % 3] * diffs[i] / std::sqrt(2. / 3 + 1e-5) + b[i % 3]);
    }
    float durations[] = {.5f, 1.5f, 2.5f, 3.5f, 0, -1};
    int rounded[6], total;
    round_clamp_durations_f32(durations, rounded, &total, 6, nullptr);
    int expected[] = {1, 2, 2, 4, 1, 1};
    require(std::equal(rounded, rounded + 6, expected) && total == 11,
            "ties-to-even duration rounding");
    std::vector<float> align(66);
    build_alignment_f32(rounded, align.data(), 6, 11, nullptr);
    for (int t = 0; t < 11; ++t) {
        float sum = 0;
        for (int i = 0; i < 6; ++i)
            sum += align[i * 11 + t];
        require(sum == 1, "alignment coverage");
    }
    for (int T : {55, 100, 305}) {
        int frames = T / 5 + 1;
        std::vector<float> signal(T), mag(11 * frames), phase(11 * frames), reconstructed(T);
        for (int i = 0; i < T; ++i)
            signal[i] = sample(i);
        stft_f32(signal.data(), mag.data(), phase.data(), T, 20, 5, nullptr);
        istft_f32(mag.data(), phase.data(), reconstructed.data(), frames, 20, 5, T, nullptr);
        for (int i = 0; i < T; ++i)
            close(reconstructed[i], signal[i], 5e-5);
    }
}
int main() {
    try {
        cpu::configure();
        conversions();
        gemms();
        convolutions();
        reductions_and_signal();
        std::cout << "PASS CPU SIMD conversions, BLAS layouts, convolution padding/residuals, "
                     "normalization and spectral roundtrip\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
