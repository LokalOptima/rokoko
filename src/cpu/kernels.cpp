// CPU signal and recurrent operators. Layout is [time, channels], shared with CUDA.
#include "kernels.h"
#include "cpu/math.h"
#include <cblas.h>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
// glibc's public x86 vector-function ABI (libmvec, linked through libm).
// Explicit calls retain finite-value checks and avoid global fast-math flags.
extern "C" __m256 _ZGVdN8v_sinf(__m256);
namespace rokoko {
static constexpr float pi = 3.14159265358979323846f;
static float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
void cast_f32_to_f16(const float *x, Half *y, int n, Stream) { cpu::to_half(x, y, n); }
void cast_f32_to_f16_pad(const float *x, Half *y, int T, int oldc, int newc, Stream) {
    for (int t = 0; t < T; ++t) {
        cpu::to_half(x + t * oldc, y + t * newc, oldc);
        std::fill(y + t * newc + oldc, y + (t + 1) * newc, Half(0));
    }
}
void gemv_tn_f16(const Half *a, int lda, const float *x, float *y, int M, int K, float alpha,
                 float beta, Stream) {
    const float *w = cpu::unpack_weight(a, size_t(lda) * M);
    cblas_sgemv(CblasColMajor, CblasTrans, K, M, alpha, w, lda, x, 1, beta, y, 1);
}
void gemv_tn_f32(const float *a, int lda, const float *x, float *y, int M, int K, float alpha,
                 float beta, Stream) {
    cblas_sgemv(CblasColMajor, CblasTrans, K, M, alpha, a, lda, x, 1, beta, y, 1);
}
void embedding_gather(const float *table, const int *ids, float *y, int N, int D, Stream) {
    for (int n = 0; n < N; ++n)
        std::copy_n(table + ids[n] * D, D, y + n * D);
}
void add_f32(const float *a, const float *b, float *y, int N, Stream) {
    for (int i = 0; i < N; ++i)
        y[i] = a[i] + b[i];
}
void scale_f32(const float *a, float *y, int N, float scale, Stream) {
    for (int i = 0; i < N; ++i)
        y[i] = a[i] * scale;
}
void layer_norm_f32(const float *x, const float *gamma, const float *beta, float *y, int N, int D,
                    float eps, Stream) {
    for (int n = 0; n < N; ++n) {
        double sum = 0, var = 0;
        const float *row = x + n * D;
        for (int i = 0; i < D; ++i)
            sum += row[i];
        float mean = float(sum / D);
        for (int i = 0; i < D; ++i) {
            double v = row[i] - mean;
            var += v * v;
        }
        float inv = 1 / std::sqrt(float(var / D) + eps);
        for (int i = 0; i < D; ++i)
            y[n * D + i] = gamma[i] * (row[i] - mean) * inv + beta[i];
    }
}
void residual_layer_norm_f32(const float *a, const float *b, const float *g, const float *beta,
                             float *y, int N, int D, float eps, Stream s) {
    add_f32(a, b, y, N * D, s);
    layer_norm_f32(y, g, beta, y, N, D, eps, s);
}
void layer_norm_channels_first_f32(const float *x, const float *g, const float *b, float *y, int C,
                                   int T, float eps, Stream s) {
    layer_norm_f32(x, g, b, y, T, C, eps, s);
}
void ada_layer_norm_f32(const float *x, const float *g, const float *b, float *y, int N, int D,
                        float eps, Stream s) {
    std::vector<float> scale(D);
    for (int i = 0; i < D; ++i)
        scale[i] = 1 + g[i];
    layer_norm_f32(x, scale.data(), b, y, N, D, eps, s);
}
void instance_norm_style_affine_f32(const float *x, const float *nw, const float *nb,
                                    const float *g, const float *b, float *y, float *, int C, int T,
                                    float eps, Stream, const float *snake) {
    // Double precision statistics avoid long-utterance cancellation. No fast-math:
    // nonfinite output remains detectable by the public pipeline.
    std::vector<double> sums(C, 0), squares(C, 0);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < C; ++c) {
            double v = x[t * C + c];
            sums[c] += v;
            squares[c] += v * v;
        }
    std::vector<float> scale(C), bias(C);
    for (int c = 0; c < C; ++c) {
        float mean = float(sums[c] / T),
              var = float(std::max(0., squares[c] / T - (sums[c] / T) * (sums[c] / T)));
        float inv = 1 / std::sqrt(var + eps);
        scale[c] = (1 + g[c]) * nw[c] * inv;
        bias[c] = (1 + g[c]) * (nb[c] - nw[c] * mean * inv) + b[c];
    }
    for (int t = 0; t < T; ++t) {
        int c = 0;
        for (; c + 8 <= C; c += 8) {
            __m256 v =
                _mm256_fmadd_ps(_mm256_loadu_ps(scale.data() + c), _mm256_loadu_ps(x + t * C + c),
                                _mm256_loadu_ps(bias.data() + c));
            if (snake) {
                __m256 a = _mm256_loadu_ps(snake + c);
                __m256 z = _ZGVdN8v_sinf(_mm256_mul_ps(a, v));
                v = _mm256_add_ps(v, _mm256_div_ps(_mm256_mul_ps(z, z), a));
            }
            _mm256_storeu_ps(y + t * C + c, v);
        }
        for (; c < C; ++c) {
            float v = scale[c] * x[t * C + c] + bias[c];
            if (snake) {
                float z = std::sin(snake[c] * v);
                v += z * z / snake[c];
            }
            y[t * C + c] = v;
        }
    }
}
void gelu_f32(const float *x, float *y, int N, Stream) {
    for (int i = 0; i < N; ++i) {
        float v = x[i];
        y[i] = 0.5f * v * (1 + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}
void leaky_relu_f32(const float *x, float *y, int N, float alpha, Stream) {
    for (int i = 0; i < N; ++i)
        y[i] = x[i] > 0 ? x[i] : alpha * x[i];
}
void softmax_f32(const float *x, float *y, int N, int D, Stream) {
    for (int n = 0; n < N; ++n) {
        float m = *std::max_element(x + n * D, x + (n + 1) * D);
        double sum = 0;
        for (int i = 0; i < D; ++i) {
            float v = std::exp(x[n * D + i] - m);
            y[n * D + i] = v;
            sum += v;
        }
        float inv = 1 / float(sum);
        for (int i = 0; i < D; ++i)
            y[n * D + i] *= inv;
    }
}
void bias_add_f32(const float *x, const float *b, float *y, int N, int D, Stream) {
    for (int n = 0; n < N; ++n)
        for (int i = 0; i < D; ++i)
            y[n * D + i] = x[n * D + i] + b[i];
}
void channel_bias_add_f32(float *y, const float *b, int C, int T, Stream s) {
    bias_add_f32(y, b, y, T, C, s);
}
void transpose_f32(const float *x, float *y, int M, int N, Stream) {
    for (int mi = 0; mi < M; mi += 32)
        for (int ni = 0; ni < N; ni += 32)
            for (int i = mi; i < std::min(mi + 32, M); ++i)
                for (int j = ni; j < std::min(ni + 32, N); ++j)
                    y[j * M + i] = x[i * N + j];
}
void conv1d_general_f32(const float *x, const float *w, const float *b, float *y, int ci, int co,
                        int ti, int K, int stride, int pad, int dilation, Stream) {
    int to = (ti + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
    for (int t = 0; t < to; ++t)
        for (int o = 0; o < co; ++o) {
            float sum = 0;
            for (int c = 0; c < ci; ++c)
                for (int k = 0; k < K; ++k) {
                    int j = t * stride - pad + k * dilation;
                    if (j >= 0 && j < ti)
                        sum += w[(o * ci + c) * K + k] * x[j * ci + c];
                }
            y[t * co + o] = sum + (b ? b[o] : 0);
        }
}
void conv1d_f32(const float *x, const float *w, const float *b, float *y, int ci, int co, int T,
                int K, Stream s) {
    conv1d_general_f32(x, w, b, y, ci, co, T, K, 1, (K - 1) / 2, 1, s);
}
void weight_norm_f32(const float *g, const float *v, float *w, int co, int fan, Stream) {
    for (int c = 0; c < co; ++c) {
        double sum = 0;
        for (int i = 0; i < fan; ++i)
            sum += double(v[c * fan + i]) * v[c * fan + i];
        float s = g[c] / std::sqrt(float(sum));
        for (int i = 0; i < fan; ++i)
            w[c * fan + i] = v[c * fan + i] * s;
    }
}
void conv_transpose1d_depthwise_f32(const float *x, const float *w, const float *b, float *y, int C,
                                    int T, int K, int stride, int pad, int outpad, Stream) {
    int to = (T - 1) * stride - 2 * pad + K + outpad;
    for (int t = 0; t < to; ++t)
        for (int c = 0; c < C; ++c) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                int p = t + pad - k;
                if (p >= 0 && p % stride == 0 && p / stride < T)
                    sum += x[(p / stride) * C + c] * w[c * K + k];
            }
            y[t * C + c] = sum + (b ? b[c] : 0);
        }
}
void upsample_nearest_1d_2x_f32(const float *x, float *y, int C, int T, Stream) {
    for (int t = 0; t < 2 * T; ++t)
        std::copy_n(x + (t / 2) * C, C, y + t * C);
}
void sigmoid_sum_f32(const float *x, float *y, int N, int D, Stream) {
    for (int n = 0; n < N; ++n) {
        float sum = 0;
        for (int d = 0; d < D; ++d)
            sum += sigmoid(x[n * D + d]);
        y[n] = sum;
    }
}
void tile_1d_f32(const float *x, float *y, int C, int T, Stream) {
    for (int t = 0; t < T; ++t)
        std::copy_n(x, C, y + t * C);
}
void reflection_pad_1d_f32(const float *x, float *y, int C, int T, int left, int right, Stream) {
    for (int t = 0; t < T + left + right; ++t) {
        int s = t < left ? left - t : (t >= left + T ? 2 * T + left - t - 2 : t - left);
        std::copy_n(x + s * C, C, y + t * C);
    }
}
void exp_f32(const float *x, float *y, int N, Stream) {
    for (int i = 0; i < N; ++i)
        y[i] = std::exp(x[i]);
}
void sin_f32(const float *x, float *y, int N, Stream) {
    for (int i = 0; i < N; ++i)
        y[i] = std::sin(x[i]);
}
// Tiny 20-point transforms use precomputed windowed DFT matrices and SIMD BLAS.
void stft_f32(const float *x, float *mag, float *phase, int T, int fft, int hop, Stream s,
              float *) {
    int frames = T / hop + 1, freq = fft / 2 + 1;
    std::vector<float> padded(T + fft), basis(size_t(2 * freq) * fft),
        windows(size_t(frames) * fft), out(size_t(frames) * 2 * freq);
    reflection_pad_1d_f32(x, padded.data(), 1, T, fft / 2, fft / 2, s);
    for (int k = 0; k < freq; ++k)
        for (int n = 0; n < fft; ++n) {
            float angle = -2 * pi * k * n / fft, win = .5f * (1 - std::cos(2 * pi * n / fft));
            basis[k * fft + n] = win * std::cos(angle);
            basis[(freq + k) * fft + n] = win * std::sin(angle);
        }
    for (int t = 0; t < frames; ++t)
        std::copy_n(padded.data() + t * hop, fft, windows.data() + t * fft);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, 2 * freq, frames, fft, 1, basis.data(),
                fft, windows.data(), fft, 0, out.data(), 2 * freq);
    for (int t = 0; t < frames; ++t)
        for (int k = 0; k < freq; ++k) {
            float r = out[t * 2 * freq + k], i = out[t * 2 * freq + freq + k];
            mag[k * frames + t] = std::sqrt(r * r + i * i);
            phase[k * frames + t] = std::atan2(i, r);
        }
}
void istft_f32(const float *mag, const float *phase, float *y, int frames, int fft, int hop, int T,
               Stream, float *) {
    int freq = fft / 2 + 1, padded = fft + hop * (frames - 1);
    std::vector<float> basis(size_t(2 * freq) * fft), spectrum(size_t(frames) * 2 * freq),
        out(size_t(frames) * fft), signal(padded, 0), ws(padded, 0), window(fft);
    for (int n = 0; n < fft; ++n) {
        window[n] = .5f * (1 - std::cos(2 * pi * n / fft));
        for (int k = 0; k < freq; ++k) {
            float angle = 2 * pi * k * n / fft, scale = (k == 0 || k == freq - 1 ? 1.f : 2.f) / fft;
            basis[k * fft + n] = scale * std::cos(angle);
            basis[(freq + k) * fft + n] = -scale * std::sin(angle);
        }
    }
    for (int t = 0; t < frames; ++t)
        for (int k = 0; k < freq; ++k) {
            int i = k * frames + t;
            spectrum[t * 2 * freq + k] = mag[i] * std::cos(phase[i]);
            spectrum[t * 2 * freq + freq + k] = mag[i] * std::sin(phase[i]);
        }
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, fft, frames, 2 * freq, 1, basis.data(),
                fft, spectrum.data(), 2 * freq, 0, out.data(), fft);
    for (int t = 0; t < frames; ++t)
        for (int n = 0; n < fft; ++n) {
            signal[t * hop + n] += out[t * fft + n] * window[n];
            ws[t * hop + n] += window[n] * window[n];
        }
    for (int t = 0; t < T; ++t) {
        int i = t + fft / 2;
        y[t] = i < padded && ws[i] > 1e-8f ? signal[i] / ws[i] : 0;
    }
}
void lstm_gates_f32(const float *g, const float *prev, float *c, float *h, int H, Stream) {
    for (int i = 0; i < H; ++i) {
        float v = sigmoid(g[H + i]) * prev[i] + sigmoid(g[i]) * std::tanh(g[2 * H + i]);
        c[i] = v;
        h[i] = sigmoid(g[3 * H + i]) * std::tanh(v);
    }
}
void im2col_1d_f32(const float *x, float *col, int ci, int ti, int K, int stride, int pad,
                   int dilation, int to, Stream) {
    for (int t = 0; t < to; ++t)
        for (int c = 0; c < ci; ++c)
            for (int k = 0; k < K; ++k) {
                int s = t * stride - pad + k * dilation;
                col[(t * ci + c) * K + k] = (s >= 0 && s < ti) ? x[s * ci + c] : 0;
            }
}
void col2im_1d_f32(const float *col, float *y, int co, int K, int ti, int stride, int pad, int to,
                   Stream) {
    for (int t = 0; t < ti; ++t)
        for (int c = 0; c < co; ++c)
            for (int k = 0; k < K; ++k) {
                int dst = t * stride - pad + k;
                if (dst >= 0 && dst < to)
                    y[dst * co + c] += col[(t * co + c) * K + k];
            }
}
void concat_channels_f32(const float *a, const float *b, float *y, int T, int ca, int cb, Stream) {
    for (int t = 0; t < T; ++t) {
        std::copy_n(a + t * ca, ca, y + t * (ca + cb));
        std::copy_n(b + t * cb, cb, y + t * (ca + cb) + ca);
    }
}
void concat3_channels_f32(const float *a, const float *b, const float *c, float *y, int T, int ca,
                          int cb, int cc, Stream) {
    int C = ca + cb + cc;
    for (int t = 0; t < T; ++t) {
        std::copy_n(a + t * ca, ca, y + t * C);
        std::copy_n(b + t * cb, cb, y + t * C + ca);
        std::copy_n(c + t * cc, cc, y + t * C + ca + cb);
    }
}
void concat4_channels_f32(const float *a, const float *b, const float *c, const float *d, float *y,
                          int T, int ca, int cb, int cc, int cd, Stream) {
    int C = ca + cb + cc + cd;
    for (int t = 0; t < T; ++t) {
        std::copy_n(a + t * ca, ca, y + t * C);
        std::copy_n(b + t * cb, cb, y + t * C + ca);
        std::copy_n(c + t * cc, cc, y + t * C + ca + cb);
        std::copy_n(d + t * cd, cd, y + t * C + ca + cb + cc);
    }
}
void pad_blocks_f32(const float *x, float *y, int n, int oldsize, int newsize, Stream) {
    for (int i = 0; i < n; ++i) {
        std::copy_n(x + i * oldsize, oldsize, y + i * newsize);
        std::fill(y + i * newsize + oldsize, y + (i + 1) * newsize, 0);
    }
}
void sinegen_phase_f32(const float *f0, const float *random, float *phase, int L, Stream) {
    int samples = L * 300;
    for (int h = 0; h < 9; ++h) {
        for (int t = 0; t < L; ++t) {
            float src = (t + .5f) * samples / float(L) - .5f;
            src = std::clamp(src, 0.f, float(samples - 1));
            int lo = int(src), hi = std::min(lo + 1, samples - 1);
            float frac = src - lo;
            float a = std::fmod(f0[lo / 300] * (h + 1) / 24000.f, 1.f),
                  b = std::fmod(f0[hi / 300] * (h + 1) / 24000.f, 1.f);
            if (a < 0)
                a += 1;
            if (b < 0)
                b += 1;
            if (lo == 0)
                a += random[h];
            if (hi == 0)
                b += random[h];
            phase[t * 9 + h] = (1 - frac) * a + frac * b;
        }
        for (int t = 1; t < L; ++t)
            phase[t * 9 + h] += phase[(t - 1) * 9 + h];
        for (int t = 0; t < L; ++t)
            phase[t * 9 + h] *= 2 * pi * 300;
    }
}
void sinegen_source_f32(const float *phase, const float *f0, const float *w, const float *b,
                        float *y, int L, int T, unsigned seed, Stream) {
    for (int t = 0; t < T; ++t) {
        float src = (t + .5f) * L / float(T) - .5f;
        src = std::clamp(src, 0.f, float(L - 1));
        int lo = int(src), hi = std::min(lo + 1, L - 1);
        float frac = src - lo, uv = f0[t / 300] > 10 ? 1.f : 0.f,
              amp = uv * .003f + (1 - uv) * .1f / 3.f, sum = b[0];
        for (int h = 0; h < 9; ++h) {
            float p = (1 - frac) * phase[lo * 9 + h] + frac * phase[hi * 9 + h];
            unsigned hash = unsigned(t) * 2654435761u + unsigned(h) * 340573321u + seed;
            hash ^= hash >> 16;
            hash *= 0x85ebca6bu;
            hash ^= hash >> 13;
            hash *= 0xc2b2ae35u;
            hash ^= hash >> 16;
            float u1 = ((hash & 0xffffffu) + 1u) / 16777217.f;
            hash = hash * 1664525u + 1013904223u;
            float u2 = (hash & 0xffffffu) / 16777216.f;
            float noise = std::sqrt(-2 * std::log(u1)) * std::cos(2 * pi * u2) * amp;
            sum += (std::sin(p) * .1f * uv + noise) * w[h];
        }
        y[t] = std::tanh(sum);
    }
}
void round_clamp_durations_f32(const float *x, int *durations, int *total, int T, Stream) {
    int sum = 0;
    for (int i = 0; i < T; ++i) {
        durations[i] = std::max(1, int(std::nearbyint(x[i])));
        sum += durations[i];
    }
    *total = sum;
}
void build_alignment_f32(const int *durations, float *y, int T, int L, Stream) {
    std::fill_n(y, size_t(T) * L, 0.f);
    int offset = 0;
    for (int i = 0; i < T; ++i) {
        std::fill_n(y + size_t(i) * L + offset, durations[i], 1.f);
        offset += durations[i];
    }
}
} // namespace rokoko
