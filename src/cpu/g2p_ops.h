// CPU counterparts of the small G2P kernels. Matrix products use OpenBLAS.
#pragma once
#include "kernels.h"
#include <algorithm>
#include <cmath>
#include <limits>
inline void g2p_embed_kernel(const int *ids, const float *emb, float *out, int T, int d) {
    rokoko::embedding_gather(emb, ids, out, T, d, nullptr);
}
inline void g2p_rms_norm_kernel(const float *x, const float *w, float *out, int T, int d,
                                float eps) {
    for (int t = 0; t < T; ++t) {
        double sum = 0;
        for (int i = 0; i < d; ++i)
            sum += double(x[t * d + i]) * x[t * d + i];
        float inv = 1 / std::sqrt(float(sum / d) + eps);
        for (int i = 0; i < d; ++i)
            out[t * d + i] = w[i] * x[t * d + i] * inv;
    }
}
inline void g2p_bias_kernel(float *x, const float *b, int d, int T) {
    rokoko::bias_add_f32(x, b, x, T, d, nullptr);
}
inline void g2p_qkv_bias_rope_kernel(float *x, const float *b, const float *rc, const float *rs,
                                     int T, int d3, int d, int heads, int dk) {
    g2p_bias_kernel(x, b, d3, T);
    for (int t = 0; t < T; ++t)
        for (int qk = 0; qk < 2; ++qk)
            for (int h = 0; h < heads; ++h) {
                float *p = x + t * d3 + qk * d + h * dk;
                for (int i = 0; i < dk / 2; ++i) {
                    float a = p[i], z = p[i + dk / 2], c = rc[t * (dk / 2) + i],
                          s = rs[t * (dk / 2) + i];
                    p[i] = a * c - z * s;
                    p[i + dk / 2] = z * c + a * s;
                }
            }
}
inline void g2p_softmax_kernel(float *x, int d, int T) { rokoko::softmax_f32(x, x, T, d, nullptr); }
inline void g2p_bias_rms_norm_kernel(float *x, const float *b, const float *w, float *y, int T,
                                     int d, float eps) {
    g2p_bias_kernel(x, b, d, T);
    g2p_rms_norm_kernel(x, w, y, T, d, eps);
}
inline void g2p_swiglu_bias_kernel(float *x, const float *b, int ff, int T) {
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < ff; ++i) {
            float g = x[t * 2 * ff + i] + b[i], u = x[t * 2 * ff + ff + i] + b[ff + i];
            x[t * 2 * ff + i] = (g / (1 + std::exp(-g))) * u;
        }
}
inline void g2p_upsample_bias_reshape_kernel(const float *x, const float *b, float *y, int T, int d,
                                             int up) {
    rokoko::bias_add_f32(x, b, y, T, d * up, nullptr);
}
inline void g2p_bias_ctc_argmax_kernel(const float *x, const float *b, int *out, int T, int n) {
    for (int t = 0; t < T; ++t) {
        int best = 0;
        float value = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; ++i)
            if (x[t * n + i] + b[i] > value) {
                value = x[t * n + i] + b[i];
                best = i;
            }
        out[t] = best;
    }
}
