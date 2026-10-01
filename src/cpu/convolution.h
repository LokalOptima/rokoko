#pragma once
#include "device.h"
#include <cstddef>

namespace rokoko::cpu {
// Float activations are rounded to FP16, matching the shared inference contract,
// then consumed as FP32 by the CPU convolution. source_ci excludes channel padding.
void convolution(const float *x, const Half *weights, const float *bias, float *y,
                 const float *residual, int ci, int co, int ti, int kernel, int stride,
                 int pad, int dilation, int source_ci);
void clear_convolutions();

struct ConvolutionCacheStats {
    size_t weights = 0;
    size_t primitives = 0;
    size_t input_capacity = 0;
};
ConvolutionCacheStats convolution_cache_stats();
} // namespace rokoko::cpu
