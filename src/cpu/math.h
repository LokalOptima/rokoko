#pragma once
#include "device.h"
#include <cstddef>
namespace rokoko::cpu {
void configure();
const float *unpack_weight(const Half *data, size_t count);
void to_float(const Half *data, float *out, size_t count);
void to_half(const float *data, Half *out, size_t count);
void round_to_half(const float *data, float *out, size_t count);
void gemm(bool transpose, int M, int N, int K, const Half *weights, int lda,
          const float *input, int ldb, float *output, int ldc, float alpha, float beta,
          const float *bias = nullptr);
void clear_weights();
} // namespace rokoko::cpu
