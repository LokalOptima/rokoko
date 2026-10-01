#pragma once
#include "device.h"
#include <cstddef>
namespace rokoko::cpu {
void configure();
const float *unpack_weight(const Half *data, size_t count);
void to_float(const Half *data, float *out, size_t count);
void to_half(const float *data, Half *out, size_t count);
void clear_weights();
} // namespace rokoko::cpu
