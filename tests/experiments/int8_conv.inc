// Experimental vocoder-only quantization. Included after the FP32 convolution
// implementation in a generated translation unit; never in the normal build.
#include "int8.h"
#include <cmath>
#include <limits>
#include <immintrin.h>
namespace rokoko::cpu {
bool int8_generator = false;
namespace {
struct Int8Conv {
    BlasPool pool;
    dnnl::engine engine{dnnl::engine::kind::cpu, 0};
    dnnl::stream stream = dnnl::threadpool_interop::make_stream(engine, &pool);
    struct Plan {
        std::array<int, 3> dimensions{};
        std::optional<Shape> shape;
        dnnl::convolution_forward operation;
        Memory packed, scales;
        Memory::desc src, dst;
        std::vector<int8_t> quantized;
    };
    std::unordered_map<const Half *, Plan> plans;
    std::vector<uint8_t> input;
    Int8Conv() { dnnl::set_primitive_cache_capacity(0); }
    void run(const float *x, const Half *w, const float *bias, float *y,
             const float *residual, int ci, int co, int ti, int kernel,
             int stride, int pad, int dilation, int source_ci) {
        if (ci != source_ci || ci <= 0 || co <= 0 || ti <= 0 || kernel <= 0 ||
            stride <= 0 || dilation <= 0 || pad < 0)
            throw std::invalid_argument("invalid experimental INT8 convolution");
        int to = (ti + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
        if (to <= 0) throw std::invalid_argument("empty INT8 convolution");
        int threads = pool.get_num_threads();
        dnnl::error::wrap_c_api(dnnl_threadpool_interop_set_max_concurrency(threads),
                              "INT8 worker configuration failed");
        auto [it, inserted] = plans.try_emplace(w);
        auto &p = it->second;
        std::array<int, 3> dimensions{co, ci, kernel};
        if (inserted) {
            p.dimensions = dimensions;
            p.scales = Memory(Memory::desc({co}, Memory::data_type::f32,
                                          Memory::format_tag::a), engine);
            auto *sc = static_cast<float *>(p.scales.get_data_handle());
            p.quantized.resize(size_t(co) * ci * kernel);
            for (int c = 0; c < co; ++c) {
                size_t start = size_t(c) * ci * kernel;
                float peak = 0;
                for (int k = 0; k < ci * kernel; ++k) {
                    float v = float(w[start + k]);
                    if (!std::isfinite(v)) throw std::runtime_error("nonfinite INT8 weight");
                    peak = std::max(peak, std::abs(v));
                }
                sc[c] = peak > 0 ? peak / 127.f : 1.f;
                for (int k = 0; k < ci * kernel; ++k)
                    p.quantized[start + k] = int8_t(std::clamp(
                        std::nearbyint(float(w[start + k]) / sc[c]), -127.f, 127.f));
            }
        }
        if (p.dimensions != dimensions) throw std::logic_error("INT8 weight dimensions changed");
        Shape shape{ti, stride, pad, dilation, threads, bool(residual)};
        if (!p.shape || *p.shape != shape) {
            auto src = Memory::desc({1, ci, ti}, Memory::data_type::u8, Memory::format_tag::nwc);
            auto dst = Memory::desc({1, co, to}, Memory::data_type::f32, Memory::format_tag::nwc);
            auto wd = Memory::desc({co, ci, kernel}, Memory::data_type::s8, Memory::format_tag::any);
            dnnl::primitive_attr attr;
            attr.set_scales_mask(DNNL_ARG_SRC, 0);
            attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1);
            attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
            if (residual) { dnnl::post_ops post; post.append_sum(1.f); attr.set_post_ops(post); }
            dnnl::convolution_forward::primitive_desc pd(
                engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
                src, wd, dst, {stride}, {dilation - 1}, {pad}, {pad}, attr);
            Memory packed = p.packed;
            if (!packed || packed.get_desc() != pd.weights_desc()) {
                auto plain = Memory::desc({co, ci, kernel}, Memory::data_type::s8,
                                           Memory::dims{int64_t(kernel) * ci, 1, ci});
                Memory raw(plain, engine, p.quantized.data());
                packed = Memory(pd.weights_desc(), engine);
                dnnl::reorder(raw, packed).execute(stream, raw, packed); stream.wait();
            }
            p.operation = dnnl::convolution_forward(pd);
            p.packed = std::move(packed); p.src = src; p.dst = dst; p.shape = shape;
        }
        size_t n = size_t(ti) * ci;
        __m256 maximum = _mm256_setzero_ps();
        __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            if (_mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q)))
                throw std::runtime_error("nonfinite INT8 input");
            maximum = _mm256_max_ps(maximum, _mm256_and_ps(v, mask));
        }
        float lanes[8]; _mm256_storeu_ps(lanes, maximum);
        float peak = 0;
        for (float v : lanes) peak = std::max(peak, v);
        for (; i < n; ++i) {
            if (!std::isfinite(x[i])) throw std::runtime_error("nonfinite INT8 input");
            peak = std::max(peak, std::abs(x[i]));
        }
        if (!std::isfinite(peak)) throw std::runtime_error("nonfinite INT8 input");
        float scale = peak > 0 ? std::max(peak / 127.f, std::numeric_limits<float>::min()) : 1.f;
        int zero = 128;
        input.resize(n);
        parallel(n, 32768, [&](size_t begin, size_t end) {
            for (size_t j = begin; j < end; ++j)
                input[j] = uint8_t(int(std::clamp(std::nearbyint(x[j] / scale), -127.f, 127.f)) + zero);
        });
        if (residual && residual != y) std::copy_n(residual, size_t(to) * co, y);
        Memory scale_memory(Memory::desc({1}, Memory::data_type::f32, Memory::format_tag::a), engine, &scale);
        Memory zero_memory(Memory::desc({1}, Memory::data_type::s32, Memory::format_tag::a), engine, &zero);
        p.operation.execute(stream,
            {{DNNL_ARG_SRC, Memory(p.src, engine, input.data())},
             {DNNL_ARG_WEIGHTS, p.packed}, {DNNL_ARG_DST, Memory(p.dst, engine, y)},
             {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scale_memory},
             {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, p.scales},
             {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, zero_memory}});
        stream.wait();
        if (bias) for (int t = 0; t < to; ++t) for (int c = 0; c < co; ++c)
            y[size_t(t) * co + c] += bias[c];
    }
};
std::unique_ptr<Int8Conv> quantized_state;
}
void convolution(const float *x, const Half *w, const float *bias, float *y,
                 const float *residual, int ci, int co, int ti, int kernel,
                 int stride, int pad, int dilation, int source_ci) {
    if (int8_generator) {
        if (!quantized_state) quantized_state = std::make_unique<Int8Conv>();
        quantized_state->run(x, w, bias, y, residual, ci, co, ti, kernel, stride, pad, dilation, source_ci);
    } else fp32_convolution(x, w, bias, y, residual, ci, co, ti, kernel, stride, pad, dilation, source_ci);
}
void clear_convolutions() { quantized_state.reset(); fp32_clear_convolutions(); }
}
