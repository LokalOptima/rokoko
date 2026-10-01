// oneDNN supplies direct AVX2 convolutions and packs immutable weights once.
// Each weight tensor retains only its most recent primitive shape. Sentence
// history cannot grow the cache; all state belongs to the single live context.
#include "convolution.h"
#include "math.h"
#include "parallel.h"
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_threadpool.hpp>
#include <cblas.h>
#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace rokoko::cpu {
namespace {
using Memory = dnnl::memory;
struct BlasPool : dnnl::threadpool_interop::threadpool_iface {
    int get_num_threads() const override { return openblas_get_num_threads(); }
    bool get_in_parallel() const override { return in_parallel(); }
    uint64_t get_flags() const override { return 0; }
    void parallel_for(int n, const std::function<void(int, int)> &function) override {
        parallel(size_t(n), 1, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
                function(int(i), n);
        });
    }
};
using Shape = std::tuple<int, int, int, int, int, bool>;
struct Entry {
    std::array<int, 3> dimensions{};
    std::optional<Shape> shape;
    Memory packed;
    Memory::desc src, dst;
    dnnl::convolution_forward operation;
};
struct State {
    BlasPool pool;
    dnnl::engine engine{dnnl::engine::kind::cpu, 0};
    dnnl::stream stream = dnnl::threadpool_interop::make_stream(engine, &pool);
    std::unordered_map<const Half *, Entry> entries;
    std::vector<float> input;
    State() {
        // Our per-weight cache owns the primitives. Avoid a second cache that
        // would retain previous sentence lengths after context destruction.
        dnnl::set_primitive_cache_capacity(0);
    }
};
std::unique_ptr<State> state;
} // namespace

void convolution(const float *x, const Half *weights, const float *bias, float *y,
                 const float *residual, int ci, int co, int ti, int kernel, int stride,
                 int pad, int dilation, int source_ci) {
    if (ci <= 0 || co <= 0 || ti <= 0 || kernel <= 0 || stride <= 0 || pad < 0 ||
        dilation <= 0 || source_ci <= 0 || source_ci > ci)
        throw std::invalid_argument("invalid CPU convolution dimensions");
    int to = (ti + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
    if (to <= 0)
        throw std::invalid_argument("empty CPU convolution output");
    if (!state)
        state = std::make_unique<State>();
    auto &s = *state;
    int threads = s.pool.get_num_threads();
    dnnl::error::wrap_c_api(dnnl_threadpool_interop_set_max_concurrency(threads),
                           "could not configure convolution workers");
    auto [it, inserted] = s.entries.try_emplace(weights);
    auto &entry = it->second;
    std::array<int, 3> dimensions{co, ci, kernel};
    if (inserted)
        entry.dimensions = dimensions;
    if (entry.dimensions != dimensions)
        throw std::logic_error("CPU convolution weight view changed dimensions");
    Shape shape{ti, stride, pad, dilation, threads, bool(residual)};
    if (!entry.shape || *entry.shape != shape) {
        auto src = Memory::desc({1, ci, ti}, Memory::data_type::f32, Memory::format_tag::nwc);
        auto dst = Memory::desc({1, co, to}, Memory::data_type::f32, Memory::format_tag::nwc);
        auto wdesc = Memory::desc({co, ci, kernel}, Memory::data_type::f32,
                                  Memory::format_tag::any);
        dnnl::primitive_attr attr;
        attr.set_fpmath_mode(dnnl::fpmath_mode::strict);
        if (residual) {
            dnnl::post_ops post;
            post.append_sum(1.f);
            attr.set_post_ops(post);
        }
        dnnl::convolution_forward::primitive_desc pd(
            s.engine, dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
            src, wdesc, dst, {stride}, {dilation - 1}, {pad}, {pad}, attr);
        Memory packed = entry.packed;
        if (!packed || packed.get_desc() != pd.weights_desc()) {
            // Bundled layout is [output, kernel, input]. Reorder also performs
            // exact FP16 -> FP32 conversion, without a second unpacked cache.
            auto plain = Memory::desc({co, ci, kernel}, Memory::data_type::f16,
                                       Memory::dims{int64_t(kernel) * ci, 1, ci});
            Memory raw(plain, s.engine, const_cast<Half *>(weights));
            packed = Memory(pd.weights_desc(), s.engine);
            dnnl::reorder(raw, packed).execute(s.stream, raw, packed);
            s.stream.wait();
        }
        entry.operation = dnnl::convolution_forward(pd);
        entry.packed = std::move(packed);
        entry.src = src;
        entry.dst = dst;
        entry.shape = shape;
    }
    s.input.resize(size_t(ti) * ci);
    if (source_ci == ci) {
        round_to_half(x, s.input.data(), s.input.size());
    } else {
        parallel(size_t(ti), size_t(std::max(1, 32768 / ci)), [&](size_t begin, size_t end) {
            for (size_t t = begin; t < end; ++t) {
                round_to_half(x + t * source_ci, s.input.data() + t * ci, source_ci);
                std::fill_n(s.input.data() + t * ci + source_ci, ci - source_ci, 0.f);
            }
        });
    }
    if (residual && residual != y)
        std::copy_n(residual, size_t(to) * co, y);
    entry.operation.execute(s.stream,
                             {{DNNL_ARG_SRC, Memory(entry.src, s.engine, s.input.data())},
                              {DNNL_ARG_WEIGHTS, entry.packed},
                              {DNNL_ARG_DST, Memory(entry.dst, s.engine, y)}});
    s.stream.wait();
    if (bias)
        for (int t = 0; t < to; ++t)
            for (int c = 0; c < co; ++c)
                y[size_t(t) * co + c] += bias[c];
}

void clear_convolutions() { state.reset(); }
ConvolutionCacheStats convolution_cache_stats() {
    ConvolutionCacheStats stats;
    if (state) {
        for (const auto &item : state->entries) {
            stats.weights += bool(item.second.packed);
            stats.primitives += bool(item.second.operation);
        }
        stats.input_capacity = state->input.capacity();
    }
    return stats;
}
} // namespace rokoko::cpu
