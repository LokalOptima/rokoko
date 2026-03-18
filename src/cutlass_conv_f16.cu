// cutlass_conv_f16.cu — Cutlass FP16 implicit GEMM Conv1d (via Conv2d H=1)
//
// Activation input: half_t (caller casts from FP32).
// Filter weights: half_t (pre-converted at startup).
// Accumulator: float.  Output: float.
// MMA instruction: 16×8×16 FP16 TensorOp on SM80+.
// Operator caching same as cutlass_conv.cu.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ---------------------------------------------------------------------------
// Common types
// ---------------------------------------------------------------------------

using ElementInputFP16   = cutlass::half_t;
using ElementFilterFP16  = cutlass::half_t;
using ElementOutputFP16  = float;
using ElementAccumFP16   = float;
using ElementComputeFP16 = float;
using LayoutNHWC         = cutlass::layout::TensorNHWC;

// ---------------------------------------------------------------------------
// FP16 TensorOp — large tiles (128x128x32)
// ---------------------------------------------------------------------------

using EpilogueOpFP16 = cutlass::epilogue::thread::LinearCombination<
    ElementOutputFP16, 4, ElementAccumFP16, ElementComputeFP16>;

using Conv2dKernelFP16Large = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputFP16, LayoutNHWC,
    ElementFilterFP16, LayoutNHWC,
    ElementOutputFP16, LayoutNHWC,
    ElementAccumFP16,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOpFP16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemmFP16Large = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernelFP16Large>;

// ---------------------------------------------------------------------------
// FP16 TensorOp — small tiles (64x64x32)
// ---------------------------------------------------------------------------

using Conv2dKernelFP16Small = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputFP16, LayoutNHWC,
    ElementFilterFP16, LayoutNHWC,
    ElementOutputFP16, LayoutNHWC,
    ElementAccumFP16,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOpFP16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemmFP16Small = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernelFP16Small>;

// ---------------------------------------------------------------------------
// Operator caches
// ---------------------------------------------------------------------------

struct ConvKey {
    int C_in, C_out, T_in, K, stride, padding, dilation;
    int mode;
    bool operator==(const ConvKey& o) const {
        return C_in == o.C_in && C_out == o.C_out && T_in == o.T_in &&
               K == o.K && stride == o.stride && padding == o.padding &&
               dilation == o.dilation && mode == o.mode;
    }
};

struct ConvKeyHash {
    size_t operator()(const ConvKey& k) const {
        size_t h = 0;
        auto mix = [&](int v) { h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2); };
        mix(k.C_in); mix(k.C_out); mix(k.T_in); mix(k.K);
        mix(k.stride); mix(k.padding); mix(k.dilation); mix(k.mode);
        return h;
    }
};

static std::unordered_map<ConvKey, ImplicitGemmFP16Large, ConvKeyHash> s_fp16_large_cache;
static std::unordered_map<ConvKey, ImplicitGemmFP16Small, ConvKeyHash> s_fp16_small_cache;

// ---------------------------------------------------------------------------
// Reshape weights: [C_out, C_in, K] float → [C_out, K, C_in] half_t (NHWC)
// ---------------------------------------------------------------------------

__global__ void reshape_weights_f16_kernel(const float* __restrict__ src,
                                             cutlass::half_t* __restrict__ dst,
                                             int C_out, int C_in, int K) {
    int total = C_out * C_in * K;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int co = idx / (C_in * K);
        int rem = idx % (C_in * K);
        int ci = rem / K;
        int k = rem % K;
        dst[co * K * C_in + k * C_in + ci] =
            cutlass::half_t(__float2half(src[idx]));
    }
}

extern "C"
void cutlass_reshape_weights_f16(const float* src, __half* dst,
                                   int C_out, int C_in, int K,
                                   cudaStream_t stream) {
    int total = C_out * C_in * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reshape_weights_f16_kernel<<<blocks, threads, 0, stream>>>(
        src, (cutlass::half_t*)dst, C_out, C_in, K);
}

// ---------------------------------------------------------------------------
// Helper: build Conv2dProblemSize
// ---------------------------------------------------------------------------

static cutlass::conv::Conv2dProblemSize make_problem(
    int C_in, int C_out, int T_in, int K,
    int stride, int padding, int dilation) {
    return cutlass::conv::Conv2dProblemSize(
        {1, 1, T_in, C_in},
        {C_out, 1, K, C_in},
        {0, 0, padding, padding},
        {1, stride},
        {1, dilation},
        cutlass::conv::Mode::kCrossCorrelation,
        1
    );
}

// ---------------------------------------------------------------------------
// Templated dispatch
// ---------------------------------------------------------------------------

template <typename GemmOp, typename CacheMap>
static int dispatch_conv(CacheMap& cache,
                          const ConvKey& key,
                          const cutlass::conv::Conv2dProblemSize& problem,
                          cutlass::half_t* x, cutlass::half_t* w,
                          float* c_ptr, float* y,
                          float* workspace, size_t workspace_bytes,
                          int C_in, int C_out, int T_in, int T_out, int K,
                          float alpha, float beta,
                          const LayoutNHWC& layout_x, const LayoutNHWC& layout_w,
                          const LayoutNHWC& layout_y, const LayoutNHWC& layout_bias,
                          cudaStream_t stream) {
    typename GemmOp::Arguments arguments;
    arguments.problem_size = problem;
    arguments.ref_A = {x, layout_x};
    arguments.ref_B = {w, layout_w};
    arguments.ref_C = {c_ptr, layout_bias};
    arguments.ref_D = {y, layout_y};
    arguments.output_op = {alpha, beta};

    auto it = cache.find(key);
    if (it != cache.end()) {
        it->second.update(arguments, workspace);
        cutlass::Status status = it->second(stream);
        return (status == cutlass::Status::kSuccess) ? 0 : -4;
    }

    GemmOp conv_op;
    cutlass::Status status = conv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return -1;

    size_t needed = conv_op.get_workspace_size(arguments);
    if (needed > workspace_bytes) return -2;

    status = conv_op.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -3;

    status = conv_op(stream);
    if (status != cutlass::Status::kSuccess) return -4;

    cache[key] = conv_op;
    return 0;
}

// ---------------------------------------------------------------------------
// cutlass_conv1d_fprop_f16
//   x: half_t [1, 1, T_in, C_in] NHWC
//   w: half_t [C_out, 1, K, C_in] NHWC
//   bias: float [C_out] or nullptr
//   y: float [1, 1, T_out, C_out] NHWC
//   residual: float [1, 1, T_out, C_out] or nullptr
// ---------------------------------------------------------------------------

static constexpr int SM_COUNT = 70;  // RTX 5070 Ti

extern "C"
int cutlass_conv1d_fprop_f16(const __half* x, const __half* w, const float* bias,
                               float* y, const float* residual,
                               float* workspace, size_t workspace_bytes,
                               int C_in, int C_out, int T_in, int K,
                               int stride, int padding, int dilation,
                               cudaStream_t stream) {
    int T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    cutlass::half_t* x_nc = const_cast<cutlass::half_t*>((const cutlass::half_t*)x);
    cutlass::half_t* w_nc = const_cast<cutlass::half_t*>((const cutlass::half_t*)w);

    float alpha = 1.0f;
    float beta;
    float* c_ptr;
    LayoutNHWC layout_bias;

    if (residual) {
        beta = 1.0f;
        c_ptr = const_cast<float*>(residual);
        layout_bias = LayoutNHWC::packed({1, 1, T_out, C_out});
    } else if (bias) {
        beta = 1.0f;
        c_ptr = const_cast<float*>(bias);
        layout_bias = LayoutNHWC(LayoutNHWC::Stride(0));
    } else {
        beta = 0.0f;
        c_ptr = y;
        layout_bias = LayoutNHWC::packed({1, 1, T_out, C_out});
    }

    LayoutNHWC layout_x = LayoutNHWC::packed({1, 1, T_in, C_in});
    LayoutNHWC layout_w = LayoutNHWC::packed({C_out, 1, K, C_in});
    LayoutNHWC layout_y = LayoutNHWC::packed({1, 1, T_out, C_out});

    auto problem = make_problem(C_in, C_out, T_in, K, stride, padding, dilation);
    int mode = residual ? 2 : (bias ? 1 : 0);
    ConvKey key{C_in, C_out, T_in, K, stride, padding, dilation, mode};

    // Require alignment: C_in % 8 == 0 for half_t TensorOp
    if ((C_in % 8 != 0) || (C_out % 4 != 0)) return -1;

    int ctas_large = ((C_out + 127) / 128) * ((T_out + 127) / 128);

    if (ctas_large >= SM_COUNT) {
        int rc = dispatch_conv<ImplicitGemmFP16Large>(
            s_fp16_large_cache, key, problem,
            x_nc, w_nc, c_ptr, y, workspace, workspace_bytes,
            C_in, C_out, T_in, T_out, K, alpha, beta,
            layout_x, layout_w, layout_y, layout_bias, stream);
        if (rc == 0) return 0;
    }

    return dispatch_conv<ImplicitGemmFP16Small>(
        s_fp16_small_cache, key, problem,
        x_nc, w_nc, c_ptr, y, workspace, workspace_bytes,
        C_in, C_out, T_in, T_out, K, alpha, beta,
        layout_x, layout_w, layout_y, layout_bias, stream);
}
