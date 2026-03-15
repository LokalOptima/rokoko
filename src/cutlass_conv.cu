// cutlass_conv.cu — Cutlass implicit GEMM Conv1d (via Conv2d with H=1)
//
// Replaces im2col + cuBLAS SGEMM with a single Cutlass kernel launch.
// Uses FP32 SIMT (CUDA cores) with NHWC layout = [T, C] for 1D.
// Fuses per-channel bias into the GEMM epilogue.
//
// Operator caching: each unique problem shape (C_in, C_out, T_in, K,
// stride, padding, dilation) gets initialize() once. Subsequent calls
// with the same shape use update() (pointer swap only) + operator().

#include <cuda_runtime.h>
#include <cstdio>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ---------------------------------------------------------------------------
// Cutlass Conv2d implicit GEMM type (FP32 SIMT, NHWC, SM80-compatible)
// ---------------------------------------------------------------------------

using ElementInput       = float;
using ElementOutput      = float;
using ElementAccumulator = float;
using ElementCompute     = float;
using LayoutNHWC         = cutlass::layout::TensorNHWC;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 1, ElementAccumulator, ElementCompute>;

using Conv2dKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInput, LayoutNHWC,         // Activation
    ElementInput, LayoutNHWC,         // Filter
    ElementOutput, LayoutNHWC,        // Output
    ElementAccumulator,
    cutlass::arch::OpClassSimt,       // CUDA cores (SIMT)
    cutlass::arch::Sm80,              // Forward-compatible with SM120
    cutlass::gemm::GemmShape<128, 128, 8>,  // Threadblock
    cutlass::gemm::GemmShape<32, 64, 8>,    // Warp
    cutlass::gemm::GemmShape<1, 1, 1>,      // Instruction (scalar for SIMT)
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,                                // Pipeline stages
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernel>;

// ---------------------------------------------------------------------------
// Operator cache: one initialized operator per unique problem shape.
// First call: initialize() + operator(). Subsequent: update() + operator().
// ---------------------------------------------------------------------------

struct ConvKey {
    int C_in, C_out, T_in, K, stride, padding, dilation;
    bool operator==(const ConvKey& o) const {
        return C_in == o.C_in && C_out == o.C_out && T_in == o.T_in &&
               K == o.K && stride == o.stride && padding == o.padding &&
               dilation == o.dilation;
    }
};

struct ConvKeyHash {
    size_t operator()(const ConvKey& k) const {
        size_t h = 0;
        auto mix = [&](int v) { h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2); };
        mix(k.C_in); mix(k.C_out); mix(k.T_in); mix(k.K);
        mix(k.stride); mix(k.padding); mix(k.dilation);
        return h;
    }
};

static std::unordered_map<ConvKey, ImplicitGemm, ConvKeyHash> s_op_cache;

// ---------------------------------------------------------------------------
// Reshape weights: [C_out, C_in, K] → [C_out, K, C_in] (NHWC filter layout)
// One-time operation during weight precomputation.
// ---------------------------------------------------------------------------

__global__ void reshape_weights_kernel(const float* __restrict__ src,
                                         float* __restrict__ dst,
                                         int C_out, int C_in, int K) {
    int total = C_out * C_in * K;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int co = idx / (C_in * K);
        int rem = idx % (C_in * K);
        int ci = rem / K;
        int k = rem % K;
        // src[co, ci, k] → dst[co, k, ci]
        dst[co * K * C_in + k * C_in + ci] = src[idx];
    }
}

extern "C"
void cutlass_reshape_weights(const float* src, float* dst,
                              int C_out, int C_in, int K,
                              cudaStream_t stream) {
    int total = C_out * C_in * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reshape_weights_kernel<<<blocks, threads, 0, stream>>>(src, dst, C_out, C_in, K);
}

// ---------------------------------------------------------------------------
// cutlass_conv1d_fprop: Conv1d forward via Cutlass Conv2d (H=1)
//   x: [T_in, C_in] (NHWC with N=1, H=1, W=T_in)
//   w: [C_out, C_in, K] stored as [C_out, 1, K, C_in] in NHWC
//   bias: [C_out] or nullptr
//   y: [T_out, C_out]
//   workspace: pre-allocated workspace buffer
// ---------------------------------------------------------------------------

extern "C"
int cutlass_conv1d_fprop(const float* x, const float* w, const float* bias,
                          float* y, float* workspace, size_t workspace_bytes,
                          int C_in, int C_out, int T_in, int K,
                          int stride, int padding, int dilation,
                          cudaStream_t stream) {
    int T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    float* x_nc = const_cast<float*>(x);
    float* w_nc = const_cast<float*>(w);
    float alpha = 1.0f;
    float beta = bias ? 1.0f : 0.0f;
    float* bias_ptr = bias ? const_cast<float*>(bias) : y;

    LayoutNHWC layout_x = LayoutNHWC::packed({1, 1, T_in, C_in});
    LayoutNHWC layout_w = LayoutNHWC::packed({C_out, 1, K, C_in});
    LayoutNHWC layout_y = LayoutNHWC::packed({1, 1, T_out, C_out});
    LayoutNHWC layout_bias = bias ? LayoutNHWC(LayoutNHWC::Stride(0)) : layout_y;

    typename ImplicitGemm::Arguments arguments;
    arguments.problem_size = cutlass::conv::Conv2dProblemSize(
        {1, 1, T_in, C_in},
        {C_out, 1, K, C_in},
        {0, 0, padding, padding},
        {1, stride},
        {1, dilation},
        cutlass::conv::Mode::kCrossCorrelation,
        1
    );
    arguments.ref_A = {x_nc, layout_x};
    arguments.ref_B = {w_nc, layout_w};
    arguments.ref_C = {bias_ptr, layout_bias};
    arguments.ref_D = {y, layout_y};
    arguments.output_op = {alpha, beta};

    // Look up cached operator for this problem shape
    ConvKey key{C_in, C_out, T_in, K, stride, padding, dilation};
    auto it = s_op_cache.find(key);

    if (it != s_op_cache.end()) {
        // Cache hit: update pointers only, skip initialize()
        it->second.update(arguments, workspace);
        cutlass::Status status = it->second(stream);
        return (status == cutlass::Status::kSuccess) ? 0 : -4;
    }

    // Cache miss: full initialization
    ImplicitGemm conv_op;

    cutlass::Status status = conv_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return -1;

    size_t needed = conv_op.get_workspace_size(arguments);
    if (needed > workspace_bytes) return -2;

    status = conv_op.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -3;

    status = conv_op(stream);
    if (status != cutlass::Status::kSuccess) return -4;

    // Cache the initialized operator
    s_op_cache[key] = conv_op;
    return 0;
}

// ---------------------------------------------------------------------------
// Query workspace size for a given conv configuration
// ---------------------------------------------------------------------------

extern "C"
size_t cutlass_conv1d_workspace_bytes(int C_in, int C_out, int T_in, int K,
                                       int stride, int padding, int dilation) {
    int T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    cutlass::conv::Conv2dProblemSize problem_size(
        {1, 1, T_in, C_in},
        {C_out, 1, K, C_in},
        {0, 0, padding, padding},
        {1, stride},
        {1, dilation},
        cutlass::conv::Mode::kCrossCorrelation,
        1
    );

    float* np = nullptr;
    typename ImplicitGemm::Arguments arguments(
        problem_size,
        {np, LayoutNHWC::packed({1, 1, T_in, C_in})},
        {np, LayoutNHWC::packed({C_out, 1, K, C_in})},
        {np, LayoutNHWC::Stride(0)},
        {np, LayoutNHWC::packed({1, 1, T_out, C_out})},
        {1.0f, 0.0f}
    );

    ImplicitGemm tmp;
    return tmp.get_workspace_size(arguments);
}
