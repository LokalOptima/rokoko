// cutlass_conv.cu — Cutlass implicit GEMM Conv1d (via Conv2d with H=1)
//
// Replaces im2col + cuBLAS SGEMM with a single Cutlass kernel launch.
// Uses TF32 TensorOp (tensor cores) with NHWC layout = [T, C] for 1D.
// Falls back to SIMT (CUDA cores) when alignment requirements aren't met.
// Fuses per-channel bias into the GEMM epilogue.
//
// Operator caching: each unique problem shape gets initialize() once.
// Subsequent calls use update() (pointer swap only) + operator().

#include <cuda_runtime.h>
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

using ElementInput       = float;
using ElementOutput      = float;
using ElementAccumulator = float;
using ElementCompute     = float;
using LayoutNHWC         = cutlass::layout::TensorNHWC;

// ---------------------------------------------------------------------------
// TF32 TensorOp path (requires C_in and C_out divisible by 4)
// Uses tensor cores with 128-bit (float4) aligned loads.
// ---------------------------------------------------------------------------

using EpilogueOpTF32 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 4, ElementAccumulator, ElementCompute>;

using Conv2dKernelTF32 = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInput, LayoutNHWC,
    ElementInput, LayoutNHWC,
    ElementOutput, LayoutNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,           // Tensor cores
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,   // Threadblock
    cutlass::gemm::GemmShape<64, 64, 16>,     // Warp
    cutlass::gemm::GemmShape<16, 8, 8>,       // TF32 MMA instruction
    EpilogueOpTF32,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,                                        // Pipeline stages
    cutlass::arch::OpMultiplyAdd,             // 1xTF32
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemmTF32 = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernelTF32>;

// ---------------------------------------------------------------------------
// SIMT fallback path (no alignment requirements)
// For convolutions with C_in or C_out not divisible by 4.
// ---------------------------------------------------------------------------

using EpilogueOpSIMT = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 1, ElementAccumulator, ElementCompute>;

using Conv2dKernelSIMT = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInput, LayoutNHWC,
    ElementInput, LayoutNHWC,
    ElementOutput, LayoutNHWC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,               // CUDA cores
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,    // Threadblock
    cutlass::gemm::GemmShape<32, 64, 8>,      // Warp
    cutlass::gemm::GemmShape<1, 1, 1>,        // Scalar instruction
    EpilogueOpSIMT,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,                                        // Pipeline stages
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;

using ImplicitGemmSIMT = cutlass::conv::device::ImplicitGemmConvolution<Conv2dKernelSIMT>;

// ---------------------------------------------------------------------------
// Operator caches: one per kernel variant, keyed by problem shape.
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

static std::unordered_map<ConvKey, ImplicitGemmTF32, ConvKeyHash> s_tf32_cache;
static std::unordered_map<ConvKey, ImplicitGemmSIMT, ConvKeyHash> s_simt_cache;

// ---------------------------------------------------------------------------
// Reshape weights: [C_out, C_in, K] → [C_out, K, C_in] (NHWC filter layout)
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
// Helper: build Arguments struct for a conv1d problem
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
// Templated dispatch: try cache hit → update, else initialize → cache
// ---------------------------------------------------------------------------

template <typename GemmOp, typename CacheMap>
static int dispatch_conv(CacheMap& cache,
                          const ConvKey& key,
                          const cutlass::conv::Conv2dProblemSize& problem,
                          float* x, float* w, float* bias_ptr,
                          float* y, float* workspace, size_t workspace_bytes,
                          int C_in, int C_out, int T_in, int T_out, int K,
                          float alpha, float beta,
                          const LayoutNHWC& layout_x, const LayoutNHWC& layout_w,
                          const LayoutNHWC& layout_y, const LayoutNHWC& layout_bias,
                          cudaStream_t stream) {
    typename GemmOp::Arguments arguments;
    arguments.problem_size = problem;
    arguments.ref_A = {x, layout_x};
    arguments.ref_B = {w, layout_w};
    arguments.ref_C = {bias_ptr, layout_bias};
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
// cutlass_conv1d_fprop: Conv1d forward via Cutlass Conv2d (H=1)
//   Tries TF32 TensorOp first (C_in, C_out must be ÷4), falls back to SIMT.
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

    auto problem = make_problem(C_in, C_out, T_in, K, stride, padding, dilation);
    ConvKey key{C_in, C_out, T_in, K, stride, padding, dilation};

    // Try TF32 TensorOp for aligned channels (÷4)
    if ((C_in % 4 == 0) && (C_out % 4 == 0)) {
        int rc = dispatch_conv<ImplicitGemmTF32>(
            s_tf32_cache, key, problem,
            x_nc, w_nc, bias_ptr, y, workspace, workspace_bytes,
            C_in, C_out, T_in, T_out, K, alpha, beta,
            layout_x, layout_w, layout_y, layout_bias, stream);
        if (rc == 0) return 0;
    }

    // SIMT fallback
    return dispatch_conv<ImplicitGemmSIMT>(
        s_simt_cache, key, problem,
        x_nc, w_nc, bias_ptr, y, workspace, workspace_bytes,
        C_in, C_out, T_in, T_out, K, alpha, beta,
        layout_x, layout_w, layout_y, layout_bias, stream);
}

// ---------------------------------------------------------------------------
// Query workspace size for a given conv configuration
// ---------------------------------------------------------------------------

extern "C"
size_t cutlass_conv1d_workspace_bytes(int C_in, int C_out, int T_in, int K,
                                       int stride, int padding, int dilation) {
    int T_out = (T_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    auto problem = make_problem(C_in, C_out, T_in, K, stride, padding, dilation);

    float* np = nullptr;
    LayoutNHWC layout_x = LayoutNHWC::packed({1, 1, T_in, C_in});
    LayoutNHWC layout_w = LayoutNHWC::packed({C_out, 1, K, C_in});
    LayoutNHWC layout_y = LayoutNHWC::packed({1, 1, T_out, C_out});

    // TF32 and SIMT may need different workspace sizes — return the max
    size_t ws = 0;
    if ((C_in % 4 == 0) && (C_out % 4 == 0)) {
        typename ImplicitGemmTF32::Arguments args(
            problem,
            {np, layout_x}, {np, layout_w},
            {np, LayoutNHWC::Stride(0)}, {np, layout_y},
            {1.0f, 0.0f});
        ImplicitGemmTF32 tmp;
        ws = tmp.get_workspace_size(args);
    }
    {
        typename ImplicitGemmSIMT::Arguments args(
            problem,
            {np, layout_x}, {np, layout_w},
            {np, LayoutNHWC::Stride(0)}, {np, layout_y},
            {1.0f, 0.0f});
        ImplicitGemmSIMT tmp;
        size_t ws2 = tmp.get_workspace_size(args);
        if (ws2 > ws) ws = ws2;
    }
    return ws;
}
