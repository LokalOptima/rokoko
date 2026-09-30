// cutlass_gemm_f16.cu — Cutlass FP16 GEMM kernels for mixed-precision inference
//
// Weights (A): half_t.  Activations (B): half_t (caller casts from FP32).
// Accumulator: float.  Output (C/D): float.
// MMA instruction: 16×8×16 FP16 TensorOp on SM80+.
//
// Same layout combinations as cutlass_gemm.cu (TN, NT, NN, batched, bias).
// Separate operator caches from the TF32 variants.

#include <cuda_runtime.h>
#include <cstdio>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ---------------------------------------------------------------------------
// Common types
// ---------------------------------------------------------------------------

using ElementHalf  = cutlass::half_t;
using Element      = float;
using Accumulator  = float;
using Compute      = float;
using LayoutRM     = cutlass::layout::RowMajor;
using LayoutCM     = cutlass::layout::ColumnMajor;

// Epilogue: float output, accumulate float, alignment 4 (128-bit stores)
using EpilogueFP16 = cutlass::epilogue::thread::LinearCombination<
    Element, 4, Accumulator, Compute>;

// Epilogue: alignment 1 for unaligned M
using EpilogueFP16_A1 = cutlass::epilogue::thread::LinearCombination<
    Element, 1, Accumulator, Compute>;

using Swizzle        = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using BatchedSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
using Arch           = cutlass::arch::Sm80;

// ---------------------------------------------------------------------------
// TN layout: A = half_t RowMajor [M,K], B = half_t ColumnMajor [K,N]
// ---------------------------------------------------------------------------

using GemmTN_FP16_Large = cutlass::gemm::device::Gemm<
    ElementHalf, LayoutRM, ElementHalf, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueFP16, Swizzle, 3>;

using GemmTN_FP16_Small = cutlass::gemm::device::Gemm<
    ElementHalf, LayoutRM, ElementHalf, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueFP16, Swizzle, 3>;

using GemmTN_FP16_Align1 = cutlass::gemm::device::Gemm<
    ElementHalf, LayoutRM, ElementHalf, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueFP16_A1, Swizzle, 3>;

using GemmTN_FP16_SIMT = cutlass::gemm::device::Gemm<
    ElementHalf, LayoutRM, ElementHalf, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassSimt, Arch,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueFP16_A1, Swizzle, 2>;  // 2 stages: pipelined (no cp_async for half)

// ---------------------------------------------------------------------------
// NN layout: A = half_t ColumnMajor [M,K], B = half_t ColumnMajor [K,N]
// ---------------------------------------------------------------------------

using GemmNN_FP16_Large = cutlass::gemm::device::Gemm<
    ElementHalf, LayoutCM, ElementHalf, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueFP16, Swizzle, 3>;

using GemmNN_FP16_Small = cutlass::gemm::device::Gemm<
    ElementHalf, LayoutCM, ElementHalf, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueFP16, Swizzle, 3>;

using GemmNN_FP16_Align1 = cutlass::gemm::device::Gemm<
    ElementHalf, LayoutCM, ElementHalf, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueFP16_A1, Swizzle, 3>;

using GemmNN_FP16_SIMT = cutlass::gemm::device::Gemm<
    ElementHalf, LayoutCM, ElementHalf, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassSimt, Arch,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueFP16_A1, Swizzle, 2>;  // 2 stages: pipelined (no cp_async for half)

// ---------------------------------------------------------------------------
// Batched TN (attention-style, if ever needed for FP16)
// ---------------------------------------------------------------------------

using GemmBatchedTN_FP16 = cutlass::gemm::device::GemmBatched<
    ElementHalf, LayoutRM, ElementHalf, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueFP16, BatchedSwizzle, 3>;

using GemmBatchedNN_FP16 = cutlass::gemm::device::GemmBatched<
    ElementHalf, LayoutCM, ElementHalf, LayoutCM, Element, LayoutCM, Accumulator,
    cutlass::arch::OpClassTensorOp, Arch,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueFP16, BatchedSwizzle, 3>;

// ---------------------------------------------------------------------------
// Operator caches (separate from TF32 caches)
// ---------------------------------------------------------------------------

struct GemmKey {
    int M, N, K, lda, ldb, ldc;
    bool operator==(const GemmKey& o) const {
        return M == o.M && N == o.N && K == o.K &&
               lda == o.lda && ldb == o.ldb && ldc == o.ldc;
    }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& k) const {
        size_t h = 0;
        auto mix = [&](int v) { h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2); };
        mix(k.M); mix(k.N); mix(k.K); mix(k.lda); mix(k.ldb); mix(k.ldc);
        return h;
    }
};

// TN caches
static std::unordered_map<GemmKey, GemmTN_FP16_Large, GemmKeyHash>  s_f16_tn_large;
static std::unordered_map<GemmKey, GemmTN_FP16_Small, GemmKeyHash>  s_f16_tn_small;
static std::unordered_map<GemmKey, GemmTN_FP16_Align1, GemmKeyHash> s_f16_tn_align1;
static std::unordered_map<GemmKey, GemmTN_FP16_SIMT, GemmKeyHash>   s_f16_tn_simt;

// NN caches
static std::unordered_map<GemmKey, GemmNN_FP16_Large, GemmKeyHash>  s_f16_nn_large;
static std::unordered_map<GemmKey, GemmNN_FP16_Small, GemmKeyHash>  s_f16_nn_small;
static std::unordered_map<GemmKey, GemmNN_FP16_Align1, GemmKeyHash> s_f16_nn_align1;
static std::unordered_map<GemmKey, GemmNN_FP16_SIMT, GemmKeyHash>   s_f16_nn_simt;

// Batched caches
struct BatchedKey {
    int M, N, K, batch;
    long long strideA, strideB, strideC;
    bool operator==(const BatchedKey& o) const {
        return M == o.M && N == o.N && K == o.K && batch == o.batch &&
               strideA == o.strideA && strideB == o.strideB && strideC == o.strideC;
    }
};

struct BatchedKeyHash {
    size_t operator()(const BatchedKey& k) const {
        size_t h = 0;
        auto mix = [&](long long v) { h ^= std::hash<long long>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2); };
        mix(k.M); mix(k.N); mix(k.K); mix(k.batch);
        mix(k.strideA); mix(k.strideB); mix(k.strideC);
        return h;
    }
};

static std::unordered_map<BatchedKey, GemmBatchedTN_FP16, BatchedKeyHash> s_f16_batched_tn;
static std::unordered_map<BatchedKey, GemmBatchedNN_FP16, BatchedKeyHash> s_f16_batched_nn;

// ---------------------------------------------------------------------------
// Tile selection
// ---------------------------------------------------------------------------

static constexpr int SM_COUNT = 70;  // RTX 5070 Ti

// ---------------------------------------------------------------------------
// Templated dispatch (same pattern as TF32)
// ---------------------------------------------------------------------------

template <typename GemmOp, typename CacheMap, typename KeyType>
static int dispatch_gemm(CacheMap& cache,
                          const KeyType& key,
                          typename GemmOp::Arguments& arguments,
                          float* workspace, size_t workspace_bytes,
                          cudaStream_t stream) {
    auto it = cache.find(key);
    if (it != cache.end()) {
        it->second.update(arguments);
        cutlass::Status status = it->second(stream);
        return (status == cutlass::Status::kSuccess) ? 0 : -4;
    }

    GemmOp gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return -1;

    size_t needed = gemm_op.get_workspace_size(arguments);
    if (needed > workspace_bytes) return -2;

    status = gemm_op.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -3;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -4;

    cache[key] = gemm_op;
    return 0;
}

// FP16 alignment: K % 8 == 0 for TensorOp with half_t operands
static bool fp16_align8(int K) { return (K % 8 == 0); }
// Epilogue align4: M % 4 == 0 for 128-bit float stores
static bool epilogue_align4(int M) { return (M % 4 == 0); }

// ---------------------------------------------------------------------------
// cutlass_gemm_tn_f16: C = alpha * A^T * B + beta * C
//   A: half_t [K, M] col-major → RowMajor [M, K]
//   B: half_t [K, N] col-major → ColumnMajor [K, N]
//   C: float [M, N] col-major → ColumnMajor [M, N]
// ---------------------------------------------------------------------------

extern "C"
int cutlass_gemm_tn_f16(int M, int N, int K,
                          const cutlass::half_t* A, int lda,
                          const cutlass::half_t* B, int ldb,
                          float* C, int ldc,
                          float alpha, float beta,
                          float* workspace, size_t workspace_bytes,
                          cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(M, N, K);
    GemmKey key{M, N, K, lda, ldb, ldc};

    if (fp16_align8(K) && epilogue_align4(M)) {
        int ctas = ((M + 127) / 128) * ((N + 127) / 128);
        if (ctas >= SM_COUNT) {
            typename GemmTN_FP16_Large::Arguments args(problem,
                {A, LayoutRM(lda)}, {B, LayoutCM(ldb)},
                {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
            if (dispatch_gemm<GemmTN_FP16_Large>(s_f16_tn_large, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
        {
            typename GemmTN_FP16_Small::Arguments args(problem,
                {A, LayoutRM(lda)}, {B, LayoutCM(ldb)},
                {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
            if (dispatch_gemm<GemmTN_FP16_Small>(s_f16_tn_small, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
    }
    if (fp16_align8(K)) {
        typename GemmTN_FP16_Align1::Arguments args(problem,
            {A, LayoutRM(lda)}, {B, LayoutCM(ldb)},
            {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
        if (dispatch_gemm<GemmTN_FP16_Align1>(s_f16_tn_align1, key, args, workspace, workspace_bytes, stream) == 0) return 0;
    }
    // SIMT fallback
    typename GemmTN_FP16_SIMT::Arguments args(problem,
        {A, LayoutRM(lda)}, {B, LayoutCM(ldb)},
        {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
    return dispatch_gemm<GemmTN_FP16_SIMT>(s_f16_tn_simt, key, args, workspace, workspace_bytes, stream);
}

// ---------------------------------------------------------------------------
// cutlass_gemm_tn_bias_f16: D = A^T * B + bias  (bias fused via stride-0 C)
// ---------------------------------------------------------------------------

extern "C"
int cutlass_gemm_tn_bias_f16(int M, int N, int K,
                               const cutlass::half_t* A, int lda,
                               const cutlass::half_t* B, int ldb,
                               float* D, int ldd,
                               const float* bias,
                               float* workspace, size_t workspace_bytes,
                               cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(M, N, K);
    GemmKey key{M, N, K, lda, ldb, -(ldd + 1)};

    if (fp16_align8(K) && epilogue_align4(M)) {
        int ctas = ((M + 127) / 128) * ((N + 127) / 128);
        if (ctas >= SM_COUNT) {
            typename GemmTN_FP16_Large::Arguments args(problem,
                {A, LayoutRM(lda)}, {B, LayoutCM(ldb)},
                {const_cast<Element*>(bias), LayoutCM(0)},
                {D, LayoutCM(ldd)}, {1.0f, 1.0f});
            if (dispatch_gemm<GemmTN_FP16_Large>(s_f16_tn_large, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
        {
            typename GemmTN_FP16_Small::Arguments args(problem,
                {A, LayoutRM(lda)}, {B, LayoutCM(ldb)},
                {const_cast<Element*>(bias), LayoutCM(0)},
                {D, LayoutCM(ldd)}, {1.0f, 1.0f});
            if (dispatch_gemm<GemmTN_FP16_Small>(s_f16_tn_small, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
    }
    if (fp16_align8(K)) {
        typename GemmTN_FP16_Align1::Arguments args(problem,
            {A, LayoutRM(lda)}, {B, LayoutCM(ldb)},
            {const_cast<Element*>(bias), LayoutCM(0)},
            {D, LayoutCM(ldd)}, {1.0f, 1.0f});
        if (dispatch_gemm<GemmTN_FP16_Align1>(s_f16_tn_align1, key, args, workspace, workspace_bytes, stream) == 0) return 0;
    }
    typename GemmTN_FP16_SIMT::Arguments args(problem,
        {A, LayoutRM(lda)}, {B, LayoutCM(ldb)},
        {const_cast<Element*>(bias), LayoutCM(0)},
        {D, LayoutCM(ldd)}, {1.0f, 1.0f});
    return dispatch_gemm<GemmTN_FP16_SIMT>(s_f16_tn_simt, key, args, workspace, workspace_bytes, stream);
}

// ---------------------------------------------------------------------------
// cutlass_gemm_nn_f16: C = alpha * A * B + beta * C
// ---------------------------------------------------------------------------

extern "C"
int cutlass_gemm_nn_f16(int M, int N, int K,
                          const cutlass::half_t* A, int lda,
                          const cutlass::half_t* B, int ldb,
                          float* C, int ldc,
                          float alpha, float beta,
                          float* workspace, size_t workspace_bytes,
                          cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(M, N, K);
    GemmKey key{M, N, K, lda, ldb, ldc};

    // NN: A ColumnMajor, alignment on M; B ColumnMajor, alignment on K
    if (fp16_align8(K) && (M % 8 == 0) && epilogue_align4(M)) {
        int ctas = ((M + 127) / 128) * ((N + 127) / 128);
        if (ctas >= SM_COUNT) {
            typename GemmNN_FP16_Large::Arguments args(problem,
                {A, LayoutCM(lda)}, {B, LayoutCM(ldb)},
                {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
            if (dispatch_gemm<GemmNN_FP16_Large>(s_f16_nn_large, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
        {
            typename GemmNN_FP16_Small::Arguments args(problem,
                {A, LayoutCM(lda)}, {B, LayoutCM(ldb)},
                {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
            if (dispatch_gemm<GemmNN_FP16_Small>(s_f16_nn_small, key, args, workspace, workspace_bytes, stream) == 0) return 0;
        }
    }
    if (fp16_align8(K) && (M % 8 == 0)) {
        typename GemmNN_FP16_Align1::Arguments args(problem,
            {A, LayoutCM(lda)}, {B, LayoutCM(ldb)},
            {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
        if (dispatch_gemm<GemmNN_FP16_Align1>(s_f16_nn_align1, key, args, workspace, workspace_bytes, stream) == 0) return 0;
    }
    // SIMT fallback
    typename GemmNN_FP16_SIMT::Arguments args(problem,
        {A, LayoutCM(lda)}, {B, LayoutCM(ldb)},
        {C, LayoutCM(ldc)}, {C, LayoutCM(ldc)}, {alpha, beta});
    return dispatch_gemm<GemmNN_FP16_SIMT>(s_f16_nn_simt, key, args, workspace, workspace_bytes, stream);
}
