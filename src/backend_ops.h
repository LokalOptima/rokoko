// Internal BLAS-layout operator ABI, implemented by CUTLASS or OpenBLAS.
#pragma once
#include "device.h"

extern "C" int backend_gemm_nt(int M, int N, int K, const float *A, int lda, const float *B,
                               int ldb, float *C, int ldc, float alpha, float beta,
                               float *workspace, size_t workspace_bytes, rokoko::Stream stream);
extern "C" int backend_gemm_batched_tn(int M, int N, int K, const float *A, int lda,
                                       long long strideA, const float *B, int ldb,
                                       long long strideB, float *C, int ldc, long long strideC,
                                       int batch_count, float alpha, float beta, float *workspace,
                                       size_t workspace_bytes, rokoko::Stream stream);
extern "C" int backend_gemm_batched_nn(int M, int N, int K, const float *A, int lda,
                                       long long strideA, const float *B, int ldb,
                                       long long strideB, float *C, int ldc, long long strideC,
                                       int batch_count, float alpha, float beta, float *workspace,
                                       size_t workspace_bytes, rokoko::Stream stream);
extern "C" int backend_gemm_tn_f16(int M, int N, int K, const rokoko::Half *A, int lda,
                                   const rokoko::Half *B, int ldb, float *C, int ldc, float alpha,
                                   float beta, float *workspace, size_t workspace_bytes,
                                   rokoko::Stream stream);
extern "C" int backend_gemm_tn_bias_f16(int M, int N, int K, const rokoko::Half *A, int lda,
                                        const rokoko::Half *B, int ldb, float *D, int ldd,
                                        const float *bias, float *workspace, size_t workspace_bytes,
                                        rokoko::Stream stream);
extern "C" int backend_gemm_nn_f16(int M, int N, int K, const rokoko::Half *A, int lda,
                                   const rokoko::Half *B, int ldb, float *C, int ldc, float alpha,
                                   float beta, float *workspace, size_t workspace_bytes,
                                   rokoko::Stream stream);
extern "C" int backend_conv1d_fprop_f16(const rokoko::Half *x, const rokoko::Half *w,
                                        const float *bias, float *y, const float *residual,
                                        float *workspace, size_t workspace_bytes, int C_in,
                                        int C_out, int T_in, int K, int stride, int padding,
                                        int dilation, rokoko::Stream stream);
extern "C" void clear_backend_gemm_cache();
extern "C" void clear_backend_gemm_f16_cache();
extern "C" void clear_backend_conv_f16_cache();
extern "C" int backend_gemm_tn(int M, int N, int K, const float *A, int lda, const float *B,
                               int ldb, float *C, int ldc, float alpha, float beta,
                               float *workspace, size_t workspace_bytes, rokoko::Stream stream);
extern "C" int backend_gemm_nn(int M, int N, int K, const float *A, int lda, const float *B,
                               int ldb, float *C, int ldc, float alpha, float beta,
                               float *workspace, size_t workspace_bytes, rokoko::Stream stream);
