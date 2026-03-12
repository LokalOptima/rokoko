#pragma once
// g2p_model_cuda.h — CUDA inference for G2P V3 Conformer CTC model.
//
// Same binary format as g2p_model.h, but runs on GPU using cublas for
// linear projections and custom kernels for RMSNorm, RoPE, softmax, SiLU.
//
// Architecture: char_emb → N × (RMSNorm→MHA→RMSNorm→SwiGLU) → upsample → head → CTC decode
//
// Optimizations over naive implementation:
//   - cublasSgemm instead of cublasLt (no descriptor create/destroy overhead)
//   - cublasSgemmStridedBatched for multi-head attention (1 call vs 4 per-head)
//   - Attention scale folded into GEMM alpha (eliminates scale kernel)
//   - Residual adds fused into GEMM via beta=1 (eliminates add kernels)
//   - Softmax batched across all heads (1 launch vs 4)
//
// Usage:
//   G2PModelCuda model;
//   model.load("data/g2p_model.bin", stream);
//   std::string phonemes = model.infer("hello world", ltHandle, stream);

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// ── CUDA kernels ────────────────────────────────────────────────────────────

// RMSNorm: out[t,i] = weight[i] * x[t,i] / sqrt(mean(x[t,:]^2) + eps)
__global__ void g2p_rms_norm_kernel(const float* __restrict__ x,
                                      const float* __restrict__ weight,
                                      float* __restrict__ out,
                                      int T, int d, float eps) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float* xt = x + t * d;
    float* ot = out + t * d;

    extern __shared__ float sdata[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        local_sum += xt[i] * xt[i];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float inv = rsqrtf(sdata[0] / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        ot[i] = weight[i] * xt[i] * inv;
}

// RoPE: apply rotary position embeddings in-place.
// data: column-major [stride, T], where head h occupies rows [h*head_dim..(h+1)*head_dim-1]
__global__ void g2p_rope_kernel(float* __restrict__ data,
                                  const float* __restrict__ cos_table,
                                  const float* __restrict__ sin_table,
                                  int T, int stride, int heads, int head_dim) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= T || h >= heads) return;

    int d2 = head_dim / 2;
    float* base = data + t * stride + h * head_dim;
    const float* rc = cos_table + t * d2;
    const float* rs = sin_table + t * d2;

    for (int i = threadIdx.x; i < d2; i += blockDim.x) {
        float x0 = base[i], x1 = base[d2 + i];
        base[i]      = x0 * rc[i] - x1 * rs[i];
        base[d2 + i] = x1 * rc[i] + x0 * rs[i];
    }
}

// Batched softmax: scores has `num_rows` rows each of length `row_len`.
// In-place: row[i] = softmax(row[i]) for each row.
__global__ void g2p_softmax_kernel(float* __restrict__ scores, int row_len, int num_rows) {
    int t = blockIdx.x;
    if (t >= num_rows) return;
    float* row = scores + t * row_len;
    extern __shared__ float sdata[];

    // Find max
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < row_len; i += blockDim.x)
        local_max = fmaxf(local_max, row[i]);
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_len; i += blockDim.x) {
        float v = expf(row[i] - max_val);
        row[i] = v;
        local_sum += v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int i = threadIdx.x; i < row_len; i += blockDim.x)
        row[i] *= inv_sum;
}

// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
__global__ void g2p_swiglu_kernel(const float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    float* __restrict__ out,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// Add bias to column-major matrix: data[M, N] += bias[M] (broadcast over columns)
__global__ void g2p_bias_kernel(float* __restrict__ data,
                                  const float* __restrict__ bias,
                                  int M, int N) {
    int col = blockIdx.x;
    if (col >= N) return;
    float* col_data = data + col * M;
    for (int i = threadIdx.x; i < M; i += blockDim.x)
        col_data[i] += bias[i];
}

// Embedding lookup: out[t, :] = emb[ids[t], :]
__global__ void g2p_embed_kernel(const int* __restrict__ ids, const float* __restrict__ emb,
                                   float* __restrict__ out, int T, int d) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float* src = emb + ids[t] * d;
    float* dst = out + t * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        dst[i] = src[i];
}

// Reshape upsample: column-major proj[d*up, T] → column-major X_up[d, T*up]
__global__ void g2p_upsample_reshape_kernel(const float* __restrict__ proj,
                                              float* __restrict__ X_up,
                                              int T, int d, int up) {
    int idx = blockIdx.x;
    if (idx >= T * up) return;
    int t = idx / up;
    int u = idx % up;
    const float* src = proj + t * (d * up) + u * d;
    float* dst = X_up + idx * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        dst[i] = src[i];
}

// CTC argmax per timestep: out[t] = argmax(logits[t, :])
__global__ void g2p_ctc_argmax_kernel(const float* __restrict__ logits,
                                        int* __restrict__ out,
                                        int T, int n_phones) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* row = logits + t * n_phones;
    int best = 0;
    float best_v = row[0];
    for (int c = 1; c < n_phones; c++) {
        if (row[c] > best_v) { best_v = row[c]; best = c; }
    }
    out[t] = best;
}


// ── Model struct ────────────────────────────────────────────────────────────

struct G2PModelCuda {
    bool load(const char* path, cudaStream_t stream);
    bool load(const void* data, size_t size, cudaStream_t stream);
    std::string infer(const std::string& text, cublasLtHandle_t ltHandle, cudaStream_t stream) const;
    void free();

    bool loaded() const { return d_ > 0; }
    size_t param_bytes() const { return total_bytes_; }

private:
    // Config
    int d_ = 0, heads_ = 0, n_layers_ = 0, ff_ = 0, up_ = 0;
    int n_chars_ = 0, n_phones_ = 0;
    int head_dim_ = 0;
    bool use_rope_ = false;

    // Vocab mappings (CPU)
    std::unordered_map<uint32_t, int> char2id_;
    std::unordered_map<int, uint32_t> id2phone_;

    // GPU weights (all contiguous in one allocation)
    float* weights_gpu_ = nullptr;
    size_t total_bytes_ = 0;

    // cuBLAS handle (created in load, used for all GEMMs)
    mutable cublasHandle_t cublas_ = nullptr;

    // Pointers into weights_gpu_
    float* char_emb_ = nullptr;    // [n_chars, d]
    float* rope_cos_ = nullptr;    // [max_pos, head_dim/2]
    float* rope_sin_ = nullptr;    // [max_pos, head_dim/2]

    struct Layer {
        float* n1_w;               // [d]
        float* qkv_w, *qkv_b;     // [3d, d], [3d]
        float* out_w, *out_b;      // [d, d], [d]
        float* n2_w;               // [d]
        float* gate_w, *gate_b;    // [ff, d], [ff]
        float* up_w, *up_b;        // [ff, d], [ff]
        float* down_w, *down_b;    // [d, ff], [d]
    };
    std::vector<Layer> layers_;

    float* up_w_ = nullptr, *up_b_ = nullptr;     // [d*up, d], [d*up]
    float* head_w_ = nullptr, *head_b_ = nullptr;  // [n_phones, d], [n_phones]

    int max_pos_ = 2048;

    // Cached workspace (avoids cudaMalloc per call)
    mutable float* workspace_ = nullptr;
    mutable size_t workspace_bytes_ = 0;

    bool load_from_file_(FILE* f, const char* label, cudaStream_t stream);
};

// ── Implementation ──────────────────────────────────────────────────────────

inline bool G2PModelCuda::load(const void* data, size_t size, cudaStream_t stream) {
    FILE* f = fmemopen(const_cast<void*>(data), size, "rb");
    if (!f) return false;
    bool ok = load_from_file_(f, "<bundle:g2p>", stream);
    fclose(f);
    return ok;
}

inline bool G2PModelCuda::load(const char* path, cudaStream_t stream) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    bool ok = load_from_file_(f, path, stream);
    fclose(f);
    return ok;
}

inline bool G2PModelCuda::load_from_file_(FILE* f, const char* label, cudaStream_t stream) {

    // Magic
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "G2P3", 4) != 0) {
        fprintf(stderr, "g2p_cuda: expected G2P3 format\n");
        return false;
    }

    // Header
    uint32_t hdr[10];
    if (fread(hdr, 4, 10, f) != 10) { return false; }
    d_ = hdr[0]; heads_ = hdr[1]; n_layers_ = hdr[2]; ff_ = hdr[3]; up_ = hdr[4];
    n_chars_ = hdr[5]; n_phones_ = hdr[6];
    uint32_t flags = hdr[9];
    use_rope_ = (flags & 1) != 0;
    bool use_qk_norm = (flags & 2) != 0;
    bool use_conv = (flags & 4) != 0;
    head_dim_ = d_ / heads_;

    bool use_rmsnorm = (flags & 8) != 0;
    if (use_qk_norm || use_conv) {
        fprintf(stderr, "g2p_cuda: QK-Norm and ConvModule not supported\n");
        return false;
    }
    if (!use_rmsnorm) {
        fprintf(stderr, "g2p_cuda: only RMSNorm supported (got LayerNorm model)\n");
        return false;
    }

    // Char vocab
    uint32_t n_cv;
    if (fread(&n_cv, 4, 1, f) != 1) { return false; }
    for (uint32_t i = 0; i < n_cv; i++) {
        uint32_t pair[2];
        if (fread(pair, 4, 2, f) != 2) { return false; }
        char2id_[pair[0]] = pair[1];
    }

    // Phone vocab
    uint32_t n_pv;
    if (fread(&n_pv, 4, 1, f) != 1) { return false; }
    for (uint32_t i = 0; i < n_pv; i++) {
        uint32_t pair[2];
        if (fread(pair, 4, 2, f) != 2) { return false; }
        id2phone_[pair[1]] = pair[0];
    }

    // Calculate total weight size
    int d2 = head_dim_ / 2;
    size_t total_floats = 0;
    total_floats += n_chars_ * d_;                      // char_emb
    total_floats += max_pos_ * d2 * 2;                  // rope cos + sin
    for (int i = 0; i < n_layers_; i++) {
        total_floats += d_;                              // n1_w
        total_floats += 3 * d_ * d_ + 3 * d_;           // qkv_w, qkv_b
        total_floats += d_ * d_ + d_;                    // out_w, out_b
        total_floats += d_;                              // n2_w
        total_floats += ff_ * d_ + ff_;                  // gate_w, gate_b
        total_floats += ff_ * d_ + ff_;                  // up_w, up_b
        total_floats += d_ * ff_ + d_;                   // down_w, down_b
    }
    total_floats += d_ * up_ * d_ + d_ * up_;           // up_w, up_b
    total_floats += n_phones_ * d_ + n_phones_;          // head_w, head_b

    total_bytes_ = total_floats * sizeof(float);

    // Allocate GPU memory (single contiguous block)
    auto err = cudaMalloc(&weights_gpu_, total_bytes_);
    if (err != cudaSuccess) {
        fprintf(stderr, "g2p_cuda: cudaMalloc failed for weights (%.1f MB): %s\n",
                total_bytes_ / (1024.0f * 1024.0f), cudaGetErrorString(err));
        return false;
    }

    // Read weights to CPU staging buffer, then upload
    std::vector<float> staging(total_floats);
    float* ptr = staging.data();
    auto read_w = [&](int n) -> float* {
        float* p = ptr;
        if (fread(p, sizeof(float), n, f) != (size_t)n) return nullptr;
        ptr += n;
        return p;
    };

    // Char embedding
    float* char_emb_cpu = read_w(n_chars_ * d_);
    if (!char_emb_cpu) { return false; }

    // Layers
    struct LayerCPU { float *n1_w, *qkv_w, *qkv_b, *out_w, *out_b, *n2_w,
                            *gate_w, *gate_b, *up_w, *up_b, *down_w, *down_b; };
    std::vector<LayerCPU> layers_cpu(n_layers_);
    for (int i = 0; i < n_layers_; i++) {
        auto& L = layers_cpu[i];
        L.n1_w   = read_w(d_);
        L.qkv_w  = read_w(3 * d_ * d_);
        L.qkv_b  = read_w(3 * d_);
        L.out_w  = read_w(d_ * d_);
        L.out_b  = read_w(d_);
        L.n2_w   = read_w(d_);
        L.gate_w = read_w(ff_ * d_);
        L.gate_b = read_w(ff_);
        L.up_w   = read_w(ff_ * d_);
        L.up_b   = read_w(ff_);
        L.down_w = read_w(d_ * ff_);
        L.down_b = read_w(d_);
        if (!L.down_b) { return false; }
    }

    float* up_w_cpu = read_w(d_ * up_ * d_);
    float* up_b_cpu = read_w(d_ * up_);
    float* head_w_cpu = read_w(n_phones_ * d_);
    float* head_b_cpu = read_w(n_phones_);
    if (!head_b_cpu) { return false; }

    // Compute RoPE tables (CPU)
    size_t rope_floats = max_pos_ * d2 * 2;
    std::vector<float> rope_staging(rope_floats);
    float* rope_cos_cpu = rope_staging.data();
    float* rope_sin_cpu = rope_staging.data() + max_pos_ * d2;
    if (use_rope_) {
        for (int t = 0; t < max_pos_; t++) {
            for (int i = 0; i < d2; i++) {
                float freq = (float)t / std::pow(10000.0f, (float)(2 * i) / head_dim_);
                rope_cos_cpu[t * d2 + i] = std::cos(freq);
                rope_sin_cpu[t * d2 + i] = std::sin(freq);
            }
        }
    }

    // Upload everything to GPU
    float* gpu = weights_gpu_;
    auto upload = [&](const float* src, int n) -> float* {
        float* dst = gpu;
        cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyHostToDevice, stream);
        gpu += n;
        return dst;
    };

    char_emb_ = upload(char_emb_cpu, n_chars_ * d_);
    rope_cos_ = upload(rope_cos_cpu, max_pos_ * d2);
    rope_sin_ = upload(rope_sin_cpu, max_pos_ * d2);

    layers_.resize(n_layers_);
    for (int i = 0; i < n_layers_; i++) {
        auto& L = layers_[i];
        auto& C = layers_cpu[i];
        L.n1_w   = upload(C.n1_w,   d_);
        L.qkv_w  = upload(C.qkv_w,  3 * d_ * d_);
        L.qkv_b  = upload(C.qkv_b,  3 * d_);
        L.out_w  = upload(C.out_w,   d_ * d_);
        L.out_b  = upload(C.out_b,   d_);
        L.n2_w   = upload(C.n2_w,    d_);
        L.gate_w = upload(C.gate_w,  ff_ * d_);
        L.gate_b = upload(C.gate_b,  ff_);
        L.up_w   = upload(C.up_w,    ff_ * d_);
        L.up_b   = upload(C.up_b,    ff_);
        L.down_w = upload(C.down_w,  d_ * ff_);
        L.down_b = upload(C.down_b,  d_);
    }
    up_w_   = upload(up_w_cpu,   d_ * up_ * d_);
    up_b_   = upload(up_b_cpu,   d_ * up_);
    head_w_ = upload(head_w_cpu, n_phones_ * d_);
    head_b_ = upload(head_b_cpu, n_phones_);

    // Create cuBLAS handle
    cublasCreate(&cublas_);
    cublasSetMathMode(cublas_, CUBLAS_TF32_TENSOR_OP_MATH);

    cudaStreamSynchronize(stream);

    fprintf(stderr, "g2p_cuda: loaded %s (d=%d, %d layers, %d heads, %d ff, %dx up, %.1f MB)\n",
            label, d_, n_layers_, heads_, ff_, up_, total_bytes_ / (1024.0f * 1024.0f));
    return true;
}

inline void G2PModelCuda::free() {
    if (weights_gpu_) { cudaFree(weights_gpu_); weights_gpu_ = nullptr; }
    if (workspace_) { cudaFree(workspace_); workspace_ = nullptr; workspace_bytes_ = 0; }
    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    d_ = 0;
}

inline std::string G2PModelCuda::infer(const std::string& text, cublasLtHandle_t /*ltHandle*/,
                                         cudaStream_t stream) const {
    if (d_ == 0) return "";

    // Encode input to char IDs
    std::vector<int> ids;
    const uint8_t* p = (const uint8_t*)text.data();
    const uint8_t* end = p + text.size();
    while (p < end) {
        uint32_t cp;
        if (*p < 0x80) { cp = *p++; }
        else if ((*p & 0xE0) == 0xC0) { cp = (*p++ & 0x1F) << 6; if (p < end) cp |= (*p++ & 0x3F); }
        else if ((*p & 0xF0) == 0xE0) { cp = (*p++ & 0x0F) << 12; if (p < end) cp |= (*p++ & 0x3F) << 6; if (p < end) cp |= (*p++ & 0x3F); }
        else { cp = (*p++ & 0x07) << 18; if (p < end) cp |= (*p++ & 0x3F) << 12; if (p < end) cp |= (*p++ & 0x3F) << 6; if (p < end) cp |= (*p++ & 0x3F); }
        auto it = char2id_.find(cp);
        ids.push_back(it != char2id_.end() ? it->second : 0);
    }

    int T = (int)ids.size();
    if (T == 0 || T > max_pos_) return "";
    int d = d_, h = heads_, dk = head_dim_, ff = ff_;
    int T_up = T * up_;

    // Workspace layout (no ffn_out needed — residual fused into GEMM):
    //   ids_gpu[T] | X[d,T] | QKV[3d,T] | attn_scores[h,T,T] | attn_out[d,T] |
    //   normed[d,T] | ffn_gate[ff,T] | ffn_up[ff,T] |
    //   X_up_proj[d*up,T] | X_up[d,T_up] | logits[n_phones,T_up] | argmax[T_up]
    size_t ws_bytes =
        T * sizeof(int) +                              // ids_gpu
        T * d * sizeof(float) +                        // X
        T * 3 * d * sizeof(float) +                    // QKV
        h * T * T * sizeof(float) +                    // attn_scores
        T * d * sizeof(float) +                        // attn_out
        T * d * sizeof(float) +                        // normed
        T * ff * sizeof(float) +                       // ffn_gate
        T * ff * sizeof(float) +                       // ffn_up
        T * d * up_ * sizeof(float) +                  // X_up_proj
        T_up * d * sizeof(float) +                     // X_up
        T_up * n_phones_ * sizeof(float) +             // logits
        T_up * sizeof(int);                            // argmax

    if (ws_bytes > workspace_bytes_) {
        if (workspace_) cudaFree(workspace_);
        auto ws_err = cudaMalloc(&workspace_, ws_bytes);
        if (ws_err != cudaSuccess) {
            fprintf(stderr, "g2p_cuda: workspace alloc failed (%zu bytes): %s\n",
                    ws_bytes, cudaGetErrorString(ws_err));
            workspace_ = nullptr; workspace_bytes_ = 0;
            return "";
        }
        workspace_bytes_ = ws_bytes;
    }

    // Assign pointers
    char* wp = (char*)workspace_;
    auto wallocf = [&](size_t n) -> float* { float* p = (float*)wp; wp += n * sizeof(float); return p; };
    auto walloci = [&](size_t n) -> int*   { int* p = (int*)wp; wp += n * sizeof(int); return p; };

    int* ids_gpu       = walloci(T);
    float* X           = wallocf(T * d);
    float* QKV         = wallocf(T * 3 * d);
    float* attn_scores = wallocf(h * T * T);
    float* attn_out    = wallocf(T * d);
    float* normed      = wallocf(T * d);
    float* ffn_gate    = wallocf(T * ff);
    float* ffn_up      = wallocf(T * ff);
    float* X_up_proj   = wallocf(T * d * up_);
    float* X_up        = wallocf(T_up * d);
    float* logits      = wallocf(T_up * n_phones_);
    int* argmax        = walloci(T_up);

    // Set stream on cublas handle
    cublasSetStream(cublas_, stream);

    // Upload input IDs
    cudaMemcpyAsync(ids_gpu, ids.data(), T * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Embedding lookup
    g2p_embed_kernel<<<T, 128, 0, stream>>>(ids_gpu, char_emb_, X, T, d);

    int block = 128;
    float one = 1.0f, zero = 0.0f;
    float scale = 1.0f / std::sqrt((float)dk);

    // Transformer layers
    for (int li = 0; li < n_layers_; li++) {
        const auto& L = layers_[li];

        // ── Self-attention ──
        // 1. RMSNorm
        g2p_rms_norm_kernel<<<T, block, block * sizeof(float), stream>>>(
            X, L.n1_w, normed, T, d, 1e-6f);

        // 2. QKV projection: QKV[3d, T] = W_qkv^T * normed + bias
        //    W stored as row-major [3d, d] = col-major [d, 3d], use OP_T
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    3 * d, T, d, &one, L.qkv_w, d, normed, d, &zero, QKV, 3 * d);
        g2p_bias_kernel<<<T, 256, 0, stream>>>(QKV, L.qkv_b, 3 * d, T);

        // 3. Apply RoPE to Q and K
        if (use_rope_) {
            dim3 rope_grid(T, h);
            g2p_rope_kernel<<<rope_grid, 64, 0, stream>>>(QKV,     rope_cos_, rope_sin_, T, 3 * d, h, dk);
            g2p_rope_kernel<<<rope_grid, 64, 0, stream>>>(QKV + d, rope_cos_, rope_sin_, T, 3 * d, h, dk);
        }

        // 4. Batched attention scores: S_h[T,T] = scale * K_h^T * Q_h (all heads at once)
        //    K_h for head h starts at QKV + d + h*dk, stride between columns = 3*d
        //    Q_h for head h starts at QKV + h*dk, stride between columns = 3*d
        //    Stride between heads = dk
        cublasSgemmStridedBatched(cublas_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, dk,
            &scale,
            QKV + d, 3 * d, (long long)dk,     // K (head 0), lda, strideA
            QKV,     3 * d, (long long)dk,     // Q (head 0), ldb, strideB
            &zero,
            attn_scores, T, (long long)T * T,  // S, ldc, strideC
            h);

        // 5. Batched softmax across all heads (h*T rows of length T)
        g2p_softmax_kernel<<<h * T, block, block * sizeof(float), stream>>>(
            attn_scores, T, h * T);

        // 6. Batched value weighted sum: O_h[dk,T] = V_h[dk,T] * S_h[T,T]
        //    V_h for head h starts at QKV + 2*d + h*dk, stride between columns = 3*d
        //    O_h for head h starts at attn_out + h*dk, stride between columns = d
        cublasSgemmStridedBatched(cublas_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dk, T, T,
            &one,
            QKV + 2 * d, 3 * d, (long long)dk,     // V (head 0), lda, strideA
            attn_scores, T,     (long long)T * T,   // S (head 0), ldb, strideB
            &zero,
            attn_out, d, (long long)dk,              // O (head 0), ldc, strideC
            h);

        // 7. Output projection with fused residual: X = W_out^T * attn_out + X + bias
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    d, T, d, &one, L.out_w, d, attn_out, d, &one, X, d);
        g2p_bias_kernel<<<T, 256, 0, stream>>>(X, L.out_b, d, T);

        // ── SwiGLU FFN ──
        // 8. RMSNorm
        g2p_rms_norm_kernel<<<T, block, block * sizeof(float), stream>>>(
            X, L.n2_w, normed, T, d, 1e-6f);

        // 9. Gate and Up projections
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    ff, T, d, &one, L.gate_w, d, normed, d, &zero, ffn_gate, ff);
        g2p_bias_kernel<<<T, 256, 0, stream>>>(ffn_gate, L.gate_b, ff, T);

        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    ff, T, d, &one, L.up_w, d, normed, d, &zero, ffn_up, ff);
        g2p_bias_kernel<<<T, 256, 0, stream>>>(ffn_up, L.up_b, ff, T);

        // 10. SwiGLU activation
        g2p_swiglu_kernel<<<(T * ff + 255) / 256, 256, 0, stream>>>(
            ffn_gate, ffn_up, ffn_gate, T * ff);

        // 11. Down projection with fused residual: X = W_down^T * ffn_gate + X + bias
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    d, T, ff, &one, L.down_w, ff, ffn_gate, ff, &one, X, d);
        g2p_bias_kernel<<<T, 256, 0, stream>>>(X, L.down_b, d, T);
    }

    // Upsample: proj[d*up, T] = W_up^T * X + bias
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                d * up_, T, d, &one, up_w_, d, X, d, &zero, X_up_proj, d * up_);
    g2p_bias_kernel<<<T, 256, 0, stream>>>(X_up_proj, up_b_, d * up_, T);

    // Reshape: X_up_proj[d*up, T] → X_up[d, T*up]
    g2p_upsample_reshape_kernel<<<T_up, 128, 0, stream>>>(X_up_proj, X_up, T, d, up_);

    // Output head: logits[n_phones, T_up] = W_head^T * X_up + bias
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                n_phones_, T_up, d, &one, head_w_, d, X_up, d, &zero, logits, n_phones_);
    g2p_bias_kernel<<<T_up, 256, 0, stream>>>(logits, head_b_, n_phones_, T_up);

    // CTC argmax
    g2p_ctc_argmax_kernel<<<(T_up + 255) / 256, 256, 0, stream>>>(logits, argmax, T_up, n_phones_);

    // Copy argmax back to CPU
    std::vector<int> argmax_cpu(T_up);
    cudaMemcpyAsync(argmax_cpu.data(), argmax, T_up * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // CTC decode on CPU
    std::string result;
    int prev = -1;
    for (int t = 0; t < T_up; t++) {
        int best = argmax_cpu[t];
        if (best != 0 && best != prev) {
            auto it = id2phone_.find(best);
            if (it != id2phone_.end()) {
                uint32_t cp = it->second;
                if (cp < 0x80) { result += (char)cp; }
                else if (cp < 0x800) { result += (char)(0xC0 | (cp >> 6)); result += (char)(0x80 | (cp & 0x3F)); }
                else if (cp < 0x10000) { result += (char)(0xE0 | (cp >> 12)); result += (char)(0x80 | ((cp >> 6) & 0x3F)); result += (char)(0x80 | (cp & 0x3F)); }
                else { result += (char)(0xF0 | (cp >> 18)); result += (char)(0x80 | ((cp >> 12) & 0x3F)); result += (char)(0x80 | ((cp >> 6) & 0x3F)); result += (char)(0x80 | (cp & 0x3F)); }
            }
        }
        prev = best;
    }

    return result;
}
