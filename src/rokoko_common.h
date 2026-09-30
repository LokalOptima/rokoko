// rokoko_common.h — Shared code between rokoko.cpp (FP32) and rokoko_f16.cpp (FP16)
//
// Contains: AlbertBuffers, TextEncoderBuffers, write_wav, compute_decode_bytes.

#pragma once
#include "audio.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "weights.h"

namespace rokoko {

// ---------------------------------------------------------------------------
// Buffer structs (arena-allocated, deterministic per T)
// ---------------------------------------------------------------------------

struct AlbertBuffers {
    float* emb = nullptr;       // [T, 128] embeddings sum
    float* hidden = nullptr;    // [T, 768] main activation
    float* qkv = nullptr;       // [T, 3*768] fused QKV
    float* attn_scores = nullptr; // [N_HEADS, T, T]
    float* attn_out = nullptr;  // [T, 768] attention output
    float* ff_mid = nullptr;    // [T, 2048] FFN intermediate
    float* ff_out = nullptr;    // [T, 768] FFN output
    float* temp = nullptr;      // [T, 768] temporary buffer
    int* token_ids = nullptr;   // [T] int32 token IDs

    void alloc(int T, GpuArena& arena) {
        emb        = arena.alloc<float>(T * 128);
        hidden     = arena.alloc<float>(T * 768);
        qkv        = arena.alloc<float>(T * 3 * 768);
        attn_scores= arena.alloc<float>(12 * T * T);
        attn_out   = arena.alloc<float>(T * 768);
        ff_mid     = arena.alloc<float>(T * 2048);
        ff_out     = arena.alloc<float>(T * 768);
        temp       = arena.alloc<float>(T * 768);
        token_ids  = arena.alloc<int>(T);
    }
};

struct TextEncoderBuffers {
    float* emb = nullptr;       // [T, 512] embedding (time-major)
    float* conv_out = nullptr;  // [T, 512] conv output / working buffer
    float* lstm_out = nullptr;  // [T, 512] LSTM output

    void alloc(int T, GpuArena& arena) {
        emb      = arena.alloc<float>(T * 512);
        conv_out = arena.alloc<float>(T * 512);
        lstm_out = arena.alloc<float>(T * 512);
    }
};

// ---------------------------------------------------------------------------
// Compute exact decode-arena bytes for given T (tokens) and L (duration frames)
// ---------------------------------------------------------------------------

inline size_t compute_decode_bytes(int T, int L) {
    int L2 = 2 * L;
    int T_audio = L2 * 300;
    int har_frames = T_audio / 5 + 1;

    auto a = [](size_t off, size_t bytes) -> size_t {
        return ((off + 255) & ~(size_t)255) + bytes;
    };

    size_t off = 0;

    // Workspace for gemm_conv1d/gemm_conv_transpose1d
    size_t max_ws_floats = (size_t)128 * 11 * har_frames;
    off = a(off, max_ws_floats * sizeof(float));
    off = a(off, size_t(2)*(T_audio+20)*sizeof(float)); // persistent STFT scratch

    // Alignment matrix + expanded encoder + shared LSTM output
    off = a(off, (size_t)T * L * sizeof(float));
    off = a(off, (size_t)L * 640 * sizeof(float));
    off = a(off, (size_t)L * 512 * sizeof(float));

    // F0/N working buffers
    off = a(off, (size_t)512 * L2 * sizeof(float));
    off = a(off, (size_t)512 * L2 * sizeof(float));
    off = a(off, (size_t)2048 * sizeof(float));

    // F0/N predictions
    off = a(off, (size_t)L2 * sizeof(float));
    off = a(off, (size_t)L2 * sizeof(float));

    // Decoder inputs
    off = a(off, (size_t)L * 512 * sizeof(float));
    off = a(off, (size_t)L * sizeof(float));
    off = a(off, (size_t)L * sizeof(float));
    off = a(off, (size_t)L * 514 * sizeof(float));

    // Decoder working buffers
    int max_ch = 1090;
    off = a(off, (size_t)max_ch * L2 * sizeof(float));
    off = a(off, (size_t)max_ch * L2 * sizeof(float));
    off = a(off, (size_t)4 * max_ch * sizeof(float));

    // Decoder blocks
    off = a(off, (size_t)L * 1024 * sizeof(float));
    off = a(off, (size_t)L * 64 * sizeof(float));
    off = a(off, (size_t)L2 * 1090 * sizeof(float));

    // Decode blocks 0-2: L*1024; block 3: L2*512
    for (int i = 0; i < 3; i++)
        off = a(off, (size_t)L * 1024 * sizeof(float));
    off = a(off, (size_t)L2 * 512 * sizeof(float));

    // Generator harmonic source
    off = a(off, (size_t)22 * har_frames * sizeof(float));
    size_t sg = off;
    sg = a(sg, (size_t)L2 * 9 * sizeof(float));
    sg = a(sg, (size_t)T_audio * sizeof(float));
    sg = a(sg, (size_t)9 * sizeof(float));

    off = a(off, (size_t)har_frames * 22 * sizeof(float));

    // Generator working pool
    size_t gen_pool_end = a(off, (size_t)5 * 128 * har_frames * sizeof(float));
    off = (sg > gen_pool_end) ? sg : gen_pool_end;
    off = a(off, (size_t)512 * sizeof(float));

    // Final audio buffer
    off = a(off, (size_t)T_audio * sizeof(float));

    return off;
}

} // namespace rokoko
