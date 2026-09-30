// rokoko_common.h — Shared code between rokoko.cpp (FP32) and rokoko_f16.cpp (FP16)
//
// Contains: AlbertBuffers, TextEncoderBuffers, write_wav, compute_decode_bytes.

#pragma once

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
// WAV I/O
// ---------------------------------------------------------------------------

inline void write_wav_to_(std::ostream& f, const float* audio, int n_samples,
                          int sample_rate) {
    int16_t bits_per_sample = 16;
    int16_t num_channels = 1;
    int32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int16_t block_align = num_channels * bits_per_sample / 8;
    int32_t data_size = n_samples * block_align;
    int32_t chunk_size = 36 + data_size;

    f.write("RIFF", 4);
    f.write(reinterpret_cast<char*>(&chunk_size), 4);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    int32_t fmt_size = 16;
    int16_t audio_format = 1;
    f.write(reinterpret_cast<char*>(&fmt_size), 4);
    f.write(reinterpret_cast<char*>(&audio_format), 2);
    f.write(reinterpret_cast<char*>(&num_channels), 2);
    f.write(reinterpret_cast<char*>(&sample_rate), 4);
    f.write(reinterpret_cast<char*>(&byte_rate), 4);
    f.write(reinterpret_cast<char*>(&block_align), 2);
    f.write(reinterpret_cast<char*>(&bits_per_sample), 2);

    f.write("data", 4);
    f.write(reinterpret_cast<char*>(&data_size), 4);

    for (int i = 0; i < n_samples; i++) {
        float s = std::max(-1.0f, std::min(1.0f, audio[i]));
        int16_t sample = (int16_t)(s * 32767.0f);
        f.write(reinterpret_cast<char*>(&sample), 2);
    }
}

inline bool write_wav(const std::string& path, const float* audio, int n_samples,
                      int sample_rate) {
    if (path == "-") {
        write_wav_to_(std::cout, audio, n_samples, sample_rate);
        std::cout.flush();
        return true;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    write_wav_to_(f, audio, n_samples, sample_rate);
    return f.good();
}

// streambuf adapter for FILE* so we can reuse write_wav_to_() with popen pipes
class stdio_streambuf : public std::streambuf {
    FILE* f_;
protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        return fwrite(s, 1, n, f_);
    }
    int overflow(int c) override {
        return (c != EOF && fputc(c, f_) != EOF) ? c : EOF;
    }
public:
    stdio_streambuf(FILE* f) : f_(f) {}
};

inline bool play_wav(const float* audio, int n_samples, int sample_rate) {
    char paplay_cmd[128], pwplay_cmd[128];
    snprintf(paplay_cmd, sizeof(paplay_cmd),
             "paplay --raw --format=s16le --rate=%d --channels=1", sample_rate);
    snprintf(pwplay_cmd, sizeof(pwplay_cmd),
             "pw-play --format=s16 --rate=%d --channels=1 -", sample_rate);

    const char* players[] = {
        "aplay -q -",
        paplay_cmd,
        pwplay_cmd,
        "ffplay -nodisp -autoexit -loglevel quiet -",
        nullptr
    };
    for (int i = 0; players[i]; i++) {
        std::string cmd = players[i];
        std::string bin = cmd.substr(0, cmd.find(' '));
        if (system(("command -v " + bin + " >/dev/null 2>&1").c_str()) != 0) continue;

        FILE* pipe = popen(cmd.c_str(), "w");
        if (!pipe) continue;

        stdio_streambuf buf(pipe);
        std::ostream os(&buf);
        write_wav_to_(os, audio, n_samples, sample_rate);
        os.flush();

        if (pclose(pipe) == 0) return true;
    }
    fprintf(stderr, "Error: no audio player found. Install alsa-utils, pulseaudio, pipewire, or ffmpeg.\n");
    return false;
}

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
