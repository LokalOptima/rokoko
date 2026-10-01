// rokoko.h — Public API for rokoko TTS library
//
// Usage:
//   #include "rokoko.h"
//   rokoko::TtsContext ctx;
//   ctx.init();
//   auto pipeline = ctx.pipeline();
//   std::vector<float> audio;
//   pipeline.synthesize("Hello world.", audio);

#pragma once

#include "weights.h"
#include "embedded.h"
#include "kernels.h"
#include "rokoko_common.h"
#include "g2p.h"
#include "normalize.h"
#include "phonemes.h"

#include <chrono>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace rokoko {

// ---------------------------------------------------------------------------
// TTS Pipeline
// ---------------------------------------------------------------------------

struct TtsPipeline {
    Weights& weights;
    G2PModelCuda& g2p;
    cudaStream_t stream;
    GpuArena& encode_arena;
    GpuArena& decode_arena;
    float* d_workspace;
    size_t ws_bytes;
    const float* voice;

    double last_preprocess_ms = 0;
    double last_g2p_ms = 0;
    double last_tts_ms = 0;

    struct Prepared {
        std::string normalized;
        std::vector<TextSpan> source_spans;
        std::vector<Chunk> chunks;
    };
    std::string prepare(const std::string& text, Prepared& out, bool phonemes=false) {
        out={}; last_preprocess_ms=last_g2p_ms=last_tts_ms=0;
        try {
            detail::utf8_len(text); // Reject malformed bytes before normalization.
            auto t0=std::chrono::steady_clock::now();
            out.normalized=phonemes?text:text_norm::preprocess_text(text);
            last_preprocess_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            out.source_spans=split_spans(out.normalized,phonemes?MAX_IPA_CHARS:g2p.max_positions());
            if (out.source_spans.empty()) return "text contains no speech";
            t0=std::chrono::steady_clock::now();
            for (auto span:out.source_spans) {
                auto part=out.normalized.substr(span.begin,span.end-span.begin);
                auto ipa=phonemes?part:g2p.infer(part,stream);
                auto chunks=chunk_ipa(ipa);
                if (chunks.empty()) return "G2P produced no output";
                for (auto& chunk:chunks) {
                    if (!has_speech_tokens(chunk.tokens)) return "text contains no speech tokens";
                    style_row(chunk.phonemes,510); // Validate before rendering or HTTP headers.
                    out.chunks.push_back(std::move(chunk));
                }
            }
            last_g2p_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            return "";
        } catch (const std::exception& e) { return e.what(); }
    }
    const float* style_for(const Chunk& chunk) const {
        return voice+256*style_row(chunk.phonemes,510);
    }
    template<typename F>
    std::string render(const Prepared& speech,F on_chunk) {
        auto t0=std::chrono::steady_clock::now();
        try {
            if (speech.chunks.empty()) return "text contains no speech";
            for (auto& chunk:speech.chunks) {
                auto audio=rokoko_infer(weights,chunk.tokens.data(),int(chunk.tokens.size()),style_for(chunk),
                                         stream,encode_arena,decode_arena,d_workspace,ws_bytes);
                if (audio.empty()) return "inference produced no audio";
                for (float x:audio) if (!std::isfinite(x)) return "inference produced nonfinite audio";
                if (!on_chunk(audio.data(),audio.size())) return "synthesis cancelled";
            }
            last_tts_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            return "";
        } catch (const std::exception& e) { return e.what(); }
    }
    template<typename F>
    std::string synthesize_streaming(const std::string& text,F on_chunk) {
        Prepared speech;auto err=prepare(text,speech);
        return err.empty()?render(speech,on_chunk):err;
    }
    std::string synthesize(const std::string& text,std::vector<float>& audio_out, bool phonemes=false) {
        audio_out.clear();Prepared speech;auto err=prepare(text,speech,phonemes);
        if (err.empty()) err=render(speech,[&](const float* p,size_t n) {audio_out.insert(audio_out.end(),p,p+n);return true;});
        if (!err.empty()) audio_out.clear();
        return err;
    }

};

// ---------------------------------------------------------------------------
// TTS Context — owns all GPU resources, provides TtsPipeline
// ---------------------------------------------------------------------------

static constexpr size_t ENCODE_ARENA_BYTES = 64 * 1024 * 1024;
static constexpr size_t WORKSPACE_BYTES    = 128 * 1024 * 1024;

// The current backend has process-global operator caches. One live context and
// serialized calls are supported; reject a second context instead of sharing pointers.
inline std::atomic<bool> context_active{false};
struct TtsContext {
    bool owns_backend=false;
    std::string last_error;
    Weights weights;
    G2PModelCuda g2p;
    cudaStream_t stream = nullptr;
    GpuArena encode_arena;
    GpuArena decode_arena;
    float* d_workspace = nullptr;
    const float* voice = nullptr;

    bool init() { return init(embedded_assets()); }

    // Borrowed asset bytes must outlive the context (used by development tools).
    bool init(const ModelAssets& assets) {
        if (owns_backend) { last_error="context already initialized"; return false; }
        bool expected=false;
        if (!context_active.compare_exchange_strong(expected,true)) { last_error="only one live TtsContext is supported"; return false; }
        owns_backend=true;last_error.clear();
        try {
            if (!assets.voice.data || assets.voice.size!=510*256*sizeof(float) ||
                reinterpret_cast<uintptr_t>(assets.voice.data)%alignof(float))
                throw std::runtime_error("invalid af_heart style data");
            voice=reinterpret_cast<const float*>(assets.voice.data);
            weights=Weights::prefetch(assets.weights.data,assets.weights.size);
            CUDA_CHECK(cudaStreamCreate(&stream));
            weights.upload(stream);
            initialize_inference(weights,stream);
            if (!g2p.load(assets.g2p.data,assets.g2p.size,stream))
                throw std::runtime_error("invalid or truncated G2P model");
            encode_arena.init(ENCODE_ARENA_BYTES);
            CUDA_CHECK(cudaMalloc(&d_workspace,WORKSPACE_BYTES));
            auto pipe=pipeline();std::vector<float> warmup;
            auto err=pipe.synthesize("Warmup.",warmup);
            if (!err.empty()) throw std::runtime_error(err);
            return true;
        } catch (const std::exception& e) {
            last_error=e.what();destroy();return false;
        }
    }

    TtsPipeline pipeline() {
        return TtsPipeline{weights, g2p, stream, encode_arena, decode_arena,
                           d_workspace, WORKSPACE_BYTES, voice};
    }

    void destroy() {
        if (!owns_backend) return;
        if (stream) cudaStreamSynchronize(stream);
        release_inference_state(weights);
        if (d_workspace) { cudaFree(d_workspace); d_workspace = nullptr; }
        decode_arena.destroy();
        encode_arena.destroy();
        voice=nullptr;
        g2p.free();
        weights.free();
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
        owns_backend=false;context_active=false;
    }

    ~TtsContext() { destroy(); }
    TtsContext() = default;
    TtsContext(const TtsContext&) = delete;
    TtsContext& operator=(const TtsContext&) = delete;
};

} // namespace rokoko
