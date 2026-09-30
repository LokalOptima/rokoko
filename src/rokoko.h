// rokoko.h — Public API for rokoko TTS library
//
// Usage:
//   #include "rokoko.h"
//   rokoko::TtsContext ctx;
//   ctx.init("~/.cache/rokoko/weights.fp16.bin",
//            "~/.cache/rokoko/g2p.bin",
//            "~/.cache/rokoko/voices");
//   auto pipeline = ctx.pipeline();
//   std::vector<float> audio;
//   pipeline.synthesize("Hello world.", "af_heart", audio);

#pragma once

#include "weights.h"
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
#include <fcntl.h>
#include <functional>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace rokoko {

// ---------------------------------------------------------------------------
// Voice map
// ---------------------------------------------------------------------------

struct VoicePack { const char* start; const char* end; };
using VoiceMap = std::unordered_map<std::string, VoicePack>;
struct VoiceMmap { void* ptr; size_t size; };

inline constexpr const char* SUPPORTED_VOICE = "af_heart";

static inline VoiceMap load_voices(const std::string& voices_dir,
                                   std::vector<VoiceMmap>& mmaps) {
    VoiceMap voices;
    std::string path = voices_dir + "/" + SUPPORTED_VOICE + ".bin";
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return voices;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size != 510 * 256 * sizeof(float)) {
        fprintf(stderr, "Warning: voice %s has invalid size\n", path.c_str());
        close(fd);
        return voices;
    }
    void* mapped = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return voices;
    mmaps.push_back({mapped, size_t(st.st_size)});
    voices[SUPPORTED_VOICE] = {static_cast<const char*>(mapped),
                              static_cast<const char*>(mapped) + st.st_size};
    return voices;
}

// ---------------------------------------------------------------------------
// Download helpers
// ---------------------------------------------------------------------------

static inline void mkdirs(const std::string& path) {
    std::string dir = path;
    for (size_t p = 1; p < dir.size(); p++) {
        if (dir[p] == '/') {
            dir[p] = '\0';
            mkdir(dir.c_str(), 0755);
            dir[p] = '/';
        }
    }
    mkdir(dir.c_str(), 0755);
}

static inline bool file_ok(const std::string& path, size_t expected_size = 0) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    if (expected_size > 0 && (size_t)st.st_size != expected_size) {
        fprintf(stderr, "Warning: %s has wrong size (%zu, expected %zu) — re-downloading\n",
                path.c_str(), (size_t)st.st_size, expected_size);
        unlink(path.c_str());
        return false;
    }
    return true;
}

static inline bool download_file(const std::string& url, const std::string& dest) {
    mkdirs(dest.substr(0, dest.rfind('/')));
    std::string tmp = dest + ".tmp";
    pid_t pid = fork();
    if (pid == 0) {
        execlp("curl", "curl", "-fL", "-#", "-o", tmp.c_str(), url.c_str(), nullptr);
        _exit(127);
    }
    int status;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        unlink(tmp.c_str());
        return false;
    }
    if (rename(tmp.c_str(), dest.c_str()) != 0) {
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

static inline std::string cache_dir() {
    const char* xdg = getenv("XDG_CACHE_HOME");
    if (xdg && xdg[0]) return std::string(xdg) + "/rokoko";
    const char* home = getenv("HOME");
    if (home && home[0]) return std::string(home) + "/.cache/rokoko";
    return ".";
}

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
    VoiceMap& voices;

    double last_preprocess_ms = 0;
    double last_g2p_ms = 0;
    double last_tts_ms = 0;

    struct Prepared {
        std::string normalized;
        std::vector<TextSpan> source_spans;
        std::vector<Chunk> chunks;
        std::string voice;
    };
    std::string prepare(const std::string& text, const std::string& voice, Prepared& out, bool phonemes=false) {
        out={}; last_preprocess_ms=last_g2p_ms=last_tts_ms=0;
        try {
            if (voice != SUPPORTED_VOICE || !voices.count(voice)) return "unknown voice '"+voice+"'";
            detail::utf8_len(text); // Reject malformed bytes before normalization.
            auto t0=std::chrono::steady_clock::now();
            out.normalized=phonemes?text:text_norm::preprocess_text(text);
            last_preprocess_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            out.voice=voice;
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
    const float* style_for(const Chunk& chunk,const std::string& voice) const {
        auto it=voices.find(voice);
        if (voice != SUPPORTED_VOICE || it==voices.end()) throw std::invalid_argument("unknown voice");
        size_t rows=(it->second.end-it->second.start)/(256*sizeof(float));
        return reinterpret_cast<const float*>(it->second.start)+256*style_row(chunk.phonemes,rows);
    }
    template<typename F>
    std::string render(const Prepared& speech,F on_chunk) {
        auto t0=std::chrono::steady_clock::now();
        try {
            if (speech.chunks.empty()) return "text contains no speech";
            for (auto& chunk:speech.chunks) {
                auto audio=rokoko_infer(weights,chunk.tokens.data(),int(chunk.tokens.size()),style_for(chunk,speech.voice),
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
    std::string synthesize_streaming(const std::string& text,const std::string& voice,F on_chunk) {
        Prepared speech;auto err=prepare(text,voice,speech);
        return err.empty()?render(speech,on_chunk):err;
    }
    std::string synthesize(const std::string& text,const std::string& voice,std::vector<float>& audio_out, bool phonemes=false) {
        audio_out.clear();Prepared speech;auto err=prepare(text,voice,speech,phonemes);
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
    std::vector<VoiceMmap> voice_mmaps;
    VoiceMap voices;

    bool init(const std::string& weights_path, const std::string& g2p_path,
              const std::string& voices_dir) {
        if (owns_backend) { last_error="context already initialized"; return false; }
        bool expected=false;
        if (!context_active.compare_exchange_strong(expected,true)) { last_error="only one live TtsContext is supported"; return false; }
        owns_backend=true;last_error.clear();
        try {
            weights=Weights::prefetch(weights_path);
            CUDA_CHECK(cudaStreamCreate(&stream));
            weights.upload(stream);
            precompute_weight_norms(weights,stream);
            if (!g2p.load(g2p_path.c_str(),stream)) throw std::runtime_error("invalid or truncated G2P model: "+g2p_path);
            voices=load_voices(voices_dir,voice_mmaps);
            if (voices.empty()) throw std::runtime_error("missing or invalid af_heart.bin in "+voices_dir);
            encode_arena.init(ENCODE_ARENA_BYTES);
            CUDA_CHECK(cudaMalloc(&d_workspace,WORKSPACE_BYTES));
            auto pipe=pipeline();std::vector<float> warmup;
            auto err=pipe.synthesize("Warmup.",SUPPORTED_VOICE,warmup);
            if (!err.empty()) throw std::runtime_error(err);
            return true;
        } catch (const std::exception& e) {
            last_error=e.what();destroy();return false;
        }
    }

    TtsPipeline pipeline() {
        return TtsPipeline{weights, g2p, stream, encode_arena, decode_arena,
                           d_workspace, WORKSPACE_BYTES, voices};
    }

    void destroy() {
        if (!owns_backend) return;
        if (stream) cudaStreamSynchronize(stream);
        release_inference_state(weights);
        if (d_workspace) { cudaFree(d_workspace); d_workspace = nullptr; }
        decode_arena.destroy();
        encode_arena.destroy();
        for (auto& vm : voice_mmaps)
            if (vm.ptr) munmap(vm.ptr, vm.size);
        voice_mmaps.clear();
        voices.clear();
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
