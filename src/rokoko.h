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

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
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

static inline VoiceMap load_voices(const std::string& voices_dir,
                                   std::vector<VoiceMmap>& mmaps) {
    VoiceMap voices;
    DIR* dir = opendir(voices_dir.c_str());
    if (!dir) return voices;

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string fname = ent->d_name;
        if (fname.size() < 5 || fname.substr(fname.size() - 4) != ".bin")
            continue;
        std::string vname = fname.substr(0, fname.size() - 4);
        std::string path = voices_dir + "/" + fname;

        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { fprintf(stderr, "Warning: cannot open voice %s\n", path.c_str()); continue; }
        struct stat st;
        fstat(fd, &st);
        if (st.st_size == 0) { fprintf(stderr, "Warning: voice %s is empty\n", path.c_str()); close(fd); continue; }
        void* mapped = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (mapped == MAP_FAILED) { fprintf(stderr, "Warning: mmap failed for voice %s\n", path.c_str()); continue; }

        mmaps.push_back({mapped, (size_t)st.st_size});
        voices[vname] = {(const char*)mapped, (const char*)mapped + st.st_size};
    }
    closedir(dir);
    return voices;
}

// ---------------------------------------------------------------------------
// Vocab table + tokenization
// ---------------------------------------------------------------------------

namespace detail {

struct VocabEntry { uint32_t codepoint; int32_t token_id; };

static const VocabEntry VOCAB[] = {
    {0x3B, 1}, {0x3A, 2}, {0x2C, 3}, {0x2E, 4}, {0x21, 5}, {0x3F, 6},
    {0x2014, 9}, {0x2026, 10}, {0x22, 11}, {0x28, 12}, {0x29, 13},
    {0x201C, 14}, {0x201D, 15}, {0x20, 16}, {0x303, 17}, {0x2A3, 18},
    {0x2A5, 19}, {0x2A6, 20}, {0x2A8, 21}, {0x1D5D, 22}, {0xAB67, 23},
    {'A', 24}, {'I', 25}, {'O', 31}, {'Q', 33}, {'S', 35}, {'T', 36},
    {'W', 39}, {'Y', 41}, {0x1D4A, 42},
    {'a', 43}, {'b', 44}, {'c', 45}, {'d', 46}, {'e', 47}, {'f', 48},
    {'h', 50}, {'i', 51}, {'j', 52}, {'k', 53}, {'l', 54}, {'m', 55},
    {'n', 56}, {'o', 57}, {'p', 58}, {'q', 59}, {'r', 60}, {'s', 61},
    {'t', 62}, {'u', 63}, {'v', 64}, {'w', 65}, {'x', 66}, {'y', 67},
    {'z', 68},
    {0x251, 69}, {0x250, 70}, {0x252, 71}, {0xE6, 72}, {0x3B2, 75},
    {0x254, 76}, {0x255, 77}, {0xE7, 78}, {0x256, 80}, {0xF0, 81},
    {0x2A4, 82}, {0x259, 83}, {0x25A, 85}, {0x25B, 86}, {0x25C, 87},
    {0x25F, 90}, {0x261, 92}, {0x265, 99}, {0x268, 101}, {0x26A, 102},
    {0x29D, 103}, {0x26F, 110}, {0x270, 111}, {0x14B, 112}, {0x273, 113},
    {0x272, 114}, {0x274, 115}, {0xF8, 116}, {0x278, 118}, {0x3B8, 119},
    {0x153, 120}, {0x279, 123}, {0x27E, 125}, {0x27B, 126}, {0x281, 128},
    {0x27D, 129}, {0x282, 130}, {0x283, 131}, {0x288, 132}, {0x2A7, 133},
    {0x28A, 135}, {0x28B, 136}, {0x28C, 138}, {0x263, 139}, {0x264, 140},
    {0x3C7, 142}, {0x28E, 143}, {0x292, 147}, {0x294, 148},
    {0x2C8, 156}, {0x2CC, 157}, {0x2D0, 158}, {0x2B0, 162}, {0x2B2, 164},
    {0x2193, 169}, {0x2192, 171}, {0x2197, 172}, {0x2198, 173}, {0x1D7B, 177},
};

static inline const std::unordered_map<uint32_t, int32_t>& vocab_map() {
    static std::unordered_map<uint32_t, int32_t> m = []() {
        std::unordered_map<uint32_t, int32_t> m;
        for (auto& e : VOCAB) m[e.codepoint] = e.token_id;
        return m;
    }();
    return m;
}

static inline uint32_t utf8_decode(const std::string& s, size_t& i) {
    uint8_t c = s[i];
    if (c < 0x80) { i++; return c; }
    uint32_t cp; int extra;
    if ((c & 0xE0) == 0xC0)      { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else                          { cp = c & 0x07; extra = 3; }
    i++;
    for (int j = 0; j < extra && i < s.size(); j++, i++)
        cp = (cp << 6) | (s[i] & 0x3F);
    return cp;
}

static inline size_t utf8_len(const std::string& s) {
    size_t n = 0, i = 0;
    while (i < s.size()) { utf8_decode(s, i); n++; }
    return n;
}

} // namespace detail

static inline std::vector<int32_t> to_tokens(const std::string& phonemes) {
    std::vector<int32_t> ids;
    ids.push_back(0); // BOS
    const auto& vm = detail::vocab_map();
    size_t i = 0;
    while (i < phonemes.size()) {
        uint32_t cp = detail::utf8_decode(phonemes, i);
        auto it = vm.find(cp);
        if (it != vm.end())
            ids.push_back(it->second);
    }
    ids.push_back(0); // EOS
    return ids;
}

// ---------------------------------------------------------------------------
// Chunking
// ---------------------------------------------------------------------------

static constexpr int MAX_IPA_CHARS = 510;

struct Chunk {
    std::string phonemes;
    std::vector<int32_t> tokens;
};

static inline size_t utf8_byte_offset(const std::string& s, size_t n_codepoints) {
    size_t i = 0, count = 0;
    while (i < s.size() && count < n_codepoints) {
        detail::utf8_decode(s, i);
        count++;
    }
    return i;
}

static inline std::vector<Chunk> chunk_ipa(const std::string& ipa) {
    std::vector<Chunk> chunks;
    size_t len = detail::utf8_len(ipa);

    if (len <= MAX_IPA_CHARS) {
        chunks.push_back({ipa, to_tokens(ipa)});
        return chunks;
    }

    size_t pos = 0;
    while (pos < ipa.size()) {
        std::string rest = ipa.substr(pos);
        size_t rest_len = detail::utf8_len(rest);
        if (rest_len <= MAX_IPA_CHARS) {
            chunks.push_back({rest, to_tokens(rest)});
            break;
        }

        size_t limit_byte = utf8_byte_offset(rest, MAX_IPA_CHARS);
        size_t split = std::string::npos;

        for (size_t j = limit_byte; j > 0; j--) {
            char c = rest[j - 1];
            if (c == '.' || c == '!' || c == '?') { split = j; break; }
        }
        if (split == std::string::npos) {
            for (size_t j = limit_byte; j > 0; j--) {
                char c = rest[j - 1];
                if (c == ',' || c == ':' || c == ';') { split = j; break; }
            }
        }
        if (split == std::string::npos) {
            for (size_t j = limit_byte; j > 0; j--) {
                if (rest[j - 1] == ' ') { split = j; break; }
            }
        }
        if (split == std::string::npos || split == 0)
            split = limit_byte;

        std::string piece = rest.substr(0, split);
        if (!piece.empty())
            chunks.push_back({piece, to_tokens(piece)});
        pos += split;
    }
    return chunks;
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

    template<typename F>
    std::string synthesize_streaming(const std::string& text, const std::string& voice,
                                      F on_chunk) {
        using clk = std::chrono::high_resolution_clock;
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        auto t0 = clk::now();
        std::string preprocessed = text_norm::preprocess_text(text);
        auto t1 = clk::now();
        last_preprocess_ms = ms(t0, t1);
        vlog("Preprocess: \"%s\" (%.1f ms)\n", preprocessed.c_str(), last_preprocess_ms);

        t0 = clk::now();
        std::string ipa = g2p.infer(preprocessed, stream);
        t1 = clk::now();
        last_g2p_ms = ms(t0, t1);
        vlog("G2P: \"%s\" (%.1f ms)\n", ipa.c_str(), last_g2p_ms);

        if (ipa.empty())
            return "G2P produced no output";

        auto chunks = chunk_ipa(ipa);

        auto vit = voices.find(voice);
        if (vit == voices.end())
            return "unknown voice '" + voice + "'";

        const float* voice_data = reinterpret_cast<const float*>(vit->second.start);
        size_t voice_size = vit->second.end - vit->second.start;
        int voice_rows = voice_size / (256 * sizeof(float));

        auto t_tts0 = clk::now();
        size_t total_samples = 0;
        for (size_t c = 0; c < chunks.size(); c++) {
            auto& chunk = chunks[c];
            int T = (int)chunk.tokens.size();

            int phoneme_count = T - 2;
            if (phoneme_count < 0) phoneme_count = 0;
            if (phoneme_count >= voice_rows) phoneme_count = voice_rows - 1;
            const float* style = voice_data + phoneme_count * 256;

            auto audio = rokoko_infer(weights, chunk.tokens.data(), T, style,
                                       stream, encode_arena,
                                       decode_arena, d_workspace, ws_bytes);
            vlog("  chunk %zu: T=%d, %zu samples (%.2fs), decode arena=%.1f MB\n",
                 c, T, audio.size(), audio.size()/24000.0, decode_arena.offset/1e6);
            total_samples += audio.size();
            if (!on_chunk(audio.data(), audio.size()))
                return "";
        }
        auto t_tts1 = clk::now();
        last_tts_ms = ms(t_tts0, t_tts1);

        double audio_sec = total_samples / 24000.0;
        double rtfx = audio_sec / (last_tts_ms / 1000.0);
        vlog("TTS: %.3f sec in %.1f ms (%.0fx realtime, %zu chunks)\n",
             audio_sec, last_tts_ms, rtfx, chunks.size());

        return "";
    }

    std::string synthesize(const std::string& text, const std::string& voice,
                           std::vector<float>& audio_out) {
        audio_out.clear();
        return synthesize_streaming(text, voice,
            [&audio_out](const float* data, size_t n) -> bool {
                audio_out.insert(audio_out.end(), data, data + n);
                return true;
            });
    }
};

// ---------------------------------------------------------------------------
// TTS Context — owns all GPU resources, provides TtsPipeline
// ---------------------------------------------------------------------------

static constexpr size_t ENCODE_ARENA_BYTES = 64 * 1024 * 1024;
static constexpr size_t WORKSPACE_BYTES    = 128 * 1024 * 1024;

struct TtsContext {
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
        using clk = std::chrono::high_resolution_clock;
        auto t_start = clk::now();

        // Prefetch weights in background while CUDA initializes
        Weights prefetched;
        std::thread prefetch_thread([&]() {
            prefetched = Weights::prefetch(weights_path);
        });
        cudaFree(0); // lazy CUDA init
        prefetch_thread.join();

        CUDA_CHECK(cudaStreamCreate(&stream));
        prefetched.upload(stream);
        weights = std::move(prefetched);

        // Load G2P model (overlapped with weight norm precomputation)
        bool g2p_ok = false;
        cudaStream_t g2p_stream;
        CUDA_CHECK(cudaStreamCreate(&g2p_stream));
        std::thread g2p_thread([&]() {
            g2p_ok = g2p.load(g2p_path.c_str(), g2p_stream);
        });
        precompute_weight_norms(weights, stream);
        g2p_thread.join();
        CUDA_CHECK(cudaStreamDestroy(g2p_stream));
        if (!g2p_ok) {
            fprintf(stderr, "Error: failed to load G2P model from %s\n", g2p_path.c_str());
            return false;
        }

        // Voices
        voices = load_voices(voices_dir, voice_mmaps);
        if (voices.empty()) {
            fprintf(stderr, "Error: no voices found in %s\n", voices_dir.c_str());
            return false;
        }

        // Arenas
        encode_arena.init(ENCODE_ARENA_BYTES);
        CUDA_CHECK(cudaMalloc(&d_workspace, WORKSPACE_BYTES));

        // Warmup
        auto vit = voices.find("af_heart");
        if (vit != voices.end()) {
            const float* voice_data = reinterpret_cast<const float*>(vit->second.start);
            auto warmup_ipa = g2p.infer("Warmup.", stream);
            auto warmup_tokens = to_tokens(warmup_ipa.empty() ? "." : warmup_ipa);
            rokoko_infer(weights, warmup_tokens.data(), (int)warmup_tokens.size(),
                         voice_data, stream, encode_arena, decode_arena,
                         d_workspace, WORKSPACE_BYTES);
            cudaStreamSynchronize(stream);
        }

        auto t_end = clk::now();
        vlog("TtsContext init: %.0f ms\n",
             std::chrono::duration<double, std::milli>(t_end - t_start).count());
        return true;
    }

    TtsPipeline pipeline() {
        return TtsPipeline{weights, g2p, stream, encode_arena, decode_arena,
                           d_workspace, WORKSPACE_BYTES, voices};
    }

    void destroy() {
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
    }

    ~TtsContext() { destroy(); }
    TtsContext() = default;
    TtsContext(const TtsContext&) = delete;
    TtsContext& operator=(const TtsContext&) = delete;
};

} // namespace rokoko
