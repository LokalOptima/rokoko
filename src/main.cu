// main.cu — Neural G2P + TTS in one binary
//
// Pipeline: text → preprocess → G2P infer → tokenize → chunk → TTS infer → WAV
//
// Build: make rokoko
// Usage: ./rokoko "Hello world." -o output.wav --voice af_heart
//        ./rokoko --serve 8080

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "rokoko.h"
#include "server.h"

bool g_verbose = false;

using namespace rokoko;

// Backend-specific: each binary provides its own weights filename + expected size
extern const char* default_weights_filename();
extern size_t default_weights_size();

// Release base URL — single source of truth for all downloads
static const char* RELEASE_BASE = "https://github.com/LokalOptima/rokoko/releases/download/v2.0.1/";

static const char* G2P_FILENAME = "g2p.bin";
static const size_t G2P_SIZE = 34641552;
static const char* VOICE_NAMES[] = {"af_heart", "af_bella", "af_nicole", "af_sky"};
static const size_t VOICE_SIZE = 522240;  // all voices are the same size

static std::string release_url(const std::string& filename) {
    return std::string(RELEASE_BASE) + filename;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    auto print_usage = [&]() {
        fprintf(stderr,
            "Usage: %s <text> [options]\n"
            "       %s --serve [port] [options]\n"
            "\n"
            "Options:\n"
            "  --voice <name>      Voice (default: af_heart)\n"
            "  -o <file>           Output WAV (default: output.wav)\n"
            "  --say               Play audio through speakers\n"
            "  --stdout            Write WAV to stdout\n"
            "  --serve [port]      HTTP server with web UI (default: 8080)\n"
            "  --host <addr>       Server bind address (default: 0.0.0.0)\n"
            "  --weights <file>    TTS weight file (default: ~/.cache/rokoko/%s)\n"
            "  --g2p <file>        G2P model file (default: ~/.cache/rokoko/g2p.bin)\n"
            "  --voices <dir>      Voice directory (default: ~/.cache/rokoko/voices)\n"
            "  -v                  Verbose output (timings, IPA, GPU info)\n"
            "  --help              Show this help\n"
            "\n"
            "Examples:\n"
            "  %s \"Hello world.\" -o hello.wav\n"
            "  %s \"Hello world.\" --say\n"
            "  %s \"Hello world.\" --stdout | aplay\n"
            "  %s --serve 8080\n",
            argv[0], argv[0], default_weights_filename(),
            argv[0], argv[0], argv[0], argv[0]);
    };

    if (argc < 2) { print_usage(); return 1; }

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            print_usage(); return 0;
        }
    }

    const char* home_env = std::getenv("HOME");
    std::string home = home_env ? home_env : ".";
    std::string cache = home + "/.cache/rokoko";
    std::string weights_path = cache + "/" + default_weights_filename();
    std::string g2p_path = cache + "/" + G2P_FILENAME;
    std::string voices_dir = cache + "/voices";
    std::string text_input;
    std::string voice_name = "af_heart";
    std::string output_path = "output.wav";
    bool say_mode = false;
    bool serve_mode = false;
    int serve_port = 8080;
    std::string serve_host = "0.0.0.0";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc)      weights_path = argv[++i];
        else if (arg == "--g2p" && i + 1 < argc)     g2p_path = argv[++i];
        else if (arg == "--voices" && i + 1 < argc)   voices_dir = argv[++i];
        else if (arg == "--voice" && i + 1 < argc)    voice_name = argv[++i];
        else if (arg == "-o" && i + 1 < argc)         output_path = argv[++i];
        else if (arg == "--stdout")                    output_path = "-";
        else if (arg == "--say")                      say_mode = true;
        else if (arg == "-v" || arg == "--verbose")   g_verbose = true;
        else if (arg == "--host" && i + 1 < argc)     serve_host = argv[++i];
        else if (arg == "--serve") {
            serve_mode = true;
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
                serve_port = std::atoi(argv[++i]);
        }
        else if (arg[0] != '-') {
            if (!text_input.empty()) {
                fprintf(stderr, "Error: unexpected argument '%s' (text already set)\n", arg.c_str());
                return 1;
            }
            text_input = arg;
        }
    }

    if (!serve_mode && text_input.empty()) {
        fprintf(stderr, "Error: provide text or use --serve for server mode\n");
        return 1;
    }

    // --- Auto-download missing or corrupt files ---
    {
        struct Download { std::string path; std::string url; std::string label; size_t expected_size; };
        std::vector<Download> needed;
        needed.push_back({weights_path, release_url(default_weights_filename()),
                          "weights", default_weights_size()});
        needed.push_back({g2p_path, release_url(G2P_FILENAME), "g2p", G2P_SIZE});
        for (size_t i = 0; i < std::size(VOICE_NAMES); i++)
            needed.push_back({voices_dir + "/" + VOICE_NAMES[i] + ".bin",
                              release_url(std::string(VOICE_NAMES[i]) + ".bin"),
                              std::string("voice ") + VOICE_NAMES[i], VOICE_SIZE});

        for (auto& f : needed) {
            if (!file_ok(f.path, f.expected_size)) {
                fprintf(stderr, "%s not found at %s — downloading...\n", f.label.c_str(), f.path.c_str());
                if (!download_file(f.url, f.path)) {
                    fprintf(stderr, "Error: failed to download %s\n", f.label.c_str());
                    return 1;
                }
            }
        }
    }

    using clk = std::chrono::high_resolution_clock;
    auto t_start = clk::now();
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // --- Load TTS weights (prefetch in background) + init CUDA ---
    Weights prefetched;
    std::thread prefetch_thread([&]() {
        prefetched = Weights::prefetch(weights_path);
    });
    cudaFree(0); // lazy CUDA init
    prefetch_thread.join();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    prefetched.upload(stream);

    // --- Load G2P model (overlapped with weight norm precomputation) ---
    G2PModelCuda g2p;
    bool g2p_ok = false;
    cudaStream_t g2p_stream;
    CUDA_CHECK(cudaStreamCreate(&g2p_stream));
    std::thread g2p_thread([&]() {
        g2p_ok = g2p.load(g2p_path.c_str(), g2p_stream);
    });
    precompute_weight_norms(prefetched, stream);
    g2p_thread.join();
    CUDA_CHECK(cudaStreamDestroy(g2p_stream));
    if (!g2p_ok) {
        fprintf(stderr, "Error: failed to load G2P model from %s\n", g2p_path.c_str());
        return 1;
    }

    auto t_init = clk::now();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    vlog("GPU: %s | Init: %.0f ms | TTS: %.0f MB | G2P: %.1f MB\n",
         prop.name, ms(t_start, t_init),
         prefetched.gpu_data_size / 1e6, g2p.param_bytes() / 1e6);

    // --- Voice map ---
    std::vector<VoiceMmap> voice_mmaps;
    VoiceMap voices = load_voices(voices_dir, voice_mmaps);
    if (voices.empty()) {
        fprintf(stderr, "Error: no voices found in %s\n", voices_dir.c_str());
        return 1;
    }
    vlog("Voices: %zu loaded from %s\n", voices.size(), voices_dir.c_str());

    // --- Pre-allocate arenas + workspace ---
    GpuArena encode_arena;
    encode_arena.init(ENCODE_ARENA_BYTES);
    GpuArena decode_arena;  // starts empty, grows on first use
    float* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, WORKSPACE_BYTES));

    // --- Pre-warm: populate Cutlass operator caches + CUDA graphs ---
    {
        auto vit = voices.find("af_heart");
        if (vit != voices.end()) {
            const float* voice_data = reinterpret_cast<const float*>(vit->second.start);
            const float* style = voice_data;  // row 0
            auto warmup_ipa = g2p.infer("Warmup.", stream);
            auto warmup_tokens = to_tokens(warmup_ipa.empty() ? "." : warmup_ipa);
            rokoko_infer(prefetched, warmup_tokens.data(), (int)warmup_tokens.size(),
                         style, stream, encode_arena, decode_arena,
                         d_workspace, WORKSPACE_BYTES);
            cudaStreamSynchronize(stream);
        }
        auto t_warm = clk::now();
        vlog("Pre-warm: %.0f ms\n", ms(t_init, t_warm));
    }

    vlog("Encode arena: %.0f MB | Workspace: %.0f MB | Decode arena: on demand\n",
         ENCODE_ARENA_BYTES / 1e6, WORKSPACE_BYTES / 1e6);

    // =======================================================================
    // Server mode
    // =======================================================================
    if (serve_mode) {
        TtsPipeline pipeline{prefetched, g2p, stream,
                             encode_arena, decode_arena, d_workspace, WORKSPACE_BYTES, voices};

        run_server(pipeline, serve_host, serve_port);

        // Cleanup (unreachable unless server stops)
        cudaFree(d_workspace);
        decode_arena.destroy();
        encode_arena.destroy();
        for (auto& vm : voice_mmaps)
            if (vm.ptr) munmap(vm.ptr, vm.size);
        g2p.free();
        prefetched.free();
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    }

    // =======================================================================
    // CLI mode (original behavior)
    // =======================================================================

    // --- Preprocess text ---
    auto t_pre0 = clk::now();
    std::string preprocessed = text_norm::preprocess_text(text_input);
    auto t_pre1 = clk::now();
    vlog("Preprocess: \"%s\" (%.1f ms)\n", preprocessed.c_str(), ms(t_pre0, t_pre1));

    // --- G2P infer ---
    auto t_g2p0 = clk::now();
    std::string ipa = g2p.infer(preprocessed, stream);
    auto t_g2p1 = clk::now();
    vlog("G2P: \"%s\" (%.1f ms)\n", ipa.c_str(), ms(t_g2p0, t_g2p1));

    if (ipa.empty()) {
        fprintf(stderr, "Error: G2P produced no output\n");
        return 1;
    }

    // --- Tokenize + chunk ---
    auto chunks = chunk_ipa(ipa);
    vlog("Chunks: %zu (total %zu IPA codepoints)\n", chunks.size(), detail::utf8_len(ipa));

    // --- Voice pack ---
    auto vit = voices.find(voice_name);
    if (vit == voices.end()) {
        fprintf(stderr, "Error: unknown voice '%s'\n", voice_name.c_str());
        return 1;
    }
    const float* voice_data = reinterpret_cast<const float*>(vit->second.start);
    size_t voice_size = vit->second.end - vit->second.start;
    int voice_rows = voice_size / (256 * sizeof(float));

    // --- TTS infer per chunk ---
    std::vector<float> all_audio;

    for (size_t c = 0; c < chunks.size(); c++) {
        auto& chunk = chunks[c];
        int T = (int)chunk.tokens.size();

        vlog("Chunk %zu: \"%s\" (%d tokens)\n", c, chunk.phonemes.c_str(), T);

        // Style vector: index by phoneme count (T - 2 excludes BOS/EOS)
        int phoneme_count = T - 2;
        if (phoneme_count < 0) phoneme_count = 0;
        if (phoneme_count >= voice_rows) phoneme_count = voice_rows - 1;
        const float* style = voice_data + phoneme_count * 256;

        auto t0 = clk::now();
        auto audio = rokoko_infer(prefetched, chunk.tokens.data(), T, style,
                                   stream, encode_arena,
                                   decode_arena, d_workspace, WORKSPACE_BYTES);
        auto t1 = clk::now();

        double gen_ms = ms(t0, t1);
        double audio_sec = audio.size() / 24000.0;
        double rtfx = audio_sec / (gen_ms / 1000.0);
        vlog("  Generated %.3f sec in %.1f ms (%.0fx realtime), decode arena=%.1f MB\n",
             audio_sec, gen_ms, rtfx, decode_arena.offset/1e6);

        all_audio.insert(all_audio.end(), audio.begin(), audio.end());
    }

    cudaFree(d_workspace);
    decode_arena.destroy();
    encode_arena.destroy();

    // --- Write output ---
    if (say_mode) {
        if (!play_wav(all_audio.data(), all_audio.size(), 24000)) return 1;
        vlog("Played %.3f sec (%zu samples)\n",
             all_audio.size() / 24000.0, all_audio.size());
    } else {
        write_wav(output_path, all_audio.data(), all_audio.size(), 24000);
        vlog("Output: %s (%.3f sec, %zu samples)\n",
             output_path.c_str(), all_audio.size() / 24000.0, all_audio.size());
    }

    // Cleanup
    for (auto& vm : voice_mmaps)
        if (vm.ptr) munmap(vm.ptr, vm.size);
    g2p.free();
    prefetched.free();
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
