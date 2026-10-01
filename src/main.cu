// main.cu — Neural G2P + TTS in one binary
//
// Pipeline: text → preprocess → G2P infer → tokenize → chunk → TTS infer → WAV
//
// Build: make rokoko
// Usage: ./rokoko "Hello world." -o output.wav
//        ./rokoko --serve 8080

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "rokoko.h"
#include "server.h"

bool g_verbose = false;

using namespace rokoko;

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
            "  -o <file>           Output WAV (default: output.wav)\n"
            "  --phonemes          Treat input as IPA (bypass normalization/G2P)\n"
            "  --say               Play audio through speakers\n"
            "  --stdout            Write WAV to stdout\n"
            "  --serve [port]      HTTP server with web UI (default: 8080)\n"
            "  --host <addr>       Server bind address (default: 0.0.0.0)\n"
            "  -v                  Verbose output (timings, IPA, GPU info)\n"
            "  --build-info        Print embedded asset identities (JSON)\n"
            "  --help              Show this help\n"
            "\n"
            "Examples:\n"
            "  %s \"Hello world.\" -o hello.wav\n"
            "  %s \"Hello world.\" --say\n"
            "  %s \"Hello world.\" --stdout | aplay\n"
            "  %s --serve 8080\n",
            argv[0], argv[0],
            argv[0], argv[0], argv[0], argv[0]);
    };

    if (argc < 2) { print_usage(); return 1; }

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--build-info") {
            auto info=embedded_build_info();
            return fwrite(info.data,1,info.size,stdout)==info.size ? 0 : 1;
        }
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            print_usage(); return 0;
        }
    }

    std::string text_input;
    std::string output_path = "output.wav";
    bool say_mode = false, phonemes_mode = false;
    bool serve_mode = false;
    int serve_port = 8080;
    std::string serve_host = "0.0.0.0";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--phonemes") phonemes_mode = true;
        else if (arg == "--stdout")                    output_path = "-";
        else if (arg == "--say")                      say_mode = true;
        else if (arg == "-v" || arg == "--verbose")   g_verbose = true;
        else if (arg == "--host" && i + 1 < argc)     serve_host = argv[++i];
        else if (arg == "--serve") {
            serve_mode = true;
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
                serve_port = std::atoi(argv[++i]);
        }
        else if (arg.empty() || arg[0] != '-') {
            if (!text_input.empty()) {
                fprintf(stderr, "Error: unexpected argument '%s' (text already set)\n", arg.c_str());
                return 1;
            }
            text_input = arg;
        } else { fprintf(stderr,"Error: unknown option or missing value: %s\n",arg.c_str()); return 1; }
    }

    if (!serve_mode && text_input.empty()) {
        fprintf(stderr, "Error: provide text or use --serve for server mode\n");
        return 1;
    }

    TtsContext ctx;
    if (!ctx.init()) {
        fprintf(stderr,"Error: %s\n",ctx.last_error.c_str());return 1;
    }
    auto pipeline=ctx.pipeline();
    if (serve_mode) { run_server(pipeline,serve_host,serve_port); return 0; }
    std::vector<float> audio;
    auto err=pipeline.synthesize(text_input,audio,phonemes_mode);
    if (!err.empty()) { fprintf(stderr,"Error: %s\n",err.c_str());return 1; }
    bool ok=say_mode?play_wav(audio.data(),audio.size(),24000):write_wav(output_path,audio.data(),audio.size(),24000);
    if (!ok) { fprintf(stderr,"Error: could not write audio output\n");return 1; }
    return 0;
}
