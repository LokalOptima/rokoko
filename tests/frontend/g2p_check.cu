// Runs rokoko's own C++ G2P (src/g2p.h) on stdin lines, one phoneme string per line.
//   tests/frontend/g2p_check model.bin [--normalize] < lines.txt
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include "device.h"
#ifdef ROKOKO_CPU
#include "cpu/math.h"
#endif

bool g_verbose = false;
#define vlog(...) do { if (g_verbose) fprintf(stderr, __VA_ARGS__); } while (0)

#include "g2p.h"
#include "normalize.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.bin [--normalize]\n", argv[0]); return 1; }
    bool normalize = argc > 2 && std::strcmp(argv[2], "--normalize") == 0;
#ifdef ROKOKO_CPU
    rokoko::cpu::configure();
#endif
    rokoko::Stream stream;
    rokoko::device::create_stream(&stream);
    rokoko::G2PModel g2p;
    if (!g2p.load(argv[1], stream)) { fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }
    std::string line;
    while (std::getline(std::cin, line)) {
        std::string text = normalize ? text_norm::preprocess_text(line) : line;
        printf("%s\n", text.empty() ? "" : g2p.infer(text, stream).c_str());
    }
    g2p.free();
    rokoko::device::destroy_stream(stream);
    return 0;
}
