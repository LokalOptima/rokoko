// A thin CPU-only runner over the production helpers. No expected values here.
#include "phonemes.h"
#include <iostream>
#include <iterator>
#include <iomanip>
#include <stdexcept>
int main(int argc, char** argv) {
    try {
        std::string mode = argc > 1 ? argv[1] : "tokens";
        std::string input((std::istreambuf_iterator<char>(std::cin)), {});
        if (mode == "vocab") {
            for (const auto& e : rokoko::detail::VOCAB)
                std::cout << e.codepoint << '\t' << e.token_id << '\n';
        } else if (mode == "tokens") {
            for (auto id : rokoko::to_tokens(input)) std::cout << id << ' ';
            std::cout << '\n';
        } else if (mode == "chunks") {
            for (const auto& chunk : rokoko::chunk_ipa(input)) {
                std::cout << std::dec << chunk.begin << " " << chunk.end << " ";
                for (unsigned char c : chunk.phonemes)
                    std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)c;
                std::cout << '\n';
            }
        } else if (mode == "style") {
            std::cout << rokoko::style_row(input, argc>2 ? std::stoul(argv[2]) : 510) << '\n';
        } else if (mode == "split") {
            for (auto span : rokoko::split_spans(input, std::stoul(argv[2])))
                std::cout << span.begin << ' ' << span.end << '\n';
        } else { throw std::invalid_argument("unknown helper mode"); }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
