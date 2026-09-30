// normalize_cli.cpp — CLI wrapper for normalize.h
// Reads lines from stdin, writes preprocess_text() output to stdout.
// Built by `make test-frontend` (see tests/frontend/README.md).

#include <iostream>
#include <string>
#include "normalize.h"

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::cout << text_norm::preprocess_text(line) << '\n';
    }
    return 0;
}
