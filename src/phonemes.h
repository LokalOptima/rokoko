// CPU-only phoneme vocabulary, tokenization and chunking.
#pragma once

#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rokoko {

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

// Reject overlong encodings, surrogates, invalid continuations and truncation.
static inline uint32_t utf8_decode(const std::string& s, size_t& i) {
    if (i >= s.size()) throw std::invalid_argument("invalid UTF-8");
    uint8_t c = (uint8_t)s[i++];
    if (c < 0x80) return c;
    int extra; uint32_t cp, minimum;
    if (c >= 0xC2 && c <= 0xDF) { extra=1; cp=c&0x1F; minimum=0x80; }
    else if (c >= 0xE0 && c <= 0xEF) { extra=2; cp=c&0x0F; minimum=0x800; }
    else if (c >= 0xF0 && c <= 0xF4) { extra=3; cp=c&0x07; minimum=0x10000; }
    else throw std::invalid_argument("invalid UTF-8");
    for (int j=0; j<extra; ++j) {
        if (i >= s.size() || ((uint8_t)s[i]&0xC0) != 0x80)
            throw std::invalid_argument("invalid UTF-8");
        cp=(cp<<6)|((uint8_t)s[i++]&0x3F);
    }
    if (cp < minimum || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
        throw std::invalid_argument("invalid UTF-8");
    return cp;
}

static inline bool is_space(uint32_t c) {
    return c == 0x20 || (c >= 9 && c <= 13) || c == 0x85 || c == 0xA0 ||
        c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
        c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
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

struct TextSpan { size_t begin, end; }; // UTF-8 byte offsets into the source
struct Chunk {
    std::string phonemes;
    std::vector<int32_t> tokens;
    size_t begin = 0, end = 0;
};

// Split in codepoints, preserving all non-whitespace content and source spans.
// Prefer sentence punctuation, then clauses, then whitespace; long words use a hard split.
static inline std::vector<TextSpan> split_spans(const std::string& text, size_t limit) {
    if (!limit) throw std::invalid_argument("split limit must be positive");
    std::vector<uint32_t> cps;
    std::vector<size_t> offsets;
    for (size_t i=0; i<text.size();) {
        offsets.push_back(i);
        cps.push_back(detail::utf8_decode(text, i));
    }
    offsets.push_back(text.size());
    std::vector<TextSpan> result;
    for (size_t begin=0; begin<cps.size();) {
        while (begin<cps.size() && detail::is_space(cps[begin])) ++begin;
        if (begin==cps.size()) break;
        size_t end=std::min(cps.size(), begin+limit);
        if (end<cps.size()) {
            bool found=false;
            for (int priority=0; priority<3 && !found; ++priority) {
                for (size_t j=end; j>begin; --j) {
                    uint32_t c=cps[j-1];
                    bool match = priority==0 ? (c=='.'||c=='!'||c=='?'||c==0x2026) :
                        priority==1 ? (c==','||c==':'||c==';'||c==0x2014) : detail::is_space(c);
                    if (match) { end=j; found=true; break; }
                }
            }
        }
        size_t trimmed=end;
        while (trimmed>begin && detail::is_space(cps[trimmed-1])) --trimmed;
        if (trimmed>begin) result.push_back({offsets[begin], offsets[trimmed]});
        begin=end;
    }
    return result;
}

static inline std::vector<Chunk> chunk_ipa(const std::string& ipa) {
    std::vector<Chunk> chunks;
    for (auto span : split_spans(ipa, MAX_IPA_CHARS)) {
        auto piece=ipa.substr(span.begin,span.end-span.begin);
        chunks.push_back({piece,to_tokens(piece),span.begin,span.end});
    }
    return chunks;
}

static inline size_t style_row(const std::string& phonemes, size_t rows) {
    size_t n=detail::utf8_len(phonemes);
    if (!n || n>MAX_IPA_CHARS || n>rows)
        throw std::invalid_argument("phoneme length outside voice pack");
    return n - 1; // Kokoro pipeline: length BEFORE unknown-symbol filtering
}

static inline bool has_speech_tokens(const std::vector<int32_t>& tokens) {
    return std::any_of(tokens.begin(),tokens.end(),[](int32_t id) { return id>=24 && id<=148; });
}

} // namespace rokoko
