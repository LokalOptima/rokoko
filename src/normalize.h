#pragma once
// text_normalize.h — Port of Python's preprocess_text() from scripts/g2p/train.py.
// Single-header C++17 implementation for standalone G2P test binaries.
//
// Pipeline (must match Python exactly):
//   1. Money:   $12.50 → twelve dollars and fifty cents
//   2. Dates:   1/15/2024 → January fifteenth, twenty twenty four
//   3. Numbers: 1234 → one thousand two hundred thirty four (standalone only)
//   4-5. Unicode → ASCII: café → cafe, strip non-printable

#include <string>
#include <regex>
#include <cstdint>
#include <cctype>

namespace text_norm {

// ── Constants ────────────────────────────────────────────────────────────────

static const char* ONES[] = {
    "", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen"
};

static const char* TENS[] = {
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety"
};

static const char* MONTH_NAMES[] = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
};

static const char* DAY_ORDINALS[] = {
    "", "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
    "twenty first", "twenty second", "twenty third", "twenty fourth",
    "twenty fifth", "twenty sixth", "twenty seventh", "twenty eighth",
    "twenty ninth", "thirtieth", "thirty first"
};

// ── Number-to-words ─────────────────────────────────────────────────────────

inline std::string two_digit(int n) {
    if (n < 20) return ONES[n];
    std::string s = TENS[n / 10];
    if (n % 10) { s += ' '; s += ONES[n % 10]; }
    return s;
}

inline std::string number_to_words(int n) {
    if (n == 0) return "zero";
    if (n < 0) return "minus " + number_to_words(-n);
    std::string parts;
    auto append = [&](const std::string& p) {
        if (!parts.empty()) parts += ' ';
        parts += p;
    };
    if (n >= 1000000) {
        append(number_to_words(n / 1000000) + " million");
        n %= 1000000;
    }
    if (n >= 1000) {
        append(number_to_words(n / 1000) + " thousand");
        n %= 1000;
    }
    if (n >= 100) {
        append(std::string(ONES[n / 100]) + " hundred");
        n %= 100;
    }
    if (n > 0) {
        append(two_digit(n));
    }
    return parts;
}

inline std::string year_to_words(int year) {
    if (year == 2000) return "two thousand";
    if (year >= 2001 && year <= 2009)
        return std::string("two thousand and ") + ONES[year - 2000];
    if (year >= 2010 && year <= 2099)
        return std::string("twenty ") + two_digit(year - 2000);
    int hi = year / 100, lo = year % 100;
    std::string hi_words = two_digit(hi);
    if (lo == 0) return hi_words + " hundred";
    if (lo < 10) return hi_words + " oh " + ONES[lo];
    return hi_words + " " + two_digit(lo);
}

// ── UTF-8 helpers ───────────────────────────────────────────────────────────

inline int utf8_byte_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

inline uint32_t decode_utf8(const std::string& s, size_t& i) {
    uint8_t c = s[i];
    if (c < 0x80) { i++; return c; }
    uint32_t cp; int extra;
    if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
    else { i++; return 0xFFFD; }
    i++;
    for (int j = 0; j < extra && i < s.size(); j++, i++) {
        if ((s[i] & 0xC0) != 0x80) return 0xFFFD;
        cp = (cp << 6) | (s[i] & 0x3F);
    }
    return cp;
}

// Check if byte at position is alpha or part of a multi-byte UTF-8 character
// (matches Python's str.isalpha() which returns True for accented letters)
inline bool is_alpha_or_multibyte(const std::string& s, size_t pos) {
    return pos < s.size() && (std::isalpha((unsigned char)s[pos]) || (uint8_t)s[pos] >= 0x80);
}

// ── 1. Money expansion ──────────────────────────────────────────────────────

struct CurrencyInfo { const char* sing; const char* plur; const char* csing; const char* cplur; };

inline int match_currency(const std::string& s, size_t i, CurrencyInfo& info) {
    if (i >= s.size()) return 0;
    if (s[i] == '$') {
        info = {"dollar", "dollars", "cent", "cents"};
        return 1;
    }
    if (i + 2 < s.size() && (uint8_t)s[i] == 0xE2 && (uint8_t)s[i+1] == 0x82 && (uint8_t)s[i+2] == 0xAC) {
        info = {"euro", "euros", "cent", "cents"};
        return 3;
    }
    if (i + 1 < s.size() && (uint8_t)s[i] == 0xC2 && (uint8_t)s[i+1] == 0xA3) {
        info = {"pound", "pounds", "penny", "pence"};
        return 2;
    }
    if (i + 1 < s.size() && (uint8_t)s[i] == 0xC2 && (uint8_t)s[i+1] == 0xA5) {
        info = {"yen", "yen", "", ""};
        return 2;
    }
    return 0;
}

inline std::string expand_money(const std::string& text) {
    std::string result;
    size_t i = 0;
    while (i < text.size()) {
        CurrencyInfo info;
        int sym_len = match_currency(text, i, info);
        if (sym_len == 0) { result += text[i++]; continue; }

        // Look ahead for optional whitespace + digits
        size_t j = i + sym_len;
        while (j < text.size() && std::isspace((unsigned char)text[j])) j++;
        if (j >= text.size() || !std::isdigit((unsigned char)text[j])) {
            for (int k = 0; k < sym_len; k++) result += text[i + k];
            i += sym_len;
            continue;
        }

        // Scan amount: \d[\d,]*(\.\d+)?
        size_t num_start = j;
        j++;
        while (j < text.size() && (std::isdigit((unsigned char)text[j]) || text[j] == ',')) j++;

        // Optional decimal
        if (j < text.size() && text[j] == '.' && j + 1 < text.size() && std::isdigit((unsigned char)text[j+1])) {
            j++; // past '.'
            while (j < text.size() && std::isdigit((unsigned char)text[j])) j++;
        }

        // Parse
        std::string clean;
        for (size_t k = num_start; k < j; k++) if (text[k] != ',') clean += text[k];

        long long whole = 0;
        int cents = 0;
        size_t dot_pos = clean.find('.');
        bool ok = true;
        if (dot_pos != std::string::npos) {
            try {
                whole = dot_pos > 0 ? std::stoll(clean.substr(0, dot_pos)) : 0;
                std::string cent_str = (clean.substr(dot_pos + 1) + "00").substr(0, 2);
                cents = std::stoi(cent_str);
            } catch (...) { ok = false; }
        } else {
            try { whole = std::stoll(clean); } catch (...) { ok = false; }
        }

        if (!ok || whole < 0 || whole > 999999999) {
            for (int k = 0; k < sym_len; k++) result += text[i + k];
            i += sym_len;
            continue;
        }

        // Build expansion
        std::string expansion;
        if (whole > 0) {
            expansion = number_to_words((int)whole) + " ";
            expansion += (whole == 1) ? info.sing : info.plur;
        }
        if (cents > 0 && info.csing[0] != '\0') {
            if (!expansion.empty()) expansion += " and ";
            expansion += number_to_words(cents) + " ";
            expansion += (cents == 1) ? info.csing : info.cplur;
        }
        if (expansion.empty()) expansion = std::string("zero ") + info.plur;

        result += expansion;
        i = j;
    }
    return result;
}

// ── 2. Date expansion ───────────────────────────────────────────────────────

inline std::string expand_date(int month, int day, int year) {
    if (month < 1 || month > 12 || day < 1 || day > 31 || year < 1000 || year > 2099)
        return "";
    return std::string(MONTH_NAMES[month - 1]) + " " + DAY_ORDINALS[day] +
           ", " + year_to_words(year);
}

inline std::string regex_replace_cb(const std::string& s, std::regex& re,
                                     std::string (*cb)(const std::smatch&)) {
    std::string out;
    std::sregex_iterator it(s.begin(), s.end(), re), end;
    size_t last = 0;
    for (; it != end; ++it) {
        auto& m = *it;
        out.append(s, last, m.position() - last);
        out += cb(m);
        last = m.position() + m.length();
    }
    out.append(s, last, s.size() - last);
    return out;
}

inline std::string expand_dates(const std::string& text) {
    std::string s = text;

    // US: M/D/YYYY
    {
        static std::regex re(R"(\b(\d{1,2})/(\d{1,2})/(\d{4})\b)");
        s = regex_replace_cb(s, re, [](const std::smatch& m) -> std::string {
            std::string r = expand_date(std::stoi(m[1]), std::stoi(m[2]), std::stoi(m[3]));
            return r.empty() ? m[0].str() : r;
        });
    }

    // ISO: YYYY-MM-DD
    {
        static std::regex re(R"(\b(\d{4})-(\d{2})-(\d{2})\b)");
        s = regex_replace_cb(s, re, [](const std::smatch& m) -> std::string {
            std::string r = expand_date(std::stoi(m[2]), std::stoi(m[3]), std::stoi(m[1]));
            return r.empty() ? m[0].str() : r;
        });
    }

    // Textual: Month DD, YYYY
    {
        static std::regex re(
            R"(\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b)");
        s = regex_replace_cb(s, re, [](const std::smatch& m) -> std::string {
            std::string name = m[1];
            int month = 0;
            for (int i = 0; i < 12; i++)
                if (name == MONTH_NAMES[i]) { month = i + 1; break; }
            std::string r = expand_date(month, std::stoi(m[2]), std::stoi(m[3]));
            return r.empty() ? m[0].str() : r;
        });
    }

    return s;
}

// ── 3. Number expansion ─────────────────────────────────────────────────────

inline std::string expand_numbers(const std::string& text) {
    std::string result;
    size_t i = 0;
    while (i < text.size()) {
        if (!std::isdigit((unsigned char)text[i])) { result += text[i++]; continue; }

        // Scan: \d[\d,]*
        size_t start = i;
        i++;
        while (i < text.size() && (std::isdigit((unsigned char)text[i]) || text[i] == ',')) i++;

        // Skip if adjacent to alpha/multibyte (70s, 3G, MP3, 5kg)
        bool adj_before = start > 0 && is_alpha_or_multibyte(text, start - 1);
        bool adj_after = i < text.size() && is_alpha_or_multibyte(text, i);
        if (adj_before || adj_after) {
            result.append(text, start, i - start);
            continue;
        }

        // Parse and expand
        std::string clean;
        for (size_t k = start; k < i; k++) if (text[k] != ',') clean += text[k];
        try {
            long long n = std::stoll(clean);
            if (n >= 0 && n <= 999999999)
                result += number_to_words((int)n);
            else
                result.append(text, start, i - start);
        } catch (...) {
            result.append(text, start, i - start);
        }
    }
    return result;
}

// ── 4-5. Unicode → ASCII (matching unidecode) ──────────────────────────────

inline const char* unidecode_codepoint(uint32_t cp) {
    // Latin-1 Supplement (U+00A0-U+00FF)
    if (cp >= 0x00A0 && cp <= 0x00FF) {
        static const char* table[96] = {
            " ",  "!",  "C/", "PS", "$?", "Y=", "|",  "SS", // A0-A7
            "\"", "(c)","a",  "<<", "!",  "",   "(r)","-",  // A8-AF
            "deg","+-", "2",  "3",  "'",  "u",  "P",  "*",  // B0-B7
            ",",  "1",  "o",  ">>", " 1/4"," 1/2"," 3/4","?",// B8-BF
            "A",  "A",  "A",  "A",  "A",  "A",  "AE", "C",  // C0-C7
            "E",  "E",  "E",  "E",  "I",  "I",  "I",  "I",  // C8-CF
            "D",  "N",  "O",  "O",  "O",  "O",  "O",  "x",  // D0-D7
            "O",  "U",  "U",  "U",  "U",  "Y",  "Th", "ss", // D8-DF
            "a",  "a",  "a",  "a",  "a",  "a",  "ae", "c",  // E0-E7
            "e",  "e",  "e",  "e",  "i",  "i",  "i",  "i",  // E8-EF
            "d",  "n",  "o",  "o",  "o",  "o",  "o",  "/",  // F0-F7
            "o",  "u",  "u",  "u",  "u",  "y",  "th", "y",  // F8-FF
        };
        return table[cp - 0x00A0];
    }

    // Latin Extended-A (U+0100-U+017F)
    if (cp >= 0x0100 && cp <= 0x017F) {
        static const char* table[128] = {
            "A","a","A","a","A","a","C","c","C","c","C","c","C","c","D","d", // 100-10F
            "D","d","E","e","E","e","E","e","E","e","E","e","G","g","G","g", // 110-11F
            "G","g","G","g","H","h","H","h","I","i","I","i","I","i","I","i", // 120-12F
            "I","i","IJ","ij","J","j","K","k","k","L","l","L","l","L","l",   // 130-13E
            "L","l","L","l","N","n","N","n","N","n","'n","NG","ng",           // 13F-14B
            "O","o","O","o","O","o","OE","oe",                               // 14C-153
            "R","r","R","r","R","r",                                         // 154-159
            "S","s","S","s","S","s","S","s",                                 // 15A-161
            "T","t","T","t","T","t",                                         // 162-167
            "U","u","U","u","U","u","U","u","U","u","U","u",                 // 168-173
            "W","w","Y","y","Y","Z","z","Z","z","Z","z","s",                 // 174-17F
        };
        return table[cp - 0x0100];
    }

    // Common punctuation & symbols
    switch (cp) {
        case 0x2013: return "-";     // en dash
        case 0x2014: return "--";    // em dash
        case 0x2018: return "'";     // left single quote
        case 0x2019: return "'";     // right single quote
        case 0x201A: return ",";     // single low-9 quote
        case 0x201C: return "\"";    // left double quote
        case 0x201D: return "\"";    // right double quote
        case 0x201E: return ",,";    // double low-9 quote
        case 0x2026: return "...";   // ellipsis
        case 0x20AC: return "EUR";   // euro sign (only if not already expanded by money step)
        case 0x2122: return "TM";    // trademark
    }

    return nullptr; // unknown → will be stripped
}

inline std::string unicode_to_ascii(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        uint8_t c = text[i];
        if (c < 0x80) {
            if (c >= 32 && c < 127) result += (char)c;
            i++;
            continue;
        }
        uint32_t cp = decode_utf8(text, i);
        const char* rep = unidecode_codepoint(cp);
        if (rep) {
            for (const char* p = rep; *p; p++)
                if (*p >= 32 && *p < 127) result += *p;
        }
    }
    return result;
}

// ── Top-level ────────────────────────────────────────────────────────────────

inline std::string preprocess_text(const std::string& text) {
    std::string s = expand_money(text);
    s = expand_dates(s);
    s = expand_numbers(s);
    s = unicode_to_ascii(s);
    return s;
}

} // namespace text_norm
