#pragma once
// normalize.h — Single-pass text normalization for TTS G2P.
// No std::regex. Character-by-character scanning with inline pattern matchers.
//
// Pipeline (single pass, priority-ordered pattern dispatch):
//   Money > Dates > Time > Telephone > Fractions > Ordinals > Percent >
//   Decimals > Numbers > Symbols > Dotted Initialisms
// Unicode → ASCII transliteration is handled inline during the main scan.

#include <string>
#include <cstdint>
#include <cctype>
#include <cstring>

namespace text_norm {

// ── Word tables ─────────────────────────────────────────────────────────────

static const char* ONES[] = {
    "", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen"
};
static const char* TENS[] = {
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety"
};
static const char* DIGIT_WORDS[] = {
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
};
static const char* MONTH_NAMES[] = {
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
};
static const int MONTH_NAME_LEN[] = { 7, 8, 5, 5, 3, 4, 4, 6, 9, 7, 8, 8 };
static const char* MONTH_ABBREVS[] = {
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
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
// Ordinal words for ones place
static const char* ORD_ONES[] = {
    "", "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth"
};
static const char* ORD_TENS[] = {
    "", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
    "sixtieth", "seventieth", "eightieth", "ninetieth"
};
// Fraction denominator words (singular)
static const char* DENOM_SING[] = {
    "", "", "half", "third", "quarter", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth"
};
static const char* DENOM_PLUR[] = {
    "", "", "halves", "thirds", "quarters", "fifths", "sixths", "sevenths",
    "eighths", "ninths", "tenths", "elevenths", "twelfths", "thirteenths",
    "fourteenths", "fifteenths", "sixteenths", "seventeenths", "eighteenths",
    "nineteenths", "twentieths"
};

// ── Number-to-words (zero-allocation, write directly to output) ─────────────

inline void emit_two_digit(std::string& out, int n) {
    if (n < 20) { out.append(ONES[n]); return; }
    out.append(TENS[n / 10]);
    if (n % 10) { out += ' '; out.append(ONES[n % 10]); }
}

inline void emit_number_to_words(std::string& out, int n) {
    if (n == 0) { out.append("zero"); return; }
    if (n < 0) { out.append("minus "); emit_number_to_words(out, -n); return; }
    bool need_space = false;
    auto sep = [&]() { if (need_space) out += ' '; need_space = true; };
    if (n >= 1000000) { sep(); emit_number_to_words(out, n / 1000000); out.append(" million"); n %= 1000000; }
    if (n >= 1000)    { sep(); emit_number_to_words(out, n / 1000); out.append(" thousand"); n %= 1000; }
    if (n >= 100)     { sep(); out.append(ONES[n / 100]); out.append(" hundred"); n %= 100; }
    if (n > 0)        { sep(); emit_two_digit(out, n); }
}

inline void emit_ordinal_words(std::string& out, int n) {
    if (n <= 0) { out.append("zeroth"); return; }
    if (n >= 1000000) {
        int rem = n % 1000000;
        emit_number_to_words(out, n / 1000000);
        if (rem == 0) { out.append(" millionth"); }
        else { out.append(" million "); emit_ordinal_words(out, rem); }
        return;
    }
    if (n >= 1000) {
        int rem = n % 1000;
        emit_number_to_words(out, n / 1000);
        if (rem == 0) { out.append(" thousandth"); }
        else { out.append(" thousand "); emit_ordinal_words(out, rem); }
        return;
    }
    if (n >= 100) {
        int rem = n % 100;
        out.append(ONES[n / 100]);
        if (rem == 0) { out.append(" hundredth"); }
        else { out.append(" hundred "); emit_ordinal_words(out, rem); }
        return;
    }
    if (n < 20) { out.append(ORD_ONES[n]); return; }
    if (n % 10 == 0) { out.append(ORD_TENS[n / 10]); return; }
    out.append(TENS[n / 10]); out += ' '; out.append(ORD_ONES[n % 10]);
}

inline void emit_year_to_words(std::string& out, int year) {
    if (year == 2000) { out.append("two thousand"); return; }
    if (year >= 2001 && year <= 2009) {
        out.append("two thousand and "); out.append(ONES[year - 2000]); return;
    }
    if (year >= 2010 && year <= 2099) {
        out.append("twenty "); emit_two_digit(out, year - 2000); return;
    }
    int hi = year / 100, lo = year % 100;
    emit_two_digit(out, hi);
    if (lo == 0) { out.append(" hundred"); }
    else if (lo < 10) { out.append(" oh "); out.append(ONES[lo]); }
    else { out += ' '; emit_two_digit(out, lo); }
}

// Bare 4-digit numbers in this range are read as years ("1969" → "nineteen
// sixty nine"). Round thousands stay cardinal ("2000" → "two thousand").
inline bool is_year_like(long long v) { return v >= 1001 && v <= 2099 && v % 1000 != 0; }

// "ninety" → "nineties", "hundred" → "hundreds" (for decades: "the 1990s").
inline void pluralize_last_word(std::string& out) {
    if (!out.empty() && out.back() == 'y') { out.pop_back(); out.append("ies"); }
    else out += 's';
}

// Denominators we read as fractions. Anything else ("9/11") is not a fraction.
inline bool is_common_denom(int d) {
    switch (d) {
        case 2: case 3: case 4: case 5: case 6: case 8: case 10:
        case 12: case 16: case 32: case 64: case 100: return true;
        default: return false;
    }
}

inline bool emit_denom_word(std::string& out, int d, bool plural) {
    if (d >= 2 && d <= 20) { out.append(plural ? DENOM_PLUR[d] : DENOM_SING[d]); return true; }
    if (d > 20 && d < 100 && d % 10 != 0) {
        out.append(TENS[d / 10]); out += ' '; out.append(ORD_ONES[d % 10]);
        if (plural) out += 's';
        return true;
    }
    if (d >= 30 && d <= 90 && d % 10 == 0) {
        out.append(ORD_TENS[d / 10]); if (plural) out += 's';
        return true;
    }
    if (d == 100) { out.append(plural ? "hundredths" : "hundredth"); return true; }
    return false;
}

// ── Scanner ─────────────────────────────────────────────────────────────────

struct Scanner {
    const char* s;
    size_t len;
    size_t pos;
    std::string out;

    Scanner(const std::string& src) : s(src.data()), len(src.size()), pos(0) {
        out.reserve(src.size() * 2);
    }

    // ── Peek helpers ────────────────────────────────────────────────────

    inline char at(size_t off) const {
        size_t i = pos + off;
        return i < len ? s[i] : '\0';
    }
    inline bool has(size_t off) const { return pos + off < len; }
    inline bool done() const { return pos >= len; }

    inline bool is_digit(size_t off) const { return has(off) && std::isdigit((unsigned char)s[pos + off]); }
    inline bool is_alpha(size_t off) const { return has(off) && std::isalpha((unsigned char)s[pos + off]); }
    inline bool is_alnum(size_t off) const { return has(off) && std::isalnum((unsigned char)s[pos + off]); }
    inline bool is_space(size_t off) const { return has(off) && std::isspace((unsigned char)s[pos + off]); }
    inline bool is_multibyte(size_t off) const { return has(off) && (uint8_t)s[pos + off] >= 0x80; }

    // Check if position is at word boundary (not preceded by alnum)
    inline bool word_start() const {
        return pos == 0 || (!std::isalnum((unsigned char)s[pos - 1]) && (uint8_t)s[pos - 1] < 0x80);
    }
    // Check if pos+off is at word end (not followed by alnum)
    inline bool word_end(size_t off) const {
        size_t i = pos + off;
        return i >= len || (!std::isalnum((unsigned char)s[i]) && (uint8_t)s[i] < 0x80);
    }

    // Match a string literal at pos+off
    inline bool match_str(size_t off, const char* lit) const {
        size_t i = pos + off;
        for (; *lit; ++lit, ++i) {
            if (i >= len || s[i] != *lit) return false;
        }
        return true;
    }
    // Case-insensitive match
    inline bool match_str_ci(size_t off, const char* lit) const {
        size_t i = pos + off;
        for (; *lit; ++lit, ++i) {
            if (i >= len || std::tolower((unsigned char)s[i]) != std::tolower((unsigned char)*lit)) return false;
        }
        return true;
    }

    // ── Scan helpers ────────────────────────────────────────────────────

    // Scan N digits at offset, return value. Returns -1 if not enough digits.
    int scan_fixed_digits(size_t off, int count) const {
        int val = 0;
        for (int i = 0; i < count; i++) {
            if (!is_digit(off + i)) return -1;
            val = val * 10 + (at(off + i) - '0');
        }
        return val;
    }

    // Scan a run of 1+ digits at offset. Returns (value, count). count=0 on failure.
    struct DigitRun { long long val; int count; };
    DigitRun scan_digits(size_t off) const {
        long long val = 0;
        int count = 0;
        while (is_digit(off + count)) {
            val = val * 10 + (at(off + count) - '0');
            count++;
            if (val > 999999999LL) break; // overflow guard
        }
        return {val, count};
    }

    // Scan digits with commas: "1,234,567". Commas must be between digits.
    // Returns (cleaned value, total chars consumed). Stops before trailing comma.
    struct AmountResult { long long whole; int cents; int consumed; bool has_decimal; };
    AmountResult scan_amount(size_t off) const {
        AmountResult r = {0, 0, 0, false};
        if (!is_digit(off)) return r;
        int i = 0;
        // Integer part: digits and commas — compute value inline
        while (is_digit(off + i)) {
            r.whole = r.whole * 10 + (at(off + i) - '0');
            i++;
            if (at(off + i) == ',' && is_digit(off + i + 1)) {
                i++; // skip comma (followed by digit)
            }
        }
        // Optional decimal
        if (at(off + i) == '.' && is_digit(off + i + 1)) {
            r.has_decimal = true;
            i++; // skip dot
            int d0 = at(off + i) - '0'; i++;
            int d1 = is_digit(off + i) ? (at(off + i) - '0') : 0;
            if (is_digit(off + i)) i++;
            // Skip remaining fractional digits
            while (is_digit(off + i)) i++;
            r.cents = d0 * 10 + d1;
        }
        r.consumed = i;
        return r;
    }

    // Skip optional whitespace at offset, return chars skipped
    int skip_space(size_t off) const {
        int n = 0;
        while (is_space(off + n)) n++;
        return n;
    }

    // ── Emit helpers ────────────────────────────────────────────────────

    void emit(const char* p) { while (*p) out.push_back(*p++); }
    void emit(const std::string& s) { out.append(s); }
    void emit_char(char c) { out.push_back(c); }
    void emit_digit_words(size_t off, int count) {
        for (int i = 0; i < count; i++) {
            if (i > 0) out.push_back(' ');
            emit(DIGIT_WORDS[at(off + i) - '0']);
        }
    }
    void advance(size_t n) { pos += n; }
    void copy_char() { out.push_back(s[pos++]); }
};

// ── Currency detection ──────────────────────────────────────────────────────

struct CurrencyInfo {
    const char* sing; const char* plur;
    const char* csing; const char* cplur;
    int bytes; // byte length of the symbol
};

inline bool match_currency(const Scanner& sc, size_t off, CurrencyInfo& info) {
    uint8_t c = (uint8_t)sc.at(off);
    if (c == '$') {
        info = {"dollar", "dollars", "cent", "cents", 1};
        return true;
    }
    // € = E2 82 AC
    if (c == 0xE2 && (uint8_t)sc.at(off+1) == 0x82 && (uint8_t)sc.at(off+2) == 0xAC) {
        info = {"euro", "euros", "cent", "cents", 3};
        return true;
    }
    // £ = C2 A3
    if (c == 0xC2 && (uint8_t)sc.at(off+1) == 0xA3) {
        info = {"pound", "pounds", "penny", "pence", 2};
        return true;
    }
    // ¥ = C2 A5
    if (c == 0xC2 && (uint8_t)sc.at(off+1) == 0xA5) {
        info = {"yen", "yen", "", "", 2};
        return true;
    }
    return false;
}

// ── Pattern matchers ────────────────────────────────────────────────────────
// Each returns true if it matched and emitted output (advancing scanner).

inline bool try_money(Scanner& sc, const CurrencyInfo& ci) {
    size_t off = ci.bytes;
    // Skip optional whitespace after symbol
    int sp = sc.skip_space(off);
    off += sp;
    // Must have a digit
    if (!sc.is_digit(off)) return false;
    auto amt = sc.scan_amount(off);
    if (amt.consumed == 0 || amt.whole < 0 || amt.whole > 999999999) return false;

    // Reject if immediately followed by alpha (e.g. "$1m" is not "$1" + "m")
    size_t end = off + amt.consumed;
    if (sc.has(end) && std::isalpha((unsigned char)sc.s[sc.pos + end])) return false;

    // Scaled amounts: "$2.5 million" → "two point five million dollars"
    static const char* SCALES[] = {"thousand", "million", "billion", "trillion"};
    if (sc.at(end) == ' ') {
        for (const char* scale : SCALES) {
            size_t slen = std::strlen(scale);
            if (!sc.match_str(end + 1, scale) || !sc.word_end(end + 1 + slen)) continue;
            emit_number_to_words(sc.out, (int)amt.whole);
            if (amt.has_decimal) {
                sc.emit(" point");
                size_t j = off;
                while (sc.at(j) != '.') j++;
                for (j++; j < end; j++) { sc.emit_char(' '); sc.emit(DIGIT_WORDS[sc.at(j) - '0']); }
            }
            sc.emit_char(' '); sc.emit(scale);
            sc.emit_char(' '); sc.emit(ci.plur);
            sc.advance(end + 1 + slen);
            return true;
        }
    }

    // Build expansion
    bool has_whole = amt.whole > 0;
    bool has_cents = amt.cents > 0 && ci.csing[0] != '\0';
    if (!has_whole && !has_cents) {
        sc.emit("zero "); sc.emit(ci.plur);
    } else {
        if (has_whole) {
            emit_number_to_words(sc.out, (int)amt.whole);
            sc.emit_char(' ');
            sc.emit(amt.whole == 1 ? ci.sing : ci.plur);
        }
        if (has_cents) {
            if (has_whole) sc.emit(" and ");
            emit_number_to_words(sc.out, amt.cents);
            sc.emit_char(' ');
            sc.emit(amt.cents == 1 ? ci.csing : ci.cplur);
        }
    }
    sc.advance(ci.bytes + sp + amt.consumed);
    return true;
}

// ── Month matching ──────────────────────────────────────────────────────────

// Returns month number (1-12) and name length, or 0 if no match.
struct MonthMatch { int month; int len; };

inline MonthMatch match_month_full(const Scanner& sc, size_t off) {
    // First-character dispatch to avoid scanning all 12 months
    char c = sc.at(off);
    const int* candidates; int count;
    // J -> Jan(0), Jun(5), Jul(6); F -> Feb(1); M -> Mar(2), May(4)
    // A -> Apr(3), Aug(7); S -> Sep(8); O -> Oct(9); N -> Nov(10); D -> Dec(11)
    static const int cJ[] = {0, 5, 6}, cF[] = {1}, cM[] = {2, 4}, cA[] = {3, 7};
    static const int cS[] = {8}, cO[] = {9}, cN[] = {10}, cD[] = {11};
    switch (c) {
        case 'J': candidates = cJ; count = 3; break;
        case 'F': candidates = cF; count = 1; break;
        case 'M': candidates = cM; count = 2; break;
        case 'A': candidates = cA; count = 2; break;
        case 'S': candidates = cS; count = 1; break;
        case 'O': candidates = cO; count = 1; break;
        case 'N': candidates = cN; count = 1; break;
        case 'D': candidates = cD; count = 1; break;
        default: return {0, 0};
    }
    for (int j = 0; j < count; j++) {
        int i = candidates[j];
        if (sc.match_str(off, MONTH_NAMES[i])) {
            int mlen = MONTH_NAME_LEN[i];
            if (!sc.is_alpha(off + mlen))
                return {i + 1, mlen};
        }
    }
    return {0, 0};
}

inline MonthMatch match_month_abbrev(const Scanner& sc, size_t off) {
    // First-character dispatch
    char c = sc.at(off);
    const int* candidates; int count;
    static const int cJ[] = {0, 5, 6}, cF[] = {1}, cM[] = {2, 4}, cA[] = {3, 7};
    static const int cS[] = {8}, cO[] = {9}, cN[] = {10}, cD[] = {11};
    switch (c) {
        case 'J': candidates = cJ; count = 3; break;
        case 'F': candidates = cF; count = 1; break;
        case 'M': candidates = cM; count = 2; break;
        case 'A': candidates = cA; count = 2; break;
        case 'S': candidates = cS; count = 1; break;
        case 'O': candidates = cO; count = 1; break;
        case 'N': candidates = cN; count = 1; break;
        case 'D': candidates = cD; count = 1; break;
        default: return {0, 0};
    }
    for (int j = 0; j < count; j++) {
        int i = candidates[j];
        if (sc.match_str(off, MONTH_ABBREVS[i])) {
            int mlen = 3;
            if (sc.at(off + mlen) == '.') mlen++;
            if (sc.is_space(off + mlen))
                return {i + 1, mlen};
        }
    }
    return {0, 0};
}

// ── Date helpers ────────────────────────────────────────────────────────────

inline bool valid_date(int month, int day, int year) {
    return month >= 1 && month <= 12 && day >= 1 && day <= 31 && year >= 1000 && year <= 2099;
}

inline void emit_date(Scanner& sc, int month, int day, int year) {
    sc.emit(MONTH_NAMES[month - 1]);
    sc.emit_char(' ');
    sc.emit(DAY_ORDINALS[day]);
    sc.emit_char(' ');
    emit_year_to_words(sc.out, year);
}

inline bool try_date_dmy_sep(Scanner& sc, char sep) {
    if (!sc.word_start()) return false;
    auto d = sc.scan_digits(0);
    if (d.count < 1 || d.count > 2) return false;
    if (sc.at(d.count) != sep) return false;
    auto m = sc.scan_digits(d.count + 1);
    if (m.count < 1 || m.count > 2) return false;
    int off2 = d.count + 1 + m.count;
    if (sc.at(off2) != sep) return false;
    int y = sc.scan_fixed_digits(off2 + 1, 4);
    if (y < 0) return false;
    int total = off2 + 5;
    if (!sc.word_end(total)) return false;
    if (!valid_date((int)m.val, (int)d.val, y)) return false;
    emit_date(sc, (int)m.val, (int)d.val, y);
    sc.advance(total);
    return true;
}
// A/B/YYYY. Expanded only when the order is unambiguous (one part > 12):
// "2/18/1975" and "18/2/1975" are both February eighteenth. When both parts
// are <= 12 ("4/5/1926") we don't guess: "four slash five slash nineteen
// twenty six".
inline bool try_date_slash(Scanner& sc) {
    if (!sc.word_start()) return false;
    auto a = sc.scan_digits(0);
    if (a.count < 1 || a.count > 2 || sc.at(a.count) != '/') return false;
    auto b = sc.scan_digits(a.count + 1);
    if (b.count < 1 || b.count > 2) return false;
    int off2 = a.count + 1 + b.count;
    if (sc.at(off2) != '/') return false;
    int y = sc.scan_fixed_digits(off2 + 1, 4);
    if (y < 0) return false;
    int total = off2 + 5;
    if (!sc.word_end(total)) return false;
    int A = (int)a.val, B = (int)b.val;
    if (A > 12 && B <= 12) {
        if (!valid_date(B, A, y)) return false;
        emit_date(sc, B, A, y);
    } else if (B > 12 && A <= 12) {
        if (!valid_date(A, B, y)) return false;
        emit_date(sc, A, B, y);
    } else if (A <= 12 && B <= 12) {
        if (A < 1 || B < 1 || !valid_date(1, 1, y)) return false;
        emit_number_to_words(sc.out, A);
        sc.emit(" slash ");
        emit_number_to_words(sc.out, B);
        sc.emit(" slash ");
        emit_year_to_words(sc.out, y);
    } else {
        return false;
    }
    sc.advance(total);
    return true;
}

inline bool try_date_iso(Scanner& sc) {
    // YYYY-MM-DD
    if (!sc.word_start()) return false;
    int y = sc.scan_fixed_digits(0, 4);
    if (y < 0 || sc.at(4) != '-') return false;
    int m = sc.scan_fixed_digits(5, 2);
    if (m < 0 || sc.at(7) != '-') return false;
    int d = sc.scan_fixed_digits(8, 2);
    if (d < 0) return false;
    if (!sc.word_end(10)) return false;
    if (!valid_date(m, d, y)) return false;
    emit_date(sc, m, d, y);
    sc.advance(10);
    return true;
}

inline bool try_date_ymd_slash(Scanner& sc) {
    // YYYY/M/D or YYYY/MM/DD
    if (!sc.word_start()) return false;
    int y = sc.scan_fixed_digits(0, 4);
    if (y < 0 || sc.at(4) != '/') return false;
    auto m = sc.scan_digits(5);
    if (m.count < 1 || m.count > 2) return false;
    int off2 = 5 + m.count;
    if (sc.at(off2) != '/') return false;
    auto d = sc.scan_digits(off2 + 1);
    if (d.count < 1 || d.count > 2) return false;
    int total = off2 + 1 + d.count;
    if (!sc.word_end(total)) return false;
    if (!valid_date((int)m.val, (int)d.val, y)) return false;
    emit_date(sc, (int)m.val, (int)d.val, y);
    sc.advance(total);
    return true;
}

inline bool try_date_textual(Scanner& sc) {
    // "January 15, 2024" or "January 15 2024" or "Jan 15, 2024" or "Jan. 15, 2024"
    if (!sc.word_start()) return false;
    // Try full month name first, then abbreviation
    auto mm = match_month_full(sc, 0);
    if (mm.month == 0) mm = match_month_abbrev(sc, 0);
    if (mm.month == 0) return false;

    size_t off = mm.len;
    off += sc.skip_space(off);
    if (!sc.is_digit(off)) return false;
    auto day = sc.scan_digits(off);
    if (day.count < 1 || day.count > 2 || day.val < 1 || day.val > 31) return false;
    off += day.count;
    size_t day_end = off;
    // Optional comma, then space and a 4-digit year
    if (sc.at(off) == ',') off++;
    int sp = sc.skip_space(off);
    int year = sp > 0 ? sc.scan_fixed_digits(off + sp, 4) : -1;
    if (year >= 0 && sc.word_end(off + sp + 4) && valid_date(mm.month, (int)day.val, year)) {
        emit_date(sc, mm.month, (int)day.val, year);
        sc.advance(off + sp + 4);
        return true;
    }
    // No year: "April 5" → "April fifth" (but not "May 5:30" or "May 5.5")
    if (!sc.word_end(day_end) || sc.at(day_end) == ':' ||
        (sc.at(day_end) == '.' && sc.is_digit(day_end + 1)))
        return false;
    sc.emit(MONTH_NAMES[mm.month - 1]);
    sc.emit_char(' ');
    sc.emit(DAY_ORDINALS[day.val]);
    sc.advance(day_end);
    return true;
}

// "7 October 2012" / "7 Oct 2012" / "7 October" → "the seventh of October twenty twelve"
inline bool try_date_day_month(Scanner& sc) {
    if (!sc.word_start()) return false;
    auto day = sc.scan_digits(0);
    if (day.count < 1 || day.count > 2 || day.val < 1 || day.val > 31) return false;
    if (sc.at(day.count) != ' ') return false;
    size_t off = day.count + 1;
    auto mm = match_month_full(sc, off);
    if (mm.month == 0) mm = match_month_abbrev(sc, off);
    if (mm.month == 0) return false;
    // "May" is also a verb: "5 May be" — require a year or sentence punctuation after it
    size_t mend = off + mm.len;
    int year = -1;
    size_t yoff = mend + (sc.at(mend) == ',' ? 1 : 0);
    if (sc.at(yoff) == ' ') {
        int y = sc.scan_fixed_digits(yoff + 1, 4);
        if (y >= 0 && sc.word_end(yoff + 5) && valid_date(mm.month, (int)day.val, y)) { year = y; mend = yoff + 5; }
    }
    if (year < 0 && mm.month == 5 && !(sc.at(mend) == '.' || sc.at(mend) == ',' || !sc.has(mend))) return false;
    sc.emit("the ");
    sc.emit(DAY_ORDINALS[day.val]);
    sc.emit(" of ");
    sc.emit(MONTH_NAMES[mm.month - 1]);
    if (year >= 0) { sc.emit_char(' '); emit_year_to_words(sc.out, year); }
    sc.advance(mend);
    return true;
}

// ── Time ────────────────────────────────────────────────────────────────────

inline bool try_time(Scanner& sc) {
    // H:MM or HH:MM, optionally followed by AM/PM/a.m./p.m.
    if (!sc.word_start()) return false;
    auto h = sc.scan_digits(0);
    if (h.count < 1 || h.count > 2) return false;
    if (sc.at(h.count) != ':') return false;
    int minute = sc.scan_fixed_digits(h.count + 1, 2);
    if (minute < 0 || minute > 59) return false;
    size_t off = h.count + 3; // past H:MM

    // Try to match AM/PM
    int sp = sc.skip_space(off);
    bool has_period = false;
    bool is_pm = false;
    int period_len = 0;

    if (sp >= 0) {
        size_t poff = off + sp;
        // "AM" / "PM"
        if (sc.match_str_ci(poff, "AM") && !sc.is_alpha(poff + 2)) {
            has_period = true; is_pm = false; period_len = sp + 2;
        } else if (sc.match_str_ci(poff, "PM") && !sc.is_alpha(poff + 2)) {
            has_period = true; is_pm = true; period_len = sp + 2;
        }
        // "a.m." / "p.m." (match_str_ci handles A.M./P.M. too)
        else if (sc.match_str_ci(poff, "a.m.")) {
            has_period = true; is_pm = false; period_len = sp + 4;
        } else if (sc.match_str_ci(poff, "p.m.")) {
            has_period = true; is_pm = true; period_len = sp + 4;
        }
    }

    int hour = (int)h.val;

    // H:MM:SS duration: "2:03:45" → "two hours three minutes and forty five seconds"
    if (!has_period && sc.at(off) == ':') {
        int second = sc.scan_fixed_digits(off + 1, 2);
        if (second < 0 || second > 59 || !sc.word_end(off + 3)) return false;
        const char* units[3][2] = {{"hour", "hours"}, {"minute", "minutes"}, {"second", "seconds"}};
        int vals[3] = {hour, minute, second};
        int nonzero = (hour > 0) + (minute > 0) + (second > 0);
        if (nonzero == 0) return false;
        int emitted = 0;
        for (int k = 0; k < 3; k++) {
            if (vals[k] == 0) continue;
            if (emitted > 0) sc.emit(emitted == nonzero - 1 ? " and " : " ");
            emit_number_to_words(sc.out, vals[k]);
            sc.emit_char(' ');
            sc.emit(units[k][vals[k] == 1 ? 0 : 1]);
            emitted++;
        }
        sc.advance(off + 3);
        return true;
    }

    if (has_period) {
        // 12-hour format
        if (hour < 1 || hour > 12) return false;
        emit_number_to_words(sc.out, hour);
        if (minute == 0) {
            sc.emit(" o'clock");
        } else if (minute < 10) {
            sc.emit(" oh ");
            emit_number_to_words(sc.out, minute);
        } else {
            sc.emit_char(' ');
            emit_number_to_words(sc.out, minute);
        }
        sc.emit(is_pm ? " P M" : " A M");
        sc.advance(off + period_len);
        return true;
    }

    if (hour > 23) return false;
    if (!sc.word_end(off)) return false;

    // 1-12 without AM/PM: "6:15" → "six fifteen", "6:00" → "six o'clock"
    if (hour >= 1 && hour <= 12) {
        emit_number_to_words(sc.out, hour);
        if (minute == 0) sc.emit(" o'clock");
        else if (minute < 10) { sc.emit(" oh "); emit_number_to_words(sc.out, minute); }
        else { sc.emit_char(' '); emit_number_to_words(sc.out, minute); }
        sc.advance(off);
        return true;
    }

    if (hour == 0) sc.emit("zero"); else emit_number_to_words(sc.out, hour);
    if (minute == 0) {
        sc.emit(" hundred");
    } else if (minute < 10) {
        sc.emit(" oh ");
        emit_number_to_words(sc.out, minute);
    } else {
        sc.emit_char(' ');
        emit_number_to_words(sc.out, minute);
    }
    sc.advance(off);
    return true;
}

inline bool try_date_dmy_dot(Scanner& sc) { return try_date_dmy_sep(sc, '.'); }

struct UnitEntry;
inline const UnitEntry* match_unit(const Scanner& sc, size_t offset);

// ── Time (period-separated: H.MM am/pm or H.MM a.m./p.m.) ─────────────────

inline bool try_time_dot(Scanner& sc) {
    // H.MM or HH.MM followed by AM/PM/a.m./p.m. (required — disambiguates from decimal)
    if (!sc.word_start()) return false;
    auto h = sc.scan_digits(0);
    if (h.count < 1 || h.count > 2) return false;
    if (sc.at(h.count) != '.') return false;
    int minute = sc.scan_fixed_digits(h.count + 1, 2);
    if (minute < 0 || minute > 59) return false;
    size_t off = h.count + 3;

    // AM/PM suffix is REQUIRED (otherwise it's a decimal)
    int sp = sc.skip_space(off);
    bool is_pm = false;
    int period_len = 0;

    size_t poff = off + sp;
    if (sc.match_str_ci(poff, "AM") && !sc.is_alpha(poff + 2)) {
        is_pm = false; period_len = sp + 2;
    } else if (sc.match_str_ci(poff, "PM") && !sc.is_alpha(poff + 2)) {
        is_pm = true; period_len = sp + 2;
    } else if (sc.match_str_ci(poff, "a.m.")) {
        is_pm = false; period_len = sp + 4;
    } else if (sc.match_str_ci(poff, "a.m")) {
        is_pm = false; period_len = sp + 3;
    } else if (sc.match_str_ci(poff, "p.m.")) {
        is_pm = true; period_len = sp + 4;
    } else if (sc.match_str_ci(poff, "p.m")) {
        is_pm = true; period_len = sp + 3;
    } else {
        // No AM/PM: a time only after a time preposition ("at 6.30", "until 10.15"),
        // and not when a unit follows ("at 3.25 meters")
        if (!sc.word_end(off) || match_unit(sc, off)) return false;
        static const char* PREPS[] = {"at", "by", "until", "till", "before", "after", "around", "from"};
        if (sc.pos < 2 || sc.s[sc.pos - 1] != ' ') return false;
        size_t e = sc.pos - 1, b = e;
        while (b > 0 && std::isalpha((unsigned char)sc.s[b - 1])) b--;
        bool prep = false;
        for (const char* p : PREPS)
            if (e - b == std::strlen(p) && std::strncmp(sc.s + b, p, e - b) == 0) { prep = true; break; }
        int hour = (int)h.val;
        if (!prep || hour < 1 || hour > 12) return false;
        emit_number_to_words(sc.out, hour);
        if (minute == 0) sc.emit(" o'clock");
        else if (minute < 10) { sc.emit(" oh "); emit_number_to_words(sc.out, minute); }
        else { sc.emit_char(' '); emit_number_to_words(sc.out, minute); }
        sc.advance(off);
        return true;
    }

    int hour = (int)h.val;
    if (hour < 1 || hour > 12) return false;

    emit_number_to_words(sc.out, hour);
    if (minute == 0) {
        sc.emit(" o'clock");
    } else if (minute < 10) {
        sc.emit(" oh ");
        emit_number_to_words(sc.out, minute);
    } else {
        sc.emit_char(' ');
        emit_number_to_words(sc.out, minute);
    }
    sc.emit(is_pm ? " P M" : " A M");
    sc.advance(off + period_len);
    return true;
}

// "9 PM", "9pm", "11 a.m." → "nine P M"
inline bool try_hour_period(Scanner& sc) {
    if (!sc.word_start()) return false;
    auto h = sc.scan_digits(0);
    if (h.count < 1 || h.count > 2 || h.val < 1 || h.val > 12) return false;
    size_t poff = h.count + sc.skip_space(h.count);
    bool is_pm; size_t plen;
    if (sc.match_str_ci(poff, "a.m.")) { is_pm = false; plen = 4; }
    else if (sc.match_str_ci(poff, "p.m.")) { is_pm = true; plen = 4; }
    else if (sc.match_str_ci(poff, "am") && sc.word_end(poff + 2)) { is_pm = false; plen = 2; }
    else if (sc.match_str_ci(poff, "pm") && sc.word_end(poff + 2)) { is_pm = true; plen = 2; }
    else return false;
    emit_number_to_words(sc.out, (int)h.val);
    sc.emit(is_pm ? " P M" : " A M");
    sc.advance(poff + plen);
    return true;
}

// ── Telephone ───────────────────────────────────────────────────────────────


inline bool try_phone_paren(Scanner& sc) {
    // (NNN) NNN-NNNN or (NNN) NNN.NNNN
    if (sc.at(0) != '(') return false;
    // 3 digits inside parens
    if (!sc.is_digit(1) || !sc.is_digit(2) || !sc.is_digit(3)) return false;
    if (sc.at(4) != ')') return false;
    size_t off = 5 + sc.skip_space(5);
    // 3 digits
    if (!sc.is_digit(off) || !sc.is_digit(off+1) || !sc.is_digit(off+2)) return false;
    char sep = sc.at(off + 3);
    if (sep != '-' && sep != '.') return false;
    // 4 digits
    if (!sc.is_digit(off+4) || !sc.is_digit(off+5) || !sc.is_digit(off+6) || !sc.is_digit(off+7)) return false;
    // Must end at word boundary
    if (sc.is_alnum(off + 8)) return false;

    sc.emit_digit_words(1, 3);
    sc.emit(" ");
    sc.emit_digit_words(off, 3);
    sc.emit(" ");
    sc.emit_digit_words(off + 4, 4);
    sc.advance(off + 8);
    return true;
}

inline bool try_phone_country(Scanner& sc) {
    // 1-NNN-NNN-NNNN
    if (!sc.word_start()) return false;
    if (sc.at(0) != '1') return false;
    char sep1 = sc.at(1);
    if (sep1 != '-' && sep1 != '.') return false;
    // NNN
    if (!sc.is_digit(2) || !sc.is_digit(3) || !sc.is_digit(4)) return false;
    char sep2 = sc.at(5);
    if (sep2 != '-' && sep2 != '.') return false;
    // NNN
    if (!sc.is_digit(6) || !sc.is_digit(7) || !sc.is_digit(8)) return false;
    char sep3 = sc.at(9);
    if (sep3 != '-' && sep3 != '.') return false;
    // NNNN
    if (!sc.is_digit(10) || !sc.is_digit(11) || !sc.is_digit(12) || !sc.is_digit(13)) return false;
    if (sc.is_alnum(14)) return false;

    sc.emit("one ");
    sc.emit_digit_words(2, 3);
    sc.emit(" ");
    sc.emit_digit_words(6, 3);
    sc.emit(" ");
    sc.emit_digit_words(10, 4);
    sc.advance(14);
    return true;
}

inline bool try_phone_7digit(Scanner& sc) {
    // NNN-NNNN
    if (!sc.word_start()) return false;
    if (!sc.is_digit(0) || !sc.is_digit(1) || !sc.is_digit(2) || sc.at(3) != '-') return false;
    if (!sc.is_digit(4) || !sc.is_digit(5) || !sc.is_digit(6) || !sc.is_digit(7)) return false;
    if (!sc.word_end(8)) return false;
    // "100-1000" is a range, not a phone number
    if (sc.scan_fixed_digits(0, 3) % 10 == 0 && sc.scan_fixed_digits(4, 4) % 10 == 0) return false;
    sc.emit_digit_words(0, 3);
    sc.emit(" ");
    sc.emit_digit_words(4, 4);
    sc.advance(8);
    return true;
}

inline bool try_phone_10digit(Scanner& sc) {
    // NNN-NNN-NNNN or NNN.NNN.NNNN
    if (!sc.word_start()) return false;
    if (!sc.is_digit(0) || !sc.is_digit(1) || !sc.is_digit(2)) return false;
    char sep1 = sc.at(3);
    if (sep1 != '-' && sep1 != '.') return false;
    if (!sc.is_digit(4) || !sc.is_digit(5) || !sc.is_digit(6)) return false;
    char sep2 = sc.at(7);
    if (sep2 != '-' && sep2 != '.') return false;
    if (!sc.is_digit(8) || !sc.is_digit(9) || !sc.is_digit(10) || !sc.is_digit(11)) return false;
    if (sc.is_alnum(12)) return false;

    sc.emit_digit_words(0, 3);
    sc.emit(" ");
    sc.emit_digit_words(4, 3);
    sc.emit(" ");
    sc.emit_digit_words(8, 4);
    sc.advance(12);
    return true;
}

// ── Fractions ───────────────────────────────────────────────────────────────

// Parse a proper fraction N/D at off. Returns chars consumed, 0 if none.
inline size_t parse_fraction(const Scanner& sc, size_t off, int& n, int& d) {
    if (off > 0 && sc.is_alnum(off - 1)) return 0;
    auto num = sc.scan_digits(off);
    if (num.count < 1 || num.count > 3) return 0;
    if (sc.at(off + num.count) != '/') return 0;
    auto den = sc.scan_digits(off + num.count + 1);
    if (den.count < 1 || den.count > 3) return 0;
    size_t total = num.count + 1 + den.count;
    if (!sc.word_end(off + total)) return 0;
    n = (int)num.val; d = (int)den.val;
    if (n <= 0 || n >= d || !is_common_denom(d)) return 0;
    return total;
}

inline void emit_fraction(std::string& out, int n, int d) {
    emit_number_to_words(out, n);
    out += ' ';
    emit_denom_word(out, d, n > 1);
}

inline bool try_fraction(Scanner& sc) {
    if (!sc.word_start()) return false;
    int n, d;
    size_t total = parse_fraction(sc, 0, n, d);
    if (!total) return false;
    emit_fraction(sc.out, n, d);
    sc.advance(total);
    return true;
}

// Numbers with a fixed spoken form.
inline bool try_number_idiom(Scanner& sc) {
    if (!sc.word_start()) return false;
    static const struct { const char* text; const char* spoken; } IDIOMS[] = {
        {"24/7", "twenty four seven"}, {"9/11", "nine eleven"},
        {"7/11", "seven eleven"}, {"50/50", "fifty fifty"},
        {"401(k)", "four oh one k"},
    };
    for (const auto& idiom : IDIOMS) {
        size_t n = std::strlen(idiom.text);
        if (sc.match_str(0, idiom.text) && sc.word_end(n)) {
            sc.emit(idiom.spoken);
            sc.advance(n);
            return true;
        }
    }
    return false;
}

// ── Ordinals ────────────────────────────────────────────────────────────────

inline bool try_ordinal(Scanner& sc) {
    if (!sc.word_start()) return false;
    auto digits = sc.scan_digits(0);
    if (digits.count < 1 || digits.val < 1 || digits.val > 999999999) return false;
    size_t off = digits.count;

    // Read suffix (2 chars)
    char s1 = std::tolower((unsigned char)sc.at(off));
    char s2 = std::tolower((unsigned char)sc.at(off + 1));
    if (!sc.word_end(off + 2)) return false; // must end after suffix

    const char* expected;
    int last_two = (int)(digits.val % 100);
    int last_one = (int)(digits.val % 10);
    if (last_two >= 11 && last_two <= 13) expected = "th";
    else if (last_one == 1) expected = "st";
    else if (last_one == 2) expected = "nd";
    else if (last_one == 3) expected = "rd";
    else expected = "th";

    if (s1 != expected[0] || s2 != expected[1]) return false;

    emit_ordinal_words(sc.out, (int)digits.val);
    sc.advance(off + 2);
    return true;
}

// ── Percent ─────────────────────────────────────────────────────────────────

inline bool try_percent(Scanner& sc) {
    // N% or N.N% (with optional space before %)
    if (!sc.word_start()) return false;
    if (!sc.is_digit(0)) return false;
    // Single-pass scan: integer part, optional decimal, then %
    long long whole = 0;
    size_t i = 0;
    while (sc.is_digit(i)) {
        whole = whole * 10 + (sc.at(i) - '0');
        i++;
        if (sc.at(i) == ',' && sc.is_digit(i + 1)) i++;
    }
    if (whole > 999999999) return false;
    size_t dot_pos = i;
    size_t frac_end = i;
    bool has_decimal = false;
    if (sc.at(i) == '.' && sc.is_digit(i + 1)) {
        has_decimal = true;
        dot_pos = i;
        frac_end = i + 1;
        while (sc.is_digit(frac_end)) frac_end++;
        i = frac_end;
    }
    size_t off = i + sc.skip_space(i);
    if (sc.at(off) != '%') return false;

    emit_number_to_words(sc.out, (int)whole);
    if (has_decimal) {
        sc.emit(" point ");
        for (size_t j = dot_pos + 1; j < frac_end; j++) {
            if (j > dot_pos + 1) sc.emit_char(' ');
            sc.emit(DIGIT_WORDS[sc.at(j) - '0']);
        }
    }
    sc.emit(" percent");
    sc.advance(off + 1); // +1 for %
    return true;
}

// ── Number + Unit ──────────────────────────────────────────────────────────
// Matches <integer_or_decimal><optional_space><unit_abbrev> and expands both.
// Must come before try_decimal/try_number to catch "2.5GB", "100kg", etc.

// attached_only: the abbreviation is also a common word, so it's a unit only
// when written without a space ("5in" yes, "19 in March" no).
struct UnitEntry { const char* abbr; const char* singular; const char* plural; bool attached_only = false; };

inline const UnitEntry* match_unit(const Scanner& sc, size_t offset) {
    static const UnitEntry UNITS[] = {
        // Temperature / angle (° is C2 B0)
        {"\xC2\xB0""F", "degree Fahrenheit", "degrees Fahrenheit"},
        {"\xC2\xB0""C", "degree Celsius", "degrees Celsius"},
        {"\xC2\xB0"" F", "degree Fahrenheit", "degrees Fahrenheit"},
        {"\xC2\xB0"" C", "degree Celsius", "degrees Celsius"},
        {"\xC2\xB0", "degree", "degrees"},
        // Length
        {"km/h", "kilometers per hour", "kilometers per hour"},
        {"mph", "miles per hour", "miles per hour"},
        {"km", "kilometer", "kilometers"},
        {"mm", "millimeter", "millimeters"},
        {"cm", "centimeter", "centimeters"},
        {"ft", "foot", "feet"},
        {"in", "inch", "inches", true},
        {"mi", "mile", "miles"},
        {"m", "meter", "meters"},
        // Weight
        {"lbs", "pounds", "pounds"},
        {"kg", "kilogram", "kilograms"},
        {"lb", "pound", "pounds"},
        {"oz", "ounce", "ounces"},
        {"mg", "milligram", "milligrams"},
        {"g", "gram", "grams"},
        // Volume
        {"ml", "milliliter", "milliliters"},
        {"mL", "milliliter", "milliliters"},
        {"gal", "gallon", "gallons"},
        {"L", "liter", "liters"},
        // Digital
        {"TB", "terabyte", "terabytes"},
        {"GB", "gigabyte", "gigabytes"},
        {"MB", "megabyte", "megabytes"},
        {"KB", "kilobyte", "kilobytes"},
        {"GHz", "gigahertz", "gigahertz"},
        {"MHz", "megahertz", "megahertz"},
        {"kHz", "kilohertz", "kilohertz"},
        {"Hz", "hertz", "hertz"},
        {"Gbps", "gigabits per second", "gigabits per second"},
        {"Mbps", "megabits per second", "megabits per second"},
        {"kbps", "kilobits per second", "kilobits per second"},
        // Speed
        {"kph", "kilometers per hour", "kilometers per hour"},
        {"fps", "frames per second", "frames per second"},
        {"rpm", "revolutions per minute", "revolutions per minute"},
        // Time
        {"ms", "millisecond", "milliseconds"},
        {"ns", "nanosecond", "nanoseconds"},
        {"hrs", "hours", "hours"},
        {"hr", "hour", "hours"},
        {"min", "minute", "minutes"},
        {"sec", "second", "seconds"},
    };

    // Skip optional space
    size_t u = offset;
    bool spaced = sc.has(u) && sc.s[sc.pos + u] == ' ';
    if (spaced) u++;

    for (const auto& unit : UNITS) {
        if (unit.attached_only && spaced) continue;
        size_t ulen = std::strlen(unit.abbr);
        bool match = true;
        for (size_t j = 0; j < ulen && match; j++) {
            if (!sc.has(u + j) || sc.s[sc.pos + u + j] != unit.abbr[j])
                match = false;
        }
        if (!match) continue;
        // Must not be followed by more alpha (e.g. "km" shouldn't match "km" in "kmart")
        size_t after = u + ulen;
        if (sc.has(after) && std::isalpha((unsigned char)sc.s[sc.pos + after]))
            continue;
        return &unit;
    }
    return nullptr;
}

inline bool try_number_unit(Scanner& sc) {
    if (!sc.word_start()) return false;
    if (!sc.is_digit(0)) return false;

    // Scan integer part
    long long whole = 0;
    size_t i = 0;
    while (sc.is_digit(i)) {
        whole = whole * 10 + (sc.at(i) - '0');
        i++;
        if (sc.at(i) == ',' && sc.is_digit(i + 1)) i++;
        if (whole > 999999999LL) return false;
    }
    if (i == 0) return false;

    // Optional decimal part
    bool has_frac = false;
    size_t frac_start = 0, frac_end = 0;
    if (sc.at(i) == '.' && sc.is_digit(i + 1)) {
        has_frac = true;
        frac_start = i + 1;
        frac_end = frac_start;
        while (sc.is_digit(frac_end)) frac_end++;
        i = frac_end;
    }

    // Must match a known unit
    const UnitEntry* unit = match_unit(sc, i);
    if (!unit) return false;

    // Skip optional space between number and unit
    size_t u = i;
    if (sc.has(u) && sc.s[sc.pos + u] == ' ') u++;
    size_t ulen = std::strlen(unit->abbr);

    // Emit expanded number
    emit_number_to_words(sc.out, (int)whole);
    if (has_frac) {
        sc.emit(" point ");
        for (size_t j = frac_start; j < frac_end; j++) {
            if (j > frac_start) sc.emit_char(' ');
            sc.emit(DIGIT_WORDS[sc.at(j) - '0']);
        }
    }

    // Emit unit word
    sc.emit_char(' ');
    bool is_one = (whole == 1 && !has_frac);
    sc.emit(is_one ? unit->singular : unit->plural);

    sc.advance(u + ulen);
    return true;
}

// ── Decimals ────────────────────────────────────────────────────────────────

inline bool try_decimal(Scanner& sc) {
    if (!sc.word_start()) return false;
    if (!sc.is_digit(0)) return false;
    // Single pass: scan integer part (digits + commas), compute value inline
    long long whole = 0;
    size_t i = 0;
    while (sc.is_digit(i)) {
        whole = whole * 10 + (sc.at(i) - '0');
        i++;
        if (sc.at(i) == ',' && sc.is_digit(i + 1)) i++;
    }
    // Must have dot followed by digit
    if (sc.at(i) != '.' || !sc.is_digit(i + 1)) return false;
    if (whole > 999999999) return false;
    // Scan fractional digits
    size_t frac_start = i + 1;
    size_t frac_end = frac_start;
    while (sc.is_digit(frac_end)) frac_end++;

    emit_number_to_words(sc.out, (int)whole);
    sc.emit(" point ");
    for (size_t j = frac_start; j < frac_end; j++) {
        if (j > frac_start) sc.emit_char(' ');
        sc.emit(DIGIT_WORDS[sc.at(j) - '0']);
    }
    // Insert space before trailing letters (e.g. "2.5GB" → "two point five GB")
    if (sc.has(frac_end) && std::isalpha((unsigned char)sc.s[sc.pos + frac_end]))
        sc.emit_char(' ');
    sc.advance(frac_end);
    return true;
}

// ── Numbers (catch-all) ─────────────────────────────────────────────────────

inline bool try_number(Scanner& sc) {
    // Scan the full digit+comma run first
    long long val = 0;
    size_t i = 0;
    while (sc.is_digit(i)) {
        val = val * 10 + (sc.at(i) - '0');
        i++;
        if (sc.at(i) == ',' && sc.is_digit(i + 1)) i++;
        if (val > 999999999LL) break;
    }
    if (i == 0) return false;

    // Check adjacency: preceded by alpha/multibyte OR followed by alpha/multibyte
    bool adj_before = sc.pos > 0 && (std::isalpha((unsigned char)sc.s[sc.pos - 1]) || (uint8_t)sc.s[sc.pos - 1] >= 0x80);
    bool adj_after = sc.has(i) && (std::isalpha((unsigned char)sc.s[sc.pos + i]) || (uint8_t)sc.s[sc.pos + i] >= 0x80);

    bool plain4 = (i == 4);  // four digits, no commas
    bool year = plain4 && is_year_like(val);

    // Decades: "1990s", "1990's", "80s" → "nineteen nineties", "eighties"
    if (!adj_before && val % 10 == 0 && (plain4 || i == 2)) {
        size_t s_off = i + (sc.at(i) == '\'' ? 1 : 0);
        if (sc.at(s_off) == 's' && sc.word_end(s_off + 1) && (i == 4 ? val >= 1000 : val >= 20)) {
            if (i == 4) emit_year_to_words(sc.out, (int)val);
            else {
                // Drop the apostrophe of "'80s" (already copied to output)
                if (sc.pos > 0 && sc.s[sc.pos - 1] == '\'' && !sc.out.empty() && sc.out.back() == '\'')
                    sc.out.pop_back();
                sc.emit(TENS[val / 10]);
            }
            pluralize_last_word(sc.out);
            sc.advance(s_off + 1);
            return true;
        }
    }

    // Abbreviated years: "'97" → "ninety seven", "'05" → "oh five"
    if (i == 2 && sc.pos > 0 && sc.s[sc.pos - 1] == '\'' && !adj_after &&
        (sc.pos == 1 || !std::isalnum((unsigned char)sc.s[sc.pos - 2])) &&
        !sc.out.empty() && sc.out.back() == '\'') {
        sc.out.pop_back();
        if (val < 10) { sc.emit("oh "); sc.emit(ONES[val]); }
        else emit_two_digit(sc.out, (int)val);
        sc.advance(i);
        return true;
    }

    // Currency written after the amount: "20$" → "twenty dollars", "20€" → "twenty euros"
    if (!adj_before) {
        const char* cur = nullptr; const char* cur_one = nullptr; size_t clen = 0;
        if (sc.at(i) == '$') { cur = "dollars"; cur_one = "dollar"; clen = 1; }
        else if ((uint8_t)sc.at(i) == 0xE2 && (uint8_t)sc.at(i + 1) == 0x82 && (uint8_t)sc.at(i + 2) == 0xAC) {
            cur = "euros"; cur_one = "euro"; clen = 3;
        }
        if (cur && val <= 999999999) {
            emit_number_to_words(sc.out, (int)val);
            sc.emit_char(' ');
            sc.emit(val == 1 ? cur_one : cur);
            sc.advance(i + clen);
            return true;
        }
    }

    // Number + capital letters: "221B" → "two hundred twenty one B", "2XL" → "two XL"
    if (!adj_before && adj_after && val <= 999999999) {
        size_t caps = 0;
        while (caps < 3 && std::isupper((unsigned char)sc.at(i + caps))) caps++;
        if (caps > 0 && sc.word_end(i + caps)) {
            emit_number_to_words(sc.out, (int)val);
            sc.emit_char(' ');
            for (size_t k = 0; k < caps; k++) sc.emit_char(sc.at(i + k));
            sc.advance(i + caps);
            return true;
        }
    }

    // Leading zeros are codes, not quantities: "000" → "zero zero zero", "007"
    if (!adj_before && !adj_after && i > 1 && sc.at(0) == '0') {
        bool digits_only = true;
        for (size_t k = 0; k < i; k++) if (!sc.is_digit(k)) digits_only = false;
        if (digits_only) {
            sc.emit_digit_words(0, (int)i);
            sc.advance(i);
            return true;
        }
    }

    if (adj_before || adj_after) {
        // Copy all digits+commas through verbatim (no expansion)
        for (size_t j = 0; j < i; j++) sc.emit_char(sc.at(j));
        sc.advance(i);
        return true;
    }

    if (val > 999999999) return false;

    if (val == 911 && i == 3) sc.emit("nine one one");
    else if (year) emit_year_to_words(sc.out, (int)val);
    else emit_number_to_words(sc.out, (int)val);

    // Scale word + unit: "150 million km" → "one hundred fifty million kilometers"
    if (sc.at(i) == ' ') {
        static const char* SCALES[] = {"thousand", "million", "billion"};
        for (const char* scale : SCALES) {
            size_t slen = std::strlen(scale);
            if (!sc.match_str(i + 1, scale) || !sc.word_end(i + 1 + slen)) continue;
            size_t after = i + 1 + slen;
            if (sc.at(after) != ' ') break;
            const UnitEntry* unit = match_unit(sc, after);
            if (!unit) break;
            sc.emit_char(' '); sc.emit(scale);
            sc.emit_char(' '); sc.emit(unit->plural);
            sc.advance(after + 1 + std::strlen(unit->abbr));
            return true;
        }
    }

    // Ranges and scores: "10-20" → "ten to twenty", "1990-1995" → years
    if (i <= 4 && sc.at(i) == '-' && sc.is_digit(i + 1)) {
        auto r = sc.scan_digits(i + 1);
        size_t r_end = i + 1 + r.count;
        bool chained = (sc.at(r_end) == '-' && sc.is_digit(r_end + 1)) ||   // "40-50-60"
                       (sc.pos >= 2 && sc.s[sc.pos - 1] == '-' && std::isdigit((unsigned char)sc.s[sc.pos - 2]));
        if (r.count <= 4 && sc.word_end(r_end) && !chained) {
            // A number "to" itself isn't a range: "20-20 vision" → "twenty twenty"
            sc.emit(r.val == val ? " " : " to ");
            if (year && r.count == 4 && is_year_like(r.val)) emit_year_to_words(sc.out, (int)r.val);
            else emit_number_to_words(sc.out, (int)r.val);
            sc.advance(i + 1 + r.count);
            return true;
        }
    }

    // Mixed numbers: "2 1/2" → "two and one half"
    if (sc.at(i) == ' ') {
        int n, d;
        size_t flen = parse_fraction(sc, i + 1, n, d);
        if (flen) {
            sc.emit(" and ");
            emit_fraction(sc.out, n, d);
            sc.advance(i + 1 + flen);
            return true;
        }
    }

    // Number-word compounds: "80-yen" → "eighty yen", "40-year-old" → "forty year-old"
    if (sc.at(i) == '-' && std::isalpha((unsigned char)sc.at(i + 1))) {
        sc.emit_char(' ');
        sc.advance(i + 1);
        return true;
    }

    // Any other number/number: "3/7" → "three slash seven",
    // except splits that add up to 100: "70/30" → "seventy thirty"
    if (sc.at(i) == '/' && sc.is_digit(i + 1)) {
        auto r = sc.scan_digits(i + 1);
        if (r.count <= 2 && val + r.val == 100 && sc.word_end(i + 1 + r.count)) {
            sc.emit_char(' ');
            emit_number_to_words(sc.out, (int)r.val);
            sc.advance(i + 1 + r.count);
            return true;
        }
        sc.emit(" slash ");
        sc.advance(i + 1);
        return true;
    }

    sc.advance(i);
    return true;
}

// ── UTF-8 / Unicode → ASCII ────────────────────────────────────────────────


inline uint32_t decode_utf8(const char* s, size_t len, size_t& i) {
    uint8_t c = (uint8_t)s[i];
    if (c < 0x80) { i++; return c; }
    uint32_t cp; int extra;
    if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
    else { i++; return 0xFFFD; }
    i++;
    for (int j = 0; j < extra && i < len; j++, i++) {
        if (((uint8_t)s[i] & 0xC0) != 0x80) return 0xFFFD;
        cp = (cp << 6) | ((uint8_t)s[i] & 0x3F);
    }
    return cp;
}

// Returns ASCII replacement string for a codepoint, or nullptr to strip.
inline const char* transliterate(uint32_t cp) {
    // Latin-1 Supplement (U+00A0-U+00FF)
    if (cp >= 0x00A0 && cp <= 0x00FF) {
        static const char* t[96] = {
            " ",  "!",  "c",  "PS", "$",  "Y=", "|",  "SS",   // A0-A7
            "\"", "(c)","a",  "<<", "!",  "",   "(r)","-",     // A8-AF
            " degrees ","+-","2","3","'",  "u",  "P",  "*",    // B0-B7
            ",",  "1",  "o",  ">>", " one quarter"," one half"," three quarters","?", // B8-BF
            "A","A","A","A","A","A","AE","C",                   // C0-C7
            "E","E","E","E","I","I","I","I",                     // C8-CF
            "D","N","O","O","O","O","O","x",                     // D0-D7
            "O","U","U","U","U","Y","Th","ss",                   // D8-DF
            "a","a","a","a","a","a","ae","c",                   // E0-E7
            "e","e","e","e","i","i","i","i",                     // E8-EF
            "d","n","o","o","o","o","o","/",                     // F0-F7
            "o","u","u","u","u","y","th","y",                   // F8-FF
        };
        return t[cp - 0x00A0];
    }
    // Latin Extended-A (U+0100-U+017F)
    if (cp >= 0x0100 && cp <= 0x017F) {
        static const char* t[128] = {
            "A","a","A","a","A","a","C","c","C","c","C","c","C","c","D","d",
            "D","d","E","e","E","e","E","e","E","e","E","e","G","g","G","g",
            "G","g","G","g","H","h","H","h","I","i","I","i","I","i","I","i",
            "I","i","IJ","ij","J","j","K","k","k","L","l","L","l","L","l",
            "L","l","L","l","N","n","N","n","N","n","'n","NG","ng",
            "O","o","O","o","O","o","OE","oe",
            "R","r","R","r","R","r",
            "S","s","S","s","S","s","S","s",
            "T","t","T","t","T","t",
            "U","u","U","u","U","u","U","u","U","u","U","u",
            "W","w","Y","y","Y","Z","z","Z","z","Z","z","s",
        };
        return t[cp - 0x0100];
    }
    // Common punctuation
    switch (cp) {
        case 0x2013: return "-";
        case 0x2014: return "--";
        case 0x2018: return "'";
        case 0x2019: return "'";
        case 0x201A: return ",";
        case 0x201C: return "\"";
        case 0x201D: return "\"";
        case 0x201E: return ",,";
        case 0x2026: return "...";
        case 0x2122: return "TM";
    }
    return nullptr; // strip
}


// ── Symbols ────────────────────────────────────────────────────────────────

// Returns word expansion for a symbol character, or nullptr if not a symbol.
// Only expands symbols that have high char-to-phoneme ratios and are
// unambiguous in context. The symbol must be standalone (not part of a
// pattern already consumed by earlier stages).
inline const char* symbol_word(char c) {
    switch (c) {
        case '&': return "and";
        case '@': return "at";
        case '+': return "plus";
        case '=': return "equals";
        default: return nullptr;
    }
}

// Expand a standalone symbol to its word form.
// Only match when the symbol is between whitespace/punctuation (not part
// of an email address, URL, or other structure consumed by earlier stages).
inline bool try_symbol(Scanner& sc) {
    const char* word = symbol_word(sc.at(0));
    if (!word) return false;
    // Only expand when the symbol is a standalone token: preceded and followed
    // by whitespace or start/end of string. This avoids expanding "AT&T", "C++",
    // "user@domain", etc.
    bool space_before = sc.pos == 0 || sc.s[sc.pos - 1] == ' ' || sc.s[sc.pos - 1] == '\t';
    bool space_after = !sc.has(1) || sc.s[sc.pos + 1] == ' ' || sc.s[sc.pos + 1] == '\t';
    if (!space_before || !space_after) return false;
    sc.emit(word);
    sc.advance(1);
    return true;
}

// ── Dotted initialisms ─────────────────────────────────────────────────────

// Match patterns like "U.S.A." or "U.S." — single uppercase letters
// separated by dots. Expands to space-separated letters: "U S A" or "U S".
// Minimum 2 letters (e.g., "A." alone is not an initialism).
inline bool try_dotted_initialism(Scanner& sc) {
    if (!sc.word_start()) return false;
    // First char must be uppercase letter
    if (!std::isupper((unsigned char)sc.at(0))) return false;
    if (sc.at(1) != '.') return false;
    // Scan the full pattern: (LETTER DOT)+
    size_t i = 0;
    int count = 0;
    while (std::isupper((unsigned char)sc.at(i)) && sc.at(i + 1) == '.') {
        count++;
        i += 2;
    }
    if (count < 2) return false;
    // Must end at word boundary (not followed by alphanumeric)
    if (sc.has(i) && std::isalnum((unsigned char)sc.s[sc.pos + i])) return false;
    // Emit space-separated letters
    for (int j = 0; j < count; j++) {
        if (j > 0) sc.emit_char(' ');
        sc.emit_char(sc.at(j * 2));
    }
    sc.advance(i);
    return true;
}

// ── Roman numerals ─────────────────────────────────────────────────────────
// Expanded only with context, because I/V/X/L/C/D/M are also ordinary words
// and letters ("I think", "Vitamin C", "DC"):
//   after a keyword ("Chapter IV", "World War II", "Super Bowl LVII") → cardinal
//   after a capitalized name, I/V/X only, 2+ letters ("Henry VIII") → "the eighth"
// Single letters only count after a keyword, and only I, V, X ("Part I").

inline int roman_digit(char c) {
    switch (c) {
        case 'I': return 1; case 'V': return 5; case 'X': return 10; case 'L': return 50;
        case 'C': return 100; case 'D': return 500; case 'M': return 1000; default: return 0;
    }
}

// Value of a canonical numeral of length len at pos, or 0 ("IIII", "VX" → 0).
inline int parse_roman(const Scanner& sc, size_t len) {
    int total = 0;
    for (size_t k = 0; k < len; k++) {
        int v = roman_digit(sc.at(k));
        int next = k + 1 < len ? roman_digit(sc.at(k + 1)) : 0;
        total += v < next ? -v : v;
    }
    if (total <= 0 || total >= 4000) return 0;
    // Re-render and compare, which rejects non-canonical spellings
    static const struct { int v; const char* s; } T[] = {
        {1000,"M"},{900,"CM"},{500,"D"},{400,"CD"},{100,"C"},{90,"XC"},
        {50,"L"},{40,"XL"},{10,"X"},{9,"IX"},{5,"V"},{4,"IV"},{1,"I"}};
    char buf[32]; size_t n = 0; int rem = total;
    for (const auto& t : T)
        while (rem >= t.v) { for (const char* p = t.s; *p; p++) buf[n++] = *p; rem -= t.v; }
    if (n != len) return 0;
    for (size_t k = 0; k < len; k++) if (buf[k] != sc.at(k)) return 0;
    return total;
}

inline bool try_roman(Scanner& sc) {
    if (sc.pos < 2 || sc.s[sc.pos - 1] != ' ') return false;
    size_t len = 0;
    while (roman_digit(sc.at(len))) len++;
    if (len == 0 || !sc.word_end(len)) return false;
    // Previous word (letters, optional trailing period for "Vol.")
    size_t e = sc.pos - 1;
    if (e > 0 && sc.s[e - 1] == '.') e--;
    size_t b = e;
    while (b > 0 && std::isalpha((unsigned char)sc.s[b - 1])) b--;
    if (b == e) return false;
    std::string prev(sc.s + b, e - b);

    static const char* KEYWORDS[] = {
        "Chapter", "Part", "Book", "Volume", "Vol", "Act", "Scene", "Section",
        "Phase", "Stage", "Level", "Round", "Episode", "Season", "Title", "Article",
        "Appendix", "War", "Bowl", "Type", "Class", "Grade", "Mark", "Series", "Unit",
        "Game", "Wars", "Fantasy", "Olympiad", "Canto", "Psalm", "Plate", "Figure",
    };
    bool keyword = false;
    for (const char* k : KEYWORDS) if (prev == k) { keyword = true; break; }

    bool ivx_only = true;
    for (size_t k = 0; k < len; k++) { char c = sc.at(k); if (c != 'I' && c != 'V' && c != 'X') ivx_only = false; }

    if (keyword) {
        if (len == 1 && !ivx_only) return false;
        int v = parse_roman(sc, len);
        if (!v) return false;
        emit_number_to_words(sc.out, v);
    } else {
        if (len < 2 || !ivx_only || !std::isupper((unsigned char)prev[0]) || prev.size() < 2) return false;
        for (size_t k = 1; k < prev.size(); k++) if (std::isupper((unsigned char)prev[k])) return false;
        int v = parse_roman(sc, len);
        if (!v) return false;
        sc.emit("the ");
        emit_ordinal_words(sc.out, v);
    }
    sc.advance(len);
    return true;
}

// ── Abbreviations and emails (pre-pass) ────────────────────────────────────
// Runs before the main scan so the expanded words go through the normal rules
// ("No. 5" → "number 5" → "number five").

inline bool is_word_char(char c) { return std::isalnum((unsigned char)c) || c == '\''; }

inline std::string expand_abbreviations(const std::string& in) {
    // before_name: expansion when followed by a capitalized word ("Dr. Smith");
    // otherwise: any other position ("Elm Dr."). prefix: a title that precedes
    // names, so "Dr. Smith" never ends a sentence; for the rest, a following
    // capital means the period also ended the sentence ("... etc. Then").
    static const struct { const char* abbr; const char* before_name; const char* otherwise; bool prefix; } ABBR[] = {
        {"Mr.", "Mister", "Mister", true}, {"Mrs.", "Missus", "Missus", true}, {"Ms.", "Miz", "Miz", true},
        {"Dr.", "Doctor", "Drive", true}, {"St.", "Saint", "Street", true}, {"Mt.", "Mount", "Mount", true},
        {"Prof.", "Professor", "Professor", true}, {"Jr.", "Junior", "Junior", false},
        {"Sr.", "Senior", "Senior", false}, {"Ave.", "Avenue", "Avenue", false},
        {"Blvd.", "Boulevard", "Boulevard", false}, {"e.g.", "for example", "for example", false},
        {"i.e.", "that is", "that is", false}, {"etc.", "et cetera", "et cetera", false},
        {"vs.", "versus", "versus", false}, {"vs", "versus", "versus", false},
        {"approx.", "approximately", "approximately", false},
    };
    std::string out;
    out.reserve(in.size() + 16);
    size_t i = 0, n = in.size();
    while (i < n) {
        bool at_word_start = i == 0 || (!is_word_char(in[i - 1]) && in[i - 1] != '.' && in[i - 1] != '@');
        if (at_word_start) {
            // "No. 5" → "number 5", "#1" → "number 1"
            if (in.compare(i, 4, "No. ") == 0 && i + 4 < n && std::isdigit((unsigned char)in[i + 4])) {
                out += "number "; i += 4; continue;
            }
            if (in[i] == '#' && i + 1 < n && std::isdigit((unsigned char)in[i + 1])) {
                out += "number "; i += 1; continue;
            }
            bool matched = false;
            for (const auto& a : ABBR) {
                size_t len = std::strlen(a.abbr);
                if (in.compare(i, len, a.abbr) != 0) continue;
                size_t j = i + len;
                if (j < n && (is_word_char(in[j]) || in[j] == '.')) continue;
                bool next_capital = j + 1 < n && in[j] == ' ' && std::isupper((unsigned char)in[j + 1]);
                out += next_capital ? a.before_name : a.otherwise;
                bool ends_sentence = j >= n || (next_capital && !a.prefix);
                if (a.abbr[len - 1] == '.' && ends_sentence) out += '.';
                i = j; matched = true; break;
            }
            if (matched) continue;
            // Email: local@domain.tld → "local at domain dot tld"
            size_t k = i;
            while (k < n && (std::isalnum((unsigned char)in[k]) || in[k] == '.' || in[k] == '_' || in[k] == '-')) k++;
            if (k > i && k < n && in[k] == '@') {
                size_t d = k + 1, dots = 0;
                while (d < n && (std::isalnum((unsigned char)in[d]) || in[d] == '-' ||
                                 (in[d] == '.' && d + 1 < n && std::isalnum((unsigned char)in[d + 1])))) {
                    if (in[d] == '.') dots++;
                    d++;
                }
                if (dots > 0 && d > k + 1) {
                    for (size_t c = i; c < d; c++) {
                        if (in[c] == '@') out += " at ";
                        else if (in[c] == '.') out += " dot ";
                        else if (in[c] == '_') out += " underscore ";
                        else out += in[c];
                    }
                    i = d; continue;
                }
            }
        }
        out += in[i++];
    }
    return out;
}

// ── Top-level ───────────────────────────────────────────────────────────────

inline std::string preprocess_text(const std::string& input) {
    const std::string text = expand_abbreviations(input);
    Scanner sc(text);

    while (!sc.done()) {
        uint8_t c = (uint8_t)sc.at(0);

        // ── Multi-byte UTF-8: currency or transliterate ─────────────────
        if (c >= 0x80) {
            // Try multi-byte currency symbols (€, £, ¥) before transliterating
            CurrencyInfo ci;
            if (match_currency(sc, 0, ci)) {
                if (try_money(sc, ci)) continue;
            }
            // Not a currency — transliterate inline
            uint32_t cp = decode_utf8(sc.s, sc.len, sc.pos);
            const char* rep = transliterate(cp);
            if (rep) {
                for (const char* p = rep; *p; p++)
                    if (*p >= 32 && *p < 127) sc.out += *p;
            }
            continue;
        }

        // ── Dollar sign ────────────────────────────────────────────────
        if (c == '$') {
            CurrencyInfo ci = {"dollar", "dollars", "cent", "cents", 1};
            if (try_money(sc, ci)) continue;
        }

        // ── Parenthesized phone ─────────────────────────────────────────
        if (c == '(' && try_phone_paren(sc)) continue;

        // ── Digit trigger ───────────────────────────────────────────────
        if (std::isdigit(c)) {
            // "COVID-19" → "COVID nineteen" (hyphen between letters and digits)
            if (sc.pos >= 2 && sc.s[sc.pos - 1] == '-' && std::isalpha((unsigned char)sc.s[sc.pos - 2]) &&
                !sc.out.empty() && sc.out.back() == '-')
                sc.out.back() = ' ';
            if (try_date_slash(sc)) continue;
            if (try_date_dmy_dot(sc)) continue;
            if (try_date_iso(sc)) continue;
            if (try_date_ymd_slash(sc)) continue;
            if (try_date_day_month(sc)) continue;
            if (try_time(sc)) continue;
            if (try_time_dot(sc)) continue;
            if (try_hour_period(sc)) continue;
            if (try_phone_country(sc)) continue;
            if (try_phone_10digit(sc)) continue;
            if (try_phone_7digit(sc)) continue;
            if (try_number_idiom(sc)) continue;
            if (try_fraction(sc)) continue;
            if (try_ordinal(sc)) continue;
            if (try_percent(sc)) continue;
            if (try_number_unit(sc)) continue;
            if (try_decimal(sc)) continue;
            if (try_number(sc)) continue;
        }

        // ── Alpha trigger (textual dates, dotted initialisms) ───────────
        if (std::isupper(c)) {
            if (try_date_textual(sc)) continue;
            if (try_dotted_initialism(sc)) continue;
            if (try_roman(sc)) continue;
        }

        // ── Symbol trigger ──────────────────────────────────────────────
        if ((c == '&' || c == '@' || c == '+' || c == '=') && try_symbol(sc)) continue;

        // ── Negative numbers: "-5" at a word start → "minus five" ─────
        if (c == '-' && sc.is_digit(1) && (sc.pos == 0 || sc.s[sc.pos - 1] == ' ' || sc.s[sc.pos - 1] == '(')) {
            sc.emit("minus ");
            sc.advance(1);
            continue;
        }

        // ── Whitespace: collapse if leading or after stripped chars ─────
        if (c == ' ' || c == '\t') {
            if (sc.out.empty() || sc.out.back() != ' ')
                sc.out += ' ';
            sc.advance(1);
            while (!sc.done() && (sc.at(0) == ' ' || sc.at(0) == '\t'))
                sc.advance(1);
            continue;
        }

        // ── ASCII control chars: strip non-printable ────────────────────
        if (c < 32 || c == 127) { sc.advance(1); continue; }

        // ── Default: bulk-copy with inline whitespace collapse ──────────
        {
            size_t start = sc.pos;
            sc.pos++;
            while (sc.pos < sc.len) {
                uint8_t ch = (uint8_t)sc.s[sc.pos];
                if (ch >= 0x80 || std::isdigit(ch) || ch == '$' || ch == '(') break;
                if (std::isupper(ch)) break;
                if (ch == '&' || ch == '@' || ch == '+' || ch == '=' || ch == '-') break;
                if (ch < 32 || ch == 127) break;
                if (ch == ' ' || ch == '\t') {
                    // Flush what we have, emit one space, skip whitespace run
                    sc.out.append(sc.s + start, sc.pos - start);
                    sc.out += ' ';
                    sc.pos++;
                    while (sc.pos < sc.len && (sc.s[sc.pos] == ' ' || sc.s[sc.pos] == '\t'))
                        sc.pos++;
                    start = sc.pos;
                    continue;
                }
                sc.pos++;
            }
            if (sc.pos > start)
                sc.out.append(sc.s + start, sc.pos - start);
        }
    }

    // Short utterances without final punctuation produce bad TTS output.
    while (!sc.out.empty() && sc.out.back() == ' ')
        sc.out.pop_back();
    if (!sc.out.empty()) {
        size_t k = sc.out.size();
        // Look through closing quotes/brackets: '"Stop."' already ends a sentence
        while (k > 0 && (sc.out[k - 1] == '"' || sc.out[k - 1] == '\'' || sc.out[k - 1] == ')')) k--;
        char last = k > 0 ? sc.out[k - 1] : '\0';
        if (last != '.' && last != '!' && last != '?')
            sc.out.push_back('.');
    }

    return std::move(sc.out);
}

} // namespace text_norm
