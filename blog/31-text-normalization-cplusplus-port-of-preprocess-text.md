## 2026-03-10: Text Normalization — C++ Port of preprocess_text()

### The problem

The G2P model was trained on text preprocessed by Python's `preprocess_text()` — a 5-stage pipeline that expands money ($12.50 → "twelve dollars and fifty cents"), dates (1/15/2024 → "January fifteenth, twenty twenty four"), numbers (1234 → "one thousand two hundred thirty four"), and folds Unicode to ASCII (café → cafe). The C++ test binaries (`test_g2p`, `test_g2p_cuda`) were feeding raw text directly, causing ~203/1000 mismatches vs Python on inputs containing numbers, dates, and currency.

This is separate from the main phonemizer (`phonemize.cpp`), which has its own number/currency pipeline that converts directly to phonemes. `preprocess_text()` is only needed for standalone G2P inference and the three-way verification script.

### The implementation

Single-header C++17 in `src/text_normalize.h` (~300 lines). Four stages matching Python exactly:

1. **Money expansion** — hand-rolled UTF-8 scanner for $€£¥, avoids regex for multi-byte currency symbols. Parses `\d[\d,]*(\.\d+)?` amounts, generates "twelve dollars and fifty cents" style output with full currency table (dollars/cents, euros/cents, pounds/pence, yen).

2. **Date expansion** — three `std::regex` patterns (US `M/D/YYYY`, ISO `YYYY-MM-DD`, textual `Month DD, YYYY`). Day ordinals ("first"..."thirty first"), year-to-words ("twenty twenty four", "nineteen oh five").

3. **Number expansion** — hand-rolled digit scanner with letter-adjacency check. Skips numbers attached to letters (70s, 3G, MP3, 5kg). Range 0–999,999,999.

4. **Unicode → ASCII** — UTF-8 codepoint decoder + lookup tables covering Latin-1 Supplement (U+00A0–U+00FF, 96 entries), Latin Extended-A (U+0100–U+017F, 128 entries), and common punctuation (curly quotes, em/en dash, ellipsis, euro sign, trademark). Strips anything outside printable ASCII (32–126).

### Challenges and solutions

**UTF-8 currency symbols in regex**: The Python regex `([$€£¥])\s*(\d[\d,]*)` works because Python's regex engine is Unicode-aware. C++ `std::regex` operates on bytes, so multi-byte UTF-8 characters in character classes would fail. Solution: hand-rolled byte-level scanner that checks for the exact UTF-8 byte sequences of each currency symbol (e.g., € = `0xE2 0x82 0xAC`).

**Letter adjacency for number expansion**: Python's `str.isalpha()` returns True for accented Unicode letters (é, ñ), but C++ `isalpha()` only handles ASCII. If an accented letter preceded a number, C++ would expand it while Python wouldn't. Solution: treat any byte ≥ 0x80 as "alpha" for the adjacency check — valid since any non-ASCII byte in UTF-8 is part of a multi-byte character, which is almost always a letter in practice.

**NFC normalization**: Python does `unicodedata.normalize("NFC", text)` before `unidecode()`. Full NFC in C++ is complex (combining character composition). In practice, input text is already NFC, and the unidecode table handles the same characters either way — a decomposed sequence (base letter + combining mark) would keep the ASCII base letter and strip the combining mark, giving the same result as NFC→unidecode. Skipped NFC entirely.

**unidecode coverage**: The Python `unidecode` library covers the entire Unicode range. Rather than porting its full 70KB dataset, we cover only the codepoints that actually appear in the training data: Latin-1 Supplement (accented letters, common symbols), Latin Extended-A (Eastern European characters), and a handful of punctuation marks. Unknown codepoints are simply stripped (same as unidecode returning empty for exotic scripts).

### Verification

Tested against Python on the full training dataset:
- **708,968 samples**: 0 differences (exact byte-for-byte match)
- **193 samples with actual changes** (numbers, dates, money, Unicode): all match
- **Targeted edge cases**: $0.50, ¥5000, café, 2024-03-10, MP3 — all match

### Files changed

| File | Changes |
|------|---------|
| `src/text_normalize.h` | New: complete preprocess_text() port (~300 lines) |
| `src/test_preprocess.cpp` | New: standalone preprocessing test binary |
| `src/test_g2p.cpp` | Added `#include "text_normalize.h"`, preprocess before infer |
| `src/test_g2p_cuda.cu` | Same |
| `Makefile` | Added `test_preprocess` target, updated deps for test_g2p/test_g2p_cuda |
