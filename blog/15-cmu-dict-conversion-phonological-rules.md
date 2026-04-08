## 2026-03-05: CMU Dict Conversion — Phonological Rules

### The problem

Our C++ phonemizer falls back to CMU Pronouncing Dictionary for words not in misaki's gold/silver lexicons (~68K words: proper nouns, technical terms). CMU uses ARPAbet; we need misaki IPA. A flat 1:1 symbol substitution gets only 47.5% exact match on the 47,750 words that overlap between CMU and our known-good dictionaries.

The gap isn't bugs — it's **allophonic variation**. The same underlying phoneme surfaces differently in context. Three rules account for most of the fixable errors.

### Rule 1: Stress-conditioned ER (biggest win)

The old code mapped all ER to `ɜɹ` regardless of stress. But unstressed /ɝ/ reduces to schwa+r:

```
ER0 → əɹ   (teacher → tˈiʧəɹ, butter → bˈʌɾəɹ)
ER1 → ˈɜɹ  (perfect → pˈɜɹfəkt)
```

We initially followed the plan's recommendation of ER0→bare `ɹ`, which came from alignment analysis counting the `ɹ` as "matching" while missing the `ə` that precedes it. Checking actual misaki output showed `əɹ` everywhere. Data beats theory.

### Rule 2: T-flapping

/t/ → [ɾ] between a vowel (or /r/) and an unstressed vowel. Blocked after /l/, /m/, /ŋ/:

```
butter  → bˈʌɾəɹ    water → wˈɔɾəɹ    party → pˈɑɹɾi
melting → mˈɛltɪŋ    (L blocks flapping)
attack  → ətˈæk      (next vowel is stressed → no flap)
```

### Rule 3: Syllabic L (but NOT syllabic N)

AH0 → `ᵊ` before word-final L when preceded by an obstruent:

```
little → lˈɪɾᵊl    puzzle → pˈʌzᵊl    middle → mˈɪdᵊl
```

The plan proposed this for both L and N. Testing showed AH0+N is a trap — "-tion" words (SH+AH0+N → `ʃən`) dominate, giving 76% false positives. We tested five variants:

| Variant | Exact match |
|---------|-------------|
| No Rule 3 | 63.0% |
| L only | 64.8% |
| **L word-final only** | **65.4%** |
| L + N word-final | 63.4% |
| All (plan's rule) | 61.6% |

Word-final restriction matters: medial AH0+L (abolition → `ˌæbəlˈɪʃən`) keeps plain ə.

### What we don't fix

| Pattern | Why not |
|---------|---------|
| T → ʔ (button → bˈʌʔn) | Glottalization is variable, can't predict from CMU |
| AH0 → ɪ (2% of cases) | Morphological (suffixes like -ist), not phonological |
| IH0 → ə (24% of cases) | Word-specific, no clean rule |
| AA ↔ AO (cot-caught) | Dialect variation, word-specific |

### Result

**47.5% → 65.4%** exact match (+8,549 words). The remaining 34.6% are CMU-vs-misaki transcription disagreements that no rule can fix. Implementation is ~300 lines in `scripts/export_cmu.py`, all rules applied in a single pass.
