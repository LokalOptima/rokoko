#!/usr/bin/env python3
"""Prepare G2P training data: normalize → phonemize → TSV.

Pipeline:
  1. Read raw text from input file(s) (one sentence per line)
  2. Normalize through normalize_cli (C++ binary, single source of truth)
  3. Phonemize normalized text with Misaki (parallel workers)
  4. Filter: drop empty phonemes and lines with ❓
  5. Dedup by normalized text
  6. Output: normalized_text \\t phonemes (TSV)

The TSV stores NORMALIZED text in column 1 (not raw). This ensures that
the trainer can load it as-is. Never normalize the prepared TSV a second time.

Usage:
    # Process natural sentences (slow — 713K lines through Misaki)
    python training/g2p/prepare_data.py \\
        --input raw/natural/all_sentences.txt \\
        --output data/base.tsv \\
        --workers 8

    # Process synthetic augmentation
    python training/g2p/prepare_data.py \\
        --input raw/synthetic/augment_heuristic.txt \\
               raw/synthetic/augment_llm.txt \\
        --output data/augment_heuristic.tsv \\
        --workers 4

    # Re-process LLM augmentation (already has phonemes, but re-run for consistency)
    python training/g2p/prepare_data.py \\
        --input-tsv raw/synthetic/augment_llm_v2.tsv \\
        --output data/augment_llm.tsv \\
        --workers 4
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time


# ── Normalizer ──────────────────────────────────────────────────────────────


from normalizer import find_normalize_cli, normalize_batch


# ── Phonemization (parallel) ────────────────────────────────────────────────
# Uses misaki.en.G2P with espeak-ng fallback for unknown words (proper nouns
# etc.). This matches how the original g2p_train_v3.tsv was generated via
# KPipeline, but without pulling in torch/kokoro.


def _init_worker():
    """Initialize Misaki G2P with espeak fallback in each worker process."""
    global _g2p
    from misaki import en, espeak
    fallback = espeak.EspeakFallback(british=False)
    _g2p = en.G2P(trf=False, british=False, fallback=fallback, unk="")


def _phonemize_chunk(chunk):
    """Phonemize a list of (index, normalized_text) pairs."""
    global _g2p
    results = []
    for idx, text in chunk:
        try:
            ph, _ = _g2p(text)
            if ph and "\u2753" not in ph:
                results.append((idx, ph))
        except Exception:
            pass
    return results


def phonemize_parallel(texts, n_workers):
    """Phonemize texts using multiple Misaki workers."""
    indexed = list(enumerate(texts))
    chunk_size = max(1, len(indexed) // (n_workers * 10))
    chunks = [indexed[i:i + chunk_size] for i in range(0, len(indexed), chunk_size)]

    phonemes = {}
    done = 0

    with multiprocessing.Pool(n_workers, initializer=_init_worker) as pool:
        for batch_results in pool.imap_unordered(_phonemize_chunk, chunks):
            for idx, ph in batch_results:
                phonemes[idx] = ph
            done += len(batch_results)
            print(f"\r  Phonemized: {done}/{len(texts)}", end="", flush=True)

    print()
    return phonemes


# ── File I/O ────────────────────────────────────────────────────────────────


def read_text_files(paths):
    """Read raw sentences from plain text files (one per line)."""
    lines = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    lines.append(line)
    return lines


def read_tsv_raw_column(path):
    """Read raw text from column 1 of a TSV file."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                lines.append(parts[0])
    return lines


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare G2P training data: normalize → phonemize → TSV"
    )
    parser.add_argument("--input", nargs="+", help="Raw text files (one sentence per line)")
    parser.add_argument("--input-tsv", help="TSV file (reads column 1 as raw text)")
    parser.add_argument("--output", required=True, help="Output TSV path")
    parser.add_argument("--workers", type=int, default=4, help="Parallel Misaki workers")
    parser.add_argument("--normalizer", help="Path to normalize_cli binary")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if os.path.exists(args.output):
        parser.error("Output already exists; choose a new dataset path")

    if not args.input and not args.input_tsv:
        parser.error("Provide --input or --input-tsv")

    # Step 1: Read raw text
    print("Reading input...")
    if args.input:
        raw_lines = read_text_files(args.input)
        sources = args.input
    else:
        raw_lines = read_tsv_raw_column(args.input_tsv)
        sources = [args.input_tsv]

    print(f"  {len(raw_lines)} raw lines from {len(sources)} file(s)")
    for s in sources:
        print(f"    {s}: {count_lines(s)} lines")

    # Step 2: Normalize
    binary = args.normalizer or find_normalize_cli()
    print(f"\nNormalizing through {binary}...")
    BATCH = 10000
    normalized = []
    for start in range(0, len(raw_lines), BATCH):
        batch = raw_lines[start:start + BATCH]
        normalized.extend(normalize_batch(binary, batch))
        print(f"\r  {min(start + BATCH, len(raw_lines))}/{len(raw_lines)}", end="", flush=True)
    print()

    # Step 3: Dedup by normalized text
    print("\nDeduplicating...")
    seen = set()
    unique = []  # (raw, normalized)
    for raw, norm in zip(raw_lines, normalized):
        key = norm.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append((raw, norm))
    print(f"  {len(raw_lines)} → {len(unique)} unique")

    # Step 4: Phonemize
    print(f"\nPhonemizing with Misaki ({args.workers} workers)...")
    t0 = time.monotonic()
    normed_texts = [norm for _, norm in unique]
    phonemes = phonemize_parallel(normed_texts, args.workers)
    elapsed = time.monotonic() - t0
    print(f"  {len(phonemes)} succeeded, {len(unique) - len(phonemes)} failed ({elapsed:.0f}s)")

    # Step 5: Write TSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, (raw, norm) in enumerate(unique):
            if i in phonemes:
                f.write(f"{norm}\t{phonemes[i]}\n")
                count += 1

    print(f"\nWrote {count} pairs to {args.output}")
    print(f"  MD5: {md5_file(args.output)}")

    # Bind generated data to its inputs and the production normalizer.
    from provenance import file_record, code_record
    manifest = {
        "sources": [file_record(path) for path in sources],
        "output": file_record(args.output),
        "normalizer_binary": file_record(binary),
        "code": code_record(),
        "phonemizer": "misaki==0.9.4, American English, espeak fallback",
        "format": "normalized_text\\tphonemes",
        "pairs": count,
    }
    with open(args.output + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # Step 6: Verification
    print("\nVerification:")
    with open(args.output, encoding="utf-8") as f:
        q_count = sum(1 for line in f if "\u2753" in line)
    print(f"  ❓ tokens: {q_count}")

    with open(args.output, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  Total lines: {len(lines)}")
    print(f"  Sample (first 5):")
    for line in lines[:5]:
        parts = line.rstrip("\n").split("\t")
        print(f"    TEXT: {parts[0][:70]}")
        print(f"    PHON: {parts[1][:70] if len(parts) > 1 else 'MISSING'}")
        print()


if __name__ == "__main__":
    main()
