#!/usr/bin/env python3
"""compare_wav.py — Mel spectrogram SNR comparison for TTS audio quality verification.

Three-way comparison: positive control, test, negative control.
Uses 80-bin log mel spectrogram (n_fft=1024, hop=256, sr=24000).

Thresholds (established for Kokoro TTS with TF32 non-determinism):
  ~35 dB mel SNR, max_mel_diff < 0.15: identical within TF32 noise
  ~29+ dB mel SNR, max_mel_diff < 0.30: FP16 quantization noise (acceptable)
  < 5 dB mel SNR: fundamentally different audio

Usage:
  # Compare two WAV files:
  python compare_wav.py ref.wav test.wav

  # Full three-way comparison (positive control, test, negative control):
  python compare_wav.py --ref-a ref_a.wav --ref-b ref_b.wav --test test.wav --neg neg.wav

  # Generate reference + test from two binaries:
  python compare_wav.py --binary-a ./rokoko_old --binary-b ./rokoko_new \\
      --text "The quick brown fox jumps over the lazy dog."

  # With paraketto STT verification:
  python compare_wav.py ref.wav test.wav --stt ~/git/LokalOptima/paraketto/paraketto.fp8 \\
      --text "Hello world."
"""
import argparse
import subprocess
import sys
import tempfile

import numpy as np


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load WAV file, return (samples_float32, sample_rate)."""
    import soundfile as sf
    audio, sr = sf.read(path)
    return audio.astype(np.float32), sr


def mel_spectrogram(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    """Compute 80-bin log mel spectrogram."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80
    )
    return np.log1p(mel)


def compute_mel_snr(mel_a: np.ndarray, mel_b: np.ndarray) -> tuple[float, float]:
    """Compute mel spectrogram SNR and max difference."""
    diff = mel_a - mel_b
    signal_power = np.mean(mel_a ** 2)
    noise_power = np.mean(diff ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-30))
    max_diff = float(np.max(np.abs(diff)))
    return float(snr), max_diff


def compare_pair(path_a: str, path_b: str) -> tuple[float, float]:
    """Compare two WAV files, return (mel_snr_dB, max_mel_diff)."""
    a, sr_a = load_wav(path_a)
    b, sr_b = load_wav(path_b)
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    mel_a = mel_spectrogram(a)
    mel_b = mel_spectrogram(b)
    return compute_mel_snr(mel_a, mel_b)


def generate_wav(binary: str, text: str, output: str) -> None:
    """Run TTS binary to generate WAV."""
    result = subprocess.run(
        [binary, text, "-o", output],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"Error running {binary}:", result.stderr, file=sys.stderr)
        sys.exit(1)


def stt_verify(paraketto: str, wav_path: str, expected_text: str) -> tuple[bool, str]:
    """Verify WAV via paraketto STT. Returns (passed, transcription)."""
    import re
    result = subprocess.run(
        [paraketto, wav_path],
        capture_output=True, text=True, timeout=30,
    )
    actual = result.stdout.strip().split("\n")[-1] if result.stdout else ""
    normalize = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return normalize(expected_text) == normalize(actual), actual


def main():
    parser = argparse.ArgumentParser(description="WAV mel spectrogram SNR comparison")
    parser.add_argument("wav_a", nargs="?", help="Reference WAV file")
    parser.add_argument("wav_b", nargs="?", help="Test WAV file")
    parser.add_argument("--ref-a", help="Reference A (for positive control)")
    parser.add_argument("--ref-b", help="Reference B (for positive control)")
    parser.add_argument("--test", help="Test WAV (for FP16 vs FP32 comparison)")
    parser.add_argument("--neg", help="Negative control WAV (different text)")
    parser.add_argument("--binary-a", help="Baseline binary (generates ref-a, ref-b)")
    parser.add_argument("--binary-b", help="Test binary (generates test WAV)")
    parser.add_argument("--text", help="Text for TTS generation / STT verification")
    parser.add_argument("--stt", help="Path to paraketto binary for STT verification")
    args = parser.parse_args()

    # Simple two-file mode
    if args.wav_a and args.wav_b and not args.ref_a:
        snr, md = compare_pair(args.wav_a, args.wav_b)
        print(f"Mel SNR: {snr:.1f} dB | max_mel_diff: {md:.4f}")
        if args.stt and args.text:
            passed, actual = stt_verify(args.stt, args.wav_b, args.text)
            print(f"STT: {'PASS' if passed else 'FAIL'} | \"{actual}\"")
        return

    # Full three-way mode
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_a = args.ref_a
        ref_b = args.ref_b
        test = args.test
        neg = args.neg

        # Generate from binaries if specified
        if args.binary_a and args.text:
            if not ref_a:
                ref_a = f"{tmpdir}/ref_a.wav"
                generate_wav(args.binary_a, args.text, ref_a)
            if not ref_b:
                ref_b = f"{tmpdir}/ref_b.wav"
                generate_wav(args.binary_a, args.text, ref_b)
        if args.binary_b and args.text:
            if not test:
                test = f"{tmpdir}/test.wav"
                generate_wav(args.binary_b, args.text, test)

        results = []
        if ref_a and ref_b:
            snr, md = compare_pair(ref_a, ref_b)
            results.append(("Positive (FP32 vs FP32)", snr, md))
            print(f"Positive control (baseline vs baseline):  SNR={snr:6.1f} dB  max_diff={md:.4f}")

        if ref_a and test:
            snr, md = compare_pair(ref_a, test)
            results.append(("Test (FP32 vs FP16)", snr, md))
            print(f"Test (baseline vs test):                  SNR={snr:6.1f} dB  max_diff={md:.4f}")

        if ref_a and neg:
            snr, md = compare_pair(ref_a, neg)
            results.append(("Negative (different text)", snr, md))
            print(f"Negative control (different text):        SNR={snr:6.1f} dB  max_diff={md:.4f}")

        if args.stt and args.text and test:
            passed, actual = stt_verify(args.stt, test, args.text)
            print(f"\nSTT verification: {'PASS' if passed else 'FAIL'} | \"{actual}\"")


if __name__ == "__main__":
    main()
