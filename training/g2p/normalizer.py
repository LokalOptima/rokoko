"""Use the production C++ normalizer; no separate training implementation."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def find_normalize_cli():
    binary = ROOT / "tests/frontend/normalize_cli"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError("Build the production normalizer with: make tests/frontend/normalize_cli")
    return str(binary)


def normalize_batch(binary_path, texts):
    if not texts:
        return []
    if any("\n" in text or "\r" in text or "\t" in text for text in texts):
        raise ValueError("Training input must contain one sentence per line, without tabs")
    result = subprocess.run([str(binary_path)], input="\n".join(texts) + "\n",
                            capture_output=True, text=True, check=True)
    lines = result.stdout.split("\n")
    if lines[-1:] != [""] or len(lines) != len(texts) + 1:
        raise RuntimeError("Normalizer returned the wrong number of lines")
    return lines[:-1]


class Normalizer:
    def __init__(self, binary_path=None):
        self.binary_path = str(Path(binary_path or find_normalize_cli()).resolve())

    def normalize_batch(self, texts):
        return normalize_batch(self.binary_path, texts)

    def normalize(self, text):
        return self.normalize_batch([text])[0]
