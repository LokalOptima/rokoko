"""Record exact dataset and source identities alongside training runs."""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def code_record():
    files = ["src/normalize.h", "training/g2p/model.py", "training/g2p/train.py",
             "training/g2p/prepare_data.py", "training/g2p/normalizer.py",
             "training/g2p/provenance.py", "training/g2p/augment_data.py"]
    result = {"sha256": {name: file_record(ROOT / name)["sha256"] for name in files}}
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    result["commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    return result
