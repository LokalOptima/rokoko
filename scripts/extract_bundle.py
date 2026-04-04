#!/usr/bin/env python3
"""Extract individual files from a .bundle archive for upload as release assets.

Extracts:
  - weights.bin (or weights.fp16.bin)  — the KOKO weight blob
  - g2p.bin                            — the G2P model
  - voices/af_heart.bin, etc.          — individual voice packs

Usage:
    uv run scripts/extract_bundle.py --bundle ~/.cache/rokoko/rokoko.bundle -o release/
    uv run scripts/extract_bundle.py --bundle ~/.cache/rokoko/rokoko.fp16.bundle -o release/ --weights-name weights.fp16.bin
"""

import argparse
import mmap
import struct
from pathlib import Path


ROKO_MAGIC = 0x4F4B4F52  # "ROKO" LE


def read_bundle(path: str) -> dict[str, bytes]:
    """Read a .bundle file and return {name: raw_bytes} for each entry."""
    with open(path, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != ROKO_MAGIC:
        raise ValueError(f"Not a bundle: bad magic {magic:#x}")
    _version, count = struct.unpack_from("<II", data, 4)
    entries = {}
    for i in range(count):
        off = 16 + i * 72
        name = data[off : off + 56].split(b"\x00", 1)[0].decode()
        entry_off, entry_sz = struct.unpack_from("<QQ", data, off + 56)
        entries[name] = bytes(data[entry_off : entry_off + entry_sz])
    return entries


def main():
    p = argparse.ArgumentParser(description="Extract bundle into individual files")
    p.add_argument("--bundle", required=True, help="Path to .bundle file")
    p.add_argument("-o", "--outdir", required=True, help="Output directory")
    p.add_argument("--weights-name", default="weights.bin",
                   help="Filename for the weights entry (default: weights.bin)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {args.bundle}")
    entries = read_bundle(args.bundle)
    print(f"  {len(entries)} entries: {', '.join(sorted(entries))}")

    for name, data in entries.items():
        if name == "weights":
            out_path = outdir / args.weights_name
        elif name == "g2p":
            out_path = outdir / "g2p.bin"
        elif name.startswith("voice/"):
            vname = name.split("/", 1)[1]
            vdir = outdir / "voices"
            vdir.mkdir(exist_ok=True)
            out_path = vdir / f"{vname}.bin"
        else:
            out_path = outdir / name
            out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_bytes(data)
        print(f"  {name} → {out_path} ({len(data) / 1e6:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
