#!/usr/bin/env python3
"""Pack weights, G2P model, and voice files into a single .bundle file.

Format:
  Header (16 bytes):
    magic    u32  "ROKO"
    version  u32  1
    count    u32  number of entries
    reserved u32  0
  Entries (72 bytes each):
    name     56 bytes (null-padded)
    offset   u64  absolute offset from file start
    size     u64  entry size in bytes
  [padding to 4096 alignment]
  [entry data, each 256-byte aligned]
"""

import argparse
import struct
from pathlib import Path

MAGIC = b"ROKO"
VERSION = 1
HEADER_ALIGN = 4096
DATA_ALIGN = 256


def align(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


def pack(output: Path, files: list[tuple[str, Path]]):
    count = len(files)
    header_size = 16 + 72 * count
    data_start = align(header_size, HEADER_ALIGN)

    # Compute offsets
    entries = []
    offset = data_start
    for name, path in files:
        size = path.stat().st_size
        entries.append((name, offset, size, path))
        offset = align(offset + size, DATA_ALIGN)

    with open(output, "wb") as f:
        # Header
        f.write(struct.pack("<4sIII", MAGIC, VERSION, count, 0))

        # Entries
        for name, off, size, _ in entries:
            name_bytes = name.encode("utf-8")[:55].ljust(56, b"\x00")
            f.write(struct.pack("<56sQQ", name_bytes, off, size))

        # Pad to data start
        f.write(b"\x00" * (data_start - f.tell()))

        # Data
        for name, off, size, path in entries:
            assert f.tell() == off, f"{name}: expected offset {off}, at {f.tell()}"
            f.write(path.read_bytes())
            # Pad to alignment
            pad = align(f.tell(), DATA_ALIGN) - f.tell()
            if pad:
                f.write(b"\x00" * pad)

    total = output.stat().st_size
    print(f"packed {count} entries -> {output} ({total / 1e6:.1f} MB)")
    for name, off, size, _ in entries:
        print(f"  {name:20s}  {size / 1e6:8.1f} MB  @ {off:#x}")


def main():
    p = argparse.ArgumentParser(description="Pack rokoko bundle")
    p.add_argument("-o", "--output", type=Path, default=Path("rokoko.bundle"))
    p.add_argument("--weights", type=Path, default=Path("weights/weights.bin"))
    p.add_argument("--g2p", type=Path, default=Path("weights/g2p_v8_model.bin"))
    p.add_argument("--voices", type=Path, default=Path("voices"), help="voices directory")
    args = p.parse_args()

    files = [
        ("weights", args.weights),
        ("g2p", args.g2p),
    ]
    for vf in sorted(args.voices.glob("*.bin")):
        files.append((f"voice/{vf.stem}", vf))

    for name, path in files:
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    pack(args.output, files)


if __name__ == "__main__":
    main()
