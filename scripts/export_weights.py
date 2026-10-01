#!/usr/bin/env python3
"""Export PyTorch weights to a flat binary file for the Rokoko CUDA backend.

Loads the KModel checkpoint, iterates model.state_dict(), and writes
all parameters to a single binary file with text header + aligned tensor data.

Weight file format (weights.bin):
  [4 bytes: "KOKO" magic]
  [4 bytes: version uint32 = 1]
  [8 bytes: header_len uint64 — byte length of the text index]
  [header_len bytes: text index, one line per tensor]
  [padding to 4096-byte alignment]
  [raw tensor data, each tensor 256-byte aligned]

Header text format (one line per tensor):
  name offset_from_data_start size_bytes dtype dim0 dim1 ...
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

MAGIC = b"KOKO"
VERSION = 1
HEADER_ALIGN = 4096
TENSOR_ALIGN = 256

DTYPE_NAMES = {
    np.dtype(np.float16): "fp16",
    np.dtype(np.float32): "fp32",
    np.dtype(np.int32): "int32",
    np.dtype(np.int64): "int64",
}


def align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) & ~(alignment - 1)


def export_weights(output_path: Path, official: Path) -> dict[str, np.ndarray]:
    """Export PyTorch model weights to a flat binary file."""
    from kokoro import KModel

    print("Loading KModel...")
    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=str(official / "config.json"),
                   model=str(official / "kokoro-v1_0.pth")).eval()
    sd = model.state_dict()
    print(f"  {len(sd)} tensors in state_dict")

    # Convert to numpy
    all_tensors: dict[str, np.ndarray] = {}
    for name, tensor in sd.items():
        all_tensors[name] = tensor.detach().cpu().numpy()

    # Print structure summary
    from collections import defaultdict
    groups = defaultdict(lambda: [0, 0])  # [count, bytes]
    for name, arr in all_tensors.items():
        prefix = name.split(".")[0]
        groups[prefix][0] += 1
        groups[prefix][1] += arr.nbytes

    print("\nModel structure:")
    for prefix in sorted(groups.keys()):
        count, nbytes = groups[prefix]
        print(f"  {prefix:20s} {count:4d} tensors  {nbytes / 1e6:8.1f} MB")

    # Keep an exact FP32 intermediate for offline conversion to the runtime format.
    print("\nKeeping FP32 weights...")
    for name, arr in list(all_tensors.items()):
        if arr.dtype == np.float64:
            all_tensors[name] = arr.astype(np.float32)
    print(f"  All float tensors stored as FP32")

    # Sort tensors by name for deterministic output
    sorted_names = sorted(all_tensors.keys())

    # Print all tensor names for reference
    print(f"\nTensor index ({len(sorted_names)} tensors):")
    for name in sorted_names:
        arr = all_tensors[name]
        print(f"  {name:80s} {str(list(arr.shape)):25s} {DTYPE_NAMES.get(arr.dtype, str(arr.dtype))}")

    # Build the data section: pack tensors with 256-byte alignment
    data_parts: list[bytes] = []
    tensor_index: list[tuple[str, int, int, str, tuple]] = []
    current_offset = 0

    for name in sorted_names:
        arr = all_tensors[name]
        raw = arr.tobytes()
        size = len(raw)
        dtype_name = DTYPE_NAMES.get(arr.dtype, str(arr.dtype))

        tensor_index.append((name, current_offset, size, dtype_name, tuple(arr.shape)))
        data_parts.append(raw)

        # Pad to next 256-byte boundary
        padded_size = align_up(size, TENSOR_ALIGN)
        if padded_size > size:
            data_parts.append(b"\x00" * (padded_size - size))
        current_offset += padded_size

    # Build text header
    header_lines = []
    for name, offset, size, dtype_name, shape in tensor_index:
        dims = " ".join(str(d) for d in shape)
        header_lines.append(f"{name} {offset} {size} {dtype_name} {dims}")
    header_text = "\n".join(header_lines).encode("utf-8")

    # Write the file
    print(f"\nWriting {output_path}...")
    with open(output_path, "wb") as f:
        # Magic + version + header_len
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(header_text)))
        f.write(header_text)

        # Pad to 4096-byte alignment
        current_pos = 4 + 4 + 8 + len(header_text)
        pad_to = align_up(current_pos, HEADER_ALIGN)
        if pad_to > current_pos:
            f.write(b"\x00" * (pad_to - current_pos))

        # Write all tensor data
        for part in data_parts:
            f.write(part)

    file_size = output_path.stat().st_size
    print(f"  Written: {file_size:,} bytes ({file_size / 1e6:.1f} MB)")

    # Summary
    print(f"\nSummary:")
    print(f"  Total tensors: {len(sorted_names)}")
    print(f"  Data size: {current_offset:,} bytes ({current_offset / 1e6:.1f} MB)")
    print(f"  File size: {file_size:,} bytes ({file_size / 1e6:.1f} MB)")

    # Per-section breakdown
    for prefix in sorted(groups.keys()):
        section = [(n, all_tensors[n]) for n in sorted_names if n.startswith(prefix + ".")]
        total = sum(a.nbytes for _, a in section)
        print(f"  {prefix}: {len(section)} tensors, {total:,} bytes ({total / 1e6:.1f} MB)")

    return all_tensors


def verify_weights(output_path: Path, original_tensors: dict[str, np.ndarray]) -> None:
    """Reload the .bin file and verify every tensor matches the original exactly."""
    print(f"\nVerifying {output_path}...")

    with open(output_path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, f"Bad magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        assert version == VERSION, f"Bad version: {version}"
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_text = f.read(header_len).decode("utf-8")

        # Skip to data start (4096-byte aligned)
        header_end = 4 + 4 + 8 + header_len
        data_start = align_up(header_end, HEADER_ALIGN)
        f.seek(data_start)
        data = f.read()

    # Parse header
    dtype_map = {"fp16": np.float16, "fp32": np.float32, "int32": np.int32, "int64": np.int64}
    errors = 0
    checked = 0
    for line in header_text.strip().split("\n"):
        parts = line.split()
        name = parts[0]
        offset = int(parts[1])
        size = int(parts[2])
        dtype_str = parts[3]
        shape = tuple(int(d) for d in parts[4:])

        np_dtype = dtype_map.get(dtype_str)
        if np_dtype is None:
            print(f"  WARNING: unknown dtype {dtype_str} for {name}")
            continue

        # Extract tensor from data
        raw = data[offset:offset + size]
        loaded = np.frombuffer(raw, dtype=np_dtype).reshape(shape)

        # Compare with original FP32 tensors.
        original = original_tensors.get(name)
        if original is None:
            print(f"  ERROR: {name} not in original tensors")
            errors += 1
            continue

        if not np.array_equal(loaded, original):
            max_diff = np.max(np.abs(loaded.astype(np.float32) - original.astype(np.float32)))
            print(f"  ERROR: {name} mismatch, max_diff={max_diff}")
            errors += 1
        checked += 1

    if errors:
        print(f"  FAILED: {errors} errors out of {checked} tensors")
        sys.exit(1)
    else:
        print(f"  OK: all {checked} tensors match exactly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True,
                        help="Local official Kokoro-82M config, checkpoint and voices")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--voice-output", type=Path,
                        help="Optional destination for af_heart.bin")
    args = parser.parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_tensors = export_weights(output_path, args.official)
    verify_weights(output_path, original_tensors)
    if args.voice_output:
        voice = torch.load(args.official / "voices/af_heart.pt",
                           map_location="cpu", weights_only=True)
        args.voice_output.parent.mkdir(parents=True, exist_ok=True)
        voice.numpy().astype("<f4").tofile(args.voice_output)
        print(f"Exported af_heart: {args.voice_output}")
    print("\nDone.")


if __name__ == "__main__":
    main()
