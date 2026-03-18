#!/usr/bin/env python3
"""Convert KOKO v1 weights to v2 (pre-baked weight-norm, FP16, NHWC, padded, LSTM bias).

Reads a .bundle or standalone .koko file, writes a standalone .koko v2 file.
The v2 file is consumed by rokoko.fp16 (FP16 binary).

Usage:
    uv run scripts/convert_v2.py [--bundle path] -o weights_v2.koko
"""

import argparse
import mmap
import struct
import sys
from pathlib import Path

import numpy as np


# ── KOKO file I/O ──────────────────────────────────────────────────────────

KOKO_MAGIC = 0x4F4B4F4B  # "KOKO" LE
ROKO_MAGIC = 0x4F4B4F52  # "ROKO" LE


def read_bundle_weights(path: str) -> bytes:
    """Extract the 'weights' entry from a .bundle file."""
    with open(path, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != ROKO_MAGIC:
        raise ValueError(f"Not a bundle: bad magic {magic:#x}")
    _version, count = struct.unpack_from("<II", data, 4)
    for i in range(count):
        off = 16 + i * 72
        name = data[off : off + 56].split(b"\x00", 1)[0].decode()
        entry_off, entry_sz = struct.unpack_from("<QQ", data, off + 56)
        if name == "weights":
            return bytes(data[entry_off : entry_off + entry_sz])
    raise ValueError("Bundle has no 'weights' entry")


def parse_koko(raw: bytes):
    """Parse a KOKO v1 weight blob. Returns {name: {array, dtype, shape}}."""
    magic = struct.unpack_from("<I", raw, 0)[0]
    assert magic == KOKO_MAGIC, f"Bad KOKO magic {magic:#x}"
    version = struct.unpack_from("<I", raw, 4)[0]
    assert version in (1, 2), f"Unsupported version {version}"
    header_len = struct.unpack_from("<Q", raw, 8)[0]
    header_text = raw[16 : 16 + header_len].decode()
    data_start = ((16 + header_len + 4095) // 4096) * 4096

    tensors = {}
    for line in header_text.strip().split("\n"):
        parts = line.split()
        name = parts[0]
        offset = int(parts[1])
        size_bytes = int(parts[2])
        dtype = parts[3]
        shape = tuple(int(x) for x in parts[4:])
        buf = raw[data_start + offset : data_start + offset + size_bytes]
        ndt = np.float32 if dtype == "fp32" else np.float16
        tensors[name] = np.frombuffer(buf, dtype=ndt).reshape(shape).copy()
    return tensors


def write_koko_v2(path: str, entries: list[tuple[str, np.ndarray]]):
    """Write a KOKO v2 file. entries = [(name, array), ...]."""
    # Build header + compute offsets (256-byte aligned)
    header_lines = []
    offset = 0
    for name, arr in entries:
        dtype_str = "fp32" if arr.dtype == np.float32 else "float16"
        size_bytes = arr.nbytes
        shape_str = " ".join(str(s) for s in arr.shape)
        header_lines.append(f"{name} {offset} {size_bytes} {dtype_str} {shape_str}")
        offset = ((offset + size_bytes + 255) // 256) * 256

    header_text = "\n".join(header_lines).encode()
    header_len = len(header_text)
    data_start = ((16 + header_len + 4095) // 4096) * 4096

    with open(path, "wb") as f:
        f.write(struct.pack("<I", KOKO_MAGIC))
        f.write(struct.pack("<I", 2))  # version=2
        f.write(struct.pack("<Q", header_len))
        f.write(header_text)
        f.write(b"\x00" * (data_start - f.tell()))

        for _name, arr in entries:
            data = arr.tobytes()
            f.write(data)
            rem = len(data) % 256
            if rem:
                f.write(b"\x00" * (256 - rem))


# ── Weight transforms ──────────────────────────────────────────────────────


def weight_norm(wg: np.ndarray, wv: np.ndarray) -> np.ndarray:
    """Apply weight norm in-place: wv = wg * wv / ||wv||. Returns normed wv."""
    C_out = wv.shape[0]
    flat = wv.reshape(C_out, -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True).clip(min=1e-12)
    return (wg.reshape(C_out, 1) * flat / norms).reshape(wv.shape).astype(np.float32)


def nhwc_reshape(w: np.ndarray) -> np.ndarray:
    """[C_out, C_in, K] → [C_out, K, C_in] (NHWC for Cutlass conv)."""
    return np.ascontiguousarray(w.transpose(0, 2, 1))


def pad_cin_nhwc(w_nhwc: np.ndarray, C_in_pad: int) -> np.ndarray:
    """Pad last dim of NHWC [C_out, K, C_in] → [C_out, K, C_in_pad]."""
    if w_nhwc.shape[-1] >= C_in_pad:
        return w_nhwc
    pad = [(0, 0), (0, 0), (0, C_in_pad - w_nhwc.shape[-1])]
    return np.pad(w_nhwc, pad, mode="constant").astype(w_nhwc.dtype)


# ── Main converter logic ───────────────────────────────────────────────────


def convert(tensors: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """Convert v1 tensors → v2 tensor list."""
    out: list[tuple[str, np.ndarray]] = []
    t = tensors  # shorthand

    def get(name):
        return t[name]

    # ── 1. Weight norm all weight_g / weight_v pairs ─────────────────────
    # Collect all (wg_name, wv_name) pairs and apply weight norm in-place.
    wg_names = [n for n in t if n.endswith(".weight_g")]
    for wg_name in wg_names:
        wv_name = wg_name.replace(".weight_g", ".weight_v")
        if wv_name in t:
            t[wv_name] = weight_norm(t[wg_name], t[wv_name])

    # ── 2. Copy all original tensors (FP32) to output ────────────────────
    for name, arr in t.items():
        out.append((name, arr))

    # ── 3. FP16 copies for GEMM weights ──────────────────────────────────

    def make_fp16(name):
        if name in t:
            out.append((name + ".f16", t[name].astype(np.float16)))

    # ALBERT
    pfx = "bert.encoder.embedding_hidden_mapping_in"
    make_fp16(f"{pfx}.weight")

    apfx = "bert.encoder.albert_layer_groups.0.albert_layers.0"
    for suffix in [
        ".attention.query.weight",
        ".attention.key.weight",
        ".attention.value.weight",
        ".attention.dense.weight",
        ".ffn.weight",
        ".ffn_output.weight",
    ]:
        make_fp16(apfx + suffix)

    make_fp16("bert_encoder.weight")

    # Text encoder conv (after weight norm)
    for i in range(3):
        make_fp16(f"text_encoder.cnn.{i}.0.weight_v")

    # LSTM weights
    def make_lstm_fp16(prefix, input_size, hidden_size):
        for sfx in [
            ".weight_ih_l0",
            ".weight_hh_l0",
            ".weight_ih_l0_reverse",
            ".weight_hh_l0_reverse",
        ]:
            make_fp16(prefix + sfx)

    make_lstm_fp16("text_encoder.lstm", 512, 256)
    for i in range(3):
        make_lstm_fp16(f"predictor.text_encoder.lstms.{i * 2}", 640, 256)
    make_lstm_fp16("predictor.lstm", 640, 256)
    make_lstm_fp16("predictor.shared", 640, 256)

    # Duration projection
    make_fp16("predictor.duration_proj.linear_layer.weight")

    # AdaLayerNorm FC
    for i in range(3):
        make_fp16(f"predictor.text_encoder.lstms.{i * 2 + 1}.fc.weight")

    # AdaIN1d FC weights + conv weights for F0/N blocks
    def make_adain_fp16(prefix, C, style_dim):
        make_fp16(prefix + ".fc.weight")

    def make_resblk_fp16(prefix, dim_in, dim_out, has_shortcut):
        make_fp16(prefix + ".conv1.weight_v")
        make_fp16(prefix + ".conv2.weight_v")
        make_adain_fp16(prefix + ".norm1", dim_in, 128)
        make_adain_fp16(prefix + ".norm2", dim_out, 128)
        if has_shortcut:
            make_fp16(prefix + ".conv1x1.weight_v")

    f0n_dims = [(512, 512, False), (512, 256, True), (256, 256, False)]
    for chain in ["predictor.F0", "predictor.N"]:
        for i, (di, do, sc) in enumerate(f0n_dims):
            make_resblk_fp16(f"{chain}.{i}", di, do, sc)

    # Decoder
    make_resblk_fp16("decoder.encode", 514, 1024, True)
    dec_dims = [(1090, 1024), (1090, 1024), (1090, 1024), (1090, 512)]
    for i, (di, do) in enumerate(dec_dims):
        make_resblk_fp16(f"decoder.decode.{i}", di, do, True)

    make_fp16("decoder.asr_res.0.weight_v")

    # Generator ups
    make_fp16("decoder.generator.ups.0.weight_v")  # [512, 256, 20]
    make_fp16("decoder.generator.ups.1.weight_v")  # [256, 128, 12]

    # Generator resblocks + noise_res
    rb_ch = [256, 256, 256, 128, 128, 128]
    rb_k = [3, 7, 11, 3, 7, 11]
    for i in range(6):
        pfx = f"decoder.generator.resblocks.{i}"
        C, K = rb_ch[i], rb_k[i]
        for j in range(3):
            make_fp16(f"{pfx}.convs1.{j}.weight_v")
            make_fp16(f"{pfx}.convs2.{j}.weight_v")
            make_adain_fp16(f"{pfx}.adain1.{j}", C, 128)
            make_adain_fp16(f"{pfx}.adain2.{j}", C, 128)

    nr_ch = [256, 128]
    nr_k = [7, 11]
    for i in range(2):
        pfx = f"decoder.generator.noise_res.{i}"
        C, K = nr_ch[i], nr_k[i]
        for j in range(3):
            make_fp16(f"{pfx}.convs1.{j}.weight_v")
            make_fp16(f"{pfx}.convs2.{j}.weight_v")
            make_adain_fp16(f"{pfx}.adain1.{j}", C, 128)
            make_adain_fp16(f"{pfx}.adain2.{j}", C, 128)

    # Generator conv_post + noise_convs[0]: [256,22,12], noise_convs[1]: [128,22,1]
    make_fp16("decoder.generator.conv_post.weight_v")
    make_fp16("decoder.generator.noise_convs.0.weight")
    make_fp16("decoder.generator.noise_convs.1.weight")
    # noise_convs[0] bias is [256], noise_convs[1] bias is [128]

    # ── 4. NHWC FP16 for Cutlass conv (K>1, aligned channels) ───────────

    def make_nhwc_f16(name, C_out, C_in, K):
        if K <= 1:
            return
        wv = t.get(name)
        if wv is None:
            return
        w_nhwc = nhwc_reshape(wv.reshape(C_out, C_in, K))
        # Aligned path: C_in % 8 == 0 and C_out % 4 == 0
        if C_in % 8 == 0 and C_out % 4 == 0:
            out.append((name + ".nhwc_f16", w_nhwc.astype(np.float16)))
        # Padded path: C_in % 8 != 0 but C_out % 4 == 0
        elif C_in % 8 != 0 and C_out % 4 == 0:
            C_in_pad = (C_in + 7) & ~7
            w_pad = pad_cin_nhwc(w_nhwc, C_in_pad)
            out.append((f"{name}.nhwc_f16_pad{C_in_pad}", w_pad.astype(np.float16)))

    # Text encoder conv (K=5, 512→512)
    for i in range(3):
        make_nhwc_f16(f"text_encoder.cnn.{i}.0.weight_v", 512, 512, 5)

    # F0/N conv weights (K=3)
    f0n_ins = [512, 512, 256]
    f0n_outs = [512, 256, 256]
    for chain in ["predictor.F0", "predictor.N"]:
        for i in range(3):
            make_nhwc_f16(f"{chain}.{i}.conv1.weight_v", f0n_outs[i], f0n_ins[i], 3)
            make_nhwc_f16(f"{chain}.{i}.conv2.weight_v", f0n_outs[i], f0n_outs[i], 3)

    # Decoder encode (514→1024, K=3)
    make_nhwc_f16("decoder.encode.conv1.weight_v", 1024, 514, 3)
    make_nhwc_f16("decoder.encode.conv2.weight_v", 1024, 1024, 3)

    # Decoder decode blocks (1090→1024/512, K=3)
    for i, (di, do) in enumerate(dec_dims):
        make_nhwc_f16(f"decoder.decode.{i}.conv1.weight_v", do, di, 3)
        make_nhwc_f16(f"decoder.decode.{i}.conv2.weight_v", do, do, 3)

    # Generator resblocks + noise_res
    for i in range(6):
        pfx = f"decoder.generator.resblocks.{i}"
        C, K = rb_ch[i], rb_k[i]
        for j in range(3):
            make_nhwc_f16(f"{pfx}.convs1.{j}.weight_v", C, C, K)
            make_nhwc_f16(f"{pfx}.convs2.{j}.weight_v", C, C, K)

    for i in range(2):
        pfx = f"decoder.generator.noise_res.{i}"
        C, K = nr_ch[i], nr_k[i]
        for j in range(3):
            make_nhwc_f16(f"{pfx}.convs1.{j}.weight_v", C, C, K)
            make_nhwc_f16(f"{pfx}.convs2.{j}.weight_v", C, C, K)

    # Generator conv_post (128→22, K=7) — C_out=22 misaligned, skip NHWC FP16 conv
    # (FP16 binary uses im2col+GEMM for this one, .f16 is enough)

    # Generator noise_convs[0] (22→256, K=12) — C_in=22 misaligned, needs padding
    make_nhwc_f16("decoder.generator.noise_convs.0.weight", 256, 22, 12)

    # ── 5. Precompute LSTM biases: bias = bih + bhh ─────────────────────

    def make_lstm_bias(prefix):
        bih_fwd = t.get(prefix + ".bias_ih_l0")
        bhh_fwd = t.get(prefix + ".bias_hh_l0")
        bih_rev = t.get(prefix + ".bias_ih_l0_reverse")
        bhh_rev = t.get(prefix + ".bias_hh_l0_reverse")
        if bih_fwd is not None and bhh_fwd is not None:
            out.append((prefix + ".bias_combined_fwd", (bih_fwd + bhh_fwd).astype(np.float32)))
        if bih_rev is not None and bhh_rev is not None:
            out.append((prefix + ".bias_combined_rev", (bih_rev + bhh_rev).astype(np.float32)))

    make_lstm_bias("text_encoder.lstm")
    for i in range(3):
        make_lstm_bias(f"predictor.text_encoder.lstms.{i * 2}")
    make_lstm_bias("predictor.lstm")
    make_lstm_bias("predictor.shared")

    return out


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Convert KOKO v1 → v2 weights")
    p.add_argument("--bundle", type=str, help="Path to .bundle file")
    p.add_argument("--koko", type=str, help="Path to standalone .koko v1 file")
    p.add_argument("-o", "--output", type=str, required=True, help="Output .koko v2 path")
    args = p.parse_args()

    if args.bundle:
        print(f"Reading weights from bundle: {args.bundle}")
        raw = read_bundle_weights(args.bundle)
    elif args.koko:
        print(f"Reading standalone KOKO: {args.koko}")
        raw = Path(args.koko).read_bytes()
    else:
        home = Path.home()
        default_bundle = home / ".cache/rokoko/rokoko.bundle"
        if default_bundle.exists():
            print(f"Reading weights from default bundle: {default_bundle}")
            raw = read_bundle_weights(str(default_bundle))
        else:
            p.error("Provide --bundle or --koko, or ensure ~/.cache/rokoko/rokoko.bundle exists")

    print("Parsing v1 tensors...")
    tensors = parse_koko(raw)
    print(f"  {len(tensors)} tensors loaded")

    print("Converting to v2...")
    entries = convert(tensors)

    # Count by type
    n_base = sum(1 for n, _ in entries if ".f16" not in n and ".nhwc" not in n and ".bias_combined" not in n)
    n_f16 = sum(1 for n, _ in entries if n.endswith(".f16"))
    n_nhwc = sum(1 for n, _ in entries if ".nhwc_f16" in n)
    n_bias = sum(1 for n, _ in entries if ".bias_combined" in n)
    total_bytes = sum(a.nbytes for _, a in entries)
    print(f"  {len(entries)} tensors: {n_base} base, {n_f16} .f16, {n_nhwc} .nhwc_f16, {n_bias} .bias_combined")
    print(f"  Total data: {total_bytes / 1e6:.1f} MB")

    print(f"Writing v2 file: {args.output}")
    write_koko_v2(args.output, entries)
    print(f"  Done: {Path(args.output).stat().st_size / 1e6:.1f} MB on disk")


if __name__ == "__main__":
    main()
