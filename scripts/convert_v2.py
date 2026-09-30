#!/usr/bin/env python3
"""Convert FP32 weights → FP16 weights.

Reads the v1 KOKO weight file, converts weights to v2 (pre-baked weight-norm,
FP16, NHWC, padded, LSTM bias), and writes a standalone v2 KOKO file.

Usage:
    uv run scripts/convert_v2.py -o weights.fp16.bin
    uv run scripts/convert_v2.py --weights path/to/weights.bin -o out.bin
"""

import argparse
import mmap
import struct
import sys
from pathlib import Path

import numpy as np


# ── KOKO file I/O ──────────────────────────────────────────────────────────

KOKO_MAGIC = 0x4F4B4F4B  # "KOKO" LE


def parse_koko(path: str):
    """Parse a KOKO v1 weight file. Returns {name: ndarray}."""
    f = open(path, "rb")
    raw = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
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
    raw.close()
    f.close()
    return tensors


def write_koko_v2(path: str, entries: list[tuple[str, np.ndarray]]):
    """Write a standalone KOKO v2 weight file."""
    header_lines = []
    offset = 0
    for name, arr in entries:
        dtype_str = "fp32" if arr.dtype == np.float32 else "fp16"
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
    """Convert v1 tensors → v2 tensor list (stripped: only what FP16 binary needs)."""
    out: list[tuple[str, np.ndarray]] = []
    t = tensors  # shorthand

    # ── 1. Weight norm all weight_g / weight_v pairs ─────────────────────
    wg_names = [n for n in t if n.endswith(".weight_g")]
    for wg_name in wg_names:
        wv_name = wg_name.replace(".weight_g", ".weight_v")
        if wv_name in t:
            t[wv_name] = weight_norm(t[wg_name], t[wv_name])

    # ── helpers ──────────────────────────────────────────────────────────

    def emit(name):
        """Emit an FP32 tensor from the original dict."""
        if name in t:
            out.append((name, t[name]))

    def emit_f16(name):
        """Emit an FP16 copy (.f16 suffix) for GEMM-only weights."""
        if name in t:
            out.append((name + ".f16", t[name].astype(np.float16)))

    def emit_nhwc_f16(name, C_out, C_in, K):
        """Emit NHWC FP16 conv weight (handles K=1, alignment, padding)."""
        wv = t.get(name)
        if wv is None:
            return
        w_nhwc = nhwc_reshape(wv.reshape(C_out, C_in, K))
        if C_in % 8 == 0 and C_out % 4 == 0:
            out.append((name + ".nhwc_f16", w_nhwc.astype(np.float16)))
        elif C_in % 8 != 0 and C_out % 4 == 0:
            C_in_pad = (C_in + 7) & ~7
            w_pad = pad_cin_nhwc(w_nhwc, C_in_pad)
            out.append((f"{name}.nhwc_f16_pad{C_in_pad}", w_pad.astype(np.float16)))

    def emit_lstm_f16(prefix):
        """Emit FP16 LSTM weights + combined bias."""
        for sfx in [".weight_ih_l0", ".weight_hh_l0",
                     ".weight_ih_l0_reverse", ".weight_hh_l0_reverse"]:
            emit_f16(prefix + sfx)
        bih_fwd = t.get(prefix + ".bias_ih_l0")
        bhh_fwd = t.get(prefix + ".bias_hh_l0")
        bih_rev = t.get(prefix + ".bias_ih_l0_reverse")
        bhh_rev = t.get(prefix + ".bias_hh_l0_reverse")
        if bih_fwd is not None and bhh_fwd is not None:
            out.append((prefix + ".bias_combined_fwd", (bih_fwd + bhh_fwd).astype(np.float32)))
        if bih_rev is not None and bhh_rev is not None:
            out.append((prefix + ".bias_combined_rev", (bih_rev + bhh_rev).astype(np.float32)))

    def emit_adain1d_fp32(prefix):
        """Emit FP32 InstanceNorm weight/bias + AdaIN1d FC bias."""
        emit(prefix + ".norm.weight")
        emit(prefix + ".norm.bias")
        emit(prefix + ".fc.bias")

    def emit_adain1d_f16(prefix):
        """Emit FP16 AdaIN1d FC weight."""
        emit_f16(prefix + ".fc.weight")

    # ── 2. Embeddings (FP32) ─────────────────────────────────────────────

    emit("bert.embeddings.word_embeddings.weight")
    emit("bert.embeddings.position_embeddings.weight")
    emit("bert.embeddings.token_type_embeddings.weight")
    emit("bert.embeddings.LayerNorm.weight")
    emit("bert.embeddings.LayerNorm.bias")

    # ── 3. ALBERT — FP32 biases/norms + FP16 weights ────────────────────

    pfx = "bert.encoder.embedding_hidden_mapping_in"
    emit(pfx + ".bias")
    emit_f16(pfx + ".weight")

    apfx = "bert.encoder.albert_layer_groups.0.albert_layers.0"
    for suffix in [".attention.query", ".attention.key", ".attention.value",
                   ".attention.dense"]:
        emit(apfx + suffix + ".bias")
        emit_f16(apfx + suffix + ".weight")
    emit(apfx + ".attention.LayerNorm.weight")
    emit(apfx + ".attention.LayerNorm.bias")
    for suffix in [".ffn", ".ffn_output"]:
        emit(apfx + suffix + ".bias")
        emit_f16(apfx + suffix + ".weight")
    emit(apfx + ".full_layer_layer_norm.weight")
    emit(apfx + ".full_layer_layer_norm.bias")

    # ── 4. bert_encoder ──────────────────────────────────────────────────

    emit("bert_encoder.bias")
    emit_f16("bert_encoder.weight")

    # ── 5. Text encoder: embedding + conv NHWC + LSTM ──────────────────

    emit("text_encoder.embedding.weight")

    for i in range(3):
        wv = f"text_encoder.cnn.{i}.0.weight_v"
        emit(f"text_encoder.cnn.{i}.0.bias")
        emit(f"text_encoder.cnn.{i}.1.gamma")
        emit(f"text_encoder.cnn.{i}.1.beta")
        emit_nhwc_f16(wv, 512, 512, 5)

    emit_lstm_f16("text_encoder.lstm")

    # ── 6. Prosody predictor ─────────────────────────────────────────────

    # DurationEncoder: 3x (BiLSTM + AdaLayerNorm)
    for i in range(3):
        emit_lstm_f16(f"predictor.text_encoder.lstms.{i * 2}")
        aln_pfx = f"predictor.text_encoder.lstms.{i * 2 + 1}"
        emit(aln_pfx + ".fc.bias")
        emit_f16(aln_pfx + ".fc.weight")

    # Duration LSTM + projection
    emit_lstm_f16("predictor.lstm")
    emit("predictor.duration_proj.linear_layer.bias")
    emit_f16("predictor.duration_proj.linear_layer.weight")

    # Shared LSTM
    emit_lstm_f16("predictor.shared")

    # F0/N chains: AdainResBlk1d blocks + projection
    # F0/N: [0] 512→512 no shortcut no upsample
    #        [1] 512→256 shortcut + upsample
    #        [2] 256→256 no shortcut no upsample
    f0n_dims = [(512, 512, False, False), (512, 256, True, True), (256, 256, False, False)]
    f0n_ins = [512, 512, 256]
    f0n_outs = [512, 256, 256]
    for chain in ["predictor.F0", "predictor.N"]:
        for i, (di, do, sc, up) in enumerate(f0n_dims):
            p = f"{chain}.{i}"
            # conv biases (FP32)
            emit(p + ".conv1.bias")
            emit(p + ".conv2.bias")
            # conv weights → NHWC FP16
            emit_nhwc_f16(p + ".conv1.weight_v", f0n_outs[i], f0n_ins[i], 3)
            emit_nhwc_f16(p + ".conv2.weight_v", f0n_outs[i], f0n_outs[i], 3)
            # AdaIN1d norms + FC
            emit_adain1d_fp32(p + ".norm1")
            emit_adain1d_fp32(p + ".norm2")
            emit_adain1d_f16(p + ".norm1")
            emit_adain1d_f16(p + ".norm2")
            if sc:
                # conv1x1 shortcut → NHWC FP16 (K=1)
                emit_nhwc_f16(p + ".conv1x1.weight_v", do, di, 1)
            if up:
                # depthwise ConvTranspose1d upsample (FP32)
                emit(p + ".pool.weight_v")
                emit(p + ".pool.bias")

    # F0/N projection (small, FP32 — used directly)
    emit("predictor.F0_proj.weight")
    emit("predictor.F0_proj.bias")
    emit("predictor.N_proj.weight")
    emit("predictor.N_proj.bias")

    # ── 7. Decoder ───────────────────────────────────────────────────────

    # F0_conv, N_conv: Conv1d(1,1,k=3) — FP32, used by conv1d_general_f32
    emit("decoder.F0_conv.weight_v")
    emit("decoder.F0_conv.bias")
    emit("decoder.N_conv.weight_v")
    emit("decoder.N_conv.bias")

    # asr_res: Conv1d(512,64,k=1) — FP32, used by conv1d_f32
    emit("decoder.asr_res.0.weight_v")
    emit("decoder.asr_res.0.bias")

    # encode: AdainResBlk1d(514→1024, shortcut)
    def emit_resblk(prefix, dim_in, dim_out, has_shortcut, has_upsample=False):
        emit(prefix + ".conv1.bias")
        emit(prefix + ".conv2.bias")
        emit_nhwc_f16(prefix + ".conv1.weight_v", dim_out, dim_in, 3)
        emit_nhwc_f16(prefix + ".conv2.weight_v", dim_out, dim_out, 3)
        emit_adain1d_fp32(prefix + ".norm1")
        emit_adain1d_fp32(prefix + ".norm2")
        emit_adain1d_f16(prefix + ".norm1")
        emit_adain1d_f16(prefix + ".norm2")
        if has_shortcut:
            emit_nhwc_f16(prefix + ".conv1x1.weight_v", dim_out, dim_in, 1)
        if has_upsample:
            emit(prefix + ".pool.weight_v")
            emit(prefix + ".pool.bias")

    emit_resblk("decoder.encode", 514, 1024, True)

    # decode[0..3]
    dec_dims = [(1090, 1024), (1090, 1024), (1090, 1024), (1090, 512)]
    for i, (di, do) in enumerate(dec_dims):
        emit_resblk(f"decoder.decode.{i}", di, do, True, has_upsample=(i == 3))

    # ── 8. Generator ─────────────────────────────────────────────────────

    # ups: ConvTranspose1d — FP16 GEMM (.f16)
    for i in range(2):
        pfx = f"decoder.generator.ups.{i}"
        emit(pfx + ".bias")
        emit_f16(pfx + ".weight_v")

    # noise_convs: [0] K=12 NHWC, [1] K=1 NHWC
    emit("decoder.generator.noise_convs.0.bias")
    emit_nhwc_f16("decoder.generator.noise_convs.0.weight", 256, 22, 12)
    emit("decoder.generator.noise_convs.1.bias")
    emit_nhwc_f16("decoder.generator.noise_convs.1.weight", 128, 22, 1)

    # resblocks + noise_res: AdaINResBlock1
    def emit_resblock1(prefix, C, K):
        for j in range(3):
            js = str(j)
            for grp in ["convs1", "convs2"]:
                emit(f"{prefix}.{grp}.{js}.bias")
                emit_nhwc_f16(f"{prefix}.{grp}.{js}.weight_v", C, C, K)
            for grp in ["adain1", "adain2"]:
                emit_adain1d_fp32(f"{prefix}.{grp}.{js}")
                emit_adain1d_f16(f"{prefix}.{grp}.{js}")
            emit(f"{prefix}.alpha1.{js}")
            emit(f"{prefix}.alpha2.{js}")

    rb_ch = [256, 256, 256, 128, 128, 128]
    rb_k = [3, 7, 11, 3, 7, 11]
    for i in range(6):
        emit_resblock1(f"decoder.generator.resblocks.{i}", rb_ch[i], rb_k[i])

    nr_ch = [256, 128]
    nr_k = [7, 11]
    for i in range(2):
        emit_resblock1(f"decoder.generator.noise_res.{i}", nr_ch[i], nr_k[i])

    # conv_post: C_out=22 misaligned → im2col+GEMM, needs .f16 (not NHWC)
    emit("decoder.generator.conv_post.bias")
    emit_f16("decoder.generator.conv_post.weight_v")

    # m_source.l_linear: small FP32
    emit("decoder.generator.m_source.l_linear.weight")
    emit("decoder.generator.m_source.l_linear.bias")

    return out


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Convert FP32 weights.bin → FP16 weights.fp16.bin")
    p.add_argument("--weights", type=str, help="Path to source weight file")
    p.add_argument("-o", "--output", type=str, required=True, help="Output path")
    args = p.parse_args()

    weights_path = args.weights
    if not weights_path:
        home = Path.home()
        default = home / ".cache/rokoko/weights.bin"
        if default.exists():
            weights_path = str(default)
        else:
            p.error("Provide --weights or ensure ~/.cache/rokoko/weights.bin exists")

    print(f"Reading weights: {weights_path}")
    tensors = parse_koko(weights_path)
    print(f"  {len(tensors)} tensors loaded")

    print("Converting to v2 (FP16)...")
    v2_entries = convert(tensors)

    n_base = sum(1 for n, _ in v2_entries if ".f16" not in n and ".nhwc" not in n and ".bias_combined" not in n)
    n_f16 = sum(1 for n, _ in v2_entries if n.endswith(".f16"))
    n_nhwc = sum(1 for n, _ in v2_entries if ".nhwc_f16" in n)
    n_bias = sum(1 for n, _ in v2_entries if ".bias_combined" in n)
    total_bytes = sum(a.nbytes for _, a in v2_entries)
    print(f"  {len(v2_entries)} tensors: {n_base} base, {n_f16} .f16, {n_nhwc} .nhwc_f16, {n_bias} .bias_combined")
    print(f"  Total weight data: {total_bytes / 1e6:.1f} MB")

    print(f"Writing: {args.output}")
    write_koko_v2(args.output, v2_entries)
    print(f"  Done: {Path(args.output).stat().st_size / 1e6:.1f} MB on disk")


if __name__ == "__main__":
    main()
