#!/usr/bin/env python3
"""Train G2P V3 Conformer model (CTC).

Usage:
    python training/g2p/train.py train --data data/base.tsv --epochs 50 --muon --compile --auto-batch
    python training/g2p/train.py test --checkpoint training/v11/best_exact.pt --text "hello world"
    python training/g2p/train.py export --checkpoint training/v11/best_exact.pt --output training/v11/g2p_model.bin

Text normalization is handled by normalize_cli (C++ binary) BEFORE data reaches
this script. The TSV files contain pre-normalized text. No preprocessing is
applied here.
"""

import argparse
import json
import math
import os
import random
import re
import struct
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import G2PModelV3
from provenance import file_record, code_record


# ── Vocabulary ───────────────────────────────────────────────────────────────

# Fixed input charset: all 95 printable ASCII characters (space through ~).
# ID 0 is reserved for PAD / CTC blank. IDs 1-95 map to chars 32-126.
CHAR_VOCAB_CHARS = "".join(chr(c) for c in range(32, 127))
assert len(CHAR_VOCAB_CHARS) == 95


class Vocab:
    """Character <-> dense integer ID. ID 0 is reserved (PAD / CTC blank)."""

    def __init__(self):
        self.ch2id = {}
        self.id2ch = {}
        self._next = 1

    def add(self, ch):
        if ch not in self.ch2id:
            self.ch2id[ch] = self._next
            self.id2ch[self._next] = ch
            self._next += 1
        return self.ch2id[ch]

    def encode(self, text):
        return [self.ch2id.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join(self.id2ch.get(i, "") for i in ids)

    def __len__(self):
        return self._next

    def to_dict(self):
        return dict(self.ch2id)

    @classmethod
    def from_dict(cls, d):
        v = cls()
        for ch, i in d.items():
            v.ch2id[ch] = i
            v.id2ch[i] = ch
        v._next = max(d.values()) + 1 if d else 1
        return v

    @classmethod
    def ascii(cls):
        """Fixed ASCII vocabulary: printable ASCII (space through ~), deterministic IDs."""
        v = cls()
        for ch in CHAR_VOCAB_CHARS:
            v.add(ch)
        return v


# ── Data ─────────────────────────────────────────────────────────────────────


_NON_LATIN_RE = re.compile(
    r'[\u0370-\u03FF'   # Greek
    r'\u0400-\u04FF'    # Cyrillic
    r'\u0530-\u058F'    # Armenian
    r'\u10A0-\u10FF'    # Georgian
    r'\u0600-\u06FF'    # Arabic
    r'\u0900-\u097F'    # Devanagari
    r'\u0E00-\u0E7F'    # Thai
    r'\u0590-\u05FF'    # Hebrew
    r'\u3000-\u9FFF'    # CJK
    r'\uAC00-\uD7AF]'   # Hangul
)

_ISBN_RE = re.compile(r'\bISBN\b')


def load_tsv(path, max_text=300, max_phone=900):
    """Load text->phonemes pairs from TSV. Groups consecutive lines with same text.

    TSV is expected to contain pre-normalized text (from normalize_cli).
    No preprocessing is applied — the data pipeline handles normalization.
    Lines with non-Latin scripts or ISBNs are dropped.
    """
    pairs = []
    cur_text = None
    cur_ph = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            raw = parts[0]
            # Drop lines with non-Latin scripts or ISBNs
            if _NON_LATIN_RE.search(raw) or _ISBN_RE.search(raw):
                cur_text = None
                cur_ph = []
                continue
            text, ph = raw, parts[1]
            if not text or not ph or ph.startswith("?"):
                cur_text = None
                cur_ph = []
                continue
            if text == cur_text:
                cur_ph.append(ph)
            else:
                if cur_text and cur_ph:
                    joined = " ".join(cur_ph)
                    if len(cur_text) <= max_text and len(joined) <= max_phone:
                        pairs.append((cur_text, joined))
                cur_text = text
                cur_ph = [ph]

    if cur_text and cur_ph:
        joined = " ".join(cur_ph)
        if len(cur_text) <= max_text and len(joined) <= max_phone:
            pairs.append((cur_text, joined))

    return pairs


class G2PDataset(Dataset):
    def __init__(self, pairs, char_vocab, phone_vocab):
        self.data = [
            (torch.tensor(char_vocab.encode(t), dtype=torch.long),
             torch.tensor(phone_vocab.encode(p), dtype=torch.long))
            for t, p in pairs
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TokenBatchSampler:
    """Batch sampler: groups similar-length sequences, shuffles batch order.

    Two modes:
    - max_tokens: variable batch sizes keeping tokens_per_batch ≈ constant
    - batch_size: fixed batch sizes (legacy)
    """

    def __init__(self, dataset, max_tokens=None, batch_size=None):
        assert max_tokens or batch_size, "Specify max_tokens or batch_size"
        lengths = [len(dataset.data[i][0]) for i in range(len(dataset))]
        sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])

        if max_tokens:
            self.batches = []
            batch = []
            max_len = 0
            for idx in sorted_indices:
                seq_len = lengths[idx]
                new_max = max(max_len, seq_len)
                if batch and (len(batch) + 1) * new_max > max_tokens:
                    self.batches.append(batch)
                    batch = [idx]
                    max_len = seq_len
                else:
                    batch.append(idx)
                    max_len = new_max
            if batch:
                self.batches.append(batch)
        else:
            self.batches = [
                sorted_indices[i : i + batch_size]
                for i in range(0, len(sorted_indices), batch_size)
            ]

        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        perm = torch.randperm(len(self.batches), generator=g).tolist()
        for idx in perm:
            yield self.batches[idx]

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch):
        self.epoch = epoch


def collate(batch):
    texts, phones = zip(*batch)
    tl = torch.tensor([len(t) for t in texts])
    pl = torch.tensor([len(p) for p in phones])
    tp = torch.zeros(len(texts), tl.max(), dtype=torch.long)
    pp = torch.zeros(len(phones), pl.max(), dtype=torch.long)
    for i, (t, p) in enumerate(zip(texts, phones)):
        tp[i, : len(t)] = t
        pp[i, : len(p)] = p
    return tp, pp, tl, pl


# ── Muon optimizer ───────────────────────────────────────────────────────────


def newton_schulz_5(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    assert G.ndim == 2
    transposed = False
    if G.shape[0] > G.shape[1]:
        G = G.T
        transposed = True
    nrm = G.norm() + eps
    G = G / nrm
    X = G
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon: momentum + Newton-Schulz orthogonalization for 2D weights."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(mu).add_(g)
                g_ns = newton_schulz_5(buf.float(), steps=ns).to(p.dtype)
                m, n = p.shape
                scale = (max(m, n) / min(m, n)) ** 0.5
                p.add_(g_ns, alpha=-lr * scale)


# ── CTC decode & metrics ────────────────────────────────────────────────────


def ctc_greedy(logits):
    preds = logits.argmax(dim=-1)
    out = []
    prev = -1
    for p in preds.tolist():
        if p != 0 and p != prev:
            out.append(p)
        prev = p
    return out


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            prev, dp[j] = dp[j], min(
                dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i - 1] == b[j - 1] else 1)
            )
    return dp[m]


# ── Training ─────────────────────────────────────────────────────────────────


def find_max_tokens(model, device, upsample, train_pairs, target_frac=0.8):
    """Probe GPU to find a max_tokens budget fitting in target_frac of VRAM.

    Binary-searches for the largest batch size at p90 sequence length,
    then returns batch_size * seq_len as the token budget.
    """
    import gc

    total = torch.cuda.get_device_properties(device).total_memory
    target = int(total * target_frac)

    text_lens = sorted(len(t) for t, _ in train_pairs)
    phone_lens = sorted(len(p) for _, p in train_pairs)
    p90 = int(len(text_lens) * 0.9)
    tl_probe = text_lens[p90]
    pl_probe = phone_lens[p90]

    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    lo, hi, best = 64, 8192, 64

    print(f"Auto-batch: probing (seq_len={tl_probe}, target={target_frac*100:.0f}% of {total/2**30:.1f} GB)...")
    while lo <= hi:
        bs = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats(device)

            texts = torch.zeros(bs, tl_probe, dtype=torch.long, device=device)
            tl_t = torch.full((bs,), tl_probe, dtype=torch.long, device=device)
            phones = torch.ones(bs, pl_probe, dtype=torch.long, device=device)
            pl_t = torch.full((bs,), pl_probe, dtype=torch.long, device=device)

            model.train()
            with torch.amp.autocast("cuda"):
                result = model(texts, tl_t)
                logits = result[0] if isinstance(result, tuple) else result
                log_probs = logits.permute(1, 0, 2).log_softmax(dim=2)
                loss = ctc(log_probs.float(), phones, tl_t * upsample, pl_t)
            loss.backward()
            model.zero_grad(set_to_none=True)

            peak = torch.cuda.max_memory_allocated(device)
            del texts, tl_t, phones, pl_t, result, logits, log_probs, loss

            if peak <= target:
                best = bs
                lo = bs + 1
            else:
                hi = bs - 1
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            hi = bs - 1
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    max_tokens = best * tl_probe
    print(f"→ max_tokens={max_tokens:,} ({best} samples × {tl_probe} seq_len)")
    torch.cuda.reset_peak_memory_stats(device)
    return max_tokens


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # Load data — supports path:N syntax for oversampling (e.g. "augment.tsv:5")
    # Oversampling is applied AFTER the train/val split to prevent leakage.
    file_pairs = []  # [(pairs_list, repeat_factor), ...]
    data_records = []
    for spec in args.data.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if ":" in spec and spec.rsplit(":", 1)[1].isdigit():
            data_path, repeat = spec.rsplit(":", 1)
            repeat = int(repeat)
        else:
            data_path, repeat = spec, 1
        data_records.append({**file_record(data_path), "repeat": repeat})
        pairs = load_tsv(data_path)
        print(f"Loaded {len(pairs)} pairs from {data_path}" +
              (f" (×{repeat})" if repeat > 1 else ""))
        file_pairs.append((pairs, repeat))

    # Dedup across all files (by text), keeping first occurrence
    seen = set()
    unique_pairs = []
    pair_repeat = []  # repeat factor for each unique pair
    for pairs, repeat in file_pairs:
        for text, ph in pairs:
            if text not in seen:
                seen.add(text)
                unique_pairs.append((text, ph))
                pair_repeat.append(repeat)

    n_unique = len(unique_pairs)
    n_effective = sum(pair_repeat)
    print(f"Unique: {n_unique} pairs, effective: {n_effective} (with oversampling)")
    if not unique_pairs:
        print("No data!", file=sys.stderr)
        return

    # Build vocabularies from unique pairs (before split — vocab must cover all data)
    char_vocab = Vocab.ascii()
    if args.resume:
        ckpt_pre = torch.load(args.resume, map_location="cpu", weights_only=False)
        phone_vocab = Vocab.from_dict(ckpt_pre["phone_vocab"])
        for _, ph in unique_pairs:
            for ch in ph:
                phone_vocab.add(ch)
        del ckpt_pre
    else:
        phone_vocab = Vocab()
        for _, ph in unique_pairs:
            for ch in ph:
                phone_vocab.add(ch)
    print(f"Char vocab: {len(char_vocab)} (fixed ASCII), Phone vocab: {len(phone_vocab)}")

    # Check expansion ratios
    ratios = [len(ph) / max(len(text), 1) for text, ph in unique_pairs]
    max_ratio = max(ratios)
    avg_ratio = sum(ratios) / len(ratios)
    print(f"Phoneme/char ratio: avg={avg_ratio:.2f}, max={max_ratio:.2f}")
    if max_ratio > args.upsample:
        n_bad = sum(1 for r in ratios if r > args.upsample)
        print(f"Warning: {n_bad} pairs exceed {args.upsample}x upsample")

    # Train/val split on UNIQUE pairs — no leakage
    indices = list(range(n_unique))
    random.seed(42)
    random.shuffle(indices)
    n_val = max(int(n_unique * args.val_fraction), 50)
    val_indices = set(indices[:n_val])

    val_pairs = [unique_pairs[i] for i in indices[:n_val]]
    # Training pairs: apply oversampling ONLY to training set
    train_pairs = []
    for i in indices[n_val:]:
        for _ in range(pair_repeat[i]):
            train_pairs.append(unique_pairs[i])
    random.shuffle(train_pairs)

    print(f"Train: {len(train_pairs)} (with oversampling), Val: {len(val_pairs)} (unique, no leakage)")

    # Dataloaders — limit workers if /dev/shm is small (e.g. Docker)
    n_workers = min(os.cpu_count() or 4, 12)
    try:
        shm = os.statvfs("/dev/shm")
        shm_mb = shm.f_frsize * shm.f_blocks / (1024 * 1024)
        if shm_mb < 512:
            n_workers = min(n_workers, 2)
            print(f"Low shared memory ({shm_mb:.0f} MB), limiting to {n_workers} workers")
    except OSError:
        pass
    pin = device == "cuda"
    train_ds = G2PDataset(train_pairs, char_vocab, phone_vocab)
    val_ds = G2PDataset(val_pairs, char_vocab, phone_vocab)

    def make_dataloaders(max_tokens=None, batch_size=None):
        train_sampler = TokenBatchSampler(train_ds, max_tokens=max_tokens, batch_size=batch_size)
        val_sampler = TokenBatchSampler(val_ds, max_tokens=max_tokens, batch_size=batch_size)
        dl_train = DataLoader(
            train_ds, batch_sampler=train_sampler, collate_fn=collate,
            num_workers=n_workers, pin_memory=pin, persistent_workers=True, prefetch_factor=4,
        )
        dl_val = DataLoader(
            val_ds, batch_sampler=val_sampler, collate_fn=collate,
            num_workers=n_workers, pin_memory=pin, persistent_workers=True, prefetch_factor=4,
        )
        batch_sizes = [len(b) for b in train_sampler.batches]
        if max_tokens:
            print(f"Token batching: max_tokens={max_tokens:,}, "
                  f"{len(batch_sizes)} batches/epoch, "
                  f"batch_size range [{min(batch_sizes)}, {max(batch_sizes)}]")
        else:
            print(f"Fixed batching: batch_size={batch_size}, {len(batch_sizes)} batches/epoch")
        return dl_train, dl_val, train_sampler

    if args.max_tokens:
        train_dl, val_dl, train_sampler = make_dataloaders(max_tokens=args.max_tokens)
    else:
        train_dl, val_dl, train_sampler = make_dataloaders(batch_size=args.batch_size)

    # Feature flags
    use_rope = not args.no_rope
    use_qk_norm = not args.no_qk_norm
    use_conv = not args.no_conv
    use_rmsnorm = not args.no_rmsnorm

    # Model
    model = G2PModelV3(
        len(char_vocab), len(phone_vocab),
        d=args.d_model, heads=args.nhead, layers=args.nlayers, ff=args.d_ff,
        up=args.upsample, kernel_size=args.kernel_size, dropout=args.dropout,
        inter_ctc_layer=args.inter_ctc_layer,
        use_rope=use_rope, use_qk_norm=use_qk_norm, use_conv=use_conv, use_rmsnorm=use_rmsnorm,
    ).to(device)
    use_inter_ctc = args.inter_ctc_layer > 0
    n_params = sum(p.numel() for p in model.parameters())
    features = []
    if use_rope: features.append("RoPE")
    if use_rmsnorm: features.append("RMSNorm")
    if use_qk_norm: features.append("QK-Norm")
    if use_conv: features.append(f"Conv(k={args.kernel_size})")
    print(f"Model V3: {n_params:,} params ({n_params * 4 / 1e6:.1f} MB fp32)")
    print(f"  features: {', '.join(features) if features else 'none (V2-style baseline)'}")
    if use_inter_ctc:
        print(f"  intermediate CTC at layer {args.inter_ctc_layer}, weight={args.inter_ctc_weight}")

    # torch.compile
    compiled = False
    if hasattr(torch, "compile") and device == "cuda" and args.compile:
        try:
            model = torch.compile(model, dynamic=True)
            compiled = True
            print("torch.compile: enabled (dynamic=True)")
        except Exception as e:
            print(f"torch.compile: failed ({e}), continuing without")

    # Optimizer — Muon for 2D block weights, AdamW for rest
    if args.muon:
        raw_model = model._orig_mod if compiled and hasattr(model, "_orig_mod") else model
        muon_params = []
        adam_params = []
        for name, p in raw_model.named_parameters():
            if p.ndim == 2 and "blocks." in name:
                muon_params.append(p)
            else:
                adam_params.append(p)
        print(f"Muon: {len(muon_params)} 2D weight tensors, AdamW: {len(adam_params)} other params")

        opt_muon = Muon(muon_params, lr=args.muon_lr)
        adam_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
        if device == "cuda":
            try:
                adam_kwargs["fused"] = True
                torch.optim.AdamW([torch.zeros(1, device=device)], **adam_kwargs)
                print("AdamW: fused=True")
            except Exception:
                del adam_kwargs["fused"]
        opt_adam = torch.optim.AdamW(adam_params, **adam_kwargs)
        optimizers = [opt_muon, opt_adam]
    else:
        opt_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)
        if device == "cuda":
            try:
                opt_kwargs["fused"] = True
                torch.optim.AdamW([torch.zeros(1, device=device)], **opt_kwargs)
                print("AdamW: fused=True")
            except Exception:
                del opt_kwargs["fused"]
        opt = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        optimizers = [opt]

    # Auto-batch: probe GPU memory for max_tokens and recreate dataloaders
    if getattr(args, "auto_batch", False) and device == "cuda" and not args.max_tokens:
        probe_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        found_max_tokens = find_max_tokens(probe_model, device, args.upsample, train_pairs)
        train_dl, val_dl, train_sampler = make_dataloaders(max_tokens=found_max_tokens)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw = model._orig_mod if compiled and hasattr(model, "_orig_mod") else model
        raw.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch} (val_loss={ckpt.get('val_loss', '?')})")

    total_steps = len(train_dl) * (args.epochs - start_epoch)
    warmup_steps = int(total_steps * 0.05)

    def make_lr_lambda(warmup, total):
        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return lr_lambda

    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, make_lr_lambda(warmup_steps, total_steps))
                  for o in optimizers]

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    label_smoothing = getattr(args, "label_smoothing", 0.0)
    if label_smoothing > 0:
        print(f"CTC label smoothing: {label_smoothing}")

    # AMP
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")

    per_sample_size = min(500, len(val_pairs))

    # Training loop
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    best_val = float("inf")
    best_exact = 0.0

    # Save full training setup (everything needed to reproduce this run)
    setup = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": args.data,
        "data_files": data_records,
        "source": code_record(),
        "model": {
            "d": args.d_model, "heads": args.nhead, "layers": args.nlayers,
            "ff": args.d_ff, "up": args.upsample, "kernel_size": args.kernel_size,
            "inter_ctc_layer": args.inter_ctc_layer,
            "use_rope": use_rope, "use_qk_norm": use_qk_norm,
            "use_conv": use_conv, "use_rmsnorm": use_rmsnorm,
        },
        "training": {
            "epochs": args.epochs, "lr": args.lr,
            "max_tokens": args.max_tokens, "batch_size": args.batch_size,
            "dropout": args.dropout, "label_smoothing": args.label_smoothing,
            "weight_decay": args.weight_decay,
            "muon": args.muon, "muon_lr": args.muon_lr,
            "compile": args.compile, "amp": use_amp,
            "val_fraction": args.val_fraction,
        },
        "data_stats": {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "char_vocab_size": len(char_vocab),
            "phone_vocab_size": len(phone_vocab),
        },
        "resume": args.resume,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    setup_path = os.path.join(out_dir, "train_setup.json")
    with open(setup_path, "w") as f:
        json.dump(setup, f, indent=2)
    print(f"Setup saved to {setup_path}")

    log_path = os.path.join(out_dir, "train_log.csv")
    log_exists = os.path.exists(log_path) and start_epoch > 0
    log_file = open(log_path, "a" if log_exists else "w")
    if not log_exists:
        log_file.write("epoch,train_loss,val_loss,per,exact,lr,secs\n")
        log_file.flush()

    # Sub-epoch logging interval
    n_total_batches = len(train_dl)
    if args.log_every > 0:
        log_interval = args.log_every
    else:
        log_interval = max(n_total_batches // 5, 1)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        n_batches = 0
        data_time = 0.0
        compute_time = 0.0

        t_data = time.perf_counter()
        for texts, phones, tl, pl in train_dl:
            t_got = time.perf_counter()
            data_time += t_got - t_data

            texts = texts.to(device, non_blocking=True)
            phones = phones.to(device, non_blocking=True)
            tl = tl.to(device, non_blocking=True)
            pl = pl.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(texts, tl)
                if use_inter_ctc and isinstance(result, tuple):
                    logits, inter_logits = result
                else:
                    logits = result if not isinstance(result, tuple) else result[0]

                log_probs = logits.permute(1, 0, 2).log_softmax(dim=2)
                input_lengths = tl * args.upsample
                ctc = ctc_loss_fn(log_probs.float(), phones, input_lengths, pl)

                if label_smoothing > 0:
                    # Uniform KL term: -mean(log_probs) over non-padded frames
                    uniform_loss = -log_probs.float().mean()
                    loss = (1 - label_smoothing) * ctc + label_smoothing * uniform_loss
                else:
                    loss = ctc

                if use_inter_ctc and isinstance(result, tuple):
                    inter_log_probs = inter_logits.permute(1, 0, 2).log_softmax(dim=2)
                    inter_loss = ctc_loss_fn(inter_log_probs.float(), phones, input_lengths, pl)
                    loss = loss + args.inter_ctc_weight * inter_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  WARNING: {loss.item()} loss at batch {n_batches}, skipping")
                for o in optimizers:
                    o.zero_grad(set_to_none=True)
                continue

            for o in optimizers:
                o.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            for o in optimizers:
                scaler.unscale_(o)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            for o in optimizers:
                scaler.step(o)
            scaler.update()
            for s in schedulers:
                s.step()

            total_loss += loss.item()
            n_batches += 1
            t_data = time.perf_counter()
            compute_time += t_data - t_got

            if n_batches % 500 == 0:
                avg = total_loss / n_batches
                pct = n_batches / len(train_dl) * 100
                print(f"  [{n_batches}/{len(train_dl)} ({pct:.0f}%)] loss={avg:.4f}", flush=True)

            # Sub-epoch CSV row
            if n_batches % log_interval == 0 and n_batches < n_total_batches:
                frac_epoch = epoch + n_batches / n_total_batches
                avg_so_far = total_loss / n_batches
                lr = optimizers[0].param_groups[0]["lr"]
                log_file.write(f"{frac_epoch:.4f},{avg_so_far:.6f},NaN,NaN,NaN,{lr:.6e},NaN\n")
                log_file.flush()

        avg_train = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for texts, phones, tl, pl in val_dl:
                texts = texts.to(device, non_blocking=True)
                phones = phones.to(device, non_blocking=True)
                tl = tl.to(device, non_blocking=True)
                pl = pl.to(device, non_blocking=True)

                result = model(texts, tl)
                logits = result[0] if isinstance(result, tuple) else result
                log_probs = logits.permute(1, 0, 2).log_softmax(dim=2)
                input_lengths = tl * args.upsample
                loss = ctc_loss_fn(log_probs.float(), phones, input_lengths, pl)
                val_loss += loss.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)

        # Sampled PER
        per_indices = random.sample(range(len(val_pairs)), per_sample_size)
        n_correct = 0
        total_per = 0.0

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for idx in per_indices:
                text, target_ph = val_pairs[idx]
                ids = torch.tensor([char_vocab.encode(text)], dtype=torch.long, device=device)
                lengths = torch.tensor([len(text)], device=device)
                result = model(ids, lengths)
                logits = result[0] if isinstance(result, tuple) else result
                pred_ids = ctc_greedy(logits[0, : len(text) * args.upsample])
                pred_str = phone_vocab.decode(pred_ids)
                if pred_str == target_ph:
                    n_correct += 1
                total_per += edit_distance(list(pred_str), list(target_ph)) / max(len(target_ph), 1)

        exact = n_correct / per_sample_size * 100
        per = total_per / per_sample_size * 100
        lr = optimizers[0].param_groups[0]["lr"]
        dt = time.time() - t0

        data_pct = 100 * data_time / (data_time + compute_time + 1e-9)
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"train={avg_train:.4f}  val={avg_val:.4f}  "
            f"PER={per:.1f}%  exact={exact:.1f}%  lr={lr:.1e}  "
            f"data={data_pct:.0f}%  {dt:.1f}s",
            flush=True,
        )
        log_file.write(f"{epoch+1},{avg_train:.6f},{avg_val:.6f},{per:.4f},{exact:.2f},{lr:.6e},{dt:.1f}\n")
        log_file.flush()

        # Save best by val loss
        ckpt_data = {
            "model": (model._orig_mod.state_dict() if compiled and hasattr(model, "_orig_mod")
                      else model.state_dict()),
            "config": {
                "d": args.d_model, "heads": args.nhead, "layers": args.nlayers,
                "ff": args.d_ff, "up": args.upsample, "kernel_size": args.kernel_size,
                "inter_ctc_layer": args.inter_ctc_layer,
                "use_rope": use_rope, "use_qk_norm": use_qk_norm,
                "use_conv": use_conv, "use_rmsnorm": use_rmsnorm,
            },
            "model_version": 3,
            "char_vocab": char_vocab.to_dict(),
            "phone_vocab": phone_vocab.to_dict(),
            "epoch": epoch,
            "val_loss": avg_val,
            "per": per,
            "exact": exact,
            "setup": setup,
        }
        if avg_val < best_val:
            best_val = avg_val
            torch.save(ckpt_data, os.path.join(out_dir, "best.pt"))
            print(f"  -> saved best (val={avg_val:.4f})")
        # Also save best by exact match (better for normalization)
        if exact > best_exact:
            best_exact = exact
            torch.save(ckpt_data, os.path.join(out_dir, "best_exact.pt"))
            print(f"  -> saved best_exact ({exact:.1f}%)")

        # Examples every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for j in range(min(3, len(val_pairs))):
                    text, target_ph = val_pairs[j]
                    ids = torch.tensor([char_vocab.encode(text)], dtype=torch.long, device=device)
                    lengths = torch.tensor([len(text)], device=device)
                    result = model(ids, lengths)
                    logits = result[0] if isinstance(result, tuple) else result
                    pred = ctc_greedy(logits[0, : len(text) * args.upsample])
                    pred_ph = phone_vocab.decode(pred)
                    mark = "ok" if pred_ph == target_ph else "MISS"
                    print(f"  {mark} \"{text}\"")
                    print(f"    pred:   {pred_ph}")
                    print(f"    target: {target_ph}")

    log_file.close()
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {os.path.join(out_dir, 'best.pt')}")
    print(f"Training log: {log_path}")


# ── Inference ────────────────────────────────────────────────────────────────


def load_model_from_ckpt(ckpt, device="cpu"):
    cfg = ckpt["config"]
    char_vocab = Vocab.from_dict(ckpt["char_vocab"])
    phone_vocab = Vocab.from_dict(ckpt["phone_vocab"])
    model = G2PModelV3(
        len(char_vocab), len(phone_vocab),
        d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"], ff=cfg["ff"], up=cfg["up"],
        kernel_size=cfg.get("kernel_size", 31),
        inter_ctc_layer=cfg.get("inter_ctc_layer", 0),
        use_rope=cfg.get("use_rope", True), use_qk_norm=cfg.get("use_qk_norm", True),
        use_conv=cfg.get("use_conv", True), use_rmsnorm=cfg.get("use_rmsnorm", True),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, char_vocab, phone_vocab, cfg


def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, char_vocab, phone_vocab, cfg = load_model_from_ckpt(ckpt, device)

    if args.text:
        texts = [args.text]
    else:
        texts = [l.strip() for l in sys.stdin if l.strip()]

    for text in texts:
        ids = torch.tensor([char_vocab.encode(text)], dtype=torch.long, device=device)
        lengths = torch.tensor([len(text)], device=device)
        with torch.no_grad():
            result = model(ids, lengths)
            logits = result[0] if isinstance(result, tuple) else result
            pred = ctc_greedy(logits[0, : len(text) * cfg["up"]])
        phonemes = phone_vocab.decode(pred)
        print(f"{text}\t{phonemes}")


# ── Eval ─────────────────────────────────────────────────────────────────────


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, char_vocab, phone_vocab, cfg = load_model_from_ckpt(ckpt, device)

    pairs = load_tsv(args.data)
    n_correct = 0
    n_total = 0
    total_per = 0.0

    for text, target_ph in pairs:
        ids = torch.tensor([char_vocab.encode(text)], dtype=torch.long, device=device)
        lengths = torch.tensor([len(text)], device=device)
        with torch.no_grad():
            result = model(ids, lengths)
            logits = result[0] if isinstance(result, tuple) else result
            pred = ctc_greedy(logits[0, : len(text) * cfg["up"]])
        pred_ph = phone_vocab.decode(pred)

        if pred_ph == target_ph:
            n_correct += 1
        else:
            if args.verbose:
                print(f"MISS \"{text}\"")
                print(f"  pred:   {pred_ph}")
                print(f"  target: {target_ph}")
        total_per += edit_distance(list(pred_ph), list(target_ph)) / max(len(target_ph), 1)
        n_total += 1

    exact = n_correct / max(n_total, 1) * 100
    per = total_per / max(n_total, 1) * 100
    print(f"Sentences: {n_total}")
    print(f"Exact match: {n_correct}/{n_total} = {exact:.1f}%")
    print(f"PER: {per:.1f}%")


# ── Export ───────────────────────────────────────────────────────────────────
#
# Binary format "G2P3":
#   [4B]  magic "G2P3"
#   [4B]  d_model
#   [4B]  n_heads
#   [4B]  n_layers
#   [4B]  d_ff
#   [4B]  upsample
#   [4B]  n_chars (embedding table rows = max_id + 1)
#   [4B]  n_phones (output classes = max_id + 1)
#   [4B]  kernel_size
#   [4B]  inter_ctc_layer (0 = none)
#   [4B]  n_char_vocab_entries
#   [n x 8B]  char vocab: (codepoint, id) pairs
#   [4B]  n_phone_vocab_entries
#   [n x 8B]  phone vocab: (codepoint, id) pairs
#   [...]  float32 weights in named_parameters() order


def export(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    char_vocab = ckpt["char_vocab"]
    phone_vocab = ckpt["phone_vocab"]

    n_chars = max(char_vocab.values()) + 1 if char_vocab else 1
    n_phones = max(phone_vocab.values()) + 1 if phone_vocab else 1

    model = G2PModelV3(
        n_chars, n_phones,
        d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"], ff=cfg["ff"], up=cfg["up"],
        kernel_size=cfg.get("kernel_size", 31),
        inter_ctc_layer=cfg.get("inter_ctc_layer", 0),
        use_rope=cfg.get("use_rope", True), use_qk_norm=cfg.get("use_qk_norm", True),
        use_conv=cfg.get("use_conv", True), use_rmsnorm=cfg.get("use_rmsnorm", True),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    with open(args.output, "wb") as f:
        f.write(b"G2P3")
        # Feature flags bitfield:
        # bit 0: use_rope, bit 1: use_qk_norm, bit 2: use_conv, bit 3: use_rmsnorm
        flags = 0
        if cfg.get("use_rope", True): flags |= 1
        if cfg.get("use_qk_norm", True): flags |= 2
        if cfg.get("use_conv", True): flags |= 4
        if cfg.get("use_rmsnorm", True): flags |= 8
        f.write(struct.pack(
            "<IIIIIIIIII",
            cfg["d"], cfg["heads"], cfg["layers"], cfg["ff"], cfg["up"],
            n_chars, n_phones,
            cfg.get("kernel_size", 31),
            cfg.get("inter_ctc_layer", 0),
            flags,
        ))

        f.write(struct.pack("<I", len(char_vocab)))
        for ch, idx in sorted(char_vocab.items(), key=lambda x: x[1]):
            f.write(struct.pack("<II", ord(ch), idx))

        f.write(struct.pack("<I", len(phone_vocab)))
        for ch, idx in sorted(phone_vocab.items(), key=lambda x: x[1]):
            f.write(struct.pack("<II", ord(ch), idx))

        for name, param in model.named_parameters():
            data = param.detach().float().cpu().numpy()
            f.write(data.tobytes())

    size = os.path.getsize(args.output)
    print(f"Exported V3 to {args.output} ({size:,} bytes, {size/1e6:.1f} MB)")

    manifest = []
    for name, param in model.named_parameters():
        manifest.append({"name": name, "shape": list(param.shape), "numel": param.numel()})
    manifest_path = args.output + ".manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="G2P V3 Conformer (CTC)")
    sub = parser.add_subparsers(dest="cmd")

    # Train
    p = sub.add_parser("train")
    p.add_argument("--data", required=True, help="Training TSV(s), comma-separated")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64, help="Fixed batch size (ignored if --max-tokens set)")
    p.add_argument("--max-tokens", type=int, default=0, help="Token budget per batch (variable batch sizes)")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--upsample", type=int, default=3)
    p.add_argument("--kernel-size", type=int, default=31, help="ConvModule depthwise kernel size")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out-dir", required=True, help="Directory for this training run")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--auto-batch", action="store_true", help="Probe GPU and set --max-tokens automatically")
    p.add_argument("--resume", help="Resume from checkpoint")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--muon", action="store_true", help="Use Muon optimizer for 2D weights")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--inter-ctc-layer", type=int, default=0, help="Intermediate CTC at this layer (0=off)")
    p.add_argument("--inter-ctc-weight", type=float, default=0.3, help="Weight for intermediate CTC loss")
    p.add_argument("--no-rope", action="store_true", help="Use learned pos embeddings instead of RoPE")
    p.add_argument("--no-qk-norm", action="store_true", help="Disable QK-Norm")
    p.add_argument("--no-conv", action="store_true", help="Disable ConvModule")
    p.add_argument("--no-rmsnorm", action="store_true", help="Use LayerNorm instead of RMSNorm")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="CTC label smoothing (blend with uniform, e.g. 0.1)")
    p.add_argument("--log-every", type=int, default=0, help="Log train loss every N batches (0=auto ~5/epoch)")

    # Test
    p = sub.add_parser("test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--text", help="Input text (or pipe via stdin)")

    # Eval
    p = sub.add_parser("eval")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--verbose", action="store_true")

    # Export
    p = sub.add_parser("export")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "test":
        test(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "export":
        export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
