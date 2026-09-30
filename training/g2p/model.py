"""G2P Model V3: Conformer + RoPE + RMSNorm + QK-Norm + SwiGLU.

Architecture:
    CharEmbed(n_chars, d) + RoPE (parameter-free, no learned pos_emb)
    -> N x ConformerBlock:
         RMSNorm -> MHSA (QK-Norm, RoPE) -> residual add
         RMSNorm -> ConvModule(depthwise, k=31) -> residual add
         RMSNorm -> SwiGLU FFN -> residual add
       [optional intermediate CTC at specified layer]
    -> Upsample linear (d -> d*up) -> reshape
    -> Output head -> CTC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RMSNorm (AMP-friendly) ──────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """RMSNorm that casts weight to input dtype for fused kernel under AMP."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight.to(x.dtype), self.eps)


# ── RoPE ─────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """RoPE: parameter-free rotary positional embeddings applied to Q and K.

    Source: Su et al., "RoFormer" (arXiv:2104.09864).
    Applied to Q and K only (not V). Encodes absolute position via rotation,
    but the Q*K dot product encodes only relative distance.
    """

    def __init__(self, dim, max_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, max_len):
        t = torch.arange(max_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (max_len, dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, T):
        if T > self.cos_cached.shape[0]:
            self._build_cache(T)
        return self.cos_cached[:T], self.sin_cached[:T]


def apply_rope(x, cos, sin):
    """Apply rotary embeddings. x: (B, heads, T, head_dim)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── SwiGLU FFN ───────────────────────────────────────────────────────────────


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: out = down(SiLU(gate(x)) * up(x))"""

    def __init__(self, d, ff, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(d, ff)
        self.up = nn.Linear(d, ff)
        self.down = nn.Linear(ff, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ── Conformer Convolution Module ─────────────────────────────────────────────


class ConvModule(nn.Module):
    """Conformer convolution module.

    Source: Gulati et al., "Conformer" (arXiv:2005.08100).
    Sub-layers (verified from reference impl github.com/sooftware/conformer):
      1. Pointwise Conv (expansion 2x)
      2. GLU (halves channels back to d)
      3. Depthwise Conv1d (kernel_size, groups=d)
      4. BatchNorm1d
      5. Swish activation
      6. Pointwise Conv (project back to d)
      7. Dropout
    """

    def __init__(self, d, kernel_size=31, dropout=0.1):
        super().__init__()
        self.pw1 = nn.Linear(d, d * 2)
        self.dw = nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d)
        h = F.glu(self.pw1(x), dim=-1)  # (B, T, d)
        h = h.transpose(1, 2)  # (B, d, T)
        h = self.dw(h)
        h = self.bn(h)
        h = F.silu(h)
        h = h.transpose(1, 2)  # (B, T, d)
        return self.drop(self.pw2(h))


# ── Conformer Block ──────────────────────────────────────────────────────────


class ConformerBlock(nn.Module):
    """Conformer block with RMSNorm, QK-Norm, RoPE, ConvModule, SwiGLU.

    QK-Norm: RMSNorm applied to Q and K after projection, before attention.
    Source: Ross Taylor, "QK Norm and the Curious Case of Logit Drift";
            HybridNorm paper (arXiv:2503.04598).

    Feature flags (for ablation):
      use_qk_norm: apply RMSNorm to Q and K before attention
      use_conv: include ConvModule between attention and FFN
      use_rmsnorm: use RMSNorm (True) or LayerNorm (False)
    """

    def __init__(self, d, heads, ff, kernel_size=31, dropout=0.1,
                 use_qk_norm=True, use_conv=True, use_rmsnorm=True):
        super().__init__()
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.use_qk_norm = use_qk_norm
        self.use_conv = use_conv

        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm

        # Attention
        self.norm1 = Norm(d)
        self.qkv = nn.Linear(d, 3 * d)
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(dropout)

        # Convolution (optional)
        if use_conv:
            self.norm2 = Norm(d)
            self.conv = ConvModule(d, kernel_size, dropout)

        # SwiGLU FFN
        self.norm_ffn = Norm(d)
        self.ffn = SwiGLUFFN(d, ff, dropout)

    def forward(self, x, key_padding_mask=None, rope_cos=None, rope_sin=None):
        B, T, _ = x.shape

        # 1. Pre-norm MHSA with optional QK-Norm + RoPE
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, heads, head_dim)
        q = q.transpose(1, 2)  # (B, heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).float() * torch.finfo(q.dtype).min

        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        h = h.transpose(1, 2).reshape(B, T, self.d)
        x = x + self.out_proj(h)

        # 2. Pre-norm ConvModule (optional)
        if self.use_conv:
            x = x + self.conv(self.norm2(x))

        # 3. Pre-norm SwiGLU FFN
        x = x + self.ffn(self.norm_ffn(x))

        return x


# ── Full Model ───────────────────────────────────────────────────────────────


class G2PModelV3(nn.Module):
    """CTC Conformer V3 for grapheme-to-phoneme conversion.

    Changes from V2:
      - RoPE replaces learned pos_emb (saves 262K params at d=256)
      - RMSNorm replaces LayerNorm (simpler, no bias)
      - QK-Norm on Q and K (training stability)
      - Conformer ConvModule between attention and FFN (local n-gram features)
      - Optional intermediate CTC loss for deep supervision
    """

    def __init__(self, n_chars, n_phones, d=256, heads=4, layers=4, ff=1024, up=3,
                 kernel_size=31, dropout=0.1, inter_ctc_layer=0,
                 use_rope=True, use_qk_norm=True, use_conv=True, use_rmsnorm=True):
        super().__init__()
        self.d = d
        self.up = up
        self.n_layers = layers
        self.inter_ctc_layer = inter_ctc_layer  # 0 = disabled
        self.use_rope = use_rope

        self.char_emb = nn.Embedding(n_chars, d, padding_idx=0)
        self.drop = nn.Dropout(dropout)

        if use_rope:
            self.rope = RotaryEmbedding(d // heads)
        else:
            self.pos_emb = nn.Embedding(1024, d)

        self.blocks = nn.ModuleList([
            ConformerBlock(d, heads, ff, kernel_size, dropout,
                          use_qk_norm=use_qk_norm, use_conv=use_conv, use_rmsnorm=use_rmsnorm)
            for _ in range(layers)
        ])
        self.upsample = nn.Linear(d, d * up)
        self.head = nn.Linear(d, n_phones)

        if inter_ctc_layer > 0:
            self.inter_upsample = nn.Linear(d, d * up)
            self.inter_head = nn.Linear(d, n_phones)

    def forward(self, x, x_lengths):
        B, T = x.shape
        h = self.char_emb(x)
        if self.use_rope:
            rope_cos, rope_sin = self.rope(T)
        else:
            h = h + self.pos_emb(torch.arange(T, device=x.device))
            rope_cos, rope_sin = None, None
        h = self.drop(h)

        mask = torch.arange(T, device=x.device).unsqueeze(0) >= x_lengths.unsqueeze(1)

        inter_logits = None
        for i, block in enumerate(self.blocks):
            h = block(h, key_padding_mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
            if self.inter_ctc_layer > 0 and (i + 1) == self.inter_ctc_layer:
                ih = self.inter_upsample(h).view(B, T * self.up, self.d)
                inter_logits = self.inter_head(ih)

        h = self.upsample(h).view(B, T * self.up, self.d)
        logits = self.head(h)

        if inter_logits is not None:
            return logits, inter_logits
        return logits
