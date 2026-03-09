"""
Rotary Position Embeddings (RoPE) with YaRN scaling.

RoPE encodes position by rotating query/key vectors in 2D subspaces:
    RoPE(x_m, m) = x_m · e^{im·θ}

For each pair of dimensions (2k, 2k+1):
    R_{θ,m} = [[cos(m·θ_k), -sin(m·θ_k)],
               [sin(m·θ_k),  cos(m·θ_k)]]

where θ_k = base^{-2k/d_h} and m is the token position.

Key properties:
    - Relative position: <R_m q, R_n k> depends only on m - n
    - Decaying attention with distance
    - No learned positional parameters

YaRN (Yet another RoPE extensioN):
    Combines NTK-aware scaling with attention temperature
    correction for extending context beyond training length.
    Likely used by Opus 4.6 for 1M context.

References:
    - Su et al., "RoFormer: Enhanced Transformer with Rotary
      Position Embedding", 2021
    - bloc97, "NTK-Aware Scaled RoPE", 2023
    - Peng et al., "YaRN: Efficient Context Window Extension
      of Large Language Models", 2023
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def _compute_default_freqs(
    head_dim: int,
    base: float = 500000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute the base frequency vector θ.

        θ_k = base^{-2k / d_h}

    Args:
        head_dim: Dimension per attention head (d_h).
        base: RoPE base frequency (default 500000 for long context).
        device: Target device.

    Returns:
        Tensor of shape [d_h // 2] with the frequency values.
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    return inv_freq


def _compute_yarn_freqs(
    head_dim: int,
    base: float = 500000.0,
    factor: float = 8.0,
    original_max_pos: int = 131072,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute YaRN-scaled frequencies for context extension.

    YaRN (Yet another RoPE extensioN) applies NTK-aware scaling
    that preserves high-frequency components better than linear
    interpolation.

    The frequency bands are partitioned into three regions:
        - High-frequency (above β_fast): no scaling (preserve local)
        - Mid-frequency: smooth blend of scaled and unscaled
        - Low-frequency (below β_slow): full scaling (extend global)

    NTK-aware scaling:
        θ'_k = (base · α^{d_h/(d_h-2)})^{-2k/d_h}
    where α = factor.

    Args:
        head_dim: Dimension per attention head.
        base: RoPE base frequency.
        factor: Context extension factor (L'/L).
        original_max_pos: Original trained context length.
        beta_fast: Upper wavelength threshold for scaling.
        beta_slow: Lower wavelength threshold for scaling.
        device: Target device.

    Returns:
        Tensor of shape [d_h // 2] with YaRN-adjusted frequencies.
    """
    # Standard frequencies
    inv_freq = _compute_default_freqs(head_dim, base, device)

    # Wavelength thresholds
    low_freq_wavelen = original_max_pos / beta_slow
    high_freq_wavelen = original_max_pos / beta_fast

    # Wavelength of each frequency band
    wavelens = 2.0 * math.pi / inv_freq

    # NTK-aware scaled frequencies
    # α = factor, θ'_k = (base · α^{d/(d-2)})^{-2k/d}
    alpha = factor
    ntk_base = base * (alpha ** (head_dim / (head_dim - 2)))
    inv_freq_scaled = 1.0 / (
        ntk_base
        ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    # Smooth interpolation between scaled and unscaled
    # High freq (short wavelength): keep original (local position matters)
    # Low freq (long wavelength): use scaled (extend reach)
    smooth = torch.clamp(
        (wavelens - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen),
        min=0.0,
        max=1.0,
    )
    inv_freq_yarn = (1.0 - smooth) * inv_freq + smooth * inv_freq_scaled

    return inv_freq_yarn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.

    Supports both standard RoPE and YaRN scaling for long
    context extension (up to 1M tokens).

    Architecture parameters (Opus 4.6 speculated):
        - head_dim: 128
        - base θ: 500,000
        - YaRN factor: 8.0 (extends 128K → 1M)
        - Original trained context: 131,072

    Usage:
        rope = RotaryEmbedding(config)
        cos, sin = rope(seq_len=4096)
        q_rotated = apply_rotary_emb(q, cos, sin)
        k_rotated = apply_rotary_emb(k, cos, sin)
    """

    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 500000.0)

        # Determine scaling type
        rope_scaling = getattr(config, "rope_scaling", None)

        if rope_scaling is not None and rope_scaling.get("type") == "yarn":
            inv_freq = _compute_yarn_freqs(
                head_dim=self.head_dim,
                base=self.rope_theta,
                factor=rope_scaling.get("factor", 8.0),
                original_max_pos=rope_scaling.get(
                    "original_max_position_embeddings", 131072
                ),
                beta_fast=rope_scaling.get("beta_fast", 32.0),
                beta_slow=rope_scaling.get("beta_slow", 1.0),
            )
            self.attention_scale = rope_scaling.get("attention_factor", 0.1)
        else:
            inv_freq = _compute_default_freqs(self.head_dim, self.rope_theta)
            self.attention_scale = 1.0

        # Register as buffer (not a parameter — no gradients)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cache for common sequence lengths
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None

    def _update_cache(self, max_pos: int, device: torch.device, dtype: torch.dtype):
        """Build cos/sin cache for positions [0, max_pos)."""
        if max_pos <= self._cached_seq_len and self._cached_cos is not None:
            return

        self._cached_seq_len = max_pos
        t = torch.arange(max_pos, device=device, dtype=torch.float32)

        # Outer product: [max_pos] × [d_h/2] → [max_pos, d_h/2]
        freqs = torch.outer(t, self.inv_freq.to(device))

        # Duplicate for pairs: [max_pos, d_h]
        emb = torch.cat([freqs, freqs], dim=-1)

        self._cached_cos = emb.cos().to(dtype)
        self._cached_sin = emb.sin().to(dtype)

    def forward(
        self,
        seq_len: int,
        offset: int = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin embeddings for the given sequence length.

        Args:
            seq_len: Number of new positions to compute.
            offset: Starting position index (from KV cache during
                    autoregressive decode). Positions returned are
                    [offset, offset + seq_len).
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple of (cos, sin), each of shape [seq_len, d_h].
        """
        if device is None:
            device = self.inv_freq.device

        # Ensure cache covers up to offset + seq_len
        self._update_cache(offset + seq_len, device, dtype)

        cos = self._cached_cos[offset : offset + seq_len]
        sin = self._cached_sin[offset : offset + seq_len]

        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension by splitting into halves and swapping.

    For x = [x_0, x_1, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]:
        returns [-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]

    This implements the rotation matrix multiplication efficiently.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to query or key tensors.

    Implements:
        RoPE(x, m) = x · cos(m·θ) + rotate_half(x) · sin(m·θ)

    This is mathematically equivalent to rotating each pair of
    dimensions (2k, 2k+1) by angle m·θ_k, giving the attention
    score a natural relative-position dependence.

    Args:
        x: Input tensor of shape [B, n_heads, seq_len, head_dim].
        cos: Cosine embeddings of shape [seq_len, head_dim].
        sin: Sine embeddings of shape [seq_len, head_dim].
        position_ids: Optional position indices for non-contiguous
                      positions (e.g., during KV cache inference).

    Returns:
        Rotated tensor of the same shape as x.
    """
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [B, 1, seq, d_h]
        sin = sin[position_ids].unsqueeze(1)
    else:
        # Standard: broadcast over batch and heads
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, d_h]
        sin = sin.unsqueeze(0).unsqueeze(0)

    return (x * cos) + (_rotate_half(x) * sin)
