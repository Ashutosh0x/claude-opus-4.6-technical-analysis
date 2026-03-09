"""
Quantization Utilities for Model Weights and KV Cache.

Implements post-training quantization (PTQ), activation-aware
weight quantization (AWQ), and KV cache quantization strategies
described in the technical analysis.

Weight quantization formats:
    GPTQ    : Hessian-based, 4-bit GPU          (Frantar et al., 2022)
    AWQ     : Activation-aware, 4-bit GPU        (Lin et al., 2023)
    GGUF    : K-quant + importance matrix, CPU    (Gerganov, 2023)
    EXL2    : Mixed-bit per group, GPU            (turboderp, 2023)
    HQQ     : Half-quadratic, no calibration      (Badri & Shaji, 2023)
    AQLM    : Additive vector quantization        (Egiazarian et al., 2024)

KV cache quantization (inference-time):
    Per-channel INT8/FP8 for keys/values reduces KV cache by 2x
    with minimal quality loss (<0.5% perplexity).

Size estimates for 2T-param model:
    BF16:   ~4.0 TB
    FP8:    ~2.0 TB
    INT4:   ~1.0 TB (Q4_K_M)
    INT2:   ~0.5 TB (IQ2_XXS)

References:
    - Frantar et al., "GPTQ: Accurate Post-Training Quantization", 2022
    - Lin et al., "AWQ: Activation-Aware Weight Quantization", 2023
    - Gerganov, "GGUF specification", 2023
"""

import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization Format Enumeration
# ---------------------------------------------------------------------------

class QuantFormat(Enum):
    """Supported quantization formats."""
    # CPU formats (GGUF)
    Q2_K     = "Q2_K"        # 2.56 bpw
    IQ2_XXS  = "IQ2_XXS"     # 2.06 bpw (importance-matrix)
    Q3_K_M   = "Q3_K_M"      # 3.44 bpw
    IQ3_XXS  = "IQ3_XXS"     # 3.07 bpw
    Q4_0     = "Q4_0"        # 4.50 bpw (legacy)
    Q4_K_M   = "Q4_K_M"      # 4.84 bpw (most popular)
    IQ4_XS   = "IQ4_XS"      # 4.25 bpw
    Q5_K_M   = "Q5_K_M"      # 5.68 bpw
    Q6_K     = "Q6_K"        # 6.57 bpw
    Q8_0     = "Q8_0"        # 8.50 bpw (near-lossless reference)
    # GPU formats
    GPTQ     = "GPTQ"        # Hessian-based 4-bit
    AWQ      = "AWQ"         # Activation-aware 4-bit
    EXL2     = "EXL2"        # Mixed-bit (ExLlamaV2)
    HQQ      = "HQQ"         # Half-quadratic (no calibration)
    AQLM     = "AQLM"        # Additive vector quantization


# Bits-per-weight for each GGUF format
GGUF_BPW = {
    QuantFormat.Q2_K:    2.56,
    QuantFormat.IQ2_XXS: 2.06,
    QuantFormat.Q3_K_M:  3.44,
    QuantFormat.IQ3_XXS: 3.07,
    QuantFormat.Q4_0:    4.50,
    QuantFormat.Q4_K_M:  4.84,
    QuantFormat.IQ4_XS:  4.25,
    QuantFormat.Q5_K_M:  5.68,
    QuantFormat.Q6_K:    6.57,
    QuantFormat.Q8_0:    8.50,
}


@dataclass
class QuantConfig:
    """Configuration for quantization."""
    format: QuantFormat = QuantFormat.Q4_K_M
    bits: int = 4
    group_size: int = 128       # weights per quantization group
    symmetric: bool = True      # symmetric vs asymmetric quantization
    use_imatrix: bool = False   # importance-matrix (IQ variants)
    calibration_samples: int = 128
    # K-quant layer differentiation
    promote_attention: bool = True   # Q/K/V at higher precision
    promote_ffn_down: bool = True    # FFN down_proj at higher precision


# ---------------------------------------------------------------------------
# Core Quantization Functions
# ---------------------------------------------------------------------------

def compute_scale_zero_point(
    tensor: torch.Tensor,
    bits: int = 4,
    symmetric: bool = True,
    group_size: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute quantization scale and zero-point for a tensor.

    PTQ formula:
        W_q = round(W / Δ) × Δ
        Δ = max(|W|) / (2^(b-1) - 1)       [symmetric]
        Δ = (max(W) - min(W)) / (2^b - 1)  [asymmetric]

    Args:
        tensor: Weight tensor to quantize.
        bits: Number of quantization bits.
        symmetric: If True, zero-point is always 0.
        group_size: If > 0, compute per-group scales.

    Returns:
        (scale, zero_point) tensors.
    """
    if group_size > 0:
        # Reshape for per-group quantization
        orig_shape = tensor.shape
        flat = tensor.reshape(-1, group_size)
    else:
        flat = tensor.reshape(1, -1)

    if symmetric:
        qmax = (1 << (bits - 1)) - 1   # e.g., 7 for 4-bit
        abs_max = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = abs_max / qmax
        zero_point = torch.zeros_like(scale)
    else:
        qmin = 0
        qmax = (1 << bits) - 1         # e.g., 15 for 4-bit
        rmin = flat.amin(dim=-1, keepdim=True)
        rmax = flat.amax(dim=-1, keepdim=True)
        scale = (rmax - rmin).clamp(min=1e-8) / qmax
        zero_point = torch.round(-rmin / scale).clamp(qmin, qmax)

    return scale, zero_point


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int = 4,
    symmetric: bool = True,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a floating-point tensor to integer representation.

    Args:
        tensor: Input tensor (float).
        bits: Target bit width.
        symmetric: Symmetric quantization.
        group_size: Per-group quantization granularity.

    Returns:
        (quantized_tensor, scale, zero_point)
    """
    scale, zp = compute_scale_zero_point(
        tensor, bits=bits, symmetric=symmetric, group_size=group_size
    )

    if group_size > 0:
        orig_shape = tensor.shape
        flat = tensor.reshape(-1, group_size)
    else:
        flat = tensor.reshape(1, -1)
        orig_shape = tensor.shape

    qmin = -(1 << (bits - 1)) if symmetric else 0
    qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1

    quantized = torch.clamp(
        torch.round(flat / scale) + zp, qmin, qmax
    ).to(torch.int8)

    return quantized.reshape(orig_shape), scale, zp


def dequantize_tensor(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int = 128,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize an integer tensor back to floating-point.

    Args:
        quantized: Integer tensor.
        scale: Scale factors.
        zero_point: Zero-point offsets.
        group_size: Must match quantization group_size.
        target_dtype: Output dtype.

    Returns:
        Dequantized floating-point tensor.
    """
    orig_shape = quantized.shape

    if group_size > 0:
        flat = quantized.reshape(-1, group_size).float()
    else:
        flat = quantized.reshape(1, -1).float()

    dequantized = (flat - zero_point) * scale
    return dequantized.reshape(orig_shape).to(target_dtype)


# ---------------------------------------------------------------------------
# AWQ — Activation-Aware Weight Quantization
# ---------------------------------------------------------------------------

class AWQQuantizer:
    """
    Activation-Aware Weight Quantization (Lin et al., 2023).

    Key insight: only ~1% of weights are "salient" — those corresponding
    to large activation magnitudes. Scale salient channels *before*
    quantization to preserve their precision.

    Formula:
        s = (max(|X_channel|) / max(|W_channel|))^α
        W'_salient = W_salient × s

    Faster than GPTQ with comparable quality at 4-bit.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        alpha: float = 0.5,
    ):
        self.bits = bits
        self.group_size = group_size
        self.alpha = alpha

    def compute_scale(
        self,
        weight: torch.Tensor,
        activation_stats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-channel AWQ scaling factors.

        Args:
            weight: Weight matrix [out, in].
            activation_stats: Per-channel activation magnitudes [in].

        Returns:
            Scale tensor [in].
        """
        w_max = weight.abs().amax(dim=0).clamp(min=1e-8)
        a_max = activation_stats.clamp(min=1e-8)

        scale = (a_max / w_max).pow(self.alpha)
        return scale

    def quantize_linear(
        self,
        weight: torch.Tensor,
        activation_stats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a linear layer's weights using AWQ.

        Args:
            weight: Weight tensor [out, in].
            activation_stats: Activation magnitude statistics [in].

        Returns:
            (quantized_weight, scale, zero_point, awq_scale)
        """
        awq_scale = self.compute_scale(weight, activation_stats)

        # Scale weights before quantization
        scaled_weight = weight * awq_scale.unsqueeze(0)

        # Standard PTQ on scaled weights
        q_weight, scale, zp = quantize_tensor(
            scaled_weight,
            bits=self.bits,
            group_size=self.group_size,
        )

        return q_weight, scale, zp, awq_scale


# ---------------------------------------------------------------------------
# KV Cache Quantization
# ---------------------------------------------------------------------------

class KVCacheQuantizer:
    """
    KV cache quantization for inference-time memory reduction.

    KV cache formula:
        M_kv = 2 × L × n_kv × d_h × S × b_kv

    At 1M tokens with BF16: ~1.25 TB
    With INT8:               ~625 GB  (2× reduction)
    With INT4:               ~312 GB  (4× reduction, noticeable degradation)

    Strategies:
        - Per-channel: different scale for each attention head
        - Per-token:   scale based on each token's KV magnitude
        - Sliding window: recent tokens BF16, older tokens INT8
        - H2O eviction: drop lowest-attention KV entries entirely
    """

    def __init__(
        self,
        kv_bits: int = 8,
        per_channel: bool = True,
        sliding_window: Optional[int] = None,
    ):
        """
        Args:
            kv_bits: Quantization bits for KV cache (8 or 4).
            per_channel: Per-head quantization scales.
            sliding_window: If set, keep last N tokens in full precision.
        """
        self.kv_bits = kv_bits
        self.per_channel = per_channel
        self.sliding_window = sliding_window

    def quantize_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Quantize a KV pair for cache storage.

        Args:
            key:   [batch, n_kv_heads, seq_len, head_dim]
            value: [batch, n_kv_heads, seq_len, head_dim]

        Returns:
            (quantized_key_dict, quantized_value_dict) each with
            'data', 'scale', 'zero_point' keys.
        """
        def _quantize_one(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
            if self.sliding_window and tensor.shape[2] > self.sliding_window:
                # Keep recent tokens in full precision
                recent = tensor[:, :, -self.sliding_window:, :]
                old = tensor[:, :, :-self.sliding_window, :]

                if self.per_channel:
                    # Per-head quantization of old tokens
                    B, H, S, D = old.shape
                    old_flat = old.reshape(B * H, S * D)
                    q, sc, zp = quantize_tensor(
                        old_flat, bits=self.kv_bits, group_size=D,
                    )
                    return {
                        "old_q": q.reshape(B, H, S, D),
                        "old_scale": sc,
                        "old_zp": zp,
                        "recent": recent,
                    }

            # Full quantization
            B, H, S, D = tensor.shape
            flat = tensor.reshape(B * H, S * D)
            q, sc, zp = quantize_tensor(
                flat, bits=self.kv_bits, group_size=D,
            )
            return {
                "data": q.reshape(B, H, S, D),
                "scale": sc,
                "zero_point": zp,
            }

        return _quantize_one(key), _quantize_one(value)

    def memory_estimate(
        self,
        num_layers: int = 160,
        num_kv_heads: int = 16,
        head_dim: int = 128,
        seq_len: int = 1_000_000,
    ) -> Dict[str, float]:
        """
        Estimate KV cache memory at various precisions.

        Formula: M = 2 × L × n_kv × d_h × S × b_kv

        Returns:
            Dict mapping precision name to size in GB.
        """
        base = 2 * num_layers * num_kv_heads * head_dim * seq_len
        return {
            "BF16":  base * 2 / (1024**3),
            "FP8":   base * 1 / (1024**3),
            "INT8":  base * 1 / (1024**3),
            "INT4":  base * 0.5 / (1024**3),
        }


# ---------------------------------------------------------------------------
# Model Size Estimator
# ---------------------------------------------------------------------------

def estimate_model_size(
    total_params: int = 2_000_000_000_000,
    quant_format: QuantFormat = QuantFormat.Q4_K_M,
) -> Dict[str, float]:
    """
    Estimate model file size for a given quantization format.

    Sharding formula:
        N_shards = ceil(Total Size / Shard Size)

    Args:
        total_params: Total parameter count.
        quant_format: Target quantization format.

    Returns:
        Dict with 'size_tb', 'num_shards_5gb', 'num_shards_20gb'.
    """
    if quant_format in GGUF_BPW:
        bpw = GGUF_BPW[quant_format]
    elif quant_format in (QuantFormat.GPTQ, QuantFormat.AWQ):
        bpw = 4.5   # typical 4-bit with group scales
    elif quant_format == QuantFormat.EXL2:
        bpw = 4.0   # variable, typical average
    elif quant_format == QuantFormat.HQQ:
        bpw = 4.0
    elif quant_format == QuantFormat.AQLM:
        bpw = 2.5
    else:
        bpw = 16.0   # BF16 default

    size_bytes = total_params * bpw / 8
    size_tb = size_bytes / (1024**4)

    return {
        "size_tb": round(size_tb, 2),
        "size_gb": round(size_bytes / (1024**3), 1),
        "num_shards_5gb": math.ceil(size_bytes / (5 * 1024**3)),
        "num_shards_20gb": math.ceil(size_bytes / (20 * 1024**3)),
        "bpw": bpw,
    }
