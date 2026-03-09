"""
Grouped Query Attention (GQA) with FlashAttention and KV-cache support.

Config:
    num_attention_heads    = 128   (query heads)
    num_key_value_heads    = 16    (KV heads)
    head_dim               = 128
    hidden_size            = 16384

GQA groups:  128 / 16 = 8  (each KV head serves 8 query heads)

Memory savings vs MHA:
    MHA KV at 1M tokens : 2 × 160 × 128 × 128 × 1M × 2 bytes = ~10 TB
    GQA KV at 1M tokens : 2 × 160 ×  16 × 128 × 1M × 2 bytes = ~1.25 TB
    Saving              : ~87.5%

References:
    - GQA: Ainslie et al. 2023 (arXiv:2305.13245)
    - FlashAttention-2: Dao 2023 (arXiv:2307.08691)
    - FlashAttention-3: Shah et al. 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with GQA and optional KV cache.

    During prefill (prompt processing):
        - Processes all T tokens in parallel
        - is_causal=True for causal masking
        - Uses FlashAttention via F.scaled_dot_product_attention

    During decode (one token at a time):
        - offset > 0; K/V appended to past_key_value cache
        - KV cache grows by 1 per step
        - Attention is over full history (causal already satisfied by cache)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size      = config.hidden_size              # 16384
        self.num_heads        = config.num_attention_heads       # 128
        self.num_kv_heads     = config.num_key_value_heads       # 16
        self.head_dim         = config.head_dim                  # 128
        self.groups           = self.num_heads // self.num_kv_heads  # 8

        assert self.hidden_size == self.num_heads * self.head_dim, (
            f"hidden_size ({self.hidden_size}) must equal "
            f"num_heads ({self.num_heads}) × head_dim ({self.head_dim})"
        )

        # Projection matrices — no bias (standard in modern LLMs)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x               : [B, T, hidden_size]
            cos, sin        : [1, 1, T, head_dim]  — from RotaryEmbedding
            attention_mask  : [B, 1, T, S] or None  (S = total seq after cache)
            past_key_value  : (K_cache, V_cache) tensors or None
            use_cache       : whether to return updated KV cache

        Returns:
            output          : [B, T, hidden_size]
            present_kv      : updated (K, V) cache or None
        """
        B, T, _ = x.shape

        # --- Project Q, K, V ---
        q = self.q_proj(x)   # [B, T, num_heads  * head_dim]
        k = self.k_proj(x)   # [B, T, num_kv_heads * head_dim]
        v = self.v_proj(x)   # [B, T, num_kv_heads * head_dim]

        # Reshape to [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # --- Apply RoPE ---
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # --- KV Cache (for autoregressive decoding) ---
        if past_key_value is not None:
            k_cache, v_cache = past_key_value
            k = torch.cat([k_cache, k], dim=2)   # append along seq dim
            v = torch.cat([v_cache, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # --- GQA: expand KV heads to match Q heads ---
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        # --- Attention ---
        # F.scaled_dot_product_attention dispatches to FlashAttention
        # when running on CUDA with compatible dtypes (BF16 / FP16)
        is_causal = (past_key_value is None) and (attention_mask is None)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )  # [B, num_heads, T, head_dim]

        # --- Merge heads and project ---
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, self.num_heads * self.head_dim)
        output = self.o_proj(attn_out)

        return output, present_kv

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, "
            f"q_heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"groups={self.groups}"
        )
