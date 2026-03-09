"""
Full Claude Opus 4.6 Model Architecture.

Components:
    ClaudeConfig    : Dataclass mirroring config.json
    RMSNorm         : Root Mean Square Layer Normalization
    TransformerLayer: One decoder block (Attention + MoE FFN)
    ClaudeModel     : Full 160-layer model with embedding + LM head

Model dimensions:
    vocab_size     = 131072
    hidden_size    = 16384
    num_layers     = 160
    num_heads      = 128  (GQA with 16 KV heads)
    num_experts    = 128  (top-2 active)
    max_seq_len    = 1048576  (1M tokens via YaRN)
    total_params   ≈ 2T
    active_params  ≈ 120–300B per token
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

from .rope import RotaryEmbedding, apply_rotary_emb
from .attention import GroupedQueryAttention
from .moe import MoELayer


# ---------------------------------------------------------------------------
# Config dataclass (mirrors config.json)
# ---------------------------------------------------------------------------

@dataclass
class ClaudeConfig:
    """All architecture hyperparameters for Claude Opus 4.6."""

    # Dimensions
    vocab_size: int             = 131072
    hidden_size: int            = 16384
    intermediate_size: int      = 65536
    num_hidden_layers: int      = 160
    head_dim: int               = 128

    # Attention
    num_attention_heads: int    = 128
    num_key_value_heads: int    = 16    # GQA

    # MoE
    num_experts: int            = 128
    num_experts_per_tok: int    = 2
    router_aux_loss_coef: float = 0.02

    # Position encoding
    max_position_embeddings: int = 1048576
    rope_theta: float            = 500000.0
    rope_scaling: Dict[str, Any] = field(default_factory=lambda: {
        "type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 131072,
        "attention_factor": 0.1,
        "beta_fast": 32,
        "beta_slow": 1,
    })

    # Misc
    rms_norm_eps: float         = 1e-5
    tie_word_embeddings: bool   = False
    use_cache: bool             = True

    # Token IDs
    bos_token_id: int           = 1
    eos_token_id: int           = 2
    pad_token_id: int           = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClaudeConfig":
        """Construct config from a dictionary (e.g. loaded from JSON)."""
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (no mean subtraction).

    y = x / RMS(x) * weight
    RMS(x) = sqrt(mean(x²) + ε)

    Preferred over LayerNorm in modern LLMs (LLaMA, Mistral, etc.)
    because it's slightly faster and equally effective.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability, then cast back
        dtype = x.dtype
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        normed = x_f32 * rms
        return (self.weight * normed).to(dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Transformer Layer (one decoder block)
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """
    Single Pre-Norm Transformer decoder block.

    Pre-Norm layout:
        x = x + Attention(RMSNorm(x))
        x = x + MoE(RMSNorm(x))

    Pre-Norm is more stable than Post-Norm at large scale
    (used in GPT-3, LLaMA, Mistral, etc.)
    """

    def __init__(self, config: ClaudeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm     = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn           = GroupedQueryAttention(config)
        self.post_attn_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp                 = MoELayer(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x              : [B, T, hidden_size]
            cos, sin       : RoPE embeddings
            attention_mask : optional mask [B, 1, T, S]
            past_key_value : optional cached (K, V) tensors
            use_cache      : return updated KV cache

        Returns:
            x          : [B, T, hidden_size]
            aux_loss   : MoE load-balancing loss (scalar)
            present_kv : updated KV cache or None
        """
        # ---- Attention sublayer ----
        residual = x
        x = self.input_layernorm(x)
        x, present_kv = self.self_attn(
            x, cos, sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = residual + x

        # ---- MoE FFN sublayer ----
        residual = x
        x = self.post_attn_layernorm(x)
        x, aux_loss = self.mlp(x)
        x = residual + x

        return x, aux_loss, present_kv


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class ClaudeModel(nn.Module):
    """
    Claude Opus 4.6 — Full autoregressive language model.

    Architecture:
        embed_tokens  →  [160 × TransformerLayer]  →  RMSNorm  →  lm_head

    Usage (training):
        model = ClaudeModel(config)
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()

    Usage (inference):
        model = ClaudeModel(config)
        outputs = model(input_ids, use_cache=True)
        next_token_logits = outputs["logits"][:, -1, :]
        past_kvs = outputs["past_key_values"]
    """

    def __init__(self, config: ClaudeConfig):
        super().__init__()
        self.config = config

        # Token embedding table
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        # Shared RoPE embeddings (one instance, all layers share)
        self.rotary_emb = RotaryEmbedding(config)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # LM head (unembedding)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Optional weight tying (embed_tokens ↔ lm_head)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * num_layers)
        self._scale_residual_projections()

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """Standard normal init, scaled for depth."""
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        """
        Scale output projections (o_proj, down_proj) of residual branches.
        Technique from GPT-2: divide by sqrt(2 * num_layers) to prevent
        variance explosion at initialization.
        """
        scale = 1.0 / math.sqrt(2 * self.config.num_hidden_layers)
        for layer in self.layers:
            layer.self_attn.o_proj.weight.data.mul_(scale)
            for expert in layer.mlp.experts:
                expert.down_proj.weight.data.mul_(scale)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_router_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids       : [B, T]  — token IDs
            attention_mask  : [B, T]  — 1=real token, 0=pad  (optional)
            labels          : [B, T]  — for language modeling loss (optional)
            past_key_values : list of (K, V) per layer (for incremental decode)
            use_cache       : whether to return past_key_values
            output_router_logits : whether to return per-layer MoE aux losses

        Returns dict with keys:
            "loss"             : CE loss + aux loss  (if labels given)
            "logits"           : [B, T, vocab_size]
            "past_key_values"  : list of (K, V) per layer  (if use_cache)
            "aux_losses"       : list of per-layer MoE losses (if requested)
        """
        B, T = input_ids.shape
        offset = 0
        if past_key_values is not None and past_key_values[0] is not None:
            offset = past_key_values[0][0].shape[2]

        # --- Embedding ---
        x = self.embed_tokens(input_ids)   # [B, T, hidden_size]

        # --- RoPE cos/sin for this chunk ---
        cos, sin = self.rotary_emb(seq_len=T, offset=offset, device=x.device, dtype=x.dtype)

        # --- Build causal 4D attention mask if padding present ---
        attn_mask_4d = self._build_4d_mask(attention_mask, B, T, offset, x.dtype, x.device)

        # --- Transformer layers ---
        present_kvs  = [] if use_cache else None
        total_aux    = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        aux_losses   = []

        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values is not None else None
            x, aux_loss, present_kv = layer(
                x, cos, sin,
                attention_mask=attn_mask_4d,
                past_key_value=pkv,
                use_cache=use_cache,
            )
            total_aux = total_aux + aux_loss
            if output_router_logits:
                aux_losses.append(aux_loss)
            if use_cache:
                present_kvs.append(present_kv)

        # --- Final norm & LM head ---
        x = self.norm(x)
        logits = self.lm_head(x).float()   # [B, T, vocab_size], fp32 for loss

        # --- Language modeling loss ---
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            loss = ce_loss + total_aux

        result: Dict[str, Any] = {"logits": logits}
        if loss is not None:
            result["loss"] = loss
            result["aux_loss"] = total_aux
        if use_cache:
            result["past_key_values"] = present_kvs
        if output_router_logits:
            result["aux_losses"] = aux_losses

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_4d_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        B: int, T: int, offset: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Convert [B, S] padding mask to [B, 1, T, S] additive attention mask.
        Returns None if no padding mask provided (FlashAttention handles causal).
        """
        if attention_mask is None:
            return None
        S = offset + T
        mask = torch.zeros(B, 1, T, S, dtype=dtype, device=device)
        pad_positions = (attention_mask == 0)   # [B, S]
        mask[:, 0, :, :] = pad_positions.unsqueeze(1).expand(-1, T, -1) * torch.finfo(dtype).min
        return mask

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count total model parameters."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
        return n

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Alias for get_num_params."""
        return self.get_num_params(non_embedding=exclude_embeddings)

    def num_active_parameters(self) -> int:
        """Estimate active parameters per token (shared + k experts)."""
        embed_params = self.embed_tokens.weight.numel()
        head_params = self.lm_head.weight.numel()

        attn_params = sum(
            p.numel() for layer in self.layers
            for p in layer.self_attn.parameters()
        )
        norm_params = sum(
            p.numel() for name, p in self.named_parameters()
            if "layernorm" in name or "norm" in name
        )

        k = self.config.num_experts_per_tok
        expert_params_per = sum(
            p.numel() for p in self.layers[0].mlp.experts[0].parameters()
        )
        active_expert_params = (
            self.config.num_hidden_layers * k * expert_params_per
        )
        router_params = sum(
            p.numel() for layer in self.layers
            for p in layer.mlp.gate.parameters()
        )

        return (
            embed_params + head_params + attn_params
            + norm_params + active_expert_params + router_params
        )
