"""
Claude Opus 4.6 — Model Architecture Package

Speculative/research implementation of a 2T-parameter MoE Transformer.

Architecture summary:
    - 160 Transformer layers
    - d_model = 16,384
    - 128 attention heads (GQA: 16 KV heads, 8:1 ratio)
    - 128 experts per MoE layer, top-2 routing
    - SwiGLU activation (d_ff = 49,152 per expert)
    - RoPE + YaRN positional encoding (1M context)
    - RMSNorm throughout
    - ViT-G/14 vision encoder for multimodal input
"""

from .rope import RotaryEmbedding, apply_rotary_emb
from .swiglu import SwiGLU
from .attention import GroupedQueryAttention
from .moe import Expert, MoELayer
from .transformer import ClaudeConfig, RMSNorm, TransformerLayer, ClaudeModel
from .vision import (
    VisionConfig,
    PatchEmbedding,
    ViTBlock,
    VisionEncoder,
    VisionProjector,
    MultimodalModel,
    GUIAction,
    preprocess_screenshot,
)
from .quantization import (
    QuantFormat,
    QuantConfig,
    AWQQuantizer,
    KVCacheQuantizer,
    quantize_tensor,
    dequantize_tensor,
    compute_scale_zero_point,
    estimate_model_size,
)
from .flash_attention import (
    OnlineSoftmax,
    flash_attention_reference,
    RingAttentionConfig,
    ChunkedPrefill,
    ActivationCheckpointing,
    InferencePhaseAnalysis,
    flash_decoding_speedup,
)
from .expert_routing import (
    ExpertChoiceRouter,
    ExpertUtilizationAnalyzer,
    RoutingComparison,
)

__all__ = [
    # Positional encoding
    "RotaryEmbedding",
    "apply_rotary_emb",
    # Activation
    "SwiGLU",
    # Attention
    "GroupedQueryAttention",
    # Mixture of Experts
    "Expert",
    "MoELayer",
    # Transformer core
    "ClaudeConfig",
    "RMSNorm",
    "TransformerLayer",
    "ClaudeModel",
    # Vision (multimodal)
    "VisionConfig",
    "PatchEmbedding",
    "ViTBlock",
    "VisionEncoder",
    "VisionProjector",
    "MultimodalModel",
    "GUIAction",
    "preprocess_screenshot",
    # Quantization
    "QuantFormat",
    "QuantConfig",
    "AWQQuantizer",
    "KVCacheQuantizer",
    "quantize_tensor",
    "dequantize_tensor",
    "compute_scale_zero_point",
    "estimate_model_size",
    # FlashAttention & Ring Attention
    "OnlineSoftmax",
    "flash_attention_reference",
    "RingAttentionConfig",
    "ChunkedPrefill",
    "ActivationCheckpointing",
    "InferencePhaseAnalysis",
    "flash_decoding_speedup",
    # Expert routing
    "ExpertChoiceRouter",
    "ExpertUtilizationAnalyzer",
    "RoutingComparison",
]
