"""
Unit tests for core model architecture.

Tests that all model components can be instantiated, run a forward pass
with correct shapes, and maintain expected mathematical properties.
"""

import math
import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Tiny config for tests (avoids multi-GB allocations)
# ---------------------------------------------------------------------------

def tiny_config():
    """Create a small ClaudeConfig for unit testing."""
    from src.model.transformer import ClaudeConfig
    return ClaudeConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        head_dim=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rope_scaling=None,
    )


def param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class TestRoPE:

    def test_import(self):
        from src.model.rope import RotaryEmbedding, apply_rotary_emb
        assert RotaryEmbedding is not None
        assert apply_rotary_emb is not None

    def test_instantiation(self):
        from src.model.rope import RotaryEmbedding
        cfg = tiny_config()
        rope = RotaryEmbedding(cfg)
        assert rope.head_dim == 64

    def test_forward_shape(self):
        from src.model.rope import RotaryEmbedding
        cfg = tiny_config()
        rope = RotaryEmbedding(cfg)
        cos, sin = rope(seq_len=32, device=torch.device("cpu"))
        assert cos.shape[-1] == 64
        assert sin.shape[-1] == 64

    def test_offset(self):
        from src.model.rope import RotaryEmbedding
        cfg = tiny_config()
        rope = RotaryEmbedding(cfg)
        cos0, sin0 = rope(seq_len=10, offset=0, device=torch.device("cpu"))
        cos5, sin5 = rope(seq_len=10, offset=5, device=torch.device("cpu"))
        assert not torch.allclose(cos0, cos5)

    def test_apply_rotary(self):
        from src.model.rope import RotaryEmbedding, apply_rotary_emb
        cfg = tiny_config()
        rope = RotaryEmbedding(cfg)
        cos, sin = rope(seq_len=16, device=torch.device("cpu"))
        q = torch.randn(1, 4, 16, 64)
        k = torch.randn(1, 2, 16, 64)
        q_rot = apply_rotary_emb(q, cos, sin)
        k_rot = apply_rotary_emb(k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        assert not torch.allclose(q, q_rot, atol=1e-5)


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------

class TestSwiGLU:

    def test_forward_shape(self):
        from src.model.swiglu import SwiGLU
        swiglu = SwiGLU(hidden_size=256, intermediate_size=512)
        x = torch.randn(2, 16, 256)
        out = swiglu(x)
        assert out.shape == (2, 16, 256)

    def test_param_count(self):
        from src.model.swiglu import SwiGLU
        swiglu = SwiGLU(hidden_size=256, intermediate_size=512)
        # gate: 256*512, up: 256*512, down: 512*256 (no bias)
        expected = 256 * 512 * 3
        actual = param_count(swiglu)
        assert actual == expected


# ---------------------------------------------------------------------------
# Grouped Query Attention
# ---------------------------------------------------------------------------

class TestGQA:

    def test_forward_shape(self):
        from src.model.attention import GroupedQueryAttention
        from src.model.rope import RotaryEmbedding
        cfg = tiny_config()
        gqa = GroupedQueryAttention(cfg)
        rope = RotaryEmbedding(cfg)
        x = torch.randn(2, 16, 256)
        cos, sin = rope(seq_len=16, device=torch.device("cpu"))
        result = gqa(x, cos, sin)
        # forward returns (output, present_kv) or just output
        out = result[0] if isinstance(result, tuple) else result
        assert out.shape == (2, 16, 256)

    def test_gqa_ratio(self):
        from src.model.attention import GroupedQueryAttention
        cfg = tiny_config()
        gqa = GroupedQueryAttention(cfg)
        assert gqa.num_heads == 4
        assert gqa.num_kv_heads == 2
        assert gqa.groups == 2


# ---------------------------------------------------------------------------
# MoE Layer
# ---------------------------------------------------------------------------

class TestMoE:

    def test_forward_shape(self):
        from src.model.moe import MoELayer
        cfg = tiny_config()
        moe = MoELayer(cfg)
        x = torch.randn(2, 8, 256)
        out, aux = moe(x)
        assert out.shape == (2, 8, 256)

    def test_aux_loss_returned(self):
        from src.model.moe import MoELayer
        cfg = tiny_config()
        moe = MoELayer(cfg)
        x = torch.randn(2, 8, 256)
        _, aux = moe(x)
        assert isinstance(aux, torch.Tensor)
        assert aux.ndim == 0  # scalar

    def test_expert_count(self):
        from src.model.moe import MoELayer
        cfg = tiny_config()
        moe = MoELayer(cfg)
        assert len(moe.experts) == 4


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:

    def test_forward_shape(self):
        from src.model.transformer import RMSNorm
        norm = RMSNorm(256)
        x = torch.randn(2, 16, 256)
        out = norm(x)
        assert out.shape == (2, 16, 256)

    def test_normalization(self):
        from src.model.transformer import RMSNorm
        norm = RMSNorm(256)
        x = torch.randn(2, 16, 256) * 10
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.2)


# ---------------------------------------------------------------------------
# ClaudeConfig
# ---------------------------------------------------------------------------

class TestConfig:

    def test_defaults(self):
        from src.model.transformer import ClaudeConfig
        cfg = ClaudeConfig()
        assert cfg.hidden_size == 16384
        assert cfg.num_hidden_layers == 160
        assert cfg.num_attention_heads == 128
        assert cfg.num_key_value_heads == 16
        assert cfg.num_experts == 128

    def test_small_config(self):
        cfg = tiny_config()
        assert cfg.hidden_size == 256
        assert cfg.num_hidden_layers == 2

    def test_from_dict(self):
        from src.model.transformer import ClaudeConfig
        d = {"vocab_size": 500, "hidden_size": 128, "num_hidden_layers": 4,
             "head_dim": 32, "num_attention_heads": 4, "num_key_value_heads": 2}
        cfg = ClaudeConfig.from_dict(d)
        assert cfg.hidden_size == 128


# ---------------------------------------------------------------------------
# Vision Encoder
# ---------------------------------------------------------------------------

class TestVision:

    def test_patch_embedding(self):
        from src.model.vision import PatchEmbedding, VisionConfig
        cfg = VisionConfig(image_size=224, patch_size=14, hidden_size=256)
        patch_emb = PatchEmbedding(cfg)
        x = torch.randn(1, 3, 224, 224)
        out = patch_emb(x)
        expected_patches = (224 // 14) ** 2
        assert out.shape == (1, expected_patches + 1, 256)

    def test_vision_encoder(self):
        from src.model.vision import VisionEncoder, VisionConfig
        cfg = VisionConfig(
            image_size=56, patch_size=14, hidden_size=128,
            num_layers=2, num_heads=4, intermediate_size=256,
        )
        encoder = VisionEncoder(cfg)
        x = torch.randn(1, 3, 56, 56)
        out = encoder(x)
        n_patches = (56 // 14) ** 2
        assert out.shape == (1, n_patches + 1, 128)

    def test_projector(self):
        from src.model.vision import VisionProjector, VisionConfig
        cfg = VisionConfig(
            hidden_size=128, projector_hidden=256, llm_hidden_size=512,
        )
        proj = VisionProjector(cfg)
        x = torch.randn(1, 17, 128)
        out = proj(x)
        assert out.shape == (1, 17, 512)
