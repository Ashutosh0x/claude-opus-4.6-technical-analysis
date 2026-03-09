"""
Vision Encoder — ViT + Projector for Multimodal Input.

Architecture:
    Image → Patch Embedding → ViT Encoder → Projector → LLM embedding space

Patch tokenization formula:
    N_visual_tokens = (H / P) × (W / P)
    where H, W = image dimensions, P = patch size

Visual token counts:
    224×224  @ P=14 →   256 tokens
    336×336  @ P=14 →   576 tokens
    672×672  @ P=14 → 2,304 tokens
    1344×1344 @ P=14 → 9,216 tokens

Vision encoder params (estimated for Opus 4.6):
    ViT-G/14    : ~1.8B params (~3.6 GB BF16)
    Projector   : ~100–500M params
    Total vision: ~1–3B params (< 0.1% of total model)

Computer Use pipeline:
    Screenshot → ViT → Visual tokens → LLM → Action(x, y, click/type/scroll)

References:
    - ViT: Dosovitskiy et al. 2020 (arXiv:2010.11929)
    - SigLIP: Zhai et al. 2023
    - LLaVA: Liu et al. 2023 (arXiv:2304.08485)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VisionConfig:
    """Vision encoder configuration."""
    image_size: int = 672              # max input resolution
    patch_size: int = 14               # patch size in pixels
    hidden_size: int = 1664            # ViT-G hidden size
    num_layers: int = 48               # ViT-G layers
    num_heads: int = 16
    intermediate_size: int = 8192
    dropout: float = 0.0
    # Projector
    projector_hidden: int = 4096
    llm_hidden_size: int = 16384       # must match LLM d_model

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_visual_tokens(self) -> int:
        """Number of visual tokens per image at max resolution."""
        return self.num_patches  # + 1 for CLS (optional)


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings using 2D convolution.

    Input:  [B, 3, H, W]  — RGB image
    Output: [B, N_patches, hidden_size]

    The conv2d with kernel=patch_size, stride=patch_size extracts
    non-overlapping patches and projects them to hidden_size in one step.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.num_patches = config.num_patches

        # Learnable position embeddings for patches
        self.position_embedding = nn.Embedding(
            config.num_patches + 1,  # +1 for optional CLS token
            config.hidden_size,
        )

        # Optional CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size) * 0.02
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W] — normalized pixel values

        Returns:
            [B, N+1, hidden_size] — patch embeddings + CLS
        """
        B = pixel_values.shape[0]

        # Extract patches via convolution
        # [B, 3, H, W] → [B, D, H/P, W/P]
        x = self.proj(pixel_values)
        # [B, D, H/P, W/P] → [B, N_patches, D]
        x = x.flatten(2).transpose(1, 2)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]

        # Add positional embeddings
        position_ids = torch.arange(
            x.shape[1], device=x.device
        ).unsqueeze(0)
        x = x + self.position_embedding(position_ids)

        return x


# ---------------------------------------------------------------------------
# ViT Encoder Block
# ---------------------------------------------------------------------------

class ViTBlock(nn.Module):
    """
    One Vision Transformer encoder block.

    Pre-norm layout (same as LLM):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


# ---------------------------------------------------------------------------
# Vision Encoder (Full ViT)
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """
    Vision Transformer encoder (ViT-G/14 scale).

    Processes images into a sequence of visual tokens that can be
    concatenated with text tokens and fed into the LLM.

    Total params: ~1.8B (ViT-G)
    Size: ~3.6 GB BF16 (< 0.1% of full model)
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.blocks = nn.ModuleList([
            ViTBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W]

        Returns:
            visual_features: [B, N_patches+1, vision_hidden_size]
        """
        x = self.patch_embed(pixel_values)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Vision-Language Projector
# ---------------------------------------------------------------------------

class VisionProjector(nn.Module):
    """
    Projects vision encoder outputs into the LLM's embedding space.

    Architecture options:
        1. Linear: single linear layer (simplest)
        2. MLP: two-layer MLP with GELU (LLaVA-1.5 style)
        3. Cross-attention: query-based (Flamingo style)

    We implement the MLP variant (LLaVA-1.5 default):
        vision_dim → projector_hidden → llm_dim

    This is the bridge between the vision encoder and the LLM.
    ~100–500M params depending on dimensions.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.projector_hidden),
            nn.GELU(),
            nn.Linear(config.projector_hidden, config.llm_hidden_size),
        )

    def forward(
        self, visual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_features: [B, N, vision_dim]

        Returns:
            projected: [B, N, llm_dim]  — ready to concat with text embeddings
        """
        return self.projector(visual_features)


# ---------------------------------------------------------------------------
# Full Multimodal Model (Vision + LLM)
# ---------------------------------------------------------------------------

class MultimodalModel(nn.Module):
    """
    Combines Vision Encoder + Projector + LLM for multimodal inference.

    Input can be:
        - Text only: standard autoregressive LLM
        - Text + images: visual tokens injected at <|image|> positions
        - Screenshots (computer use): processed as images

    Visual token context cost:
        effective_text_context = 1,000,000 - N_images × N_visual_tokens/image

    Token cost of a screenshot at 1920×1080 @ P=14:
        N = ceil(1920/14) × ceil(1080/14) = 137 × 78 ≈ 10,686 tokens
        (Usually downsampled to 672×672 → 2,304 tokens)
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        projector: VisionProjector,
        language_model,  # ClaudeModel
        image_token_id: int = 128011,  # <|image|> token ID
    ):
        super().__init__()
        self.vision = vision_encoder
        self.projector = projector
        self.llm = language_model
        self.image_token_id = image_token_id

    def _merge_visual_tokens(
        self,
        input_ids: torch.Tensor,        # [B, T]
        text_embeds: torch.Tensor,       # [B, T, D]
        visual_embeds: torch.Tensor,     # [B, N_img, N_patches, D]
    ) -> torch.Tensor:
        """
        Replace <|image|> token embeddings with visual token embeddings.

        Before: [tok, tok, <img>, tok, tok, <img>, tok]
        After:  [tok, tok, vis1, vis2, ..., visN, tok, tok, vis1, ..., visN, tok]
        """
        B, T, D = text_embeds.shape
        merged = text_embeds.clone()

        for b in range(B):
            img_positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            for img_idx, pos in enumerate(img_positions):
                if img_idx < visual_embeds.shape[1]:
                    # Replace the <|image|> token with visual tokens
                    # In practice, this requires expanding the sequence
                    # Here we just replace the single token position
                    # (production implementations expand the tensor)
                    merged[b, pos] = visual_embeds[b, img_idx].mean(dim=0)

        return merged

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass for multimodal input.

        Args:
            input_ids    : [B, T] — token IDs with <|image|> placeholders
            pixel_values : [B, N_images, 3, H, W] — images (or None for text-only)
            attention_mask: [B, T]
            labels       : [B, T] — for training loss

        Returns:
            dict with "logits", "loss" (if labels), etc.
        """
        # Get text embeddings
        text_embeds = self.llm.embed_tokens(input_ids)

        # Process images if provided
        if pixel_values is not None and pixel_values.numel() > 0:
            B, N_img = pixel_values.shape[:2]
            # Flatten batch × images for vision encoder
            flat_pixels = pixel_values.view(-1, *pixel_values.shape[2:])
            visual_features = self.vision(flat_pixels)
            visual_embeds = self.projector(visual_features)
            # Reshape: [B*N_img, N_patches, D] → [B, N_img, N_patches, D]
            N_patches = visual_embeds.shape[1]
            D = visual_embeds.shape[2]
            visual_embeds = visual_embeds.view(B, N_img, N_patches, D)

            # Merge visual tokens into text sequence
            text_embeds = self._merge_visual_tokens(
                input_ids, text_embeds, visual_embeds
            )

        # Run through LLM (skip embedding since we already have embeds)
        # In practice, we'd modify ClaudeModel to accept embeddings directly
        return self.llm(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Computer Use — Screenshot-to-Action Processor
# ---------------------------------------------------------------------------

@dataclass
class GUIAction:
    """
    A GUI action output from the computer use pipeline.

    Action space:
        click        : coordinate [x, y], button
        double_click : coordinate [x, y]
        type         : text string
        key          : key combination (e.g., "ctrl+c")
        scroll       : coordinate, direction, amount
        screenshot   : capture current screen
        drag         : start [x, y], end [x, y]
    """
    action_type: str       # "click", "type", "key", "scroll", "screenshot", "drag"
    coordinate: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    key: Optional[str] = None
    button: str = "left"   # "left", "right", "middle"
    direction: str = "down"
    amount: int = 3
    end_coordinate: Optional[Tuple[int, int]] = None


def preprocess_screenshot(
    screenshot_bytes: bytes,
    target_size: int = 672,
) -> torch.Tensor:
    """
    Preprocess a screenshot for the vision encoder.

    1. Decode image bytes to tensor
    2. Resize to target_size × target_size
    3. Normalize to [-1, 1] range

    At 672×672 with P=14: 2,304 visual tokens per screenshot.
    Cost: 2,304 × $5/M ≈ $0.012 per screenshot (input).

    A 50-step GUI session:
        C ≈ 50 × 2304 × $5/M + 50 × 200 × $25/M
        = $0.576 + $0.25 = $0.83
    """
    # Placeholder — in production, use PIL/torchvision
    # This returns a dummy tensor for API compatibility
    return torch.randn(1, 3, target_size, target_size)
