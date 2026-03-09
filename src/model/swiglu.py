"""
SwiGLU Activation Function.

SwiGLU (Shazeer, 2020) is the standard activation in modern
transformer FFNs, preferred over ReLU/GELU for its superior
quality at matched compute:

    SwiGLU(x) = (Swish(x · W_gate) ⊙ x · W_up) · W_down

where Swish(x) = x · σ(βx) and σ is the sigmoid function.

Key advantages over ReLU:
    - ~1–2% better perplexity at same compute
    - Smoother gradients → more stable training
    - Multiplicative gating provides richer expressivity

The SwiGLU FFN requires THREE weight matrices per expert
(gate, up, down) instead of two, but the intermediate dimension
is adjusted to keep total FLOPs comparable:

    P_ffn = 3 × d_model × d_ff

For Opus 4.6 (speculated):
    d_model = 16,384
    d_ff = 65,536 (= 4 × d_model)
    P_ffn per expert = 3 × 16,384 × 65,536 ≈ 3.22B params

References:
    - Shazeer, "GLU Variants Improve Transformer", 2020
    - Dauphin et al., "Language Modeling with Gated Convolutional
      Networks", 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Architecture:
        input (d_model)
            ├─→ gate_proj (d_model → d_ff)  ──→ Swish/SiLU
            │                                      │
            ├─→ up_proj   (d_model → d_ff)  ──→   ⊙ (element-wise multiply)
            │                                      │
            └──────────────────────────────────→ down_proj (d_ff → d_model)
                                                   │
                                                output (d_model)

    Dimensions for Opus 4.6:
        - hidden_size (d_model): 16,384
        - intermediate_size (d_ff): 65,536
        - Parameters per FFN: ~3.22B

    Note: When used inside MoE, each expert is a separate
    SwiGLU instance. With 128 experts, total FFN params
    ≈ 128 × 3.22B ≈ 412B per layer.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        """
        Args:
            hidden_size: Model dimension (d_model).
            intermediate_size: FFN intermediate dimension (d_ff).
            bias: Whether to use bias in linear layers
                  (modern LLMs typically don't).
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Three weight matrices for SwiGLU
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Computes: down(SiLU(gate(x)) ⊙ up(x))

        Where SiLU(x) = x · σ(x) is the Swish activation
        (β=1 variant, which is standard in practice).

        Args:
            x: Input tensor of shape [..., hidden_size].

        Returns:
            Output tensor of shape [..., hidden_size].
        """
        # Gate branch: x → W_gate → SiLU
        gate = F.silu(self.gate_proj(x))

        # Up branch: x → W_up
        up = self.up_proj(x)

        # Element-wise multiply and project down
        return self.down_proj(gate * up)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"params={self.num_params():,}"
        )

    def num_params(self) -> int:
        """Total parameter count for this FFN."""
        return sum(p.numel() for p in self.parameters())
