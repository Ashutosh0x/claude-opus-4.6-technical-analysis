"""
Loss Functions for Claude Opus 4.6 Training.

Training uses a combination of losses:

Primary:
    L_LM = -(1/T) Σ_{t=1}^{T} log P_θ(x_t | x_{<t})
    Standard next-token prediction cross-entropy.

MoE Auxiliary:
    L_total = L_LM + α·L_balance + β·L_z
    where:
        L_balance = E · Σ f_i · p_i  (load balancing, α = 0.02)
        L_z = mean(log Σ exp(z_i))²  (router stability, β = 0.001)

Perplexity (evaluation metric):
    PPL = exp(L_LM)
    Frontier models: PPL ≈ 5–8 on standard benchmarks

References:
    - Fedus et al., "Switch Transformers" (balance loss), 2022
    - Zoph et al., "ST-MoE" (z-loss), 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class LanguageModelingLoss(nn.Module):
    """
    Cross-entropy loss for causal language modeling.

    L_LM = -(1/T) Σ_{t=1}^{T} log P_θ(x_t | x_{<t})

    Shifts logits and labels by one position so that
    position t predicts token at position t+1.

    Uses label smoothing optionally to prevent overconfidence.
    """

    def __init__(
        self,
        vocab_size: int,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute next-token prediction loss.

        Args:
            logits: Model output [B, T, vocab_size].
            labels: Target token IDs [B, T]. Use -100 for padding.

        Returns:
            Scalar cross-entropy loss.
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return loss


class MoELoss(nn.Module):
    """
    Combined loss for MoE training.

    L_total = L_LM + α·L_balance + β·L_z

    The auxiliary losses prevent:
        - Expert collapse (some experts never used)
        - Router instability (exploding logit magnitudes)
    """

    def __init__(
        self,
        vocab_size: int,
        balance_coef: float = 0.02,
        z_loss_coef: float = 0.001,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            vocab_size: Vocabulary size for CE loss.
            balance_coef: Weight α for load balancing loss.
            z_loss_coef: Weight β for router z-loss.
            label_smoothing: Label smoothing factor.
        """
        super().__init__()
        self.lm_loss = LanguageModelingLoss(
            vocab_size=vocab_size,
            label_smoothing=label_smoothing,
        )
        self.balance_coef = balance_coef
        self.z_loss_coef = z_loss_coef

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total MoE training loss.

        Args:
            logits: Model output [B, T, vocab_size].
            labels: Target token IDs [B, T].
            aux_loss: Sum of MoE aux losses from all layers.

        Returns:
            Dict with 'total_loss', 'lm_loss', 'aux_loss'.
        """
        lm_loss = self.lm_loss(logits, labels)

        total_loss = lm_loss + aux_loss

        return {
            "total_loss": total_loss,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
        }


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from cross-entropy loss.

    PPL = exp(L_LM)

    Frontier models typically achieve PPL ≈ 5–8 on
    standard benchmarks.

    Domain-specific PPL (estimated):
        English prose: ~5–7
        Code (Python):  ~3–5
        Mathematics:    ~8–12
        Legal text:     ~6–9
        Low-resource languages: ~15–30
    """
    return torch.exp(loss)
