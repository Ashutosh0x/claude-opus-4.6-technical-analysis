"""
Knowledge Distillation and Model Compression.

Implements distillation (Opus -> Sonnet/Haiku), structured pruning,
and expert utilization analysis for MoE model compression.

Knowledge Distillation Loss (Hinton et al., 2015):
    L_distill = (1 - α) CE(y, y_hat) + α T² KL(P_teacher^T || P_student^T)
    where T = temperature, α = interpolation weight.

Model family distillation:
    Opus 4.6   (2-5T total)  → Pre-trained from scratch
    Sonnet 4.6 (~200-500B?)  → Likely distilled from Opus
    Haiku 4.6  (~30-70B?)    → Likely distilled from Sonnet/Opus

Structured Pruning:
    W_pruned = W ⊙ M,  M_ij = 1[|W_ij| > θ]

Expert Utilization Entropy:
    H = -Σ p_i log p_i
    H = log(E) → perfectly balanced
    H << log(E) → some experts dominate (collapse risk)

References:
    - Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
    - Sanh et al., "DistilBERT", 2019
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Knowledge Distillation Loss
# ---------------------------------------------------------------------------

class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for training a smaller student model.

    Formula:
        L = (1 - α) CE(y, y_hat) + α T² KL(P_teacher || P_student)

    where:
        P_teacher = softmax(z_teacher / T)
        P_student = softmax(z_student / T)
        T = temperature (higher → softer distributions)
        α = interpolation weight between hard and soft targets

    Soft targets carry "dark knowledge" — relationships between
    classes that the hard labels don't capture.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        vocab_size: Optional[int] = None,
    ):
        """
        Args:
            temperature: Softening temperature T.
            alpha: Weight for distillation loss (1-α for hard loss).
            vocab_size: Vocabulary size for cross-entropy.
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.vocab_size = vocab_size

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation + hard-target loss.

        Args:
            student_logits: [B, T, V] student model output.
            teacher_logits: [B, T, V] teacher model output (detached).
            labels: [B, T] ground-truth token IDs.

        Returns:
            Dict with 'total_loss', 'distill_loss', 'hard_loss'.
        """
        T = self.temperature

        # Soft distributions
        teacher_probs = F.log_softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        # KL divergence (on softened distributions)
        # KL(P_teacher || P_student) = Σ P_teacher * (log P_teacher - log P_student)
        distill_loss = F.kl_div(
            student_log_probs,
            teacher_probs.exp(),
            reduction="batchmean",
            log_target=False,
        ) * (T * T)    # Scale by T² as per Hinton

        # Hard-target cross-entropy
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        # Combined loss
        total_loss = (
            (1 - self.alpha) * hard_loss
            + self.alpha * distill_loss
        )

        return {
            "total_loss": total_loss,
            "distill_loss": distill_loss,
            "hard_loss": hard_loss,
        }


# ---------------------------------------------------------------------------
# Distillation Trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """
    Trainer for distilling a large teacher model into a smaller student.

    Claude model family distillation:
        Opus 4.6   (2-5T total)  → Teacher (pre-trained from scratch)
        Sonnet 4.6 (~200-500B?)  → Student (distilled from Opus)
        Haiku 4.6  (~30-70B?)    → Student (distilled from Sonnet/Opus)

    Training procedure:
        1. Forward pass through both teacher & student on same batch
        2. Compute distillation loss (soft + hard targets)
        3. Backprop through student only (teacher is frozen)
        4. Optional: progressive layer-wise distillation
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        self.teacher = teacher
        self.student = student
        self.loss_fn = DistillationLoss(
            temperature=temperature, alpha=alpha
        )

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        One training step of distillation.

        Args:
            input_ids: [B, T] token IDs.
            labels: [B, T] target token IDs.

        Returns:
            Loss dict from DistillationLoss.
        """
        # Teacher forward (no gradient)
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)

        # Student forward
        student_logits = self.student(input_ids)

        return self.loss_fn(student_logits, teacher_logits, labels)


# ---------------------------------------------------------------------------
# Structured Pruning
# ---------------------------------------------------------------------------

class StructuredPruning:
    """
    Structured pruning for model compression.

    Formula:
        W_pruned = W ⊙ M,  M_ij = 1[|W_ij| > θ]

    Pruning strategies:
        - Magnitude: Remove weights below threshold
        - Movement: Remove weights that move toward zero during training
        - Expert pruning: Remove least-used MoE experts entirely
    """

    @staticmethod
    def magnitude_prune(
        tensor: torch.Tensor,
        sparsity: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune weights by magnitude (unstructured).

        Args:
            tensor: Weight tensor.
            sparsity: Fraction of weights to prune (0.5 = 50%).

        Returns:
            (pruned_tensor, binary_mask)
        """
        threshold = torch.quantile(
            tensor.abs().float(), sparsity
        )
        mask = (tensor.abs() >= threshold).float()
        return tensor * mask, mask

    @staticmethod
    def compute_sparsity(tensor: torch.Tensor) -> float:
        """Fraction of zero weights."""
        total = tensor.numel()
        zeros = (tensor == 0).sum().item()
        return zeros / total if total > 0 else 0.0

    @staticmethod
    def prune_experts(
        expert_usage: torch.Tensor,
        num_to_prune: int = 8,
    ) -> list:
        """
        Identify least-used experts for removal.

        Args:
            expert_usage: [E] usage count or average routing probability.
            num_to_prune: Number of experts to remove.

        Returns:
            List of expert indices to prune.
        """
        _, indices = torch.topk(expert_usage, num_to_prune, largest=False)
        return indices.tolist()

    @staticmethod
    def estimated_speedup(sparsity: float) -> float:
        """
        Estimated inference speedup from weight sparsity.

        Real speedup depends on hardware support for sparse operations.
        Typical: ~50% sparsity → 1.3-1.5× speedup with structured sparsity.
        """
        if sparsity < 0.3:
            return 1.0
        return 1.0 + (sparsity - 0.3) * 1.5


# ---------------------------------------------------------------------------
# Model Size Estimator for Compression
# ---------------------------------------------------------------------------

@dataclass
class CompressionEstimate:
    """Estimate compression results for Claude model family."""

    @staticmethod
    def distillation_specs() -> Dict[str, Dict[str, Any]]:
        """Speculated sizes for the Claude 4.6 family."""
        return {
            "Opus 4.6": {
                "total_params": "2-5T",
                "active_params": "120-300B",
                "method": "Pre-trained from scratch",
                "price_input": "$5/M",
                "price_output": "$25/M",
            },
            "Sonnet 4.6": {
                "total_params": "~200-500B",
                "active_params": "~200-500B (dense)",
                "method": "Likely distilled from Opus",
                "price_input": "$3/M",
                "price_output": "$15/M",
            },
            "Haiku 4.6": {
                "total_params": "~30-70B",
                "active_params": "~30-70B (dense)",
                "method": "Likely distilled from Sonnet/Opus",
                "price_input": "$0.25/M",
                "price_output": "$1.25/M",
            },
        }
