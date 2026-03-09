"""
Optimizer and Learning Rate Scheduler.

AdamW optimizer with warmup + cosine decay schedule:

    η(t) = { η_max · (t / T_warmup)                          if t ≤ T_warmup
           { η_min + 0.5(η_max - η_min)(1 + cos(π(t-T_w)/(T_total-T_w)))  otherwise

Typical hyperparameters (frontier models):
    Peak LR (η_max):    1e-4 to 3e-4
    Final LR (η_min):   η_max/10 to η_max/100
    Warmup steps:       2,000–5,000
    β₁, β₂ (Adam):     0.9, 0.95
    Weight decay:       0.1
    Gradient clipping:  1.0 (max grad norm)
    Batch size:         2–16M tokens

Batch size scaling:
    B_total = B_micro × N_accum × N_data_parallel

References:
    - Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
      (AdamW), 2019
    - Hoffmann et al., "Training Compute-Optimal LLMs" (Chinchilla
      scaling), 2022
"""

import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """
    Build AdamW optimizer with weight decay applied only to
    weight matrices (not biases, norms, or embeddings).

    This is critical for training stability — weight decay on
    norms/biases hurts performance.

    Args:
        model: The model to optimize.
        lr: Peak learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam β₁ and β₂.
        eps: Adam epsilon.

    Returns:
        Configured AdamW optimizer.
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't decay biases, norms, or embeddings
        if (
            param.ndim == 1
            or "bias" in name
            or "layernorm" in name
            or "norm" in name
            or "embed" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
    )

    return optimizer


def build_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 2000,
    total_steps: int = 500000,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Build warmup + cosine decay learning rate schedule.

    Phase 1 — Linear warmup (0 → T_warmup):
        η(t) = η_max · t / T_warmup

    Phase 2 — Cosine decay (T_warmup → T_total):
        η(t) = η_min + 0.5(η_max - η_min)(1 + cos(π·(t-T_w)/(T-T_w)))

    For Opus 4.6 training (~90 days on 32K GPUs):
        total_steps ≈ 500K–1M gradient steps
        warmup_steps ≈ 2000–5000
        min_lr = peak_lr / 10

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Ratio η_min / η_max.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        # Phase 1: Linear warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        # Phase 2: Cosine annealing
        progress = (current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        progress = min(progress, 1.0)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def build_wsd_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 2000,
    stable_steps: int = 400000,
    decay_steps: int = 100000,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    WSD (Warmup-Stable-Decay) schedule.

    An alternative to cosine decay used by some frontier models
    (e.g., MiniCPM, some DeepSeek variants):

        Phase 1: Linear warmup (0 → T_warmup)
        Phase 2: Constant LR (T_warmup → T_stable)
        Phase 3: Cosine decay (T_stable → T_total)

    Advantage: Allows checkpoint saving at end of stable phase
    and then decaying with different data mixes (annealing).

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Warmup duration.
        stable_steps: Duration at peak LR.
        decay_steps: Decay duration.
        min_lr_ratio: Final LR ratio.

    Returns:
        LambdaLR scheduler.
    """
    total_steps = warmup_steps + stable_steps + decay_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        elif current_step < warmup_steps + stable_steps:
            return 1.0
        else:
            progress = (current_step - warmup_steps - stable_steps) / max(
                1, decay_steps
            )
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
