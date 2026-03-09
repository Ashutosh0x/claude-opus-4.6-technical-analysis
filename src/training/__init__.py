"""
Claude Opus 4.6 — Training Pipeline

Components:
    Trainer            — Main FSDP training loop with BF16 mixed precision
    LanguageModelingLoss — Cross-entropy loss with label smoothing
    MoELoss            — Auxiliary load-balancing + router z-loss
    CheckpointManager  — Distributed checkpoint save/load/resume
    build_optimizer     — AdamW with cosine or WSD learning rate schedule
"""

from .loss import LanguageModelingLoss, MoELoss
from .optimizer import build_optimizer
from .checkpoint import CheckpointManager
from .trainer import Trainer
from .distillation import (
    DistillationLoss,
    DistillationTrainer,
    StructuredPruning,
    CompressionEstimate,
)

__all__ = [
    "LanguageModelingLoss",
    "MoELoss",
    "build_optimizer",
    "CheckpointManager",
    "Trainer",
    # Distillation & compression
    "DistillationLoss",
    "DistillationTrainer",
    "StructuredPruning",
    "CompressionEstimate",
]
