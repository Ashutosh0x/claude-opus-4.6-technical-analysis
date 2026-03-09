"""
Claude Opus 4.6 — Alignment & RLHF

Reward modeling, Direct Preference Optimization (DPO),
and Constitutional AI (CAI / RLAIF) data generation.
"""

from .reward_model import RewardModelConfig, RewardHead, RewardModel, best_of_n
from .dpo import DPOConfig, DPOTrainer, OnlineDPOTrainer, compute_log_probs
from .constitutional_ai import (
    CAIConfig,
    ConstitutionalAIGenerator,
    DEFAULT_CONSTITUTION,
)

__all__ = [
    # Reward modeling
    "RewardModelConfig",
    "RewardHead",
    "RewardModel",
    "best_of_n",
    # DPO
    "DPOConfig",
    "DPOTrainer",
    "OnlineDPOTrainer",
    "compute_log_probs",
    # Constitutional AI
    "CAIConfig",
    "ConstitutionalAIGenerator",
    "DEFAULT_CONSTITUTION",
]
