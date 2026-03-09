"""
Reward Model Training for RLHF.

A separate model R_φ trained on human preference data:

    L_RM = -E_{(x, y_w, y_l) ~ D} [ log σ( R_φ(x, y_w) - R_φ(x, y_l) ) ]

where:
    y_w = preferred (winning) response
    y_l = rejected (losing) response
    σ   = sigmoid function

The reward model learns to assign higher scalar scores to
human-preferred outputs. It serves two purposes:

    1. In PPO:  Provides the reward signal for RL optimization
    2. In eval: Scores candidate responses for best-of-N sampling

Architecture:
    - Same backbone as the language model (or a smaller version)
    - Replace LM head with a scalar output head (hidden → 1)
    - Trained on ~100K–500K preference pairs
    - Typically 10–50B params (much smaller than the policy model)

Training data format:
    [
        {"prompt": "...", "chosen": "...", "rejected": "..."},
        ...
    ]

References:
    - InstructGPT: Ouyang et al. 2022 (arXiv:2203.02155)
    - RLHF: Christiano et al. 2017 (arXiv:1706.03741)
    - Bradley-Terry model: Bradley & Terry 1952
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from ..model.transformer import ClaudeModel, ClaudeConfig


# ---------------------------------------------------------------------------
# Reward Model Config
# ---------------------------------------------------------------------------

@dataclass
class RewardModelConfig:
    """Configuration for the reward model."""
    base_model_config: ClaudeConfig = None   # backbone architecture
    hidden_size: int = 16384                 # must match backbone
    reward_head_hidden: int = 4096           # intermediate size
    dropout: float = 0.1
    use_length_penalty: bool = True          # penalize verbose responses
    length_penalty_weight: float = 0.01


# ---------------------------------------------------------------------------
# Reward Head — maps final hidden state to scalar reward
# ---------------------------------------------------------------------------

class RewardHead(nn.Module):
    """
    Scalar reward head on top of the language model backbone.

    Architecture:
        hidden_state [B, T, D] → pooled [B, D] → reward [B, 1]

    Pooling strategy: take the hidden state at the EOS/last token
    position (standard for reward models).

    The reward is unbounded — it's only meaningful in comparison:
        R(x, y_w) > R(x, y_l)
    """

    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.reward_head_hidden)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.reward_head_hidden, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D] — last hidden states from backbone

        Returns:
            rewards: [B] — scalar reward per sequence
        """
        # Pool: take last token's hidden state
        pooled = hidden_states[:, -1, :]    # [B, D]
        x = self.dropout(torch.tanh(self.dense(pooled)))
        reward = self.out_proj(x).squeeze(-1)   # [B]
        return reward


# ---------------------------------------------------------------------------
# Reward Model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    Full reward model: backbone + scalar head.

    During training:
        1. Encode both chosen and rejected responses
        2. Get scalar rewards for each
        3. Compute Bradley-Terry loss: L = -log σ(r_w - r_l)

    During inference (best-of-N):
        1. Generate N candidate responses
        2. Score each with reward model
        3. Return highest-scoring candidate
    """

    def __init__(
        self,
        backbone: ClaudeModel,
        config: RewardModelConfig,
    ):
        super().__init__()
        self.backbone = backbone
        self.reward_head = RewardHead(config)
        self.config = config

        # Freeze backbone initially (optional — can unfreeze later)
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get scalar reward for a sequence.

        Args:
            input_ids     : [B, T] — tokenized prompt + response
            attention_mask: [B, T] — 1=real, 0=pad

        Returns:
            rewards: [B] — scalar reward per sequence
        """
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        # outputs["logits"] is [B, T, V], but we need hidden states
        # In practice, we'd modify the backbone to return hidden states
        # Here we approximate by using the pre-LM-head state
        # (The reward head replaces the LM head)
        hidden = self.backbone.norm(
            # Get the hidden states before the LM head
            # This requires access to intermediate outputs
            outputs.get("hidden_states", outputs["logits"])
        )
        return self.reward_head(hidden)

    def forward(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute preference loss.

        L_RM = -log σ(R(x, y_w) - R(x, y_l))

        Args:
            chosen_ids   : [B, T] — tokenized prompt + chosen response
            rejected_ids : [B, T] — tokenized prompt + rejected response
            chosen_mask  : [B, T] — attention mask for chosen
            rejected_mask: [B, T] — attention mask for rejected

        Returns:
            dict with "loss", "chosen_reward", "rejected_reward", "accuracy"
        """
        r_chosen  = self.get_reward(chosen_ids, chosen_mask)    # [B]
        r_rejected = self.get_reward(rejected_ids, rejected_mask)  # [B]

        # Optional length penalty: discourage verbose responses
        if self.config.use_length_penalty:
            chosen_len  = chosen_mask.sum(dim=-1).float() if chosen_mask is not None else chosen_ids.shape[1]
            rejected_len = rejected_mask.sum(dim=-1).float() if rejected_mask is not None else rejected_ids.shape[1]
            r_chosen  = r_chosen - self.config.length_penalty_weight * chosen_len
            r_rejected = r_rejected - self.config.length_penalty_weight * rejected_len

        # Bradley-Terry loss: L = -log σ(r_w - r_l)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        # Accuracy: fraction where r_chosen > r_rejected
        accuracy = (r_chosen > r_rejected).float().mean()

        return {
            "loss": loss,
            "chosen_reward": r_chosen.mean(),
            "rejected_reward": r_rejected.mean(),
            "reward_margin": (r_chosen - r_rejected).mean(),
            "accuracy": accuracy,
        }


# ---------------------------------------------------------------------------
# Best-of-N Sampling with Reward Model
# ---------------------------------------------------------------------------

@torch.inference_mode()
def best_of_n(
    policy_model: ClaudeModel,
    reward_model: RewardModel,
    tokenizer,
    prompt: str,
    n: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
) -> Tuple[str, float]:
    """
    Generate N responses and return the one scored highest by the reward model.

    Best-of-N is a simple inference-time alignment technique:
        - No RL training required
        - Quality scales with N: reward ∝ log(N) × σ_reward
        - Cost scales linearly with N

    At N=64, best-of-N approaches RLHF quality without any RL training
    (Gao et al., 2022 — "Scaling Laws for Reward Model Overoptimization").

    Args:
        policy_model : the language model
        reward_model : the reward scorer
        tokenizer    : tokenizer
        prompt       : input prompt
        n            : number of candidates
        max_new_tokens : max tokens per candidate
        temperature  : sampling temperature (higher = more diverse)

    Returns:
        best_response: str  — highest-scored response
        best_reward:   float — reward of the best response
    """
    device = next(policy_model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    candidates: List[Tuple[str, float]] = []

    for _ in range(n):
        # Generate one candidate
        output_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = policy_model(output_ids)["logits"][:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            output_ids = torch.cat([output_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

        # Score with reward model
        reward = reward_model.get_reward(output_ids).item()
        response = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        candidates.append((response, reward))

    # Return best
    best = max(candidates, key=lambda x: x[1])
    return best
