"""
Direct Preference Optimization (DPO).

DPO eliminates the reward model entirely — aligning the policy
directly from preference data:

    L_DPO = -E[ log σ( β·log(π_θ(y_w|x)/π_ref(y_w|x))
                      - β·log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]

where:
    π_θ   = policy being trained
    π_ref = frozen reference policy (SFT model)
    y_w   = preferred response
    y_l   = rejected response
    β     = KL penalty coefficient (typically 0.1–0.5)

DPO comes from the insight that the optimal RLHF policy satisfies:

    r*(x, y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)

So we can implicitly define a reward through the policy ratio,
and optimize directly without fitting a separate reward model.

Variants:
    - IPO (Azar et al. 2023): replaces log-sigmoid with squared loss
    - KTO (Ethayarajh et al. 2024): works with unpaired preferences
    - ORPO (Hong et al. 2024): combines SFT + DPO in one stage

Why DPO over PPO:
    ✓ No reward model to train
    ✓ No value function / critic
    ✓ No RL infrastructure (no rollouts, no advantage estimation)
    ✓ More stable training (no reward hacking)
    ✗ Potentially weaker for complex multi-turn alignment

References:
    - DPO: Rafailov et al. 2023 (arXiv:2305.18290)
    - IPO: Azar et al. 2023 (arXiv:2310.12036)
    - KTO: Ethayarajh et al. 2024 (arXiv:2402.01306)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from ..model.transformer import ClaudeModel, ClaudeConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DPOConfig:
    """DPO training hyperparameters."""
    beta: float = 0.1                # KL penalty coefficient
    label_smoothing: float = 0.0     # label smoothing for robustness
    loss_type: str = "sigmoid"       # "sigmoid" (DPO), "hinge" (IPO), "kto"
    reference_free: bool = False     # if True, skip π_ref (simpler but weaker)
    # Learning rate (typically lower than SFT)
    lr: float = 5e-7
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# Log-probability computation
# ---------------------------------------------------------------------------

def compute_log_probs(
    model: ClaudeModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-token log probabilities of the labels under the model.

    log π(y|x) = Σ_t log P(y_t | x, y_{<t})

    We only sum over the response tokens (not the prompt tokens).
    Labels should have -100 for prompt positions.

    Args:
        model         : language model
        input_ids     : [B, T] — full sequence (prompt + response)
        labels        : [B, T] — token IDs for loss, -100 for prompt positions
        attention_mask: [B, T] — 1=real, 0=pad

    Returns:
        log_probs: [B] — total log probability of each response
    """
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs["logits"]   # [B, T, V]

    # Shift: logits predict the NEXT token
    shift_logits = logits[:, :-1, :]     # [B, T-1, V]
    shift_labels = labels[:, 1:]          # [B, T-1]

    # Per-token log probs
    log_probs_all = F.log_softmax(shift_logits.float(), dim=-1)
    per_token_logps = log_probs_all.gather(
        dim=-1, index=shift_labels.unsqueeze(-1).clamp(min=0)
    ).squeeze(-1)   # [B, T-1]

    # Mask out prompt tokens (labels == -100)
    loss_mask = (shift_labels != -100).float()  # [B, T-1]
    per_token_logps = per_token_logps * loss_mask

    # Sum over response tokens → total log prob
    return per_token_logps.sum(dim=-1)   # [B]


# ---------------------------------------------------------------------------
# DPO Trainer
# ---------------------------------------------------------------------------

class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    Usage:
        trainer = DPOTrainer(policy_model, ref_model, config)
        for batch in dataloader:
            loss_dict = trainer.compute_loss(batch)
            loss_dict["loss"].backward()
            optimizer.step()

    The reference model (π_ref) is the SFT model, frozen. The policy
    model (π_θ) starts as a copy of π_ref and is updated by DPO.
    """

    def __init__(
        self,
        policy_model: ClaudeModel,
        ref_model: ClaudeModel,
        config: DPOConfig,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.config = config

        # Freeze reference model
        self.ref.eval()
        for param in self.ref.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the DPO loss.

        L_DPO = -E[ log σ( β·(log π(y_w|x)/π_ref(y_w|x)
                           - log π(y_l|x)/π_ref(y_l|x)) ) ]

        Simplified:
            L = -log σ( β·(Δ_chosen - Δ_rejected) )
            where Δ_i = log π_θ(y_i|x) - log π_ref(y_i|x)

        Args:
            chosen_ids    : [B, T] — prompt + preferred response
            rejected_ids  : [B, T] — prompt + rejected response
            chosen_labels : [B, T] — labels (-100 for prompt tokens)
            rejected_labels: [B, T]
            chosen_mask   : [B, T] — attention mask
            rejected_mask : [B, T]

        Returns:
            dict with "loss", "chosen_reward", "rejected_reward",
            "accuracy", "margin"
        """
        # --- Policy log probs ---
        policy_chosen_logps = compute_log_probs(
            self.policy, chosen_ids, chosen_labels, chosen_mask
        )
        policy_rejected_logps = compute_log_probs(
            self.policy, rejected_ids, rejected_labels, rejected_mask
        )

        # --- Reference log probs (no grad) ---
        with torch.no_grad():
            if self.config.reference_free:
                ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
            else:
                ref_chosen_logps = compute_log_probs(
                    self.ref, chosen_ids, chosen_labels, chosen_mask
                )
                ref_rejected_logps = compute_log_probs(
                    self.ref, rejected_ids, rejected_labels, rejected_mask
                )

        # --- Log ratios (implicit rewards) ---
        # r̂(x, y) = β·log(π_θ(y|x) / π_ref(y|x))
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # Implicit rewards
        chosen_rewards = self.config.beta * chosen_logratios
        rejected_rewards = self.config.beta * rejected_logratios

        # --- DPO Loss ---
        logits = chosen_rewards - rejected_rewards   # [B]

        if self.config.loss_type == "sigmoid":
            # Standard DPO: L = -log σ(logits)
            if self.config.label_smoothing > 0:
                # Smooth labels: y ∈ [ε, 1-ε] instead of {0, 1}
                eps = self.config.label_smoothing
                loss = (
                    -F.logsigmoid(logits) * (1 - eps)
                    - F.logsigmoid(-logits) * eps
                ).mean()
            else:
                loss = -F.logsigmoid(logits).mean()

        elif self.config.loss_type == "hinge":
            # IPO: L = (logits - 1/(2β))²
            # This is more robust to outliers
            loss = (logits - 1.0 / (2 * self.config.beta)).pow(2).mean()

        elif self.config.loss_type == "kto":
            # KTO: works with unpaired data (only chosen OR rejected)
            # Simplified version for paired data
            kl_chosen = (policy_chosen_logps - ref_chosen_logps).mean().detach()
            kl_rejected = (policy_rejected_logps - ref_rejected_logps).mean().detach()
            chosen_loss = 1.0 - F.sigmoid(
                self.config.beta * (chosen_logratios - kl_rejected)
            )
            rejected_loss = 1.0 - F.sigmoid(
                self.config.beta * (kl_chosen - rejected_logratios)
            )
            loss = (chosen_loss.mean() + rejected_loss.mean()) / 2

        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        # --- Metrics ---
        accuracy = (logits > 0).float().mean()

        return {
            "loss": loss,
            "chosen_reward": chosen_rewards.mean().detach(),
            "rejected_reward": rejected_rewards.mean().detach(),
            "accuracy": accuracy.detach(),
            "margin": logits.mean().detach(),
            # For monitoring KL divergence from reference
            "kl_chosen": (policy_chosen_logps - ref_chosen_logps).mean().detach(),
            "kl_rejected": (policy_rejected_logps - ref_rejected_logps).mean().detach(),
        }


# ---------------------------------------------------------------------------
# Online DPO (with on-policy rejection sampling)
# ---------------------------------------------------------------------------

class OnlineDPOTrainer(DPOTrainer):
    """
    Online DPO: generates new rejected samples on-the-fly.

    Instead of using a fixed dataset of (chosen, rejected) pairs,
    online DPO generates rejected samples from the current policy
    and uses the reference model or a reward model to rank them.

    This reduces distribution shift between the training data and
    the current policy, improving training stability.

    Algorithm:
        1. For each prompt, generate K responses from π_θ
        2. Score with reward model (or π_ref ranking)
        3. Best = chosen, worst = rejected
        4. Train with standard DPO loss on these pairs
    """

    def __init__(self, *args, reward_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_model = reward_model

    @torch.inference_mode()
    def generate_pairs(
        self,
        prompts: list,
        tokenizer,
        k: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
    ):
        """
        Generate (chosen, rejected) pairs from the current policy.

        Args:
            prompts       : list of prompt strings
            tokenizer     : tokenizer
            k             : number of candidates per prompt
            max_new_tokens: max tokens per candidate
            temperature   : sampling temp

        Returns:
            list of dicts with "prompt", "chosen", "rejected"
        """
        device = next(self.policy.parameters()).device
        pairs = []

        for prompt in prompts:
            candidates = []
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            for _ in range(k):
                out_ids = input_ids.clone()
                for _ in range(max_new_tokens):
                    logits = self.policy(out_ids)["logits"][:, -1, :]
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, 1)
                    out_ids = torch.cat([out_ids, next_tok], dim=1)
                    if next_tok.item() == tokenizer.eos_token_id:
                        break

                response = tokenizer.decode(
                    out_ids[0, input_ids.shape[1]:], skip_special_tokens=True
                )
                # Score with reward model or reference log-prob
                if self.reward_model:
                    score = self.reward_model.get_reward(out_ids).item()
                else:
                    score = compute_log_probs(
                        self.ref, out_ids,
                        out_ids.clone(),  # simplified — would need proper labels
                    ).item()

                candidates.append((response, score))

            # Best = chosen, worst = rejected
            candidates.sort(key=lambda x: x[1], reverse=True)
            pairs.append({
                "prompt": prompt,
                "chosen": candidates[0][0],
                "rejected": candidates[-1][0],
            })

        return pairs
