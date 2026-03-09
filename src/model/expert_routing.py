"""
Expert Routing Strategies for MoE Transformers.

Implements both Token-Choice (standard) and Expert-Choice (Zhou et al., 2022)
routing strategies for the Mixture-of-Experts layer.

Token-Choice (standard, used in Mixtral, likely Opus 4.6):
    Each token selects its top-k experts via the gating network.
    Problem: load imbalance — popular experts get overwhelmed.
    Solution: auxiliary load-balancing loss.

Expert-Choice (Zhou et al., 2022):
    Each expert selects its top-C tokens (capacity C).
    Guaranteed balanced load, but some tokens may be unprocessed.

Capacity formula (Expert-Choice):
    C = k × T / E
    where T = total tokens, E = number of experts, k = top-k.

Expert utilization entropy:
    H = -Σ p_i log p_i
    H = log(E) → perfectly balanced
    H << log(E) → expert collapse risk

References:
    - Fedus et al., "Switch Transformers", 2022
    - Zhou et al., "Mixture-of-Experts with Expert Choice Routing", 2022
    - Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse
      Expert Models", 2022
"""

import math
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expert-Choice Router
# ---------------------------------------------------------------------------

class ExpertChoiceRouter(nn.Module):
    """
    Expert-Choice routing (Zhou et al., 2022).

    Instead of each token picking its top-k experts, each expert
    picks its top-C tokens from the batch.

    Formula:
        tokens(E_i) = TopC(G(X)_i, C)
        C = k × T / E

    Advantages:
        - Guaranteed perfect load balance (no aux loss needed)
        - No token dropping due to overflow (unlike token-choice)
    Disadvantages:
        - Some tokens may be unprocessed
        - Less flexible routing patterns
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Router gate: hidden → expert logits
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute expert-choice routing.

        Args:
            x: [B, T, D] input hidden states.

        Returns:
            expert_indices:  [E, C] — which tokens each expert processes
            expert_weights:  [E, C] — gating weights for selected tokens
            routing_probs:   [N, E] — full routing probabilities (for analysis)

            where N = B*T, C = capacity per expert.
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        # Compute routing probabilities
        router_logits = self.gate(x_flat)                 # [N, E]
        routing_probs = F.softmax(router_logits, dim=-1)  # [N, E]

        # Capacity per expert
        capacity = self.top_k * N // self.num_experts

        # Each expert selects its top-C tokens
        # Transpose so we select along the token dimension per expert
        probs_t = routing_probs.t()   # [E, N]

        expert_weights, expert_indices = torch.topk(
            probs_t, k=capacity, dim=-1
        )  # both [E, C]

        # Renormalize weights per expert
        expert_weights = expert_weights / expert_weights.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)

        return expert_indices, expert_weights, routing_probs

    def capacity_per_expert(self, num_tokens: int) -> int:
        """
        Compute capacity C for expert-choice routing.

        C = k × T / E
        """
        return self.top_k * num_tokens // self.num_experts


# ---------------------------------------------------------------------------
# Expert Utilization Analysis
# ---------------------------------------------------------------------------

class ExpertUtilizationAnalyzer:
    """
    Analyze expert utilization patterns in MoE models.

    Key metrics:
        - Utilization entropy: H = -Σ p_i log p_i
        - Expert overlap / correlation matrix
        - Load imbalance ratio
    """

    @staticmethod
    def utilization_entropy(
        routing_probs: torch.Tensor,
    ) -> float:
        """
        Compute expert utilization entropy.

        H = -Σ p_i log p_i

        Interpretation:
            H = log(E)  → perfectly balanced (all experts equally used)
            H << log(E) → some experts dominate (collapse risk)

        Args:
            routing_probs: [N, E] routing probabilities across N tokens.

        Returns:
            Entropy value (higher = more balanced).
        """
        # Average probability per expert
        p = routing_probs.mean(dim=0)   # [E]
        p = p.clamp(min=1e-10)

        entropy = -(p * p.log()).sum().item()
        return entropy

    @staticmethod
    def max_entropy(num_experts: int) -> float:
        """Maximum possible entropy (perfectly balanced)."""
        return math.log(num_experts)

    @staticmethod
    def balance_ratio(
        routing_probs: torch.Tensor,
    ) -> float:
        """
        Ratio of actual entropy to maximum entropy.

        1.0 = perfectly balanced, 0.0 = fully collapsed.
        """
        entropy = ExpertUtilizationAnalyzer.utilization_entropy(routing_probs)
        max_ent = ExpertUtilizationAnalyzer.max_entropy(routing_probs.shape[-1])
        return entropy / max_ent if max_ent > 0 else 0.0

    @staticmethod
    def correlation_matrix(
        routing_decisions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute expert co-activation correlation matrix.

        C_ij = Corr(1[expert i active], 1[expert j active])

        Args:
            routing_decisions: [N, E] binary or soft assignment matrix.

        Returns:
            [E, E] correlation matrix.
        """
        # Binarize if not already
        active = (routing_decisions > 0).float()   # [N, E]

        # Compute correlation
        N = active.shape[0]
        mean = active.mean(dim=0, keepdim=True)       # [1, E]
        centered = active - mean                       # [N, E]

        cov = (centered.t() @ centered) / max(N - 1, 1)   # [E, E]
        std = centered.std(dim=0, keepdim=True).clamp(min=1e-8)
        corr = cov / (std.t() @ std)

        return corr

    @staticmethod
    def find_dead_experts(
        routing_probs: torch.Tensor,
        threshold: float = 0.001,
    ) -> list:
        """
        Identify experts that are rarely or never activated.

        Args:
            routing_probs: [N, E] routing probabilities.
            threshold: Minimum average probability to be considered alive.

        Returns:
            List of dead expert indices.
        """
        avg_prob = routing_probs.mean(dim=0)
        dead = (avg_prob < threshold).nonzero(as_tuple=True)[0]
        return dead.tolist()


# ---------------------------------------------------------------------------
# Comparison: Token-Choice vs Expert-Choice
# ---------------------------------------------------------------------------

@dataclass
class RoutingComparison:
    """
    Side-by-side comparison of routing strategies.

    Token-Choice:
        Routing:        Token picks top-k experts
        Load balance:   Requires aux loss
        Token dropping: No (but overflow possible)
        Used by:        Mixtral, likely Opus 4.6

    Expert-Choice:
        Routing:        Expert picks top-C tokens
        Load balance:   Guaranteed balanced
        Token dropping: Yes (some tokens unprocessed)
        Used by:        Switch Transformer, V-MoE
    """

    @staticmethod
    def summary() -> Dict[str, Dict[str, str]]:
        return {
            "token_choice": {
                "routing": "Token picks top-k experts",
                "load_balance": "Requires auxiliary loss",
                "token_dropping": "No (overflow possible)",
                "used_by": "Mixtral, likely Opus 4.6",
            },
            "expert_choice": {
                "routing": "Expert picks top-C tokens",
                "load_balance": "Guaranteed balanced",
                "token_dropping": "Yes (some tokens unprocessed)",
                "used_by": "Switch Transformer, V-MoE",
            },
        }
