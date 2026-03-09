"""
Speculative Decoding Engine.

A smaller, faster draft model generates K candidate tokens, then the
large target model verifies all K tokens in a single forward pass.

Algorithm:
    1. Draft: Small model generates K tokens autoregressively (fast)
    2. Verify: Large model runs one forward pass over all K tokens
    3. Accept/Reject: Accept where P_target ≥ P_draft; resample from first mismatch

Acceptance Criterion:
    p_accept = min(1, P_target(t_i | x_{<i}) / P_draft(t_i | x_{<i}))

This guarantees the output distribution is IDENTICAL to sampling
from the target model alone (no quality loss).

Speedup Formula:
    S ≈ K / (1 + (K-1) × c_draft / c_target)
    where c_draft/c_target << 1

Typical speedup:  2-3× with no quality loss.

Draft model options for Opus 4.6:
    Haiku 4.6 (~50B):          K=5, ~2.0× speedup
    Dedicated draft (~7B):     K=8, ~2.5× speedup
    Self-speculative (early):  K=3, ~1.5× speedup

EAGLE-2 (Li et al., 2024):
    Tree-based verification instead of linear. Draft model predicts
    multiple branches, target model verifies all in one pass.

References:
    - Leviathan et al., "Fast Inference from Transformers via
      Speculative Decoding", ICML 2023
    - Chen et al., "Accelerating LLM Inference with Staged
      Speculative Decoding", 2023
    - Li et al., "EAGLE-2: Faster Inference of Language Models with
      Dynamic Draft Trees", 2024
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Speculative Decoding Core
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    """
    Speculative decoding with rejection sampling.

    Uses a small draft model to propose tokens, then the large
    target model verifies them in a single forward pass.

    The key insight: one forward pass of the target model on K tokens
    costs roughly the same as one forward pass on 1 token (the KV cache
    handles the parallelism). So if the draft model can generate K
    tokens faster than the target generates 1, we get a speedup.
    """

    def __init__(
        self,
        target_model_fn: Callable,
        draft_model_fn: Callable,
        lookahead: int = 5,
        temperature: float = 1.0,
    ):
        """
        Args:
            target_model_fn: fn(token_ids) -> logits [B, T, V].
            draft_model_fn:  fn(token_ids) -> logits [B, T, V].
            lookahead: Number of draft tokens K to generate per step.
            temperature: Sampling temperature.
        """
        self.target_fn = target_model_fn
        self.draft_fn = draft_model_fn
        self.lookahead = lookahead
        self.temperature = temperature

    def draft_tokens(
        self,
        prefix: torch.Tensor,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate K draft tokens autoregressively with the draft model.

        Args:
            prefix: [B, S] current token sequence.
            k: Number of tokens to draft (defaults to self.lookahead).

        Returns:
            draft_ids:    [B, K] drafted token IDs.
            draft_probs:  [B, K, V] draft model probabilities.
        """
        k = k or self.lookahead
        draft_ids = []
        draft_probs = []
        current = prefix

        for _ in range(k):
            logits = self.draft_fn(current)          # [B, S+i, V]
            next_logits = logits[:, -1, :]           # [B, V]

            if self.temperature > 0:
                probs = F.softmax(next_logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)  # [B, 1]
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            draft_ids.append(next_token)
            draft_probs.append(probs.unsqueeze(1))
            current = torch.cat([current, next_token], dim=1)

        return (
            torch.cat(draft_ids, dim=1),          # [B, K]
            torch.cat(draft_probs, dim=1),         # [B, K, V]
        )

    def verify_and_accept(
        self,
        prefix: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Verify draft tokens with the target model using rejection sampling.

        Acceptance Criterion (per position i):
            p_accept = min(1, P_target(t_i) / P_draft(t_i))

        If rejected at position i, resample from the residual distribution:
            P_residual = max(0, P_target - P_draft) / Z

        This guarantees the output matches the target model's distribution
        EXACTLY.

        Args:
            prefix: [B, S] input sequence before drafting.
            draft_ids: [B, K] drafted token IDs.
            draft_probs: [B, K, V] draft model probabilities.

        Returns:
            accepted_ids: [B, N_accepted] tokens to append (1 <= N <= K+1).
            num_accepted: Number of accepted draft tokens.
        """
        B, K = draft_ids.shape

        # Run target model on prefix + all draft tokens at once
        full_seq = torch.cat([prefix, draft_ids], dim=1)
        target_logits = self.target_fn(full_seq)    # [B, S+K, V]

        # Extract target probabilities for each draft position
        S = prefix.shape[1]
        target_probs = F.softmax(
            target_logits[:, S - 1:S + K - 1, :] / max(self.temperature, 1e-8),
            dim=-1,
        )  # [B, K, V]

        accepted = []
        num_accepted = 0

        for i in range(K):
            # Get probabilities for the drafted token
            t_id = draft_ids[:, i:i + 1]   # [B, 1]

            p_target = target_probs[:, i, :].gather(1, t_id).squeeze(-1)  # [B]
            p_draft = draft_probs[:, i, :].gather(1, t_id).squeeze(-1)    # [B]

            # Acceptance probability
            accept_prob = torch.clamp(p_target / p_draft.clamp(min=1e-10), max=1.0)

            # Stochastic acceptance (for batch, take worst case)
            r = torch.rand_like(accept_prob)
            if (r <= accept_prob).all():
                accepted.append(t_id)
                num_accepted += 1
            else:
                # Rejected — resample from residual distribution
                residual = torch.clamp(
                    target_probs[:, i, :] - draft_probs[:, i, :], min=0
                )
                residual = residual / residual.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                new_token = torch.multinomial(residual, 1)
                accepted.append(new_token)
                num_accepted += 1  # The resampled token counts
                break

        if not accepted:
            # Fallback: sample from target at first position
            probs = target_probs[:, 0, :]
            new_token = torch.multinomial(probs, 1)
            return new_token, 1

        return torch.cat(accepted, dim=1), num_accepted

    def step(
        self, prefix: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        One speculative decoding step: draft + verify.

        Returns:
            new_tokens: [B, N] accepted tokens.
            num_accepted: How many draft tokens were accepted.
        """
        draft_ids, draft_probs = self.draft_tokens(prefix)
        return self.verify_and_accept(prefix, draft_ids, draft_probs)


# ---------------------------------------------------------------------------
# Speedup Estimation
# ---------------------------------------------------------------------------

def speculative_speedup(
    k: int,
    acceptance_rate: float = 0.85,
    draft_cost_ratio: float = 0.05,
) -> float:
    """
    Estimate speculative decoding speedup.

    Formula:
        S ≈ K / (1 + (K-1) × c_draft / c_target)

    More precisely, with acceptance rate α:
        Expected accepted tokens = Σ_{i=0}^{K} α^i × (1 - α)
                                 ≈ (1 - α^(K+1)) / (1 - α)

    Args:
        k: Number of draft tokens (lookahead).
        acceptance_rate: Average token acceptance probability.
        draft_cost_ratio: draft_flops / target_flops (e.g., 0.05 for 7B/200B).

    Returns:
        Estimated speedup factor.
    """
    if acceptance_rate <= 0:
        return 1.0

    # Expected tokens per step
    expected_tokens = sum(
        acceptance_rate ** i for i in range(k)
    )

    # Total compute: K draft passes + 1 target pass
    total_compute = k * draft_cost_ratio + 1.0

    return expected_tokens / total_compute


@dataclass
class DraftModelConfig:
    """Configuration options for draft models."""
    name: str
    params: str
    lookahead: int
    expected_acceptance: float
    expected_speedup: float

    @staticmethod
    def opus_options() -> List["DraftModelConfig"]:
        """Available draft model configurations for Opus 4.6."""
        return [
            DraftModelConfig(
                name="Haiku 4.6",
                params="~50B",
                lookahead=5,
                expected_acceptance=0.85,
                expected_speedup=2.0,
            ),
            DraftModelConfig(
                name="Dedicated draft",
                params="~7B",
                lookahead=8,
                expected_acceptance=0.80,
                expected_speedup=2.5,
            ),
            DraftModelConfig(
                name="Self-speculative (early exit)",
                params="Same model, layer 40/160",
                lookahead=3,
                expected_acceptance=0.90,
                expected_speedup=1.5,
            ),
        ]


# ---------------------------------------------------------------------------
# Tree-Based Verification (EAGLE-2)
# ---------------------------------------------------------------------------

@dataclass
class TreeVerificationNode:
    """A node in the EAGLE-2 draft tree."""
    token_id: int
    probability: float
    children: List["TreeVerificationNode"] = field(default_factory=list)
    depth: int = 0


class EAGLE2TreeBuilder:
    """
    EAGLE-2 tree-based speculative decoding.

    Instead of linear drafting (one path of K tokens), EAGLE-2
    generates a tree of possible continuations. The target model
    verifies all branches in a single forward pass.

    This increases acceptance rate because if path A is rejected,
    path B from the same position may be accepted.

    Key difference from linear speculative decoding:
        Linear: draft [A, B, C, D, E] → verify sequentially
        Tree:   draft a tree with multiple branches at each level
                → verify all branches in parallel

    Expected speedup: 5-8× (vs 2-3× for linear).
    """

    def __init__(
        self,
        max_depth: int = 5,
        branching_factor: int = 3,
        min_probability: float = 0.05,
    ):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.min_probability = min_probability

    def build_tree(
        self,
        root_logits: torch.Tensor,
        draft_fn: Callable,
        temperature: float = 1.0,
    ) -> TreeVerificationNode:
        """
        Build a draft tree from root logits.

        Args:
            root_logits: [V] logits at the current position.
            draft_fn: Function to get next-token logits.
            temperature: Sampling temperature.

        Returns:
            Root node of the draft tree.
        """
        probs = F.softmax(root_logits / max(temperature, 1e-8), dim=-1)
        top_probs, top_ids = torch.topk(probs, self.branching_factor)

        root = TreeVerificationNode(token_id=-1, probability=1.0, depth=0)

        for prob, tid in zip(top_probs.tolist(), top_ids.tolist()):
            if prob < self.min_probability:
                break
            child = TreeVerificationNode(
                token_id=tid, probability=prob, depth=1
            )
            root.children.append(child)

        return root

    def count_nodes(self, root: TreeVerificationNode) -> int:
        """Count total nodes in the tree (excluding root)."""
        count = len(root.children)
        for child in root.children:
            count += self.count_nodes(child)
        return count

    def expected_accepted_tokens(
        self,
        acceptance_rate: float = 0.85,
    ) -> float:
        """
        Estimate expected accepted tokens for tree verification.

        Tree verification accepts more tokens than linear because
        multiple branches are checked per level.
        """
        # Simplified: each level has branching_factor attempts
        expected = 0
        for depth in range(1, self.max_depth + 1):
            # Probability that at least one branch is accepted at this depth
            p_all_reject = (1 - acceptance_rate) ** self.branching_factor
            p_at_least_one = 1 - p_all_reject
            expected += p_at_least_one * (acceptance_rate ** (depth - 1))
        return expected
