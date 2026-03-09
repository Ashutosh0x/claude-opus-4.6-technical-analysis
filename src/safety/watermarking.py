"""
Statistical Watermarking for LLM Output Detection.

Implements the Kirchenbauer et al. (2023) watermarking scheme, which
embeds a statistically detectable signal in generated text by biasing
token selection toward a pseudo-random "green list".

Algorithm:
    Generation (embedding):
        1. At each step, hash the previous token: h = Hash(t_{i-1})
        2. Partition vocabulary into green/red using h as seed:
           Green list = {t : h(t) < |V|/2}
        3. Add bias δ to green-list logits:
           z'_i = z_i + δ  if t_i ∈ green list

    Detection:
        1. For generated text, count green-list tokens |G|
        2. Under null hypothesis (no watermark): E[|G|] = T/2
        3. Z-score test:
           z = (|G| - T/2) / sqrt(T/4)
        4. If z > 4 → almost certainly watermarked (p < 10⁻⁵)

Quality impact:
    δ = 1.0: minimal perplexity increase (~0.1%)
    δ = 2.0: small quality trade-off, stronger detection
    δ → ∞ : forced green-only tokens (very detectable, quality loss)

References:
    - Kirchenbauer et al., "A Watermark for Large Language Models",
      ICML 2023 (arXiv:2301.10226)
    - Kirchenbauer et al., "On the Reliability of Watermarks for
      Large Language Models", ICLR 2024
"""

import hashlib
import math
import logging
from dataclasses import dataclass
from typing import Optional, List, Set, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Watermark Logits Processor (Generation Side)
# ---------------------------------------------------------------------------

class WatermarkLogitsProcessor:
    """
    Inject a statistical watermark during token generation.

    At each decoding step:
        1. Hash the previous token to get a seed
        2. Use seed to partition vocab into green/red sets
        3. Add bias δ to green-list token logits

    The bias makes green-list tokens slightly more likely without
    dramatically changing the output distribution.
    """

    def __init__(
        self,
        vocab_size: int = 131072,
        delta: float = 1.0,
        green_fraction: float = 0.5,
        hash_key: int = 42,
    ):
        """
        Args:
            vocab_size: Size of the tokenizer vocabulary.
            delta: Logit bias added to green-list tokens.
            green_fraction: Fraction of vocab in the green list.
            hash_key: Secret key for green-list generation.
        """
        self.vocab_size = vocab_size
        self.delta = delta
        self.green_fraction = green_fraction
        self.hash_key = hash_key

    def _get_green_list(self, prev_token_id: int) -> Set[int]:
        """
        Generate the green list for a given previous token.

        Uses Hash(prev_token_id, hash_key) to seed a deterministic
        partition of the vocabulary.

        Args:
            prev_token_id: The previous token ID.

        Returns:
            Set of green-list token IDs.
        """
        # Deterministic hash of (key, prev_token)
        seed_bytes = f"{self.hash_key}:{prev_token_id}".encode()
        hash_val = int(hashlib.sha256(seed_bytes).hexdigest(), 16)

        # Use hash to seed a pseudo-random generator
        rng = torch.Generator()
        rng.manual_seed(hash_val % (2**63))

        # Random permutation of vocabulary
        perm = torch.randperm(self.vocab_size, generator=rng)

        # First green_fraction tokens in permutation are "green"
        green_count = int(self.vocab_size * self.green_fraction)
        return set(perm[:green_count].tolist())

    def __call__(
        self,
        logits: torch.Tensor,
        prev_token_id: int,
    ) -> torch.Tensor:
        """
        Apply watermark bias to logits.

        Args:
            logits: [V] or [B, V] unnormalized logits.
            prev_token_id: Previous token for green-list seeding.

        Returns:
            Modified logits with green-list bias.
        """
        green_set = self._get_green_list(prev_token_id)

        # Build bias tensor
        bias = torch.zeros_like(logits)
        green_indices = torch.tensor(
            list(green_set), dtype=torch.long, device=logits.device
        )

        if logits.dim() == 1:
            bias[green_indices] = self.delta
        else:
            bias[:, green_indices] = self.delta

        return logits + bias


# ---------------------------------------------------------------------------
# Watermark Detector (Detection Side)
# ---------------------------------------------------------------------------

class WatermarkDetector:
    """
    Detect statistical watermark in generated text.

    Detection algorithm:
        1. For each token t_i, compute the green list using t_{i-1}
        2. Count how many tokens fall in their respective green lists
        3. Under null hypothesis (no watermark): E[|G|] = T/2
        4. Z-score: z = (|G| - T/2) / sqrt(T/4)
        5. If z > threshold → watermarked

    Z-score interpretation:
        z > 2:  suggestive (p < 0.023)
        z > 3:  strong evidence (p < 0.0013)
        z > 4:  very strong (p < 3.2×10⁻⁵)
        z > 5:  conclusive (p < 2.9×10⁻⁷)
    """

    def __init__(
        self,
        vocab_size: int = 131072,
        green_fraction: float = 0.5,
        hash_key: int = 42,
        z_threshold: float = 4.0,
    ):
        """
        Args:
            vocab_size: Vocabulary size (must match generator).
            green_fraction: Green list fraction (must match generator).
            hash_key: Secret key (must match generator).
            z_threshold: Z-score threshold for watermark detection.
        """
        self.vocab_size = vocab_size
        self.green_fraction = green_fraction
        self.hash_key = hash_key
        self.z_threshold = z_threshold

    def _get_green_list(self, prev_token_id: int) -> Set[int]:
        """Generate green list (same as generator)."""
        seed_bytes = f"{self.hash_key}:{prev_token_id}".encode()
        hash_val = int(hashlib.sha256(seed_bytes).hexdigest(), 16)

        rng = torch.Generator()
        rng.manual_seed(hash_val % (2**63))

        perm = torch.randperm(self.vocab_size, generator=rng)
        green_count = int(self.vocab_size * self.green_fraction)
        return set(perm[:green_count].tolist())

    def detect(
        self,
        token_ids: List[int],
    ) -> "WatermarkResult":
        """
        Test whether a sequence of tokens is watermarked.

        Args:
            token_ids: List of token IDs to analyze.

        Returns:
            WatermarkResult with z-score and detection decision.
        """
        if len(token_ids) < 2:
            return WatermarkResult(
                z_score=0.0,
                green_count=0,
                total_tokens=0,
                is_watermarked=False,
                p_value=1.0,
            )

        green_count = 0
        total = len(token_ids) - 1   # Skip first token (no prev)

        for i in range(1, len(token_ids)):
            prev_id = token_ids[i - 1]
            curr_id = token_ids[i]
            green_set = self._get_green_list(prev_id)

            if curr_id in green_set:
                green_count += 1

        # Z-score test
        # Under null: E[green] = T × green_fraction
        #            Var[green] = T × green_fraction × (1 - green_fraction)
        expected = total * self.green_fraction
        variance = total * self.green_fraction * (1 - self.green_fraction)
        std = math.sqrt(max(variance, 1e-10))

        z_score = (green_count - expected) / std

        # One-sided p-value
        p_value = 0.5 * math.erfc(z_score / math.sqrt(2))

        return WatermarkResult(
            z_score=z_score,
            green_count=green_count,
            total_tokens=total,
            is_watermarked=(z_score > self.z_threshold),
            p_value=p_value,
        )


@dataclass
class WatermarkResult:
    """Result of watermark detection analysis."""
    z_score: float
    green_count: int
    total_tokens: int
    is_watermarked: bool
    p_value: float

    @property
    def green_fraction_observed(self) -> float:
        """Observed fraction of green-list tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.green_count / self.total_tokens

    @property
    def confidence(self) -> str:
        """Human-readable confidence level."""
        z = abs(self.z_score)
        if z > 5:
            return "conclusive"
        elif z > 4:
            return "very strong"
        elif z > 3:
            return "strong"
        elif z > 2:
            return "suggestive"
        else:
            return "not detected"
