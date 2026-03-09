"""
Tokenizer Utilities.

Helper functions for token counting, encoding, and cost estimation.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


# Token-to-cost relationship:
#   Cost per word ≈ F_lang × P_per_token
# where F_lang = tokenizer fertility for the language.

FERTILITY_TABLE = {
    "english":  1.3,    # ~750K words in 1M context
    "spanish":  1.5,    # ~667K words
    "french":   1.5,    # ~667K words
    "german":   1.8,    # ~556K words
    "chinese":  1.75,   # ~571K chars (per character)
    "japanese": 2.5,    # ~400K chars
    "korean":   2.0,    # ~500K chars
    "arabic":   2.0,    # ~500K words
    "hindi":    3.5,    # ~286K words
    "russian":  2.0,    # ~500K words
}

# Pricing per million tokens (Opus 4.6)
PRICING = {
    "opus_input": 5.00,
    "opus_output": 25.00,
    "opus_cache_write": 6.25,
    "opus_cache_read": 0.50,
    "sonnet_input": 3.00,
    "sonnet_output": 15.00,
    "haiku_input": 0.25,
    "haiku_output": 1.25,
}


def estimate_tokens(text: str, language: str = "english") -> int:
    """
    Rough estimate of token count from text.

    Uses fertility table for language-specific estimation.
    For precise counts, use the actual tokenizer.

    Args:
        text: Input text.
        language: Language of the text.

    Returns:
        Estimated token count.
    """
    words = len(text.split())
    fertility = FERTILITY_TABLE.get(language.lower(), 1.5)
    return int(words * fertility)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
    model: str = "opus",
    cached_input_tokens: int = 0,
) -> float:
    """
    Estimate API cost for a request.

    Cost formula:
        Cost = (N_input × P_input + N_output × P_output
                + N_thinking × P_output) / 10^6

    IMPORTANT: Thinking tokens are billed at OUTPUT rates ($25/M),
    not input rates. A complex query with 50K thinking tokens
    costs $1.25 in thinking alone.

    Args:
        input_tokens: Number of input tokens (uncached).
        output_tokens: Number of output tokens.
        thinking_tokens: Number of thinking tokens.
        model: Model tier ("opus", "sonnet", or "haiku").
        cached_input_tokens: Tokens served from cache (10× cheaper).

    Returns:
        Estimated cost in USD.
    """
    p_input = PRICING.get(f"{model}_input", 5.00)
    p_output = PRICING.get(f"{model}_output", 25.00)
    p_cache_read = PRICING.get(f"{model}_cache_read", 0.50)

    cost = (
        (input_tokens - cached_input_tokens) * p_input
        + cached_input_tokens * p_cache_read
        + (output_tokens + thinking_tokens) * p_output
    ) / 1_000_000

    return cost


def estimate_agent_cost(
    num_turns: int,
    avg_input_tokens: int = 5000,
    avg_output_tokens: int = 500,
    avg_thinking_tokens: int = 2000,
    cache_hit_rate: float = 0.7,
    model: str = "opus",
) -> float:
    """
    Estimate cost of a multi-turn agent session.

    Agent loop token growth:
        C_input(i) = C_system + Σ_{j=1}^{i-1} (C_output(j) + C_tool_result(j))

    Each turn re-sends all previous messages, so input grows
    quadratically with turn count.

    Example: 20-turn agent loop, 50K-token conversations:
        Without caching: ~$6.00
        With caching:    ~$1.50 (70% savings)

    Args:
        num_turns: Number of agent loop iterations.
        avg_input_tokens: Average input per turn.
        avg_output_tokens: Average output per turn.
        avg_thinking_tokens: Average thinking per turn.
        cache_hit_rate: Fraction of input served from cache.
        model: Model tier.

    Returns:
        Estimated total session cost in USD.
    """
    total_cost = 0.0

    for turn in range(num_turns):
        # Input grows with turn number (re-sent context)
        turn_input = avg_input_tokens * (1 + turn * 0.3)
        cached = int(turn_input * cache_hit_rate) if turn > 0 else 0

        turn_cost = estimate_cost(
            input_tokens=int(turn_input),
            output_tokens=avg_output_tokens,
            thinking_tokens=avg_thinking_tokens,
            model=model,
            cached_input_tokens=cached,
        )
        total_cost += turn_cost

    return total_cost
