"""
Claude Opus 4.6 — Tokenizer

BPE tokenizer with ~131K vocabulary, trained via SentencePiece.
Includes cost estimation utilities and fertility analysis.
"""

from .tokenizer_utils import estimate_token_cost, compute_fertility

__all__ = [
    "estimate_token_cost",
    "compute_fertility",
]
