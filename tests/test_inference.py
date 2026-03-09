"""
Unit tests for inference engine components.

Tests thinking mode enums, KV cache, budget enforcement,
and basic engine instantiation without requiring model weights.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# ThinkingMode & ThinkingBudget
# ---------------------------------------------------------------------------

class TestThinkingMode:

    def test_enum_values(self):
        from src.inference.fast_mode import ThinkingMode
        assert ThinkingMode.FAST is not None
        assert ThinkingMode.LOW is not None
        assert ThinkingMode.MEDIUM is not None
        assert ThinkingMode.HIGH is not None
        assert ThinkingMode.MAX is not None

    def test_budget_for_mode(self):
        from src.inference.fast_mode import ThinkingBudget, ThinkingMode
        budget = ThinkingBudget.for_mode(ThinkingMode.FAST)
        assert budget.max_tokens == 0
        assert budget.min_tokens == 0

    def test_budget_high(self):
        from src.inference.fast_mode import ThinkingBudget, ThinkingMode
        budget = ThinkingBudget.for_mode(ThinkingMode.HIGH)
        assert budget.max_tokens == 30_000
        assert budget.min_tokens == 2_000

    def test_budget_max(self):
        from src.inference.fast_mode import ThinkingBudget, ThinkingMode
        budget = ThinkingBudget.for_mode(ThinkingMode.MAX)
        assert budget.max_tokens == 128_000


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

class TestKVCache:

    def test_instantiation(self):
        from src.inference.fast_mode import KVCache
        cache = KVCache(
            num_layers=4,
            max_seq_len=256,
            device="cpu",
            dtype=torch.float32,
        )
        assert cache is not None
        assert cache.num_layers == 4

    def test_initial_length(self):
        from src.inference.fast_mode import KVCache
        cache = KVCache(
            num_layers=2,
            max_seq_len=128,
            device="cpu",
            dtype=torch.float32,
        )
        assert cache._length == 0


# ---------------------------------------------------------------------------
# ThinkingModeEngine
# ---------------------------------------------------------------------------

class TestThinkingModeEngine:

    def test_import(self):
        from src.inference.thinking_mode import ThinkingModeEngine
        assert ThinkingModeEngine is not None

    def test_compactor_import(self):
        from src.inference.thinking_mode import ThinkingCompactor
        assert ThinkingCompactor is not None


# ---------------------------------------------------------------------------
# FastModeEngine
# ---------------------------------------------------------------------------

class TestFastModeEngine:

    def test_import(self):
        from src.inference.fast_mode import FastModeEngine
        assert FastModeEngine is not None
