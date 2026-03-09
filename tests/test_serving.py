"""
Unit tests for serving, evaluation, alignment, safety, and distributed modules.

Tests component instantiation, data structures, and logic
without requiring actual model weights or GPU hardware.
"""

import json
import time
import pytest
import torch
import torch.nn as nn


# ===================================================================
# SERVING
# ===================================================================

class TestPageTable:
    """Tests for PagedAttention page table."""

    def test_allocate_and_free(self):
        from src.serving.batch_scheduler import PageTable
        pt = PageTable(total_pages=100, block_size=16)
        pages = pt.allocate(num_tokens=48)
        assert len(pages) == 3  # ceil(48/16) = 3
        assert pt.utilization > 0
        pt.free(pages)
        assert pt.utilization == 0

    def test_prefix_cache(self):
        from src.serving.batch_scheduler import PageTable
        pt = PageTable(total_pages=100, block_size=16)
        pages = pt.allocate(num_tokens=32)
        prefix_hash = hash((1, 2, 3, 4))
        pt.share_prefix(prefix_hash, pages)
        cached = pt.get_cached_prefix(prefix_hash)
        assert cached is not None
        assert len(cached) == len(pages)


class TestInferenceRequest:
    """Tests for inference request data structure."""

    def test_creation(self):
        from src.serving.batch_scheduler import InferenceRequest, RequestState
        req = InferenceRequest(
            request_id="test-001",
            input_ids=[1, 2, 3, 4, 5],
            max_new_tokens=100,
        )
        assert req.state == RequestState.WAITING
        assert req.total_tokens == 5
        assert req.ttft_ms is None

    def test_token_tracking(self):
        from src.serving.batch_scheduler import InferenceRequest
        req = InferenceRequest(request_id="test-002", input_ids=[1, 2])
        req.output_ids = [10, 20, 30]
        assert req.total_tokens == 5  # 2 input + 3 output


class TestContinuousBatchingScheduler:
    """Tests for the continuous batching scheduler."""

    def test_add_request(self):
        from src.serving.batch_scheduler import (
            ContinuousBatchingScheduler, InferenceRequest,
        )
        scheduler = ContinuousBatchingScheduler(max_batch_size=4)
        req = InferenceRequest(
            request_id="r1", input_ids=list(range(100)),
        )
        scheduler.add_request(req)
        assert len(scheduler.waiting_queue) == 1

    def test_cancel_request(self):
        from src.serving.batch_scheduler import (
            ContinuousBatchingScheduler, InferenceRequest,
        )
        scheduler = ContinuousBatchingScheduler()
        req = InferenceRequest(request_id="r1", input_ids=[1, 2])
        scheduler.add_request(req)
        assert scheduler.cancel_request("r1") is True
        assert len(scheduler.waiting_queue) == 0


# ===================================================================
# API SERVER
# ===================================================================

class TestSSEEventBuilder:
    """Tests for SSE event formatting."""

    def test_format_event(self):
        from src.serving.api_server import SSEEventBuilder
        event = SSEEventBuilder.format_event("ping", {"type": "ping"})
        assert "event: ping" in event
        assert "data: " in event
        assert event.endswith("\n\n")

    def test_text_delta(self):
        from src.serving.api_server import SSEEventBuilder
        event = SSEEventBuilder.text_delta(0, "Hello")
        parsed = json.loads(event.split("data: ")[1].strip())
        assert parsed["delta"]["text"] == "Hello"
        assert parsed["index"] == 0


class TestUsage:
    """Tests for token usage and cost calculation."""

    def test_cost_calculation(self):
        from src.serving.api_server import Usage
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        # $5/M input + $25/M output = $30
        assert abs(usage.cost_usd - 30.0) < 0.01


class TestRateLimiter:
    """Tests for the API rate limiter."""

    def test_allows_first_request(self):
        from src.serving.api_server import RateLimiter
        limiter = RateLimiter(rpm=10, tpm=100_000, max_concurrent=5)
        allowed, retry = limiter.check()
        assert allowed is True
        assert retry is None

    def test_concurrent_limit(self):
        from src.serving.api_server import RateLimiter
        limiter = RateLimiter(rpm=100, tpm=1_000_000, max_concurrent=2)
        limiter.acquire()
        limiter.acquire()
        allowed, _ = limiter.check()
        assert allowed is False
        limiter.release()
        allowed, _ = limiter.check()
        assert allowed is True


class TestAPIRequest:
    """Tests for API request parsing."""

    def test_from_dict(self):
        from src.serving.api_server import APIRequest
        data = {
            "model": "claude-opus-4-6-20260301",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
        }
        req = APIRequest.from_dict(data)
        assert req.stream is True
        assert req.thinking_enabled is True
        assert req.thinking_budget == 5000
        assert len(req.messages) == 1


# ===================================================================
# EVALUATION
# ===================================================================

class TestEloCalculator:
    """Tests for Arena Elo rating system."""

    def test_initial_rating(self):
        from src.evaluation.benchmarks import EloCalculator
        elo = EloCalculator(initial_elo=1200.0)
        assert elo.get_rating("new_model") == 1200.0

    def test_update(self):
        from src.evaluation.benchmarks import EloCalculator
        elo = EloCalculator(initial_elo=1200.0, k_factor=32.0)
        new_a, new_b = elo.update("model_a", "model_b", outcome=1.0)
        # Winner gains rating, loser drops
        assert new_a > 1200.0
        assert new_b < 1200.0

    def test_leaderboard(self):
        from src.evaluation.benchmarks import EloCalculator
        elo = EloCalculator()
        elo.update("opus", "sonnet", 1.0)
        elo.update("opus", "gpt5", 1.0)
        board = elo.leaderboard()
        assert board[0][0] == "opus"  # highest rated


class TestContaminationDetector:
    """Tests for benchmark contamination detection."""

    def test_no_contamination(self):
        from src.evaluation.benchmarks import ContaminationDetector
        detector = ContaminationDetector(n=5, threshold=0.5)
        result = detector.check_contamination(
            benchmark_examples=["What is 2+2?", "Capital of France?"],
            training_documents=["The quick brown fox jumps over the lazy dog"],
        )
        assert result["contamination_rate"] == 0.0

    def test_high_overlap(self):
        from src.evaluation.benchmarks import ContaminationDetector
        detector = ContaminationDetector(n=5, threshold=0.3)
        text = "What is the capital of France? The answer is Paris."
        result = detector.check_contamination(
            benchmark_examples=[text],
            training_documents=[text],  # exact copy
        )
        assert result["contamination_rate"] > 0


# ===================================================================
# ALIGNMENT
# ===================================================================

class TestDPOConfig:
    """Tests for DPO configuration."""

    def test_defaults(self):
        from src.alignment.dpo import DPOConfig
        cfg = DPOConfig()
        assert cfg.beta == 0.1
        assert cfg.loss_type == "sigmoid"


class TestConstitution:
    """Tests for Constitutional AI principles."""

    def test_constitution_loaded(self):
        from src.alignment.constitutional_ai import DEFAULT_CONSTITUTION
        assert len(DEFAULT_CONSTITUTION) > 0
        for principle in DEFAULT_CONSTITUTION:
            assert "name" in principle
            assert "critique_prompt" in principle
            assert "revision_prompt" in principle


# ===================================================================
# SAFETY
# ===================================================================

class TestSafetyCategory:
    """Tests for safety category enumeration."""

    def test_categories(self):
        from src.safety.classifiers import SafetyCategory
        # Should have multiple harm categories
        categories = list(SafetyCategory)
        assert len(categories) >= 5


# ===================================================================
# DISTRIBUTED
# ===================================================================

class TestParallelismConfig:
    """Tests for distributed parallelism configuration."""

    def test_total_gpus(self):
        from src.distributed.parallelism import ParallelismConfig
        cfg = ParallelismConfig()
        # Default: TP=8 * PP=40 * DP=4 * EP=16 = 20,480
        assert cfg.total_gpus == 20_480

    def test_global_batch_size(self):
        from src.distributed.parallelism import ParallelismConfig
        cfg = ParallelismConfig()
        assert cfg.global_batch_size > 0


class TestPipelineStages:
    """Tests for pipeline stage partitioning."""

    def test_build_stages(self):
        from src.distributed.parallelism import build_pipeline_stages
        stages = build_pipeline_stages(num_layers=160, pp_degree=40)
        assert len(stages) == 40
        # All layers covered
        total_layers = sum(s.num_layers for s in stages)
        assert total_layers == 160

    def test_balanced_stages(self):
        from src.distributed.parallelism import build_pipeline_stages
        stages = build_pipeline_stages(num_layers=160, pp_degree=40)
        # 160 / 40 = 4 layers each
        for stage in stages:
            assert stage.num_layers == 4


class TestMemoryEstimator:
    """Tests for per-GPU memory estimation."""

    def test_fits_in_80gb(self):
        from src.distributed.parallelism import (
            estimate_memory_per_gpu, ParallelismConfig,
        )
        cfg = ParallelismConfig()
        mem = estimate_memory_per_gpu(cfg)
        assert "total_gb" in mem
        assert "fits_in_80gb" in mem

    def test_flop_estimator(self):
        from src.distributed.parallelism import estimate_training_flops
        flops = estimate_training_flops()
        assert flops["total_flops"] > 0
        assert flops["training_days"] > 0
        assert flops["estimated_cost_usd"] > 0
