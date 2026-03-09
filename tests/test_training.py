"""
Unit tests for training pipeline components.

Tests loss functions, optimizer construction, and checkpoint
serialization without requiring GPU or actual training data.
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class TestLanguageModelingLoss:

    def test_import(self):
        from src.training.loss import LanguageModelingLoss
        assert LanguageModelingLoss is not None

    def test_forward(self):
        from src.training.loss import LanguageModelingLoss
        loss_fn = LanguageModelingLoss(vocab_size=1000)
        logits = torch.randn(2, 16, 1000)
        labels = torch.randint(0, 1000, (2, 16))
        loss = loss_fn(logits, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_label_smoothing(self):
        from src.training.loss import LanguageModelingLoss
        loss_no_smooth = LanguageModelingLoss(vocab_size=100, label_smoothing=0.0)
        loss_smooth = LanguageModelingLoss(vocab_size=100, label_smoothing=0.1)
        logits = torch.randn(2, 16, 100)
        labels = torch.randint(0, 100, (2, 16))
        l1 = loss_no_smooth(logits, labels)
        l2 = loss_smooth(logits, labels)
        assert l1.item() > 0
        assert l2.item() > 0


class TestMoELoss:

    def test_import(self):
        from src.training.loss import MoELoss
        assert MoELoss is not None

    def test_forward(self):
        from src.training.loss import MoELoss
        moe_loss = MoELoss(vocab_size=1000)
        logits = torch.randn(2, 16, 1000)
        labels = torch.randint(0, 1000, (2, 16))
        aux_loss = torch.tensor(0.05)
        result = moe_loss(logits, labels, aux_loss)
        assert "total_loss" in result
        assert "lm_loss" in result
        assert result["total_loss"].item() > 0


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class TestOptimizer:

    def test_build_optimizer(self):
        from src.training.optimizer import build_optimizer
        model = nn.Linear(256, 256)
        optimizer = build_optimizer(model, lr=1e-4)
        assert optimizer is not None
        assert len(optimizer.param_groups) >= 1


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class TestCheckpoint:

    def test_import(self):
        from src.training.checkpoint import CheckpointManager
        assert CheckpointManager is not None

    def test_instantiation(self, tmp_path):
        from src.training.checkpoint import CheckpointManager
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        assert mgr is not None
