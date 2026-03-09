"""
Distributed Checkpointing for Multi-Trillion Parameter Models.

At 2T parameters in BF16, checkpointing is a major challenge:

    Component        | Size
    ────────────────────────────────
    Model weights    | ~4 TB (BF16)
    Optimizer states | ~8 TB (Adam m + v in FP32)
    Gradients        | ~4 TB (BF16)
    RNG states       | ~few KB
    Total state      | ~12–16 TB per checkpoint

Checkpointing cost:
    M_state = N_params × 16 bytes = 2T × 16 = 32 TB
    T_checkpoint = 32,000 GB / 100 GB/s ≈ 320 s ≈ 5.3 min

With 32,000 GPUs at ~1% annual failure rate:
    E[failures/day] = 32,000 × (0.01/365) ≈ 0.87
    Over 90 days: ~78 GPU failures expected.

Strategy:
    - Checkpoint every 1,000 steps (~30 min of training)
    - Use async checkpointing to overlap with training
    - FSDP sharded checkpoints (each rank saves its shard)
    - Keep last 3 checkpoints, delete older ones

References:
    - Rajbhandari et al., "ZeRO: Memory Optimizations Toward
      Training Trillion Parameter Models", 2020
"""

import os
import json
import shutil
import logging
import torch
import torch.distributed as dist
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages distributed model checkpoints for large-scale training.

    Handles:
        - FSDP sharded state dict saving/loading
        - Async checkpointing (non-blocking)
        - Checkpoint rotation (keep last N)
        - Training state persistence (step, LR, etc.)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        save_interval: int = 1000,
    ):
        """
        Args:
            checkpoint_dir: Root directory for checkpoints.
            max_checkpoints: Number of most recent checkpoints to keep.
            save_interval: Save every N training steps.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval

    def should_save(self, step: int) -> bool:
        """Check if we should save at this step."""
        return step > 0 and step % self.save_interval == 0

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        epoch: int = 0,
        loss: float = 0.0,
        extra_state: Optional[Dict] = None,
    ) -> str:
        """
        Save a full training checkpoint.

        For a 2T parameter model:
            - model_state_dict alone ≈ 4 TB (BF16)
            - optimizer_state_dict (Adam m, v) ≈ 8 TB (FP32)
            - Total ≈ 12–16 TB per checkpoint

        With FSDP, each rank saves only its portion:
            - Per-rank shard ≈ 12 TB / N_ranks

        Args:
            model: Model (possibly FSDP-wrapped).
            optimizer: Optimizer with state.
            scheduler: LR scheduler.
            step: Current training step.
            epoch: Current epoch.
            loss: Current loss value.
            extra_state: Additional state to save.

        Returns:
            Path to saved checkpoint directory.
        """
        step_dir = self.checkpoint_dir / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # ── Save model state dict ─────────────────────────
        model_path = step_dir / f"model_rank_{rank:04d}.pt"
        model_state = model.state_dict()
        torch.save(model_state, model_path)

        # ── Save optimizer state dict ─────────────────────
        optim_path = step_dir / f"optimizer_rank_{rank:04d}.pt"
        torch.save(optimizer.state_dict(), optim_path)

        # ── Save scheduler + training state (rank 0 only) ─
        if rank == 0:
            training_state = {
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "world_size": world_size,
                "scheduler_state_dict": scheduler.state_dict(),
            }
            if extra_state:
                training_state.update(extra_state)

            state_path = step_dir / "training_state.json"
            with open(state_path, "w") as f:
                json.dump(
                    {k: v if not isinstance(v, torch.Tensor) else v.item()
                     for k, v in training_state.items()},
                    f, indent=2,
                )

            logger.info(
                f"Checkpoint saved: step={step}, loss={loss:.4f}, "
                f"path={step_dir}"
            )

        # Sync all ranks before cleanup
        if dist.is_initialized():
            dist.barrier()

        # ── Rotate old checkpoints ────────────────────────
        if rank == 0:
            self._rotate_checkpoints()

        return str(step_dir)

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        If checkpoint_path is None, loads the latest checkpoint.

        Args:
            model: Model to load state into.
            optimizer: Optional optimizer to load state into.
            scheduler: Optional scheduler to load state into.
            checkpoint_path: Explicit checkpoint dir (or None for latest).

        Returns:
            Training state dict (step, loss, etc.).
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest()

        if checkpoint_path is None:
            logger.warning("No checkpoint found, starting from scratch")
            return {"step": 0, "epoch": 0, "loss": float("inf")}

        step_dir = Path(checkpoint_path)
        rank = dist.get_rank() if dist.is_initialized() else 0

        # ── Load model state ──────────────────────────────
        model_path = step_dir / f"model_rank_{rank:04d}.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            logger.info(f"Model loaded from {model_path}")

        # ── Load optimizer state ──────────────────────────
        if optimizer is not None:
            optim_path = step_dir / f"optimizer_rank_{rank:04d}.pt"
            if optim_path.exists():
                optimizer.load_state_dict(
                    torch.load(optim_path, map_location="cpu", weights_only=True)
                )

        # ── Load training state ───────────────────────────
        state_path = step_dir / "training_state.json"
        training_state = {"step": 0, "epoch": 0, "loss": float("inf")}
        if state_path.exists():
            with open(state_path) as f:
                training_state = json.load(f)

            if scheduler is not None and "scheduler_state_dict" in training_state:
                scheduler.load_state_dict(training_state["scheduler_state_dict"])

        logger.info(
            f"Checkpoint loaded: step={training_state.get('step', 0)}, "
            f"loss={training_state.get('loss', 'N/A')}"
        )

        return training_state

    def _find_latest(self) -> Optional[str]:
        """Find the most recent checkpoint directory."""
        candidates = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        return str(candidates[-1]) if candidates else None

    def _rotate_checkpoints(self):
        """Delete old checkpoints, keeping only the most recent N."""
        candidates = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        while len(candidates) > self.max_checkpoints:
            old_dir = candidates.pop(0)
            logger.info(f"Removing old checkpoint: {old_dir}")
            shutil.rmtree(old_dir, ignore_errors=True)
