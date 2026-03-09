"""
Main Training Loop for Claude Opus 4.6.

Training infrastructure (estimated):
    GPU Count:       20,000–60,000 NVIDIA H100
    Training:        3–6 months
    Training Data:   20–40+ trillion tokens
    Interconnect:    NVLink / NVSwitch
    Estimated FLOPs: ~3.6 × 10^26

Training FLOPs estimate (Kaplan approximation):
    C ≈ 6 × N × D
    For N = 2 × 10^12 and D = 30 × 10^12:
    C ≈ 6 × 2e12 × 30e12 = 3.6 × 10^26 FLOPs

Parallelism strategy:
    Data Parallel:    ZeRO Stage 3 / FSDP  — 256–512 groups
    Tensor Parallel:  Within-node           — 4–8 GPUs
    Pipeline Parallel: Across nodes         — 8–16 stages
    Expert Parallel:  MoE routing           — 64–128 GPUs
    Total GPUs ≈ N_DP × N_TP × N_PP ≈ 256 × 8 × 16 = 32,768

Activation checkpointing:
    Without: M_act = L × B × S × d_model × b ≈ 140 TB
    Selective (√L checkpoints): ~20% compute overhead

Mixed precision:
    Forward: BF16 (same range as FP32, sufficient precision)
    Master weights: FP32 (for optimizer updates)
    Loss scaling: Not needed with BF16 (unlike FP16)

References:
    - Narayanan et al., "Megatron-LM" (parallelism), 2021
    - Rajbhandari et al., "ZeRO" (memory optimization), 2020
"""

import os
import time
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from typing import Dict, Optional

from ..model.transformer import ClaudeModel, TransformerLayer
from .loss import MoELoss, compute_perplexity
from .optimizer import build_optimizer, build_cosine_schedule
from .checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class Trainer:
    """
    Distributed trainer for Claude Opus 4.6.

    Supports:
        - FSDP (Fully Sharded Data Parallel) for memory efficiency
        - BF16 mixed precision training
        - Gradient accumulation for large effective batch sizes
        - Activation checkpointing (selective)
        - Async distributed checkpointing
        - MoE auxiliary loss tracking

    Effective batch size:
        B_total = B_micro × N_accum × N_data_parallel

    For Opus 4.6 training:
        B_micro = 1–2 (sequences per GPU)
        N_accum = 16–32 (gradient accumulation steps)
        N_data_parallel = 256–512
        → B_total ≈ 4–16M tokens per step
    """

    def __init__(self, config):
        """
        Args:
            config: Training configuration with fields:
                - Model architecture params
                - Training hyperparams (lr, warmup, etc.)
                - Data paths and batch sizes
                - Checkpoint settings
        """
        self.config = config
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # ─── Initialize distributed ──────────────────────
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        self.device = torch.device(f"cuda:{self.local_rank}")

        # ─── Build model ─────────────────────────────────
        self.model = self._build_model()

        # ─── Build optimizer and scheduler ────────────────
        self.optimizer = build_optimizer(
            self.model,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
        )
        self.scheduler = build_cosine_schedule(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        # ─── Loss function ───────────────────────────────
        self.loss_fn = MoELoss(
            vocab_size=config.vocab_size,
            balance_coef=config.router_aux_loss_coef,
        )

        # ─── Checkpoint manager ──────────────────────────
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_checkpoints=config.max_checkpoints,
            save_interval=config.save_interval,
        )

        # ─── Training state ──────────────────────────────
        self.global_step = 0
        self.epoch = 0

        # ─── Gradient accumulation ───────────────────────
        self.grad_accum_steps = config.gradient_accumulation_steps

    def _build_model(self) -> nn.Module:
        """
        Build and wrap model with FSDP.

        FSDP shards model parameters, gradients, and optimizer
        states across all data-parallel ranks, reducing per-GPU
        memory by ~N_DP×.

        Mixed precision policy:
            - param_dtype: BF16 (weights during forward)
            - reduce_dtype: BF16 (gradient all-reduce)
            - buffer_dtype: BF16
        """
        # Build base model
        model = ClaudeModel(self.config)

        if self.rank == 0:
            total_params = model.num_parameters()
            active_params = model.num_active_parameters()
            logger.info(
                f"Model initialized: "
                f"{total_params/1e9:.1f}B total params, "
                f"{active_params/1e9:.1f}B active params/token"
            )

        # Move to GPU
        model = model.to(self.device)

        # Wrap with FSDP for distributed training
        if self.world_size > 1:
            # BF16 mixed precision — no loss scaling needed
            bf16_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

            # Auto-wrap at TransformerLayer granularity
            auto_wrap = transformer_auto_wrap_policy(
                transformer_layer_cls={TransformerLayer}
            )

            model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=bf16_policy,
                auto_wrap_policy=auto_wrap,
                device_id=self.local_rank,
                limit_all_gathers=True,
            )

            # Enable activation checkpointing
            # Selective: checkpoint every √L ≈ 13 layers
            # ~20% compute overhead, massive memory savings
            checkpoint_every = max(1, int(self.config.num_hidden_layers ** 0.5))
            for i, layer in enumerate(model.module.layers
                                       if hasattr(model, "module")
                                       else model.layers):
                if i % checkpoint_every == 0:
                    layer = torch.utils.checkpoint.checkpoint(
                        layer, use_reentrant=False
                    )

        return model

    def train_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Execute one training step (possibly with gradient accumulation).

        Mixed precision training with BF16:
            - Forward pass runs in BF16 (autocast)
            - Master weights stay in FP32 (optimizer)
            - BF16 has same exponent range as FP32, so no loss
              scaling needed (unlike FP16)

        Args:
            batch: Dict with 'input_ids' [B, T] and optionally
                   'labels' [B, T] and 'attention_mask' [B, T].

        Returns:
            Dict with loss values and learning rate.
        """
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass in BF16
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
            aux_loss = outputs["aux_loss"]

            # Scale for gradient accumulation
            loss = loss / self.grad_accum_steps

        # Backward pass
        loss.backward()

        return {
            "loss": (loss * self.grad_accum_steps).item(),
            "aux_loss": aux_loss.item(),
            "ppl": compute_perplexity(
                loss * self.grad_accum_steps
            ).item(),
        }

    def optimizer_step(self):
        """
        Apply accumulated gradients.

        Gradient clipping at max_norm=1.0 prevents
        training instability from rare large gradients.
        """
        # Gradient clipping
        if hasattr(self.model, "clip_grad_norm_"):
            # FSDP provides its own grad norm clipping
            self.model.clip_grad_norm_(self.config.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.max_grad_norm,
            )

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def train(self, dataloader: DataLoader):
        """
        Main training loop.

        Structure:
            for each epoch:
                for each batch:
                    1. Forward pass (BF16)
                    2. Backward pass
                    3. Every N steps: clip grads + optimizer step
                    4. Every M steps: checkpoint

        At scale (32K GPUs, 90 days):
            ~500K–1M total gradient steps
            ~20–40T tokens processed
        """
        self.model.train()
        accum_loss = 0.0

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch

            if self.rank == 0:
                logger.info(f"Starting epoch {epoch}")

            for batch_idx, batch in enumerate(dataloader):
                # ── Training step ─────────────────────────
                metrics = self.train_step(batch)
                accum_loss += metrics["loss"]

                # ── Optimizer step (every N accumulation steps)
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.optimizer_step()
                    avg_loss = accum_loss / self.grad_accum_steps
                    accum_loss = 0.0

                    # Log every 10 steps
                    if self.global_step % 10 == 0 and self.rank == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {self.global_step:>7d} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"PPL: {metrics['ppl']:.2f} | "
                            f"Aux: {metrics['aux_loss']:.4f} | "
                            f"LR: {lr:.2e}"
                        )

                    # ── Checkpoint ─────────────────────────
                    if self.ckpt_manager.should_save(self.global_step):
                        self.ckpt_manager.save(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            step=self.global_step,
                            epoch=self.epoch,
                            loss=avg_loss,
                        )

                    # ── Early termination ──────────────────
                    if self.global_step >= self.config.total_steps:
                        if self.rank == 0:
                            logger.info(
                                f"Training complete at step "
                                f"{self.global_step}"
                            )
                        return

        if self.rank == 0:
            logger.info("All epochs complete")

    def resume(self, checkpoint_path: Optional[str] = None):
        """Resume training from a checkpoint."""
        state = self.ckpt_manager.load(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_path=checkpoint_path,
        )
        self.global_step = state.get("step", 0)
        self.epoch = state.get("epoch", 0)

        if self.rank == 0:
            logger.info(
                f"Resumed from step {self.global_step}, "
                f"epoch {self.epoch}"
            )
