"""
FlashAttention, Ring Attention, and Activation Checkpointing.

Educational reference implementations of the key attention
optimizations used in frontier model inference and training.

FlashAttention (Dao et al., 2022-2024):
    IO-aware exact attention that avoids materializing the full
    n×n attention matrix in GPU HBM. Uses tiling with online
    softmax to compute exact attention in O(n²d/M) HBM accesses.

    Versions:
        FA-1:  2-4× speedup (tiled exact attention)
        FA-2:  5-9× speedup (better work partitioning)
        FA-3:  1.5-2× over FA-2 (FP8, warp scheduling, H100)

Ring Attention (Liu et al., 2023):
    Distributes sequence across GPUs in a ring topology for
    sequences exceeding single-GPU memory. Communication is
    overlapped with computation.

Chunked Prefill (Agrawal et al., 2024 — Sarathi):
    Splits 1M-token inputs into chunks for incremental KV cache
    building, allowing interleaved decode for other requests.

FlashDecoding (Dao et al., 2023):
    Parallelizes decode across the KV cache sequence dimension
    for 2-8× speedup on long sequences (S > 64K).

Activation Checkpointing:
    Selective layer recomputation to reduce activation memory from
    O(L) to O(sqrt(L)) with ~20% compute overhead.

References:
    - Dao et al., "FlashAttention: Fast and Memory-Efficient
      Exact Attention", NeurIPS 2022
    - Liu et al., "Ring Attention with Blockwise Transformers", 2023
    - Milakov & Gimelshein, "Online normalizer calculation for
      softmax", 2018
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Online Softmax  (the core FlashAttention insight)
# ---------------------------------------------------------------------------

class OnlineSoftmax:
    """
    Online (streaming) softmax computation.

    Standard softmax requires two passes:
        1. Compute max over all elements
        2. Exponentiate and normalize

    Online softmax maintains running statistics in a single pass,
    enabling tiled attention without inter-tile communication.

    Running updates (Milakov & Gimelshein 2018):
        m^(i) = max(m^(i-1), rowmax(S^(i)))
        l^(i) = exp(m^(i-1) - m^(i)) * l^(i-1)
               + rowsum(exp(S^(i) - m^(i)))
        O^(i) = diag(exp(m^(i-1) - m^(i)))^-1 * O^(i-1)
               + exp(S^(i) - m^(i)) * V^(i)

    Result: each tile can be computed independently in SRAM
    without materializing the full n×n matrix.
    """

    @staticmethod
    def softmax_online(
        scores_blocks: List[torch.Tensor],
        values_blocks: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute attention output using online softmax over blocks.

        Args:
            scores_blocks: List of [B, H, Q_block, K_block] score tiles.
            values_blocks: List of [B, H, K_block, D] value tiles.

        Returns:
            output: [B, H, Q, D] — exact same result as standard attention.
        """
        # Initialize running statistics
        B, H, Q, _ = scores_blocks[0].shape
        D = values_blocks[0].shape[-1]

        m = torch.full((B, H, Q, 1), float("-inf"),
                       device=scores_blocks[0].device,
                       dtype=scores_blocks[0].dtype)
        l = torch.zeros(B, H, Q, 1,
                        device=scores_blocks[0].device,
                        dtype=scores_blocks[0].dtype)
        o = torch.zeros(B, H, Q, D,
                        device=scores_blocks[0].device,
                        dtype=scores_blocks[0].dtype)

        for s_block, v_block in zip(scores_blocks, values_blocks):
            # s_block: [B, H, Q, K_block]
            # v_block: [B, H, K_block, D]

            # Update running max
            m_new = torch.maximum(m, s_block.amax(dim=-1, keepdim=True))

            # Rescale previous accumulator
            correction = torch.exp(m - m_new)
            l = correction * l + torch.exp(s_block - m_new).sum(dim=-1, keepdim=True)
            o = correction * o + torch.exp(s_block - m_new) @ v_block

            m = m_new

        # Normalize
        return o / l


# ---------------------------------------------------------------------------
# FlashAttention Reference (pure PyTorch, educational)
# ---------------------------------------------------------------------------

def flash_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 256,
    causal: bool = True,
) -> torch.Tensor:
    """
    Reference FlashAttention implementation using tiled online softmax.

    This is a pure-PyTorch educational version. In production, use
    torch.nn.functional.scaled_dot_product_attention which dispatches
    to the optimized CUDA kernel (FlashAttention-2/3).

    Complexity:
        Standard:       O(n²) HBM reads/writes
        FlashAttention: O(n²d/M) HBM accesses (M = SRAM size)

    Args:
        q: [B, H, N, D] queries
        k: [B, H, S, D] keys
        v: [B, H, S, D] values
        block_size: Tile size (fits in SRAM)
        causal: Apply causal mask

    Returns:
        [B, H, N, D] attention output (numerically identical to standard)
    """
    B, H, N, D = q.shape
    S = k.shape[2]
    scale = 1.0 / math.sqrt(D)

    # Initialize online softmax accumulators
    m = torch.full((B, H, N, 1), float("-inf"), device=q.device, dtype=q.dtype)
    l = torch.zeros(B, H, N, 1, device=q.device, dtype=q.dtype)
    o = torch.zeros(B, H, N, D, device=q.device, dtype=q.dtype)

    # Tile over key/value blocks
    for j_start in range(0, S, block_size):
        j_end = min(j_start + block_size, S)
        k_block = k[:, :, j_start:j_end, :]   # [B, H, Bk, D]
        v_block = v[:, :, j_start:j_end, :]

        # Compute attention scores for this block
        scores = (q @ k_block.transpose(-2, -1)) * scale  # [B, H, N, Bk]

        if causal:
            # Create causal mask for this block
            row_idx = torch.arange(N, device=q.device).unsqueeze(1)
            col_idx = torch.arange(j_start, j_end, device=q.device).unsqueeze(0)
            mask = row_idx < col_idx
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Online softmax update
        m_new = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
        correction = torch.exp(m - m_new)
        p = torch.exp(scores - m_new)

        l = correction * l + p.sum(dim=-1, keepdim=True)
        o = correction * o + p @ v_block

        m = m_new

    return o / l


# ---------------------------------------------------------------------------
# Ring Attention
# ---------------------------------------------------------------------------

@dataclass
class RingAttentionConfig:
    """Configuration for ring attention across GPUs.

    For 1M tokens across 4 GPUs:
        M_per_GPU = M_total / N_GPUs = 1.25 TB / 4 = 312.5 GB

    Communication is overlapped with computation: while GPU i
    computes attention on its local block, it sends its KV cache
    to GPU i+1 in the ring.
    """
    num_gpus: int = 4
    total_seq_len: int = 1_000_000
    overlap_communication: bool = True

    @property
    def seq_per_gpu(self) -> int:
        return self.total_seq_len // self.num_gpus

    def partition_sequence(
        self, tokens: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Split a token sequence across GPUs."""
        chunk_size = self.seq_per_gpu
        return [
            tokens[:, i * chunk_size:(i + 1) * chunk_size]
            for i in range(self.num_gpus)
        ]

    def kv_memory_per_gpu(
        self,
        num_layers: int = 160,
        num_kv_heads: int = 16,
        head_dim: int = 128,
        dtype_bytes: int = 2,
    ) -> float:
        """Estimate KV cache memory per GPU in GB."""
        total = (
            2 * num_layers * num_kv_heads * head_dim
            * self.seq_per_gpu * dtype_bytes
        )
        return total / (1024**3)


# ---------------------------------------------------------------------------
# Chunked Prefill
# ---------------------------------------------------------------------------

class ChunkedPrefill:
    """
    Chunked prefill for very long inputs (e.g., 1M tokens).

    Instead of processing the entire prompt in one pass, split into
    chunks and build the KV cache incrementally. This enables:
        1. Generation before fully processing the input
        2. Interleaving prefill with decode for other requests
        3. Fitting within GPU memory constraints

    Typical chunk size: ~32K tokens.
    """

    def __init__(self, chunk_size: int = 32768):
        self.chunk_size = chunk_size

    def split_input(
        self, input_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Split input tokens into chunks for incremental prefill.

        Args:
            input_ids: [B, total_seq_len]

        Returns:
            List of [B, chunk_size] tensors.
        """
        total = input_ids.shape[1]
        chunks = []
        for start in range(0, total, self.chunk_size):
            end = min(start + self.chunk_size, total)
            chunks.append(input_ids[:, start:end])
        return chunks

    def prefill_chunks(
        self,
        chunks: List[torch.Tensor],
        model_forward_fn: Callable,
    ) -> List[torch.Tensor]:
        """
        Process chunks sequentially, building KV cache incrementally.

        Args:
            chunks: List of token chunks.
            model_forward_fn: Function(chunk, kv_cache) -> (logits, kv_cache).

        Returns:
            List of per-chunk logits.
        """
        kv_cache = None
        all_logits = []

        for i, chunk in enumerate(chunks):
            logits, kv_cache = model_forward_fn(chunk, kv_cache)
            all_logits.append(logits)
            logger.debug(
                f"Prefill chunk {i + 1}/{len(chunks)}: "
                f"{chunk.shape[1]} tokens processed"
            )

        return all_logits


# ---------------------------------------------------------------------------
# FlashDecoding
# ---------------------------------------------------------------------------

def flash_decoding_speedup(
    seq_len: int,
    num_parallel_blocks: int = 8,
) -> float:
    """
    Estimate FlashDecoding speedup over standard sequential decode.

    Standard decode: T ∝ S   (sequential over keys)
    FlashDecoding:   T ∝ S/P (parallel across P thread blocks)

    Speedup: 2-8× for long sequences (S > 64K).

    Args:
        seq_len: Current KV cache length.
        num_parallel_blocks: Thread blocks for parallel reduction.

    Returns:
        Estimated speedup factor.
    """
    if seq_len < 1024:
        return 1.0   # No benefit for short sequences

    # Ideal speedup is min(P, S/block_size), with overhead
    block_size = max(seq_len // num_parallel_blocks, 256)
    effective_parallelism = min(num_parallel_blocks, seq_len // block_size)

    # Account for reduction overhead (~10-15%)
    overhead = 0.85
    return effective_parallelism * overhead


# ---------------------------------------------------------------------------
# Activation Checkpointing
# ---------------------------------------------------------------------------

class ActivationCheckpointing:
    """
    Selective activation checkpointing for memory-efficient training.

    The memory problem:
        M_activations = L × B × S × d_model × b
        For Opus 4.6: 160 × 4096 × 8192 × 16384 × 2 ≈ 140 TB (impossible)

    Solution: recompute activations during backward pass.

    Strategies:
        No checkpointing:        O(L) memory, 0% overhead
        Full checkpointing:      O(1) memory, ~33% overhead
        Selective (√L):          O(√L) memory, ~20% overhead

    Selective checkpointing (checkpoint every √160 ≈ 13 layers)
    is the standard for frontier models.
    """

    def __init__(
        self,
        num_layers: int = 160,
        strategy: str = "selective",
    ):
        """
        Args:
            num_layers: Total transformer layers.
            strategy: "none", "full", or "selective".
        """
        self.num_layers = num_layers
        self.strategy = strategy

    @property
    def checkpoint_interval(self) -> int:
        """Layers between checkpoints."""
        if self.strategy == "none":
            return self.num_layers  # Never checkpoint
        elif self.strategy == "full":
            return 1                # Every layer
        else:
            return max(1, int(math.sqrt(self.num_layers)))  # √L

    def should_checkpoint(self, layer_idx: int) -> bool:
        """Whether to checkpoint activations at this layer."""
        if self.strategy == "none":
            return False
        if self.strategy == "full":
            return True
        return layer_idx % self.checkpoint_interval == 0

    def memory_reduction(self) -> float:
        """Estimated memory reduction factor vs no checkpointing."""
        if self.strategy == "none":
            return 1.0
        elif self.strategy == "full":
            return 1.0 / self.num_layers
        else:
            return self.checkpoint_interval / self.num_layers

    def compute_overhead(self) -> float:
        """Estimated compute overhead (fraction of forward pass)."""
        if self.strategy == "none":
            return 0.0
        elif self.strategy == "full":
            return 0.33          # ~33%
        else:
            return 0.20          # ~20% for selective

    def wrap_layer(
        self,
        layer: nn.Module,
        layer_idx: int,
    ) -> nn.Module:
        """
        Wrap a transformer layer with gradient checkpointing if needed.

        Uses torch.utils.checkpoint.checkpoint for automatic
        recomputation during backward pass.

        Args:
            layer: The transformer layer module.
            layer_idx: Index of this layer.

        Returns:
            Wrapped or original module.
        """
        if not self.should_checkpoint(layer_idx):
            return layer

        class CheckpointedLayer(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, *args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    self.wrapped, *args, use_reentrant=False, **kwargs,
                )

        return CheckpointedLayer(layer)


# ---------------------------------------------------------------------------
# Prefill vs Decode Analysis
# ---------------------------------------------------------------------------

@dataclass
class InferencePhaseAnalysis:
    """
    Analysis of prefill vs decode inference phases.

    Prefill (prompt processing):
        - Processes all input tokens in parallel
        - Matrix-matrix multiply (GEMM) → compute-bound
        - GPU utilization >70%
        - Determines TTFT (Time to First Token)

    Decode (token generation):
        - Generates one token at a time
        - Matrix-vector multiply (GEMV) → memory-bandwidth-bound
        - GPU utilization ~5-15%
        - Determines per-token latency

    Inference FLOPs per token:
        C_inference ≈ 2 × N_active
        For Opus 4.6: 2 × 200B = 400 GFLOPs/token
    """
    active_params: int = 200_000_000_000    # 200B active

    @property
    def flops_per_token(self) -> int:
        """FLOPs per output token (decode phase)."""
        return 2 * self.active_params

    def is_compute_bound(self, phase: str) -> bool:
        """Whether the phase is compute-bound (vs memory-bandwidth-bound)."""
        return phase.lower() == "prefill"

    def roofline_balance_point(
        self,
        peak_flops: float = 990e12,      # H100 BF16 TFLOPS
        memory_bw: float = 3.35e12,       # H100 HBM bandwidth
    ) -> float:
        """
        H100 roofline balance point (FLOPs/byte).

        If arithmetic intensity < balance → memory-bound (decode)
        If arithmetic intensity > balance → compute-bound (prefill)
        """
        return peak_flops / memory_bw

    def estimated_throughput(
        self,
        phase: str,
        num_gpus: int = 8,
    ) -> float:
        """Estimated tokens/second for the given phase."""
        if phase == "prefill":
            # Compute-bound: ~100K tokens/s on 8xH100
            return 100_000 * num_gpus / 8
        else:
            # Memory-bound: ~50-100 tokens/s per request
            return 80
