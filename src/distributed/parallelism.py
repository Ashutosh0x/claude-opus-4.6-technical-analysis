"""
Distributed Training Parallelism — TP, PP, DP, EP, FSDP, ZeRO.

For a 2T-parameter MoE model in BF16, the memory and compute budget
requires combining ALL major parallelism strategies:

┌──────────────────────────────────────────────────────────────────┐
│ Dimension       │ Splits    │ What it shards                    │
├──────────────────────────────────────────────────────────────────┤
│ Tensor Parallel │ TP=8      │ Self-attn & FFN weight matrices   │
│ Pipeline Paral. │ PP=40     │ Transformer layers across GPUs    │
│ Expert Parallel │ EP=16     │ MoE experts across nodes          │
│ Data Parallel   │ DP=4      │ Replicate model, shard data       │
│ FSDP / ZeRO-3  │ =DP       │ Shard optimizer + gradients + wts │
│ Seq. Parallel   │ =TP       │ Layer norm, dropout on TP group   │
│ Context Paral.  │ CP=4      │ Ring attention for 1M context     │
└──────────────────────────────────────────────────────────────────┘

Total GPU count:
    N_GPUs = TP × PP × DP × EP = 8 × 40 × 4 × 16 = 20,480

Memory budget per GPU (H100 80GB):
    Model params  : 2T × 2B ÷ 20,480 ≈ 195 GB → sharded to ~15 GB/GPU
    Optimizer     : cleaned with ZeRO-3 → ~3 GB/GPU
    Activations   : with gradient checkpointing → ~20 GB/GPU
    KV cache      : training N/A; inference → separate budget
    Buffer        : ~5 GB
    Total         : ~43 GB/GPU (within 80GB budget)

References:
    - Megatron-LM: Shoeybi et al. 2019 / Narayanan et al. 2021
    - DeepSpeed ZeRO: Rajbhandari et al. 2020
    - FSDP: Zhao et al. 2023 (PyTorch)
    - Ring Attention: Liu et al. 2023
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parallelism Configuration
# ---------------------------------------------------------------------------

@dataclass
class ParallelismConfig:
    """
    Full distributed training configuration for Opus 4.6.

    These values determine how the model is partitioned across
    the cluster. Every weight tensor, activation, and optimizer
    state is assigned to specific GPU(s) based on these settings.
    """
    # Parallelism degrees
    tensor_parallel: int = 8       # TP — Megatron column/row split
    pipeline_parallel: int = 40    # PP — layers split across stages
    data_parallel: int = 4         # DP — FSDP sharding group size
    expert_parallel: int = 16      # EP — MoE expert placement
    context_parallel: int = 4      # CP — Ring Attention groups
    sequence_parallel: bool = True # SP — fused with TP

    # Micro-batching for pipeline
    num_micro_batches: int = 32
    gradient_accumulation_steps: int = 16

    # ZeRO optimization stage
    zero_stage: int = 3  # 0=disabled, 1=optimizer, 2=+gradients, 3=+parameters

    # Activation checkpointing
    activation_checkpointing: bool = True
    checkpoint_every_n_layers: int = 1  # full recompute

    # Communication
    use_nccl: bool = True
    use_nvlink: bool = True      # intra-node (NVLink 4.0 = 900 GB/s)
    use_infiniband: bool = True  # inter-node (NDR = 400 Gb/s per port)
    gradient_compression: bool = False  # FP16 all-reduce

    @property
    def total_gpus(self) -> int:
        return (
            self.tensor_parallel
            * self.pipeline_parallel
            * self.data_parallel
            * self.expert_parallel
        )

    @property
    def global_batch_size(self) -> int:
        """Effective batch size across all data-parallel ranks."""
        return (
            self.data_parallel
            * self.num_micro_batches
            * self.gradient_accumulation_steps
        )


# ---------------------------------------------------------------------------
# Tensor Parallelism — Megatron-style Column/Row Splitting
# ---------------------------------------------------------------------------

class TensorParallelismMode(str, Enum):
    """How to split weight matrices across TP group."""
    COLUMN = "column"   # Split along output dim  → each GPU: [D, D/TP]
    ROW = "row"         # Split along input dim   → each GPU: [D/TP, D]


@dataclass
class TensorParallelPlan:
    """
    Defines how each linear layer is split across TP ranks.

    GQA Attention with TP=8:
        W_q [16384, 16384] → column split: each GPU [16384, 2048]
        W_k [16384, 2048]  → column split: each GPU [16384,  256]
        W_v [16384, 2048]  → column split: each GPU [16384,  256]
        W_o [16384, 16384] → row split:    each GPU [2048, 16384]

    MoE FFN with TP=8 (per expert):
        W_gate [16384, 49152] → column split: each GPU [16384, 6144]
        W_up   [16384, 49152] → column split: each GPU [16384, 6144]
        W_down [49152, 16384] → row split:    each GPU [6144, 16384]

    All-reduce after row-parallel layers.
    No all-reduce needed after column-parallel (just scatter).
    """
    layer_plans: Dict[str, TensorParallelismMode] = field(default_factory=dict)

    @staticmethod
    def default_plan() -> "TensorParallelPlan":
        """Default TP plan for Claude Opus 4.6."""
        return TensorParallelPlan(layer_plans={
            "attention.q_proj": TensorParallelismMode.COLUMN,
            "attention.k_proj": TensorParallelismMode.COLUMN,
            "attention.v_proj": TensorParallelismMode.COLUMN,
            "attention.o_proj": TensorParallelismMode.ROW,
            "mlp.gate_proj": TensorParallelismMode.COLUMN,
            "mlp.up_proj": TensorParallelismMode.COLUMN,
            "mlp.down_proj": TensorParallelismMode.ROW,
        })


# ---------------------------------------------------------------------------
# Pipeline Parallelism — 1F1B Schedule
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    """
    One pipeline stage containing a subset of transformer layers.

    With PP=40 and 160 layers:
        Stage 0: Layers [0,  3]  → embedding + first 4 layers
        Stage 1: Layers [4,  7]
        ...
        Stage 39: Layers [156, 159] → last 4 layers + LM head
    """
    stage_id: int
    start_layer: int
    end_layer: int  # exclusive
    device_id: int  # which GPU this stage runs on

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer


class PipelineSchedule(str, Enum):
    """Pipeline execution schedule."""
    NAIVE = "naive"           # Fill → Drain (high bubble)
    ONE_F_ONE_B = "1f1b"     # Interleaved forward-backward
    INTERLEAVED = "interleaved"  # Megatron's interleaved
    ZERO_BUBBLE = "zero_bubble"  # Qi et al. 2024


@dataclass
class PipelineConfig:
    """
    Pipeline parallelism configuration.

    Bubble ratio:
        Naive:       (PP - 1) / total_microbatches ≈ 55% bubble (unusable)
        1F1B:        (PP - 1) / total_microbatches ≈ 6% bubble
        Interleaved: (PP - 1) / (chunks × total_microbatches) ≈ 2% bubble
        Zero-bubble: ~0% bubble (uses point-to-point scheduling)

    For PP=40, M=32:
        Naive:    39/32    = 122% → invalid (more bubble than compute)
        1F1B:     39/32    = ~6%  (with high M, bubble scales down)
        Interl:   39/(4×32) ≈ 0.3% (with 4 chunks per stage)
    """
    schedule: PipelineSchedule = PipelineSchedule.INTERLEAVED
    num_chunks: int = 4  # interleaved chunks per stage
    send_recv_overlap: bool = True  # overlap p2p with compute


def build_pipeline_stages(
    num_layers: int,
    pp_degree: int,
    tp_degree: int = 1,
) -> List[PipelineStage]:
    """
    Partition transformer layers across pipeline stages.

    Strategy: balanced partitioning (equal layers per stage).
    First and last stages may have slightly different compute
    due to embedding/LM head.
    """
    layers_per_stage = num_layers // pp_degree
    remainder = num_layers % pp_degree

    stages = []
    current_layer = 0
    for stage_id in range(pp_degree):
        # Distribute remainder layers to first N stages
        n_layers = layers_per_stage + (1 if stage_id < remainder else 0)
        stages.append(PipelineStage(
            stage_id=stage_id,
            start_layer=current_layer,
            end_layer=current_layer + n_layers,
            device_id=stage_id * tp_degree,  # first GPU in TP group
        ))
        current_layer += n_layers

    return stages


# ---------------------------------------------------------------------------
# Expert Parallelism — MoE Expert Placement
# ---------------------------------------------------------------------------

@dataclass
class ExpertPlacement:
    """
    Mapping of experts to EP ranks.

    With 128 experts and EP=16:
        Each EP rank holds 128/16 = 8 experts.

    Token routing requires all-to-all communication:
        - Each GPU sends tokens to the EP rank holding the target expert
        - After expert computation, results sent back (all-to-all)

    Communication cost per layer:
        C = 2 × B × S × D × (EP-1)/EP   (two all-to-all ops)

    For B=1, S=8192, D=16384, EP=16:
        C ≈ 2 × 1 × 8192 × 16384 × 15/16 × 2 bytes (BF16)
        ≈ 503 MB per layer  (×160 layers = ~80 GB per step)
    """
    expert_to_rank: Dict[int, int] = field(default_factory=dict)
    rank_to_experts: Dict[int, List[int]] = field(default_factory=dict)

    @staticmethod
    def uniform(
        num_experts: int = 128, ep_degree: int = 16
    ) -> "ExpertPlacement":
        """Uniformly distribute experts across EP ranks."""
        experts_per_rank = num_experts // ep_degree
        expert_to_rank = {}
        rank_to_experts = {}

        for rank in range(ep_degree):
            start = rank * experts_per_rank
            end = start + experts_per_rank
            experts = list(range(start, end))
            rank_to_experts[rank] = experts
            for e in experts:
                expert_to_rank[e] = rank

        return ExpertPlacement(
            expert_to_rank=expert_to_rank,
            rank_to_experts=rank_to_experts,
        )


# ---------------------------------------------------------------------------
# Context Parallelism — Ring Attention for 1M Context
# ---------------------------------------------------------------------------

@dataclass
class ContextParallelConfig:
    """
    Ring Attention configuration for ultra-long sequences.

    With a 1M-token context and CP=4:
        Each CP rank processes 250K contiguous tokens.
        KV blocks rotate around the ring so every rank attends
        to every position over CP steps.

    Memory savings:
        Without CP: O(S²) attention → 1M² = 1T elements (impossible)
        With CP=4:  O((S/CP)²) per rank → 250K² = 62.5B elements
                    With FlashAttention: tiled, fits in SRAM

    Communication pattern:
        Step t: rank i sends KV[i] to rank (i+1) % CP
                                                → ring topology

    References:
        - Ring Attention: Liu et al. 2023
        - Striped Attention: Brandon et al. 2023
    """
    cp_degree: int = 4
    ring_topology: str = "ring"          # "ring" or "allgather"
    overlap_comm_compute: bool = True     # overlap KV send with current compute
    chunk_size: int = 65536              # tokens per ring step


# ---------------------------------------------------------------------------
# Communication Estimator
# ---------------------------------------------------------------------------

def estimate_communication_volume(
    config: ParallelismConfig,
    hidden_size: int = 16384,
    num_layers: int = 160,
    batch_tokens: int = 8_000_000,  # global tokens per step
    dtype_bytes: int = 2,           # BF16
) -> Dict[str, float]:
    """
    Estimate communication volume per training step (in GB).

    Major communication patterns:
        1. TP all-reduce : 2 × 2D × B_local / TP  per layer
        2. PP p2p send   : activation_size per micro-batch transition
        3. EP all-to-all : 2 × B × S × D per MoE layer
        4. DP all-reduce : parameter gradient all-reduce (with ZeRO)
        5. CP ring send  : KV cache per ring step
    """
    TP = config.tensor_parallel
    PP = config.pipeline_parallel
    DP = config.data_parallel
    EP = config.expert_parallel
    D = hidden_size
    L = num_layers

    # Tokens per DP rank
    tokens_per_dp = batch_tokens // DP

    # 1. TP all-reduce: 2D per token, 2× per layer (after attn + after FFN)
    # Ring all-reduce: 2 × (TP-1)/TP × message_size
    tp_volume_per_layer = 2 * 2 * D * tokens_per_dp * dtype_bytes * (TP - 1) / TP
    tp_total = tp_volume_per_layer * L

    # 2. PP p2p: activation tensor per micro-batch boundary
    pp_activation_size = tokens_per_dp * D * dtype_bytes  # per micro-batch
    pp_total = pp_activation_size * config.num_micro_batches * 2  # fwd + bwd

    # 3. EP all-to-all: tokens × D, twice per MoE layer
    ep_volume_per_layer = 2 * tokens_per_dp * D * dtype_bytes * (EP - 1) / EP
    ep_total = ep_volume_per_layer * L

    # 4. DP gradient all-reduce (with ZeRO-3: only 1/DP of params)
    total_params_bytes = 2_000_000_000_000 * dtype_bytes  # 2T × 2B
    dp_total = total_params_bytes / DP  # each rank all-reduces its shard

    # 5. CP ring: KV cache per ring step
    kv_heads = 16
    head_dim = 128
    seq_per_cp = tokens_per_dp // config.context_parallel
    cp_kv_size = 2 * kv_heads * head_dim * seq_per_cp * dtype_bytes * L
    cp_total = cp_kv_size * (config.context_parallel - 1)

    return {
        "tp_allreduce_gb": tp_total / 1e9,
        "pp_p2p_gb": pp_total / 1e9,
        "ep_alltoall_gb": ep_total / 1e9,
        "dp_allreduce_gb": dp_total / 1e9,
        "cp_ring_gb": cp_total / 1e9,
        "total_gb": (tp_total + pp_total + ep_total + dp_total + cp_total) / 1e9,
        "total_gpus": config.total_gpus,
    }


# ---------------------------------------------------------------------------
# Memory Estimator
# ---------------------------------------------------------------------------

def estimate_memory_per_gpu(
    config: ParallelismConfig,
    total_params: int = 2_000_000_000_000,
    hidden_size: int = 16384,
    num_layers: int = 160,
    seq_len: int = 8192,
    micro_batch_size: int = 1,
    dtype_bytes: int = 2,
) -> Dict[str, float]:
    """
    Estimate GPU memory usage per device.

    Memory components:
        1. Model parameters (sharded)
        2. Optimizer states (sharded by ZeRO stage)
        3. Gradients (sharded by ZeRO stage)
        4. Activations (with gradient checkpointing)
        5. Communication buffers

    Target: fit within 80 GB (H100).
    """
    TP = config.tensor_parallel
    PP = config.pipeline_parallel
    DP = config.data_parallel
    EP = config.expert_parallel
    N = config.total_gpus

    # 1. Model parameters
    # Active params per token ≈ 330B (Opus 4.6 MoE)
    # But all params must be held for gradient computation
    params_per_gpu = total_params / N * dtype_bytes
    params_gb = params_per_gpu / 1e9

    # 2. Optimizer states (AdamW: 2 states per param)
    # ZeRO-3: optimizer states sharded across DP group
    if config.zero_stage >= 1:
        opt_per_gpu = total_params * 8 / N  # 8 bytes: 2×FP32 states
    else:
        opt_per_gpu = total_params * 8 / TP  # only TP sharded
    opt_gb = opt_per_gpu / 1e9

    # 3. Gradients
    if config.zero_stage >= 2:
        grad_per_gpu = total_params * dtype_bytes / N
    else:
        grad_per_gpu = total_params * dtype_bytes / (TP * PP)
    grad_gb = grad_per_gpu / 1e9

    # 4. Activations (with gradient checkpointing)
    # With full recompute, only store 2 × hidden_size per layer boundary
    layers_per_pp = num_layers // PP
    if config.activation_checkpointing:
        # Only layer boundaries stored: L/PP × activation_size
        act_per_gpu = layers_per_pp * micro_batch_size * seq_len * hidden_size * dtype_bytes
    else:
        # Full activations (much more — ~10× higher)
        act_per_gpu = layers_per_pp * micro_batch_size * seq_len * hidden_size * dtype_bytes * 12
    act_gb = act_per_gpu / 1e9

    # 5. Communication buffers (~2 GB typical)
    buffer_gb = 2.0

    total_gb = params_gb + opt_gb + grad_gb + act_gb + buffer_gb

    return {
        "params_gb": params_gb,
        "optimizer_gb": opt_gb,
        "gradients_gb": grad_gb,
        "activations_gb": act_gb,
        "buffers_gb": buffer_gb,
        "total_gb": total_gb,
        "fits_in_80gb": total_gb <= 80.0,
    }


# ---------------------------------------------------------------------------
# Training FLOP Estimator
# ---------------------------------------------------------------------------

def estimate_training_flops(
    total_params: int = 2_000_000_000_000,
    active_params: int = 330_000_000_000,
    total_tokens: int = 30_000_000_000_000,
    overhead_factor: float = 1.1,
) -> Dict[str, float]:
    """
    Estimate total FLOPs for training.

    Kaplan scaling law approximation:
        FLOPs ≈ 6 × N_active × T  (forward + backward)

    For Opus 4.6:
        FLOPs ≈ 6 × 330B × 30T = 5.94 × 10^25 ≈ 6 × 10^25

    Training time at cluster efficiency:
        H100 peak:   989 TFLOPS (BF16)
        MFU:         ~35–45%
        Effective:   ~395 TFLOPS/GPU

        Time = FLOPs / (N_GPUs × effective_throughput)
             = 6e25 / (20,480 × 395e12)
             = 6e25 / 8.1e18
             ≈ 7.4M seconds ≈ 86 days
    """
    # Total FLOPs
    total_flops = 6 * active_params * total_tokens * overhead_factor

    # Cluster throughput
    num_gpus = 20_480
    peak_tflops = 989   # H100 BF16 peak
    mfu = 0.40          # Model FLOPs Utilization
    effective_per_gpu = peak_tflops * 1e12 * mfu
    cluster_flops = num_gpus * effective_per_gpu

    training_seconds = total_flops / cluster_flops
    training_days = training_seconds / 86400

    # Cost estimate
    cost_per_gpu_hour = 3.50  # approximate H100 rental $/hr
    total_gpu_hours = num_gpus * training_seconds / 3600
    total_cost = total_gpu_hours * cost_per_gpu_hour

    return {
        "total_flops": total_flops,
        "total_flops_str": f"{total_flops:.2e}",
        "cluster_gpus": num_gpus,
        "mfu": mfu,
        "effective_tflops_per_gpu": effective_per_gpu / 1e12,
        "training_seconds": training_seconds,
        "training_days": training_days,
        "estimated_cost_usd": total_cost,
        "estimated_cost_str": f"${total_cost / 1e6:.1f}M",
    }
