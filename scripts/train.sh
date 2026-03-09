#!/bin/bash
# ============================================================
# Claude Opus 4.6 — Distributed Training Launch Script
# ============================================================
#
# Launches training on 32,768 GPUs using torchrun (elastic).
#
# Infrastructure:
#   GPU Count:       32,768 NVIDIA H100 80GB
#   Interconnect:    NVLink + NVSwitch + InfiniBand
#   Training Time:   ~90 days
#   Total FLOPs:     ~3.6 × 10^26
#
# Parallelism:
#   Data Parallel:    512 groups (ZeRO Stage 3 / FSDP)
#   Tensor Parallel:  8 GPUs (within node, NVLink)
#   Pipeline Parallel: 8 stages (across nodes)
#   Expert Parallel:  128 GPUs (one expert per GPU)
#
# Batch size:
#   B_micro × N_accum × N_DP = 2 × 16 × 512 = 16,384 seq/step
#   At 8192 tokens/seq: ~134M tokens per gradient step
#
# Usage:
#   sbatch scripts/train.sh             # SLURM cluster
#   bash scripts/train.sh               # Single node (testing)
# ============================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────
CONFIG_FILE="configs/opus_4_6.yaml"
NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

echo "============================================"
echo "Claude Opus 4.6 Training"
echo "============================================"
echo "Nodes:         ${NUM_NODES}"
echo "GPUs/node:     ${GPUS_PER_NODE}"
echo "Total GPUs:    ${TOTAL_GPUS}"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "Config:        ${CONFIG_FILE}"
echo "============================================"

# ── Environment ───────────────────────────────────────────
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Enable TF32 for faster matmuls on H100
export NVIDIA_TF32_OVERRIDE=1

# ── Launch Training ───────────────────────────────────────
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NUM_NODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m src.training.trainer \
    --config ${CONFIG_FILE}

echo "Training complete."
