"""
Claude Opus 4.6 — Distributed Training & Parallelism

Tensor, pipeline, data, expert, and context parallelism configurations.
Memory estimators, communication volume calculators, and FLOP estimators
for 20,480-GPU training clusters.
"""

from .parallelism import (
    ParallelismConfig,
    TensorParallelismMode,
    TensorParallelPlan,
    PipelineStage,
    PipelineSchedule,
    PipelineConfig,
    ExpertPlacement,
    ContextParallelConfig,
    build_pipeline_stages,
    estimate_communication_volume,
    estimate_memory_per_gpu,
    estimate_training_flops,
)

__all__ = [
    "ParallelismConfig",
    "TensorParallelismMode",
    "TensorParallelPlan",
    "PipelineStage",
    "PipelineSchedule",
    "PipelineConfig",
    "ExpertPlacement",
    "ContextParallelConfig",
    "build_pipeline_stages",
    "estimate_communication_volume",
    "estimate_memory_per_gpu",
    "estimate_training_flops",
]
