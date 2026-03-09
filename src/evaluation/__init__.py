"""
Claude Opus 4.6 — Evaluation

Benchmarks, Arena Elo, Needle-in-a-Haystack, contamination detection,
and sycophancy evaluation.
"""

from .benchmarks import (
    BenchmarkResult,
    EloCalculator,
    NeedleInAHaystackTest,
    ContaminationDetector,
    SycophancyEvaluator,
    BENCHMARK_TARGETS,
    run_evaluation_suite,
)

__all__ = [
    "BenchmarkResult",
    "EloCalculator",
    "NeedleInAHaystackTest",
    "ContaminationDetector",
    "SycophancyEvaluator",
    "BENCHMARK_TARGETS",
    "run_evaluation_suite",
]
