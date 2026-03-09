"""
Claude Opus 4.6 — Safety

Multi-head safety classifiers, content filtering, and jailbreak detection.
Covers CBRN, CSAM, deception, violence, and prompt-injection categories.
"""

from .classifiers import (
    SafetyCategory,
    ClassifierResult,
    InputClassifier,
    OutputClassifier,
    SafetyPipeline,
)
from .watermarking import (
    WatermarkLogitsProcessor,
    WatermarkDetector,
    WatermarkResult,
)

__all__ = [
    "SafetyCategory",
    "ClassifierResult",
    "InputClassifier",
    "OutputClassifier",
    "SafetyPipeline",
    # Watermarking
    "WatermarkLogitsProcessor",
    "WatermarkDetector",
    "WatermarkResult",
]
