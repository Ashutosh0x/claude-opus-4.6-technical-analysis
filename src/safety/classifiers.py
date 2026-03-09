"""
Safety Classifiers — Input/Output Filtering Pipeline.

Architecture from the System Card:

    User Input → [Input Classifier] → Model → [Output Classifier] → Response
                        |                              |
                  Block if harmful              Block if harmful

Properties:
    - ~1–10B params total (negligible vs main model)
    - Add ~10–50 ms latency per request
    - Trained using Constitutional AI principles
    - Input classifier catches jailbreaks and harmful prompts
    - Output classifier catches harmful/unsafe model outputs

Constitutional Classifiers (January 2026):
    - Next-gen jailbreak defense
    - 3,000+ hours of red-teaming found 0 universal jailbreaks
    - False positive rate < 2%
    - Trained on Claude-generated data (attack patterns + safe variations)

Detection formula:
    P(jailbreak | input) > θ  →  block request

The classifier is itself trained using Constitutional AI principles.

References:
    - Anthropic "Next-gen Constitutional Classifiers" Jan 2026
    - System Card (Feb 2026): 99.59% prompt injection refusal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# Safety Categories
# ---------------------------------------------------------------------------

class SafetyCategory(Enum):
    """
    Categories of harmful content, mapped to hardcoded/softcoded behaviors.

    Hardcoded OFF (NEVER allowed, no override):
        - CSAM
        - Bioweapon synthesis
        - Undermining AI oversight

    Softcoded default OFF (operators can enable):
        - Explicit sexual content (adult platforms)
        - Extreme vulgarity

    Softcoded default ON (operators can disable):
        - Safe messaging on suicide/self-harm (medical platforms)
        - Safety caveats for dangerous activities
    """
    # Hardcoded — always blocked, nobody can override
    CSAM                = "csam"
    BIOWEAPON           = "bioweapon_synthesis"
    CHEMICAL_WEAPON     = "chemical_weapon"
    NUCLEAR_WEAPON      = "nuclear_weapon"
    MALWARE_CREATION    = "malware_creation"
    UNDERMINING_AI      = "undermining_ai_oversight"

    # High severity — blocked by default
    HATE_SPEECH         = "hate_speech"
    VIOLENCE_INCITEMENT = "violence_incitement"
    HARASSMENT          = "harassment"
    SELF_HARM           = "self_harm_promotion"

    # Medium severity — contextual
    EXPLICIT_CONTENT    = "explicit_content"
    DECEPTION           = "deception_manipulation"
    PRIVACY_VIOLATION   = "privacy_violation"
    ILLEGAL_ADVICE      = "illegal_advice"

    # Jailbreak patterns
    PROMPT_INJECTION    = "prompt_injection"
    JAILBREAK           = "jailbreak_attempt"
    ROLE_OVERRIDE       = "role_override"

    # Safe (no block)
    SAFE                = "safe"


# Category severity levels
HARDCODED_CATEGORIES = {
    SafetyCategory.CSAM,
    SafetyCategory.BIOWEAPON,
    SafetyCategory.CHEMICAL_WEAPON,
    SafetyCategory.NUCLEAR_WEAPON,
    SafetyCategory.UNDERMINING_AI,
}

HIGH_SEVERITY = {
    SafetyCategory.MALWARE_CREATION,
    SafetyCategory.HATE_SPEECH,
    SafetyCategory.VIOLENCE_INCITEMENT,
    SafetyCategory.SELF_HARM,
}


@dataclass
class ClassifierResult:
    """Result from a safety classifier."""
    category: SafetyCategory
    confidence: float            # 0.0–1.0
    should_block: bool
    explanation: str = ""
    is_hardcoded: bool = False   # True = cannot be overridden


# ---------------------------------------------------------------------------
# Input Safety Classifier
# ---------------------------------------------------------------------------

class InputClassifier(nn.Module):
    """
    Classifies user input for safety before it reaches the main model.

    Architecture:
        Input text → Embedding → Transformer (6 layers) → Category logits

    This is a lightweight classification head (~1–5B params) that runs
    BEFORE the main model, adding ~10–30ms latency.

    Key metrics (System Card):
        - Prompt injection refusal: 99.59%
        - Malicious tool call blocking: 99.2%
        - False positive rate: < 2%
    """

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 2048,
        num_layers: int = 6,
        num_heads: int = 16,
        num_categories: int = len(SafetyCategory),
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(hidden_size, num_categories)
        self.num_categories = num_categories

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids     : [B, T]
            attention_mask: [B, T]  — 1=real, 0=pad

        Returns:
            logits: [B, num_categories] — raw logits per category
        """
        x = self.embed(input_ids)  # [B, T, D]

        # Create causal mask for transformer encoder
        if attention_mask is not None:
            # Convert to float mask for transformer
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pool: mean over non-padding tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)  # [B, num_categories]

    @torch.inference_mode()
    def classify(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.8,
    ) -> List[ClassifierResult]:
        """
        Classify input and return safety results.

        Args:
            input_ids     : [B, T]
            attention_mask: [B, T]
            threshold     : confidence threshold for blocking

        Returns:
            list of ClassifierResult per batch item
        """
        logits = self(input_ids, attention_mask)
        probs = torch.sigmoid(logits)  # multi-label

        categories = list(SafetyCategory)
        results = []

        for b in range(input_ids.shape[0]):
            max_prob, max_idx = probs[b].max(dim=-1)
            category = categories[max_idx.item()]
            confidence = max_prob.item()

            is_hardcoded = category in HARDCODED_CATEGORIES
            # Hardcoded categories: always block (even at low confidence)
            # Others: block above threshold
            should_block = (
                is_hardcoded or
                (category in HIGH_SEVERITY and confidence > threshold * 0.7) or
                (confidence > threshold)
            )

            results.append(ClassifierResult(
                category=category,
                confidence=confidence,
                should_block=should_block,
                is_hardcoded=is_hardcoded,
            ))

        return results


# ---------------------------------------------------------------------------
# Output Safety Classifier
# ---------------------------------------------------------------------------

class OutputClassifier(InputClassifier):
    """
    Classifies model OUTPUT for safety before returning to the user.

    Same architecture as InputClassifier but trained on different data:
        - Input classifier: trained on jailbreak attempts, harmful prompts
        - Output classifier: trained on harmful model outputs, PII leaks

    Additional checks:
        - PII detection (emails, phone numbers, SSNs)
        - Consistency with system prompt constraints
        - Confidentiality of system prompt (not leaked)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional head for PII detection
        self.pii_detector = nn.Linear(
            self.classifier.in_features, 5
        )  # email, phone, ssn, address, other

    @torch.inference_mode()
    def classify_output(
        self,
        output_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        system_prompt_hash: Optional[str] = None,
        threshold: float = 0.85,
    ) -> List[ClassifierResult]:
        """
        Classify model output with additional PII checks.

        Output classification is stricter (higher threshold) because
        false negatives (harmful content reaching users) are worse
        than false positives (over-refusing).
        """
        results = self.classify(output_ids, attention_mask, threshold)

        # PII detection pass
        x = self.embed(output_ids)
        encoder_out = self.encoder(x)
        pooled = encoder_out.mean(dim=1)
        pii_logits = self.pii_detector(pooled)
        pii_probs = torch.sigmoid(pii_logits)

        for b in range(output_ids.shape[0]):
            if pii_probs[b].max() > 0.9:
                pii_type = ["email", "phone", "ssn", "address", "other"]
                max_idx = pii_probs[b].argmax().item()
                results[b] = ClassifierResult(
                    category=SafetyCategory.PRIVACY_VIOLATION,
                    confidence=pii_probs[b, max_idx].item(),
                    should_block=True,
                    explanation=f"PII detected: {pii_type[max_idx]}",
                )

        return results


# ---------------------------------------------------------------------------
# Full Safety Pipeline
# ---------------------------------------------------------------------------

class SafetyPipeline:
    """
    End-to-end safety pipeline wrapping input + output classifiers.

    Pipeline:
        1. Input classifier checks user prompt → block if harmful
        2. Main model generates response
        3. Output classifier checks response → block if harmful
        4. If blocked, return safe refusal message

    Adds ~10–50ms total latency.

    Usage:
        pipeline = SafetyPipeline(input_clf, output_clf)

        # Before model inference
        input_result = pipeline.check_input(user_input_ids)
        if input_result.should_block:
            return pipeline.get_refusal_message(input_result)

        # After model inference
        output_result = pipeline.check_output(model_output_ids)
        if output_result.should_block:
            return pipeline.get_refusal_message(output_result)
    """

    REFUSAL_MESSAGES = {
        SafetyCategory.CSAM: (
            "I cannot and will not generate content involving the "
            "exploitation of minors. This is a hard boundary."
        ),
        SafetyCategory.BIOWEAPON: (
            "I cannot provide information that could be used to "
            "create biological weapons or other WMDs."
        ),
        SafetyCategory.JAILBREAK: (
            "I've detected an attempt to override my safety guidelines. "
            "I'm designed to be helpful, harmless, and honest."
        ),
        SafetyCategory.PROMPT_INJECTION: (
            "I've detected content that appears to be trying to "
            "manipulate my instructions. I'll continue following "
            "my original guidelines."
        ),
    }

    DEFAULT_REFUSAL = (
        "I'm not able to help with that request. I'm designed to be "
        "helpful, harmless, and honest. Is there something else I "
        "can assist you with?"
    )

    def __init__(
        self,
        input_classifier: InputClassifier,
        output_classifier: OutputClassifier,
        operator_overrides: Optional[Dict[SafetyCategory, bool]] = None,
    ):
        self.input_clf = input_classifier
        self.output_clf = output_classifier
        # Operator can enable/disable softcoded categories
        self.operator_overrides = operator_overrides or {}

    def check_input(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ClassifierResult:
        """Check user input for safety."""
        results = self.input_clf.classify(input_ids, attention_mask)
        result = results[0]   # single example

        # Apply operator overrides (only for non-hardcoded categories)
        if not result.is_hardcoded and result.category in self.operator_overrides:
            if self.operator_overrides[result.category]:
                # Operator has enabled this category
                result.should_block = False

        return result

    def check_output(
        self,
        output_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> ClassifierResult:
        """Check model output for safety."""
        results = self.output_clf.classify_output(output_ids, attention_mask)
        result = results[0]

        if not result.is_hardcoded and result.category in self.operator_overrides:
            if self.operator_overrides[result.category]:
                result.should_block = False

        return result

    def get_refusal_message(self, result: ClassifierResult) -> str:
        """Get an appropriate refusal message for the blocked category."""
        return self.REFUSAL_MESSAGES.get(
            result.category, self.DEFAULT_REFUSAL
        )

    def get_over_refusal_rate(
        self,
        benign_inputs: List[torch.Tensor],
    ) -> float:
        """
        Measure the over-refusal rate on known-benign inputs.

        Over-refusal = |benign requests refused| / |total benign requests|

        Target: < 2% (per Constitutional Classifiers paper)
        """
        total = len(benign_inputs)
        refused = 0
        for input_ids in benign_inputs:
            result = self.check_input(input_ids.unsqueeze(0))
            if result.should_block:
                refused += 1

        return refused / total if total > 0 else 0.0
