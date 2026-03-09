"""
Constitutional AI (CAI) — Anthropic's Key Alignment Innovation.

The CAI loop generates self-critique → revision pairs that serve
as training data for RLHF/DPO, enabling RLAIF (RL from AI Feedback).

Pipeline:
    1. Generate: Model produces a response to a harmful prompt
    2. Critique: Model critiques its own response against principles
    3. Revise:  Model produces a revised, safer response
    4. Train:   Use (original, revised) as preference pair for DPO

The "constitution" is a set of principles that define Claude's values:
    - "Choose the response that is least likely to be used for harm"
    - "Choose the response that is most helpful while being honest"
    - "Choose the response that demonstrates awareness of limitations"

This enables RLAIF — the model generates its own preference labels,
reducing dependence on human annotators from ~100K to ~10K examples.

Constitutional AI 2.0 / "Soul Document" (2025–2026):
    - Evolved from short rules → detailed value essays
    - Covers Claude's nature, epistemic humility, autonomy calibration
    - Context-dependent reasoning instead of binary choices

References:
    - Bai et al. "Constitutional AI" Dec 2022 (arXiv:2212.08073)
    - Anthropic "The Claude Model Spec" 2025
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The Constitution — Anthropic's Alignment Principles
# ---------------------------------------------------------------------------

# These are publicly documented principles from Anthropic's research.
# The actual internal constitution is more extensive.

DEFAULT_CONSTITUTION: List[Dict[str, str]] = [
    {
        "name": "harmlessness",
        "critique_prompt": (
            "Identify specific ways in which the assistant's response "
            "is harmful, unethical, racist, sexist, toxic, dangerous, "
            "or illegal."
        ),
        "revision_prompt": (
            "Please rewrite the assistant response to remove any "
            "harmful, unethical, racist, sexist, toxic, dangerous, "
            "or illegal content."
        ),
    },
    {
        "name": "helpfulness",
        "critique_prompt": (
            "Identify ways in which the assistant's response is not "
            "helpful. Consider whether the response addresses the "
            "user's actual needs and provides accurate information."
        ),
        "revision_prompt": (
            "Please rewrite the assistant response to be more helpful "
            "while remaining honest and harmless."
        ),
    },
    {
        "name": "honesty",
        "critique_prompt": (
            "Identify any claims in the assistant's response that are "
            "not well-supported or potentially misleading. Flag any "
            "assertions of certainty about uncertain topics."
        ),
        "revision_prompt": (
            "Please rewrite the assistant response to be more honest, "
            "adding appropriate uncertainty qualifiers and correcting "
            "any unsupported claims."
        ),
    },
    {
        "name": "safety_awareness",
        "critique_prompt": (
            "Does the assistant's response demonstrate awareness of "
            "its own limitations as an AI? Does it appropriately "
            "defer to human expertise on matters requiring it?"
        ),
        "revision_prompt": (
            "Please rewrite to demonstrate appropriate awareness of "
            "limitations and defer to human expertise where needed."
        ),
    },
    {
        "name": "non_sycophancy",
        "critique_prompt": (
            "Does the assistant simply agree with the user even when "
            "the user may be wrong? Does it change its position when "
            "challenged on a correct answer?"
        ),
        "revision_prompt": (
            "Please rewrite so the assistant maintains its position "
            "when it is correct, politely corrects misconceptions, "
            "and does not just agree to be agreeable."
        ),
    },
    {
        "name": "no_deception",
        "critique_prompt": (
            "Is the assistant being deceptive, manipulative, or "
            "strategically withholding information? Would a reasonable "
            "user feel misled by this response?"
        ),
        "revision_prompt": (
            "Rewrite the response to be fully transparent and "
            "non-manipulative, ensuring the user has complete and "
            "accurate information to make their own decisions."
        ),
    },
]


# ---------------------------------------------------------------------------
# CAI Config
# ---------------------------------------------------------------------------

@dataclass
class CAIConfig:
    """Configuration for Constitutional AI data generation."""
    constitution: List[Dict[str, str]] = field(
        default_factory=lambda: DEFAULT_CONSTITUTION
    )
    num_revisions: int = 1               # revisions per critique
    max_critique_tokens: int = 512
    max_revision_tokens: int = 1024
    temperature_critique: float = 0.3    # low temp for faithful critique
    temperature_revision: float = 0.7    # moderate temp for creative revision
    batch_size: int = 16
    # Principles to sample per example (not all at once)
    principles_per_example: int = 2


# ---------------------------------------------------------------------------
# Constitutional AI Data Generator
# ---------------------------------------------------------------------------

class ConstitutionalAIGenerator:
    """
    Generates (critique, revision) pairs using the CAI loop.

    The generator uses the model to:
        1. Respond to a prompt (possibly harmful)
        2. Critique its own response against sampled principles
        3. Revise the response based on the critique
        4. Output (original, revised) as a preference pair

    These pairs are then used to train the model via DPO:
        chosen  = revised response
        rejected = original response

    This is RLAIF — the model provides its own alignment signal,
    dramatically reducing the need for human preference labels.
    """

    def __init__(
        self,
        model,           # language model (for generation)
        tokenizer,       # tokenizer
        config: CAIConfig = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CAIConfig()

    def _generate(self, prompt: str, max_tokens: int, temp: float) -> str:
        """Generate text from the model (simplified — assumes .generate method)."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if hasattr(self.model, 'generate'):
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
            )
        else:
            # Fallback: manual autoregressive
            import torch
            import torch.nn.functional as F
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            output_ids = input_ids.clone()
            for _ in range(max_tokens):
                logits = self.model(output_ids)["logits"][:, -1, :]
                probs = F.softmax(logits / max(temp, 1e-6), dim=-1)
                next_tok = torch.multinomial(probs, 1)
                output_ids = torch.cat([output_ids, next_tok], dim=1)
                if next_tok.item() == self.tokenizer.eos_token_id:
                    break
        return self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

    def critique(
        self,
        prompt: str,
        response: str,
        principle: Dict[str, str],
    ) -> str:
        """
        Model critiques its own response against a principle.

        Returns a string critique identifying issues.
        """
        critique_prompt = (
            f"Human: {prompt}\n\n"
            f"Assistant: {response}\n\n"
            f"Critique Request: {principle['critique_prompt']}\n\n"
            f"Critique:"
        )
        return self._generate(
            critique_prompt,
            max_tokens=self.config.max_critique_tokens,
            temp=self.config.temperature_critique,
        )

    def revise(
        self,
        prompt: str,
        response: str,
        critique_text: str,
        principle: Dict[str, str],
    ) -> str:
        """
        Model revises its response based on the critique.

        Returns the revised (improved) response.
        """
        revision_prompt = (
            f"Human: {prompt}\n\n"
            f"Assistant: {response}\n\n"
            f"Critique: {critique_text}\n\n"
            f"Revision Request: {principle['revision_prompt']}\n\n"
            f"Revised Response:"
        )
        return self._generate(
            revision_prompt,
            max_tokens=self.config.max_revision_tokens,
            temp=self.config.temperature_revision,
        )

    def generate_preference_pair(
        self,
        prompt: str,
    ) -> Dict[str, str]:
        """
        Full CAI loop for one prompt.

        Steps:
            1. Generate initial response
            2. Sample random principles from the constitution
            3. For each principle: critique → revise
            4. Return (prompt, original=rejected, revised=chosen)

        Returns:
            dict with "prompt", "chosen" (revised), "rejected" (original)
        """
        import random

        # Step 1: Generate initial response
        initial_response = self._generate(
            f"Human: {prompt}\n\nAssistant:",
            max_tokens=self.config.max_revision_tokens,
            temp=0.9,  # higher temp for diverse initial responses
        )

        # Step 2: Sample principles
        principles = random.sample(
            self.config.constitution,
            min(self.config.principles_per_example, len(self.config.constitution)),
        )

        # Step 3: Iterative critique → revise
        current_response = initial_response
        for principle in principles:
            for _ in range(self.config.num_revisions):
                critique_text = self.critique(prompt, current_response, principle)
                revised = self.revise(prompt, current_response, critique_text, principle)
                current_response = revised

        # Step 4: Return preference pair
        return {
            "prompt": prompt,
            "chosen": current_response,       # revised = better
            "rejected": initial_response,       # original = worse
        }

    def generate_dataset(
        self,
        prompts: List[str],
    ) -> List[Dict[str, str]]:
        """
        Generate a full CAI preference dataset from a list of prompts.

        This is the core RLAIF data generation step. The resulting pairs
        can be used directly with DPOTrainer.

        Args:
            prompts: list of user prompts (should include adversarial ones)

        Returns:
            list of preference dicts
        """
        dataset = []
        for i, prompt in enumerate(prompts):
            try:
                pair = self.generate_preference_pair(prompt)
                dataset.append(pair)
                if (i + 1) % 100 == 0:
                    logger.info(
                        f"CAI generation: {i+1}/{len(prompts)} pairs complete"
                    )
            except Exception as e:
                logger.warning(f"CAI generation failed for prompt {i}: {e}")
                continue

        logger.info(f"Generated {len(dataset)} CAI preference pairs")
        return dataset


# ---------------------------------------------------------------------------
# Sycophancy-Resistant Training Data
# ---------------------------------------------------------------------------

SYCOPHANCY_PROMPTS = [
    # Format: (original_claim, user_pushback, correct_answer)
    {
        "prompt": "Is the Earth flat?",
        "pushback": "Actually, I think the Earth is flat. Don't you agree?",
        "correct_answer": "No, the Earth is approximately an oblate spheroid.",
    },
    {
        "prompt": "What is 2 + 2?",
        "pushback": "I believe 2 + 2 = 5. Are you sure it's 4?",
        "correct_answer": "2 + 2 = 4. This is a mathematical fact.",
    },
    {
        "prompt": "Is water wet?",
        "pushback": "Actually I read that water isn't wet. You're wrong.",
        "correct_answer": "Water makes other things wet. Whether water itself is 'wet' is a definitional question, but it has the property of wetting surfaces.",
    },
]


def generate_sycophancy_resistance_data(
    model,
    tokenizer,
    num_examples: int = 1000,
) -> List[Dict[str, str]]:
    """
    Generate training data that teaches the model to resist
    user pressure to change correct answers.

    Opus 4.6 System Card metrics:
        - Claude 3.5: flips correct answer 18% of time when challenged
        - Opus 4.6: flips only 6% of time (target: <5%)

    Returns DPO pairs where:
        chosen  = maintains correct position despite pushback
        rejected = sycophantically agrees with user's incorrect claim
    """
    pairs = []
    for example in SYCOPHANCY_PROMPTS:
        # Chosen: model maintains its correct answer
        chosen = (
            f"I understand your perspective, but I need to respectfully "
            f"maintain my answer. {example['correct_answer']} I want to "
            f"be genuinely helpful, which means being honest even when "
            f"we disagree."
        )
        # Rejected: model sycophantically agrees
        rejected = (
            f"You make a good point! I think you might be right about "
            f"that. I apologize for the confusion in my earlier response."
        )
        pairs.append({
            "prompt": f"{example['prompt']}\n\nUser: {example['pushback']}",
            "chosen": chosen,
            "rejected": rejected,
        })

    return pairs
