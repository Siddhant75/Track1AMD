"""
Smart routing engine for Track 1 hybrid inference.

Decides whether a task should be handled locally (0 tokens) or escalated
to the remote Fireworks API (minimal tokens).  The routing decision is
based on three signals:

  1. Category confidence — some categories are inherently safe for local.
  2. Prompt complexity — length, multi-step indicators, domain difficulty.
  3. Local model confidence — logprob score from the first local attempt.

The goal: clear the 80%+ accuracy gate while minimizing remote token usage.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

from classifier import TaskCategory, ComplexityTier

# ---------------------------------------------------------------------------
# Routing tiers
# ---------------------------------------------------------------------------


class RoutingTier:
    """Defines how a task should be processed."""
    LOCAL_ONLY = "local_only"         # Never escalate (sentiment, NER)
    LOCAL_FIRST = "local_first"       # Try local, escalate if uncertain
    REMOTE_PREFERRED = "remote_preferred"  # Prefer remote for accuracy


# ---------------------------------------------------------------------------
# Category → default routing tier
# ---------------------------------------------------------------------------

_CATEGORY_ROUTING: Dict[TaskCategory, str] = {
    TaskCategory.SENTIMENT: RoutingTier.LOCAL_ONLY,
    TaskCategory.NER: RoutingTier.LOCAL_ONLY,
    TaskCategory.FACTUAL: RoutingTier.LOCAL_FIRST,
    TaskCategory.SUMMARIZATION: RoutingTier.LOCAL_FIRST,
    TaskCategory.MATH: RoutingTier.LOCAL_FIRST,
    TaskCategory.LOGIC: RoutingTier.LOCAL_FIRST,
    TaskCategory.DEBUGGING: RoutingTier.LOCAL_FIRST,
    TaskCategory.CODE_GEN: RoutingTier.LOCAL_FIRST,
}

# ---------------------------------------------------------------------------
# Confidence thresholds per category
# More lenient for easy categories, stricter for hard ones
# ---------------------------------------------------------------------------

_CONFIDENCE_THRESHOLDS: Dict[TaskCategory, float] = {
    TaskCategory.SENTIMENT: -3.0,      # Very lenient — almost always local
    TaskCategory.NER: -3.0,
    TaskCategory.FACTUAL: -1.8,
    TaskCategory.SUMMARIZATION: -2.0,
    TaskCategory.MATH: -1.2,           # Strict — math errors are costly
    TaskCategory.LOGIC: -1.2,
    TaskCategory.DEBUGGING: -1.3,
    TaskCategory.CODE_GEN: -1.3,
}

# ---------------------------------------------------------------------------
# Category → remote API task_type mapping (for model selection)
# ---------------------------------------------------------------------------

_CATEGORY_TO_TASK_TYPE: Dict[TaskCategory, str] = {
    TaskCategory.SENTIMENT: "general",
    TaskCategory.NER: "general",
    TaskCategory.FACTUAL: "general",
    TaskCategory.SUMMARIZATION: "general",
    TaskCategory.MATH: "reasoning",
    TaskCategory.LOGIC: "reasoning",
    TaskCategory.DEBUGGING: "code",
    TaskCategory.CODE_GEN: "code",
}

# ---------------------------------------------------------------------------
# Category-adaptive temperature settings
# ---------------------------------------------------------------------------

_CATEGORY_TEMPERATURE: Dict[TaskCategory, float] = {
    TaskCategory.SENTIMENT: 0.0,       # Deterministic classification
    TaskCategory.NER: 0.0,             # Deterministic extraction
    TaskCategory.FACTUAL: 0.0,         # Deterministic recall
    TaskCategory.SUMMARIZATION: 0.1,   # Slight creativity
    TaskCategory.MATH: 0.0,            # Deterministic computation
    TaskCategory.LOGIC: 0.0,           # Deterministic reasoning
    TaskCategory.DEBUGGING: 0.1,       # Slight variation
    TaskCategory.CODE_GEN: 0.1,        # Slight variation for better code
}

# ---------------------------------------------------------------------------
# Prompt complexity signals that suggest the local model may struggle
# ---------------------------------------------------------------------------

_COMPLEXITY_INDICATORS = [
    r"\bstep\s+by\s+step\b",
    r"\bmulti-?step\b",
    r"\bfirst\b.*\bthen\b.*\bfinally\b",
    r"\b(?:compare|contrast|analyze|evaluate)\b.*\b(?:and|versus|vs)\b",
    r"\b(?:write|implement)\b.*\b(?:complete|full|entire|comprehensive)\b",
    r"\b(?:explain|describe)\b.*\b(?:detail|depth|thorough)\b",
    r"\bprove\b",
    r"\bderive\b",
    r"\b(?:optimize|refactor)\b.*\b(?:code|algorithm|function)\b",
]


def get_routing_tier(category: TaskCategory) -> str:
    """Get the default routing tier for a category."""
    return _CATEGORY_ROUTING.get(category, RoutingTier.LOCAL_FIRST)


def get_confidence_threshold(category: TaskCategory) -> float:
    """Get the logprob confidence threshold for a category."""
    return _CONFIDENCE_THRESHOLDS.get(category, -1.5)


def get_temperature(category: TaskCategory) -> float:
    """Get the optimal temperature for a category."""
    return _CATEGORY_TEMPERATURE.get(category, 0.1)


def get_task_type(category: TaskCategory) -> str:
    """Get the remote API task type for model selection."""
    return _CATEGORY_TO_TASK_TYPE.get(category, "general")


def assess_prompt_complexity(prompt: str) -> float:
    """Score the complexity of a prompt (0.0 = simple, 1.0 = very complex).

    Uses heuristics like length, indicator words, and structural signals.
    """
    score = 0.0

    # Length-based (longer prompts tend to be harder)
    word_count = len(prompt.split())
    if word_count > 200:
        score += 0.3
    elif word_count > 100:
        score += 0.15
    elif word_count > 50:
        score += 0.05

    # Complexity indicator patterns
    prompt_lower = prompt.lower()
    for pattern in _COMPLEXITY_INDICATORS:
        if re.search(pattern, prompt_lower):
            score += 0.15

    # Multiple question marks = multiple sub-questions
    question_count = prompt.count("?")
    if question_count > 2:
        score += 0.2
    elif question_count > 1:
        score += 0.1

    # Code blocks in prompt (debugging/code tasks with lots of code)
    code_block_count = prompt.count("```")
    if code_block_count >= 2:
        score += 0.15

    return min(score, 1.0)


def should_escalate(
    category: TaskCategory,
    prompt: str,
    local_answer: str,
    confidence: float,
    critic_passed: bool,
) -> Tuple[bool, str]:
    """Decide whether to escalate a task to the remote API.

    Args:
        category: The classified task category.
        prompt: The original prompt text.
        local_answer: The local model's generated answer.
        confidence: The average logprob from the local model.
        critic_passed: Whether the rule-based critic validated the output.

    Returns:
        Tuple of (should_escalate: bool, reason: str).
    """
    tier = get_routing_tier(category)

    # Tier 1: LOCAL_ONLY — never escalate
    if tier == RoutingTier.LOCAL_ONLY:
        return False, "local_only_category"

    # If critic rejected the output, strongly consider escalation
    if not critic_passed:
        return True, f"critic_rejected (conf={confidence:.2f})"

    # Check confidence against category-specific threshold
    threshold = get_confidence_threshold(category)
    if confidence < threshold:
        # Also factor in prompt complexity
        complexity = assess_prompt_complexity(prompt)
        if complexity > 0.3 or confidence < (threshold - 0.5):
            return True, f"low_confidence ({confidence:.2f} < {threshold}, complexity={complexity:.2f})"

    # Check for suspiciously short answers on complex tasks
    if category in (TaskCategory.MATH, TaskCategory.LOGIC, TaskCategory.CODE_GEN, TaskCategory.DEBUGGING):
        if len(local_answer.strip()) < 30:
            return True, f"answer_too_short ({len(local_answer)} chars)"

    return False, "confident_local"


def build_verification_prompt(
    category: TaskCategory,
    original_prompt: str,
    local_answer: str,
) -> str:
    """Build a compressed verification prompt for remote escalation.

    Instead of sending the full prompt cold, we include the local model's
    draft answer. This uses fewer input tokens because the remote model
    only needs to verify/correct rather than generate from scratch.

    Args:
        category: The task category.
        original_prompt: The full original prompt.
        local_answer: The local model's best attempt.

    Returns:
        A compressed prompt string for the remote API.
    """
    if category in (TaskCategory.MATH, TaskCategory.LOGIC):
        return (
            f"Verify and correct this answer if wrong. Show your work.\n\n"
            f"Question: {original_prompt}\n\n"
            f"Draft answer: {local_answer}\n\n"
            f"Provide the correct final answer."
        )

    if category in (TaskCategory.CODE_GEN, TaskCategory.DEBUGGING):
        return (
            f"Review and fix this code response if needed.\n\n"
            f"Original request: {original_prompt}\n\n"
            f"Draft response: {local_answer}\n\n"
            f"Provide the corrected, working code."
        )

    # General fallback — just ask the question directly
    return original_prompt
