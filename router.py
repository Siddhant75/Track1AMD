"""
Smart routing engine for Track 1 hybrid inference.

Decides whether a task should be handled locally (0 tokens) or escalated
to the remote Fireworks API (minimal tokens).  The routing decision is
based on three signals:

  1. Category confidence — some categories are inherently safe for local.
  2. Prompt complexity and constraints — negative constraints force DeepSeek.
  3. Local model confidence — logprob score from the first local attempt.

The goal: clear the 80%+ accuracy gate while minimizing remote token usage.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple, Literal

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
    TaskCategory.SENTIMENT: RoutingTier.LOCAL_FIRST, # Changed to LOCAL_FIRST for strict constraints
    TaskCategory.NER: RoutingTier.LOCAL_FIRST,
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
    TaskCategory.SENTIMENT: -2.0,
    TaskCategory.NER: -2.0,
    TaskCategory.FACTUAL: -1.5,
    TaskCategory.SUMMARIZATION: -1.5,
    TaskCategory.MATH: -1.0,           # Strict — math errors are costly
    TaskCategory.LOGIC: -1.0,
    TaskCategory.DEBUGGING: -1.0,
    TaskCategory.CODE_GEN: -1.0,
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

_CONSTRAINT_INDICATORS = [
    r"\bexactly\b.*\b(?:sentences?|words?|bullets?|points?|paragraphs?)\b",
    r"\b(?:no|not)\s+(?:more|longer)\s+than\b",
    r"\bunder\b.*\bwords?\b",
    r"\bone-sentence\b",
    r"\bat\s+most\b",
    r"\bmust\s+be\b",
    r"\bmaximum\b",
    r"\bminimum\b",
]


def has_constraints(prompt: str) -> bool:
    """Check if the prompt contains strict formatting constraints."""
    prompt_lower = prompt.lower()
    for pattern in _CONSTRAINT_INDICATORS:
        if re.search(pattern, prompt_lower):
            return True
    return False

def select_local_model(prompt: str) -> Literal["gemma", "deepseek"]:
    """Select the best local model for the task.
    
    DeepSeek handles constraints (via <think> loop).
    Gemma handles everything else (faster, better factual knowledge).
    """
    if has_constraints(prompt):
        return "deepseek"
    return "gemma"

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
            
    # Constraint indicators severely bump complexity
    for pattern in _CONSTRAINT_INDICATORS:
        if re.search(pattern, prompt_lower):
            score += 0.3

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
    validator_passed: bool = True,
) -> Tuple[bool, str]:
    """Decide whether to escalate a task to the remote API.

    Args:
        category: The classified task category.
        prompt: The original prompt text.
        local_answer: The local model's generated answer.
        confidence: The average logprob from the local model.
        critic_passed: Whether the rule-based critic validated the output.
        validator_passed: Whether the Python constraint validator passed.

    Returns:
        Tuple of (should_escalate: bool, reason: str).
    """
    tier = get_routing_tier(category)

    # If python validator forcefully rejected the output, ALWAYS escalate
    if not validator_passed:
        return True, "validator_rejected (failed strict constraints)"

    # If critic rejected the output, strongly consider escalation
    if not critic_passed:
        return True, f"critic_rejected (conf={confidence:.2f})"

    # Check confidence against category-specific threshold
    threshold = get_confidence_threshold(category)
    if confidence < threshold:
        return True, f"low_confidence ({confidence:.2f} < {threshold})"
        
    complexity = assess_prompt_complexity(prompt)
    if complexity > 0.4 and confidence < (threshold + 0.5):
        return True, f"high_complexity_moderate_confidence (c={complexity:.2f}, conf={confidence:.2f})"

    # Check for suspiciously short answers on complex tasks
    if category in (TaskCategory.MATH, TaskCategory.LOGIC, TaskCategory.CODE_GEN, TaskCategory.DEBUGGING):
        if len(local_answer.strip()) < 30:
            return True, f"answer_too_short ({len(local_answer)} chars)"

    return False, "confident_local"


