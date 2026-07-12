"""
Lightweight keyword/regex-based task classifier.

Maps each prompt to one of 8 categories and assigns a complexity tier
(LOW = zero-shot local, HIGH = few-shot local) without any ML inference.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class TaskCategory(str, Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"
    DEBUGGING = "debugging"
    LOGIC = "logic"
    CODE_GEN = "code_gen"


class ComplexityTier(str, Enum):
    LOW = "low"
    HIGH = "high"


# Maps each category to its complexity tier
CATEGORY_COMPLEXITY: dict[TaskCategory, ComplexityTier] = {
    TaskCategory.FACTUAL: ComplexityTier.LOW,
    TaskCategory.SENTIMENT: ComplexityTier.HIGH,
    TaskCategory.SUMMARIZATION: ComplexityTier.HIGH,
    TaskCategory.NER: ComplexityTier.HIGH,
    TaskCategory.MATH: ComplexityTier.HIGH,
    TaskCategory.DEBUGGING: ComplexityTier.HIGH,
    TaskCategory.LOGIC: ComplexityTier.HIGH,
    TaskCategory.CODE_GEN: ComplexityTier.HIGH,
}

# ---------------------------------------------------------------------------
# Pattern banks – order matters (more specific categories checked first)
# ---------------------------------------------------------------------------

_DEBUGGING_PATTERNS = [
    r"\b(debug|fix\s+(?:the|this|my)|error|bug|traceback|exception|fault|crash|wrong\s+output)\b",
    r"what['\u2019]?s?\s+wrong",
    r"find\s+the\s+(bug|error|issue|problem)",
    r"correct\s+the\s+(code|program|function|script)",
    r"fix\s+(?:the\s+)?(?:following|below|this)\s+code",
]

_CODE_GEN_PATTERNS = [
    r"\b(write|create|generate|implement|build)\s+(?:a\s+|the\s+|me\s+)?(?:python|javascript|java|c\+\+|typescript|rust|go|ruby|code|function|program|script|class|method)\b",
    r"\b(?:code|function|program|script|class)\s+(?:that|which|to)\b",
    r"\bimplement\b.*\b(?:algorithm|solution|interface|api)\b",
]

_SENTIMENT_PATTERNS = [
    r"\bsentiment\b",
    r"\b(?:analyze|determine|classify|identify|detect)\b.*\b(?:tone|sentiment|feeling|emotion|mood|opinion)\b",
    r"\b(?:positive|negative|neutral)\s+(?:sentiment|tone|feeling)\b",
    r"\bis\s+(?:the|this)\s+(?:text|review|sentence|statement|comment)\s+(?:positive|negative|neutral)\b",
]

_NER_PATTERNS = [
    r"\bnamed\s+entit",
    r"\bner\b",
    r"\b(?:extract|identify|find|list|recognize)\b.*\b(?:names|entities|people|persons|locations|places|organizations|companies)\b",
    r"\b(?:entity|entities)\s+(?:extraction|recognition|tagging)\b",
]

_SUMMARIZATION_PATTERNS = [
    r"\b(?:summarize|summarise|summary|condense|shorten|recap|paraphrase)\b",
    r"\btl;?dr\b",
    r"\bgive\s+(?:a\s+)?(?:brief|short|concise)\s+(?:overview|summary|description)\b",
    r"\bin\s+(?:a\s+)?(?:few|one|two|three)\s+(?:words|sentences?|lines?)\b",
]

_MATH_PATTERNS = [
    r"\b(?:calculate|compute|solve|evaluate|simplify)\b",
    r"\b(?:math|equation|arithmetic|algebra|geometry|calculus|trigonometry)\b",
    r"\b(?:integral|derivative|limit|logarithm|factorial|percentage|ratio|fraction)\b",
    r"\d+\s*[\+\-\*\/\^%]\s*\d+",
    r"\b(?:what\s+is|find|evaluate)\b.*\d+\s*[\+\-\*\/\^%]",
    r"\b(?:sum|product|difference|quotient|remainder|square\s+root)\s+of\b",
    r"\bhow\s+(?:much|many)\b.*\b(?:total|sum|left|remain|cost|price)\b",
]

_LOGIC_PATTERNS = [
    r"\b(?:logic|logical)\b",
    r"\b(?:reasoning|deduce|deduction|infer|inference)\b",
    r"\b(?:puzzle|riddle|brain\s*teaser)\b",
    r"\bif\s+.{5,}?\s+then\b",
    r"\b(?:premise|conclusion|syllogism|paradox|fallacy)\b",
    r"\b(?:true|false)\b.*\b(?:statement|claim|assertion|proposition)\b",
    r"\b(?:must\s+be|necessarily|follows\s+that|therefore|thus|hence)\b",
    r"\b(?:can\s+we|could\s+we)\s+(?:conclude|infer|deduce|say|determine)\b",
    r"\b(?:all|every|each|some|no|none)\s+\w+\s+(?:are|is|have|has)\b.*\b(?:all|every|some|no|can)\b",
    r"\b(?:valid|invalid)\s+(?:argument|reasoning|conclusion|deduction)\b",
    r"\b(?:what\s+can\s+(?:be|we)\s+(?:concluded|inferred|deduced))\b",
]

# Ordered pattern groups: checked from most to least specific
_PATTERN_GROUPS: list[tuple[TaskCategory, list[str]]] = [
    (TaskCategory.DEBUGGING, _DEBUGGING_PATTERNS),
    (TaskCategory.CODE_GEN, _CODE_GEN_PATTERNS),
    (TaskCategory.SENTIMENT, _SENTIMENT_PATTERNS),
    (TaskCategory.NER, _NER_PATTERNS),
    (TaskCategory.SUMMARIZATION, _SUMMARIZATION_PATTERNS),
    (TaskCategory.MATH, _MATH_PATTERNS),
    (TaskCategory.LOGIC, _LOGIC_PATTERNS),
]


def classify(prompt: str, category_hint: Optional[str] = None) -> TaskCategory:
    """Classify a task prompt into one of the 8 categories.

    If the evaluation harness provides a *category_hint*, it is respected first.
    Otherwise, regex-based heuristics are applied in priority order.
    Falls back to FACTUAL if no patterns match.
    """
    # ------------------------------------------------------------------
    # 1. Respect an explicit category hint from the harness
    # ------------------------------------------------------------------
    if category_hint:
        hint = category_hint.lower().strip()
        for cat in TaskCategory:
            if cat.value in hint or hint in cat.value:
                return cat
        # Loose matching for common synonyms
        _HINT_MAP = {
            "code": TaskCategory.CODE_GEN,
            "coding": TaskCategory.CODE_GEN,
            "programming": TaskCategory.CODE_GEN,
            "generation": TaskCategory.CODE_GEN,
            "debug": TaskCategory.DEBUGGING,
            "fix": TaskCategory.DEBUGGING,
            "mathematics": TaskCategory.MATH,
            "calculation": TaskCategory.MATH,
            "entity": TaskCategory.NER,
            "extraction": TaskCategory.NER,
            "summary": TaskCategory.SUMMARIZATION,
            "opinion": TaskCategory.SENTIMENT,
            "emotion": TaskCategory.SENTIMENT,
            "reason": TaskCategory.LOGIC,
            "reasoning": TaskCategory.LOGIC,
            "knowledge": TaskCategory.FACTUAL,
            "trivia": TaskCategory.FACTUAL,
            "qa": TaskCategory.FACTUAL,
        }
        for key, cat in _HINT_MAP.items():
            if key in hint:
                return cat

    # ------------------------------------------------------------------
    # 2. Regex-based classification on the prompt text
    # ------------------------------------------------------------------
    prompt_lower = prompt.lower()
    for category, patterns in _PATTERN_GROUPS:
        for pattern in patterns:
            if re.search(pattern, prompt_lower):
                return category

    # ------------------------------------------------------------------
    # 3. Fallback: treat unknown prompts as factual Q&A
    # ------------------------------------------------------------------
    return TaskCategory.FACTUAL


def get_complexity(category: TaskCategory) -> ComplexityTier:
    """Return the complexity tier for a given category."""
    return CATEGORY_COMPLEXITY[category]
