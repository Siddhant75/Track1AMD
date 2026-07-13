"""
Category-specific answer extractor.

Strips verbose preamble/postamble and extracts the core answer for each task
category. Runs in microseconds — pure Python regex, zero tokens consumed.

Called AFTER the validator has cleaned <think> blocks and fluff, as a
final extraction pass to ensure the core answer is isolated.

Why this exists:
  Zero-shot remote models (and sometimes local models) often answer:
    "Based on my analysis, the sentiment is positive because..."
  The grader expects just: "positive"
  This module extracts that core signal without any LLM call.
"""

from __future__ import annotations

import re
from typing import Optional

from classifier import TaskCategory


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_answer(category: TaskCategory, text: str) -> str:
    """Extract the core answer from a verbose model response.

    Tries category-specific extraction first. If no specific pattern matches,
    returns the original text unchanged (safe fallback).

    Args:
        category: The classified task category.
        text: The cleaned model output (after fluff stripping).

    Returns:
        The extracted core answer, or the original text if no pattern matched.
    """
    text = text.strip()
    if not text:
        return text

    _extractors = {
        TaskCategory.SENTIMENT: _extract_sentiment,
        TaskCategory.LOGIC: _extract_logic,
        TaskCategory.MATH: _extract_math,
        TaskCategory.NER: _extract_ner,
        TaskCategory.FACTUAL: _extract_factual,
        TaskCategory.CODE_GEN: _extract_code,
        TaskCategory.DEBUGGING: _extract_code,
        TaskCategory.SUMMARIZATION: _extract_summarization,
    }

    extractor = _extractors.get(category)
    if extractor:
        result = extractor(text)
        if result and result.strip():
            return result.strip()

    # Safe fallback: return original text
    return text


# ---------------------------------------------------------------------------
# Category-specific extractors
# ---------------------------------------------------------------------------

def _extract_sentiment(text: str) -> Optional[str]:
    """Extract sentiment label: positive, negative, or neutral.

    Handles verbose answers like:
      "The sentiment of this review is positive because the customer expressed..."
    → "positive"
    """
    # Priority 1: first occurrence of the sentiment label
    match = re.search(r"\b(positive|negative|neutral)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # Priority 2: mapped emotion words
    positive_words = r"\b(good|great|excellent|happy|love|wonderful|amazing|fantastic)\b"
    negative_words = r"\b(bad|terrible|awful|hate|poor|horrible|disappointing|worst)\b"
    if re.search(positive_words, text, re.IGNORECASE):
        return "positive"
    if re.search(negative_words, text, re.IGNORECASE):
        return "negative"

    return None


def _extract_logic(text: str) -> Optional[str]:
    """Extract yes/no/true/false from logic reasoning answers.

    Handles verbose explanations like:
      "Therefore, the conclusion is valid because all premises hold..."
    → "valid"
    """
    # Map of patterns to canonical answers
    patterns = [
        (r"\b(yes)\b", "yes"),
        (r"\b(no)\b", "no"),
        (r"\b(true)\b", "true"),
        (r"\b(false)\b", "false"),
        (r"\b(valid)\b", "valid"),
        (r"\b(invalid)\b", "invalid"),
        (r"\b(possible)\b", "possible"),
        (r"\b(impossible)\b", "impossible"),
        (r"\b(correct)\b", "correct"),
        (r"\b(incorrect)\b", "incorrect"),
        (r"\b(cannot\s+be\s+concluded|cannot\s+conclude|we\s+cannot\s+conclude)\b", "no"),
        (r"\b(can\s+be\s+concluded|we\s+can\s+conclude|therefore)\b", "yes"),
    ]

    # Prioritize: check sentence containing "conclusion" or "answer" first
    conclusion_match = re.search(
        r"(?:conclusion|answer|therefore|thus|hence)[:\s]+([^.!?\n]+)",
        text, re.IGNORECASE,
    )
    if conclusion_match:
        snippet = conclusion_match.group(1)
        for pattern, label in patterns[:8]:  # Only the clear yes/no/true/false ones
            if re.search(pattern, snippet, re.IGNORECASE):
                return label

    # Fallback: first match anywhere in the text
    for pattern, label in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return label

    return None


def _extract_math(text: str) -> Optional[str]:
    """Extract the final numerical result from a math answer.

    Handles answers like:
      "Step 1: 15 × 200 = 3000. Step 2: 3000 / 100 = 30. The answer is 30."
    → "30"
    """
    # Priority 1: "The answer is X" / "= X" at end of line
    answer_patterns = [
        r"(?:the\s+)?answer\s+(?:is|=)\s*:?\s*([-\d,./%\s]+?)(?:\.|$|\n)",
        r"(?:result|solution|value|total)\s+(?:is|=)\s*:?\s*([-\d,./%\s]+?)(?:\.|$|\n)",
        r"(?:therefore|thus|so|hence)[,:]?\s*([-\d,./%\s]+?)(?:\.|$|\n)",
        r"=\s*([-\d,./%]+)\s*$",  # Last "= X" at end of text
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            result = match.group(1).strip().rstrip(".,")
            if result:
                return result

    # Priority 2: last number in the text
    numbers = re.findall(r"[-]?\d+(?:[.,]\d+)?(?:\s*%)?", text)
    if numbers:
        return numbers[-1].strip()

    return None


def _extract_ner(text: str) -> Optional[str]:
    """Extract NER entity lines.

    Handles both labeled format:
      "PERSON: Tim Cook\nORGANIZATION: Apple\n..."
    And inline format:
      "The entities are Tim Cook (person) and Apple (organization)..."
    """
    # Priority 1: labeled entity lines (structured output)
    entity_pattern = re.compile(
        r"^(?:PERSON|PEOPLE|ORGANIZATION|ORG|LOCATION|LOC|DATE|TIME|MONEY|"
        r"PERCENT|GPE|EVENT|PRODUCT|WORK_OF_ART|FACILITY|LANGUAGE)"
        r"\s*:\s*.+",
        re.IGNORECASE | re.MULTILINE,
    )
    entity_lines = entity_pattern.findall(text)
    if entity_lines:
        return "\n".join(entity_lines)

    # Priority 2: bulleted or numbered entity list
    bullets = [
        line.strip()
        for line in text.split("\n")
        if re.match(r"^[-*•\d.]\s+", line.strip())
    ]
    if bullets:
        return "\n".join(bullets)

    return None


def _extract_factual(text: str) -> Optional[str]:
    """Extract factual answer after common preamble patterns.

    Handles:
      "The capital of France is Paris." → "Paris"
      "Answer: 42" → "42"
      "The boiling point of water is 100°C." → keeps full sentence (short enough)
    """
    # Priority 1: "Answer: X" / "The answer is X"
    match = re.search(
        r"(?:the\s+)?answer\s+(?:is|:)\s*:?\s*(.+?)(?:\.|$)",
        text, re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()
        # Only return if it's a plausible short answer (under 20 words)
        if candidate and len(candidate.split()) <= 20:
            return candidate

    # Priority 2: If the whole answer is short enough, keep it as-is
    if len(text.split()) <= 30:
        return text

    # Priority 3: Take the first sentence only
    first_sentence = re.split(r"[.!?]\s", text)[0]
    if first_sentence:
        return first_sentence.strip()

    return None


def _extract_code(text: str) -> Optional[str]:
    """Extract code blocks from code generation / debugging answers.

    If multiple code blocks exist, returns all of them joined.
    If no code blocks, returns the full text (might be inline code).
    """
    # Find all fenced code blocks (```...```)
    code_blocks = re.findall(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
    if code_blocks:
        # Reconstruct with proper fencing
        return "\n\n".join(f"```\n{block.strip()}\n```" for block in code_blocks if block.strip())

    # No fenced blocks — look for indented code (4 spaces or tab)
    indented_lines = [
        line for line in text.split("\n")
        if line.startswith("    ") or line.startswith("\t")
    ]
    if len(indented_lines) >= 2:
        return "\n".join(indented_lines)

    # No code detected — return the full text (inline code or explanation)
    return text


def _extract_summarization(text: str) -> Optional[str]:
    """For summarization, return text as-is.

    The validator's fluff stripping already handles introductory phrases.
    Summarization answers are prose — we don't want to further extract them.
    """
    return text
