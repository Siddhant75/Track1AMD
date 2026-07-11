"""
Rule-based local critic for validating model outputs.

Runs fast Python checks against the local model's response to catch
obviously broken or off-format answers *before* they are written to
results.json.  If the critic rejects an output, the orchestrator can
retry with a tweaked prompt instead of wasting remote API tokens.
"""

from __future__ import annotations

import re
from typing import Tuple

from classifier import TaskCategory


def validate_output(
    category: TaskCategory,
    prompt: str,
    output: str,
) -> Tuple[bool, str]:
    """Validate a model output against category-specific rules.

    Args:
        category: The classified category of the task.
        prompt: The original user prompt (used for length comparisons).
        output: The model's generated answer.

    Returns:
        A tuple of (is_valid, reason).
    """
    # ------------------------------------------------------------------
    # Universal checks
    # ------------------------------------------------------------------
    if not output or not output.strip():
        return False, "Empty output"

    output_stripped = output.strip()

    # ------------------------------------------------------------------
    # Category-specific validators
    # ------------------------------------------------------------------

    if category == TaskCategory.SENTIMENT:
        output_lower = output_stripped.lower()
        valid_labels = ["positive", "negative", "neutral"]
        if any(label in output_lower for label in valid_labels):
            return True, "Contains valid sentiment label"
        return False, "No valid sentiment label (positive/negative/neutral) found"

    if category == TaskCategory.NER:
        # NER output should mention at least some entity or explicitly say none
        if len(output_stripped) >= 2:
            return True, "NER output present"
        return False, "NER output too short"

    if category == TaskCategory.SUMMARIZATION:
        # Summary should be shorter than the input and non-trivial
        if len(output_stripped) < 10:
            return False, "Summary too short (< 10 chars)"
        if len(output_stripped) > len(prompt) * 2:
            # Allow some tolerance — summaries can sometimes be longer if
            # the prompt itself is very short
            return True, "Summary present (length check relaxed)"
        return True, "Valid summary"

    if category == TaskCategory.MATH:
        # Should contain a number or a clear answer indicator
        if re.search(r"-?\d+\.?\d*", output_stripped):
            return True, "Contains numeric answer"
        answer_keywords = ["answer", "result", "solution", "equals", "=", "is"]
        if any(kw in output_stripped.lower() for kw in answer_keywords):
            return True, "Contains answer keyword"
        return False, "No numeric answer or answer keyword found"

    if category == TaskCategory.CODE_GEN:
        code_indicators = [
            "def ", "class ", "function ", "import ", "return ",
            "print(", "console.", "var ", "let ", "const ",
            "for ", "while ", "if ", "{", "}", "=>",
            "```",
        ]
        if any(indicator in output_stripped for indicator in code_indicators):
            return True, "Contains code content"
        return False, "No code-like content found"

    if category == TaskCategory.DEBUGGING:
        # Should contain an explanation AND corrected code
        if len(output_stripped) < 20:
            return False, "Debugging response too short (< 20 chars)"
        # Check for at least some code-like content
        code_indicators = ["def ", "class ", "function ", "return ", "print(", "```"]
        has_code = any(ind in output_stripped for ind in code_indicators)
        has_explanation = len(output_stripped) > 50
        if has_code or has_explanation:
            return True, "Contains debugging response"
        return False, "Response lacks code or explanation"

    if category == TaskCategory.LOGIC:
        if len(output_stripped) < 10:
            return False, "Logic response too short (< 10 chars)"
        return True, "Contains reasoning"

    if category == TaskCategory.FACTUAL:
        if len(output_stripped) < 2:
            return False, "Factual answer too short"
        return True, "Contains answer"

    # Default: accept anything non-empty
    return True, "Default pass"
