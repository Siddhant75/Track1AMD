"""
Output Validator and Auto-Corrector for strict grading constraints.

Two responsibilities:
  1. Strip <think> blocks from DeepSeek local model outputs.
  2. Strip introductory/concluding "fluff" from zero-shot remote model outputs
     so that format constraints (exact words, bullets, sentences) pass cleanly.

If an output cannot be auto-corrected, returns (False, answer) to signal
that the critic should escalate this to the next tier.
"""

import re
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_sentences(text: str) -> int:
    """Rough heuristic for counting sentences."""
    sentences = re.split(r'[.!?](?:\s+|$)', text.strip())
    return len([s for s in sentences if s.strip()])


def _get_sentences(text: str) -> List[str]:
    """Get list of sentences, keeping their trailing punctuation."""
    return re.findall(r'[^.!?]+[.!?]*', text)


def _count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def _strip_fluff(text: str) -> str:
    """Remove common LLM introductory/concluding fluff patterns.

    Zero-shot remote models often add phrases like:
      - "Here is the answer: ..."
      - "Sure! ..."
      - "Of course, ..."
      - "Here are the bullet points: ..."
    This function strips those leading and trailing patterns to expose
    the core answer for clean constraint validation.
    """
    text = text.strip()

    # Strip common introductory phrases (greedy match to end of first line)
    intro_patterns = [
        r"^(?:here(?:'s| is| are)(?: the| a| my)?[^:\n]*:?\s*)\n?",
        r"^(?:sure[!,.]?\s*)",
        r"^(?:of course[!,.]?\s*)",
        r"^(?:certainly[!,.]?\s*)",
        r"^(?:absolutely[!,.]?\s*)",
        r"^(?:great[!,.]?\s*)",
        r"^(?:the answer is:?\s*)",
        r"^(?:answer:?\s*)",
        r"^(?:result:?\s*)",
        r"^(?:response:?\s*)",
        r"^(?:below (?:is|are)[^:\n]*:?\s*)\n?",
        r"^(?:as requested[,.]?\s*)",
    ]
    for pattern in intro_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # Strip common concluding fluff on the last line
    outro_patterns = [
        r"\n\s*(?:i hope (?:this|that) helps?[!.]?)\s*$",
        r"\n\s*(?:let me know if (?:you|there)[^.!?\n]*[.!?]?)\s*$",
        r"\n\s*(?:feel free to[^.!?\n]*[.!?]?)\s*$",
        r"\n\s*(?:please (?:let|note|remember)[^.!?\n]*[.!?]?)\s*$",
    ]
    for pattern in outro_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    return text


def _extract_bullets(text: str) -> List[str]:
    """Extract bullet point lines from a text block.
    Handles -, *, •, and numbered lists (1. 2. 3.).
    """
    bullets = []
    for line in text.split("\n"):
        line = line.strip()
        if (
            line.startswith("-")
            or line.startswith("*")
            or line.startswith("•")
            or re.match(r"^\d+[.)]\s", line)
        ):
            bullets.append(line)
    return bullets


# ---------------------------------------------------------------------------
# Main validation + correction entry point
# ---------------------------------------------------------------------------

def validate_and_correct(prompt: str, local_answer: str) -> Tuple[bool, str]:
    """Validate the answer against constraints in the prompt.

    This function handles both local model outputs (with <think> tags)
    and remote model outputs (with introductory/concluding fluff).

    Returns:
        Tuple of (is_valid: bool, corrected_answer: str).
    """
    prompt_lower = prompt.lower()
    answer_stripped = local_answer.strip()

    # ------------------------------------------------------------------
    # Step 1: Remove <think> blocks (DeepSeek local model)
    # ------------------------------------------------------------------
    if "</think>" in answer_stripped:
        # DeepSeek sometimes omits the opening <think> tag but includes closing
        answer_cleaned = answer_stripped.split("</think>")[-1].strip()
    elif "<think>" in answer_stripped:
        # Opening tag but no closing = answer is truncated mid-thought
        answer_cleaned = ""
    else:
        answer_cleaned = answer_stripped

    # ------------------------------------------------------------------
    # Step 2: Strip introductory/concluding fluff from remote model answers
    # ------------------------------------------------------------------
    answer_cleaned = _strip_fluff(answer_cleaned)

    # If we've stripped everything into nothing, return the original cleaned
    if not answer_cleaned:
        return False, local_answer.strip()

    # ------------------------------------------------------------------
    # Step 3: Enforce "exactly N sentences"
    # ------------------------------------------------------------------
    sentence_match = re.search(
        r'\bexactly\s+(one|two|three|four|five|1|2|3|4|5)\s+sentences?\b',
        prompt_lower,
    )
    if not sentence_match:
        sentence_match = re.search(
            r'\b(one|two|three|four|five|1|2|3|4|5)-sentence\b',
            prompt_lower,
        )

    if sentence_match:
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
        }
        target_sentences = word_to_num[sentence_match.group(1)]
        current_sentences = _count_sentences(answer_cleaned)

        if current_sentences > target_sentences:
            sents = [s.strip() for s in _get_sentences(answer_cleaned) if s.strip()]
            if len(sents) >= target_sentences:
                corrected = " ".join(sents[:target_sentences])
                return True, corrected
        elif current_sentences < target_sentences:
            return False, answer_cleaned

    # ------------------------------------------------------------------
    # Step 4: Enforce word limits ("no more than N words", "under N words")
    # ------------------------------------------------------------------
    word_limit_match = re.search(
        r'(?:no longer than|no more than|under|at most|maximum)\s+(\d+)\s+words',
        prompt_lower,
    )
    if word_limit_match:
        limit = int(word_limit_match.group(1))
        if "bullet" in prompt_lower:
            bullets = _extract_bullets(answer_cleaned)
            if bullets:
                for b in bullets:
                    if _count_words(b) > limit:
                        return False, answer_cleaned
        else:
            if _count_words(answer_cleaned) > limit:
                # Attempt aggressive truncation: keep only first `limit` words
                words = answer_cleaned.split()
                return True, " ".join(words[:limit])

    # ------------------------------------------------------------------
    # Step 5: Enforce "exactly N words"
    # ------------------------------------------------------------------
    exact_word_match = re.search(
        r'\bexactly\s+(one|two|three|four|five|\d+)\s+words?\b',
        prompt_lower,
    )
    if exact_word_match:
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        }
        raw = exact_word_match.group(1)
        target_words = word_to_num.get(raw, int(raw) if raw.isdigit() else None)

        if target_words is not None:
            current_words = _count_words(answer_cleaned)
            if current_words != target_words:
                if current_words > target_words:
                    words = [w for w in re.split(r"\s+", answer_cleaned) if w]
                    if len(words) >= target_words:
                        # For "exactly 1 word": prefer the last substantive word
                        # (models often lead with "The answer is: <word>")
                        if target_words == 1:
                            # Try to find the last non-stop-word token
                            candidates = [
                                w.strip(".,!?;:\"'") for w in words if w.strip(".,!?;:\"'")
                            ]
                            if candidates:
                                return True, candidates[-1]
                        return True, " ".join(words[:target_words])
                return False, answer_cleaned

    # ------------------------------------------------------------------
    # Step 6: Enforce "exactly N bullet points"
    # ------------------------------------------------------------------
    bullet_match = re.search(
        r'\bexactly\s+(one|two|three|four|five|\d+)\s+bullet\s+points?\b',
        prompt_lower,
    )
    if not bullet_match:
        bullet_match = re.search(
            r'\b(one|two|three|four|five|\d+)\s+bullet\s+points?\b',
            prompt_lower,
        )

    if bullet_match:
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        }
        raw = bullet_match.group(1)
        target_bullets = word_to_num.get(raw, int(raw) if raw.isdigit() else None)

        if target_bullets is not None:
            bullets = _extract_bullets(answer_cleaned)

            if not bullets:
                # No bullets found — the model answered in prose. Fail so remote handles it.
                return False, answer_cleaned

            if len(bullets) > target_bullets:
                corrected = "\n".join(bullets[:target_bullets])
                return True, corrected
            elif len(bullets) < target_bullets:
                return False, answer_cleaned

    return True, answer_cleaned
