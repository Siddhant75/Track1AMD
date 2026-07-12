"""
Output Validator and Auto-Corrector for strict grading constraints.

Checks local model outputs against constraints detected in the prompt.
Can auto-correct minor formatting errors (e.g., stripping extra sentences).
If an output fails a strict constraint and cannot be corrected, it returns False.
"""

import re
from typing import Tuple

def _count_sentences(text: str) -> int:
    """Rough heuristic for counting sentences."""
    # Split by period, exclamation, or question mark followed by space or end
    sentences = re.split(r'[.!?](?:\s+|$)', text.strip())
    return len([s for s in sentences if s.strip()])

def _get_sentences(text: str) -> list[str]:
    """Get list of sentences, keeping punctuation."""
    # Find all sentences with their punctuation
    return re.findall(r'[^.!?]+[.!?]*', text)

def _count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())

def validate_and_correct(prompt: str, local_answer: str) -> Tuple[bool, str]:
    """
    Validate the local answer against constraints in the prompt.
    Returns (is_valid, corrected_answer).
    """
    prompt_lower = prompt.lower()
    answer_stripped = local_answer.strip()
    
    # Remove <think> blocks before counting/validating
    if '</think>' in answer_stripped:
        # DeepSeek sometimes omits the opening <think> tag but includes the closing tag
        answer_cleaned = answer_stripped.split('</think>')[-1].strip()
    else:
        answer_cleaned = re.sub(r'<think>.*?</think>', '', answer_stripped, flags=re.DOTALL).strip()
        # Fallback if only opening tag exists
        answer_cleaned = re.sub(r'<think>.*', '', answer_cleaned, flags=re.DOTALL).strip()
    
    # 1. Exactly N sentences
    sentence_match = re.search(r'\bexactly\s+(one|two|three|four|five|1|2|3|4|5)\s+sentences?\b', prompt_lower)
    if not sentence_match:
        sentence_match = re.search(r'\b(one|two|three|four|five|1|2|3|4|5)-sentence\b', prompt_lower)
        
    if sentence_match:
        word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                       '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        target_sentences = word_to_num[sentence_match.group(1)]
        
        current_sentences = _count_sentences(answer_cleaned)
        
        if current_sentences > target_sentences:
            # Try to auto-correct by taking only the first N sentences
            sents = [s.strip() for s in _get_sentences(answer_cleaned) if s.strip()]
            if len(sents) >= target_sentences:
                corrected = " ".join(sents[:target_sentences])
                return True, corrected
        elif current_sentences < target_sentences:
            # Cannot auto-correct missing sentences
            return False, answer_cleaned
            
    # 2. Word count limits (e.g., "no longer than 15 words" or "under 15 words")
    word_limit_match = re.search(r'(?:no longer than|no more than|under|at most|maximum)\s+(\d+)\s+words', prompt_lower)
    if word_limit_match:
        limit = int(word_limit_match.group(1))
        
        # We need to check if the overall answer is under the limit, OR if it's bullet points,
        # each bullet point is under the limit.
        if "bullet" in prompt_lower:
            bullets = [line for line in answer_cleaned.split('\n') if line.strip().startswith('-') or line.strip().startswith('*')]
            if bullets:
                for b in bullets:
                    if _count_words(b) > limit:
                        # Auto-correct is risky here, better to fail and escalate
                        return False, answer_cleaned
        else:
            if _count_words(answer_cleaned) > limit:
                # Too long. Could truncate, but might lose meaning. Fail and escalate.
                return False, answer_cleaned
                
    # 2.5 Exactly N words
    exact_word_match = re.search(r'\bexactly\s+(one|two|three|four|five|1|2|3|4|5)\s+words?\b', prompt_lower)
    if exact_word_match:
        word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                       '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        target_words = word_to_num[exact_word_match.group(1)]
        current_words = _count_words(answer_cleaned)
        if current_words != target_words:
            # Maybe it contains punctuation that makes it hard to count? 
            # If it's a short response, let's just fail and let Fireworks handle it
            # if we can't trivially extract the exact word(s)
            
            # Auto-correction: if current_words > target_words, try taking the first N words
            if current_words > target_words:
                words = [w for w in re.split(r'\s+', answer_cleaned) if w]
                if len(words) >= target_words:
                    corrected = " ".join(words[:target_words])
                    # Ensure we don't accidentally truncate a negative to a positive (e.g. "not happy" -> "not")
                    # Given the risk, if it's "exactly one word", taking the last word is sometimes better
                    if target_words == 1:
                        # Just fail, let remote handle it perfectly
                        return False, answer_cleaned
                    return True, corrected
            return False, answer_cleaned
                
    # 3. Exactly N bullet points
    bullet_match = re.search(r'\bexactly\s+(one|two|three|four|five|1|2|3|4|5)\s+bullet\s+points?\b', prompt_lower)
    if bullet_match:
        word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                       '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        target_bullets = word_to_num[bullet_match.group(1)]
        
        # Count lines starting with - or * or numbers
        bullets = [line.strip() for line in answer_cleaned.split('\n') 
                   if line.strip().startswith('-') or line.strip().startswith('*') or re.match(r'^\d+\.', line.strip())]
                   
        if len(bullets) > target_bullets:
            # Auto-correct by taking first N bullets
            corrected = "\n".join(bullets[:target_bullets])
            return True, corrected
        elif len(bullets) < target_bullets:
            return False, answer_cleaned
            
    return True, answer_cleaned
