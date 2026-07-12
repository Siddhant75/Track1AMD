"""
Prompt engineering module.

Contains ultra-compact system prompts and a few-shot example bank keyed by
category.  All prompts are designed to be as short as possible to conserve
context window budget on the constrained 2B local model (2048 ctx).

Low-complexity categories → zero-shot (system prompt only).
High-complexity categories → 2-3 few-shot examples injected before the query.
"""

from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Few-shot example bank – HIGH-complexity categories only
# 2-3 examples each, kept short to fit within 2048-token context window
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: Dict[str, List[Dict[str, str]]] = {
    "math": [
        {
            "user": "What is 15% of 200?",
            "assistant": "15% of 200 = 0.15 × 200 = 30.\n\nThe answer is 30.",
        },
        {
            "user": "Solve for x: 3x + 7 = 22",
            "assistant": "3x + 7 = 22\n3x = 22 - 7 = 15\nx = 15 / 3 = 5\n\nThe answer is x = 5.",
        },
        {
            "user": "A train travels 120 km in 2 hours. What is its speed in km/h?",
            "assistant": "Speed = Distance / Time = 120 km / 2 h = 60 km/h.\n\nThe answer is 60 km/h.",
        },
    ],
    "logic": [
        {
            "user": "All cats are animals. Some animals are pets. Can we conclude that all cats are pets?",
            "assistant": "No. While all cats are animals, only some animals are pets. We cannot conclude that all cats are pets — this is the fallacy of the undistributed middle.",
        },
        {
            "user": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
            "assistant": "Not necessarily. The ground being wet is consistent with rain, but other causes (sprinklers, spills) could produce the same result. This is the fallacy of affirming the consequent.",
        },
    ],
    "debugging": [
        {
            "user": "Fix this Python code:\ndef add(a, b):\n    return a * b",
            "assistant": "Bug: The function uses `*` (multiplication) instead of `+` (addition).\n\nFixed code:\n```python\ndef add(a, b):\n    return a + b\n```",
        },
        {
            "user": "Fix this code:\nfor i in range(5):\nprint(i)",
            "assistant": "Bug: The `print(i)` line is not indented inside the for loop.\n\nFixed code:\n```python\nfor i in range(5):\n    print(i)\n```",
        },
    ],
    "code_gen": [
        {
            "user": "Write a Python function to check if a string is a palindrome.",
            "assistant": '```python\ndef is_palindrome(s: str) -> bool:\n    """Check if a string is a palindrome (case-insensitive, ignoring spaces)."""\n    cleaned = s.lower().replace(" ", "")\n    return cleaned == cleaned[::-1]\n```',
        },
        {
            "user": "Write a Python function to find the factorial of a number.",
            "assistant": '```python\ndef factorial(n: int) -> int:\n    """Return n! for non-negative integers."""\n    if n < 0:\n        raise ValueError("n must be non-negative")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```',
        },
    ],
    "summarization": [
        {
            "user": "Summarize the following text in exactly two sentences.\n\nThe Apollo 11 mission launched on July 16, 1969, carrying astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins. Four days later, Armstrong and Aldrin became the first humans to walk on the Moon, an event watched by millions worldwide. They collected samples and deployed instruments before returning safely to Earth.",
            "assistant": "In 1969, the Apollo 11 mission successfully landed the first humans on the Moon. Astronauts Armstrong and Aldrin explored the lunar surface before returning safely to Earth.",
        },
        {
            "user": "Summarize this article using exactly three bullet points.\n\nThe global economy showed signs of recovery this quarter. Tech stocks rebounded strongly after last year's dip. However, inflation remains a persistent concern for many central banks. Consumer spending has slightly decreased as interest rates climb.",
            "assistant": "- The global economy is recovering, led by a strong rebound in tech stocks.\n- Inflation continues to be a major concern for central banks worldwide.\n- Consumer spending has dropped slightly due to rising interest rates.",
        },
    ],
    "sentiment": [
        {
            "user": "Is this review positive or negative? Explain your reasoning.\n\nThe food at the restaurant was absolutely delicious, but the service was incredibly slow and the waiters were rude.",
            "assistant": "Negative because the service was incredibly slow and rude BUT the food was absolutely delicious.",
        },
        {
            "user": "Determine if the sentiment is positive, negative, or neutral.\n\nThe new smartphone has a stunning display and lightning-fast processor, although the battery life is quite disappointing.",
            "assistant": "Positive because the display and processor are excellent BUT the battery life is disappointing.",
        },
    ],
    "ner": [
        {
            "user": "Extract all named entities from this text.\n\nOn Tuesday, Tim Cook announced that Apple will be opening a new campus in Austin next year.",
            "assistant": "PERSON: Tim Cook\nORGANIZATION: Apple\nLOCATION: Austin\nDATE: Tuesday, next year",
        },
        {
            "user": "Find the entities in the following sentence: Microsoft CEO Satya Nadella visited London on October 24, 2023.",
            "assistant": "PERSON: Satya Nadella\nORGANIZATION: Microsoft\nLOCATION: London\nDATE: October 24, 2023",
        },
    ],
}


def build_messages(
    category: str,
    user_prompt: str,
    use_few_shot: bool = False,
) -> List[Dict[str, str]]:
    """Build a chat-completion message list for the given category and prompt.

    Gemma 2 does NOT support the ``system`` role in its chat template.
    Instead, the system instruction is prepended to the first ``user``
    message to achieve the same effect without triggering a template error.

    Args:
        category: One of the 8 category strings (e.g. "math", "sentiment").
        user_prompt: The raw user prompt from tasks.json.
        use_few_shot: If True, inject few-shot examples for the category.

    Returns:
        A list of message dicts ready for model consumption.
    """
    system_content = "You are a helpful AI assistant. Follow the user's instructions exactly, paying strict attention to any length, formatting, or reasoning constraints requested."
    messages: List[Dict[str, str]] = []

    # Inject few-shot examples for high-complexity categories
    if use_few_shot and category in FEW_SHOT_EXAMPLES:
        examples = FEW_SHOT_EXAMPLES[category]
        for i, example in enumerate(examples):
            if i == 0:
                # Prepend system instruction to the first example's user turn
                messages.append({
                    "role": "user",
                    "content": f"Instructions: {system_content}\n\n{example['user']}",
                })
            else:
                messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})

        # The actual user prompt goes last
        messages.append({"role": "user", "content": user_prompt})
    else:
        # Zero-shot: prepend system instruction directly to user prompt
        messages.append({
            "role": "user",
            "content": f"Instructions: {system_content}\n\n{user_prompt}",
        })

    return messages
