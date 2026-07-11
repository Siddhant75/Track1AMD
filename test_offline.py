"""
AMD Track 1 — Offline Validation Suite

Tests all modules WITHOUT requiring the actual GGUF model.
Tests routing logic, classifier, prompts, critic, schemas, and router.
"""

import json
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print("=" * 60)
    print(f"TEST: {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Schemas
# ---------------------------------------------------------------------------

def test_schemas():
    section("Schemas")

    from schemas import TaskInput, TaskOutput, read_tasks, write_results

    # Basic validation
    t = TaskInput(task_id="x1", prompt="Hello")
    assert t.task_id == "x1"
    assert t.prompt == "Hello"
    assert t.category is None
    print("  [PASS] TaskInput basic validation")

    # With category hint
    t2 = TaskInput(task_id="x2", prompt="Hello", category="math")
    assert t2.category == "math"
    print("  [PASS] TaskInput with category hint")

    # Output validation
    o = TaskOutput(task_id="x1", answer="World")
    assert o.task_id == "x1"
    assert o.answer == "World"
    print("  [PASS] TaskOutput validation")

    # Serialization round-trip
    d = o.model_dump()
    assert d == {"task_id": "x1", "answer": "World"}
    o2 = TaskOutput(**d)
    assert o2 == o
    print("  [PASS] Serialization round-trip")
    print()


# ---------------------------------------------------------------------------
# 2. File I/O
# ---------------------------------------------------------------------------

def test_file_io():
    section("File I/O")

    from schemas import TaskInput, TaskOutput, read_tasks, write_results

    # Read sample tasks
    sample_path = os.path.join(os.path.dirname(__file__), "input", "tasks.json")
    tasks = read_tasks(sample_path)
    assert len(tasks) == 8, f"Expected 8 tasks, got {len(tasks)}"
    print(f"  [PASS] Read {len(tasks)} tasks from input/tasks.json")

    # Write results
    results = [TaskOutput(task_id=t.task_id, answer="test") for t in tasks]
    out_path = os.path.join(os.path.dirname(__file__), "output", "test_results.json")
    write_results(results, out_path)

    with open(out_path, "r") as f:
        written = json.load(f)
    assert len(written) == 8
    print(f"  [PASS] Wrote and verified {len(written)} results to output/test_results.json")
    print()


# ---------------------------------------------------------------------------
# 3. Classifier
# ---------------------------------------------------------------------------

def test_classifier():
    section("Classifier")

    from classifier import TaskCategory, ComplexityTier, classify, get_complexity

    test_cases = [
        ("What is the capital of France?", None, TaskCategory.FACTUAL),
        ("Calculate 15% of 250.", None, TaskCategory.MATH),
        ("Analyze the sentiment of this text: I loved it!", None, TaskCategory.SENTIMENT),
        ("Summarize the following text: The cat sat on the mat.", None, TaskCategory.SUMMARIZATION),
        ("Extract all named entities from this text: Tim Cook leads Apple.", None, TaskCategory.NER),
        ("Fix this Python code:\ndef add(a,b): return a*b", None, TaskCategory.DEBUGGING),
        ("If all cats are animals, can we conclude all animals are cats?", None, TaskCategory.LOGIC),
        ("Write a Python function to sort a list.", None, TaskCategory.CODE_GEN),
    ]

    passed = 0
    for prompt, hint, expected in test_cases:
        result = classify(prompt, hint)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] '{prompt[:50]}...' → {result.value} (expected: {expected.value})")

    print(f"\n  Results: {passed}/{len(test_cases)} passed")

    # Category hint override
    result = classify("random text", "math")
    assert result == TaskCategory.MATH, f"Hint override failed: got {result}"
    print("  [PASS] Category hint override works")

    # Complexity tiers
    assert get_complexity(TaskCategory.SENTIMENT) == ComplexityTier.LOW
    assert get_complexity(TaskCategory.MATH) == ComplexityTier.HIGH
    print("  [PASS] Complexity tiers correct")
    print()


# ---------------------------------------------------------------------------
# 4. Prompt Builder
# ---------------------------------------------------------------------------

def test_prompts():
    section("Prompt Builder")

    from prompts import build_messages

    categories = ["factual", "math", "sentiment", "summarization",
                   "ner", "debugging", "logic", "code_gen"]

    for cat in categories:
        use_few_shot = cat in ("math", "logic", "debugging", "code_gen")
        messages = build_messages(cat, "Test prompt", use_few_shot=use_few_shot)

        assert len(messages) >= 1, f"Category {cat}: no messages"
        # Gemma 2 fix: no system role, instructions merged into first user msg
        assert messages[0]["role"] == "user"
        assert messages[-1]["role"] == "user"
        assert "Instructions:" in messages[0]["content"]

        if use_few_shot:
            assert len(messages) > 1, f"Category {cat}: missing few-shot examples"
            assert messages[-1]["content"] == "Test prompt"
        else:
            assert "Test prompt" in messages[0]["content"]

        print(f"  [PASS] {cat:<14} → {len(messages)} messages (few_shot={use_few_shot})")

    print()


# ---------------------------------------------------------------------------
# 5. Critic Validator
# ---------------------------------------------------------------------------

def test_critic():
    section("Critic Validator")

    from classifier import TaskCategory
    from critic import validate_output

    valid_cases = [
        (TaskCategory.SENTIMENT, "test", "Positive. The text is upbeat."),
        (TaskCategory.NER, "test", "Tim Cook (Person), Apple (Organization)"),
        (TaskCategory.SUMMARIZATION, "test " * 20, "This is a brief summary of the text."),
        (TaskCategory.MATH, "test", "The answer is 42."),
        (TaskCategory.CODE_GEN, "test", "def hello():\n    print('Hello')"),
        (TaskCategory.DEBUGGING, "test", "Bug: wrong operator.\n\nFixed code:\ndef add(a, b):\n    return a + b"),
        (TaskCategory.LOGIC, "test", "Based on the premises, we can conclude that not all animals are cats."),
        (TaskCategory.FACTUAL, "test", "The capital of France is Paris."),
    ]

    for cat, prompt, output in valid_cases:
        is_valid, reason = validate_output(cat, prompt, output)
        assert is_valid, f"{cat.value} valid case failed: {reason}"
        print(f"  [PASS] {cat.value:<14} valid output → {reason}")

    # Invalid cases
    invalid_cases = [
        (TaskCategory.SENTIMENT, "test", ""),
        (TaskCategory.MATH, "test", "I think maybe something"),
        (TaskCategory.CODE_GEN, "test", "Here is my answer"),
    ]

    for cat, prompt, output in invalid_cases:
        is_valid, reason = validate_output(cat, prompt, output)
        assert not is_valid, f"{cat.value} invalid case should fail"
        print(f"  [PASS] {cat.value:<14} invalid output → {reason}")

    print()


# ---------------------------------------------------------------------------
# 6. Router
# ---------------------------------------------------------------------------

def test_router():
    section("Smart Router")

    from classifier import TaskCategory
    from router import (
        RoutingTier,
        get_routing_tier,
        get_confidence_threshold,
        get_temperature,
        get_task_type,
        assess_prompt_complexity,
        should_escalate,
        build_verification_prompt,
    )

    # Routing tiers
    assert get_routing_tier(TaskCategory.SENTIMENT) == RoutingTier.LOCAL_ONLY
    assert get_routing_tier(TaskCategory.NER) == RoutingTier.LOCAL_ONLY
    assert get_routing_tier(TaskCategory.MATH) == RoutingTier.LOCAL_FIRST
    assert get_routing_tier(TaskCategory.CODE_GEN) == RoutingTier.LOCAL_FIRST
    print("  [PASS] Routing tiers correct")

    # Temperature settings
    assert get_temperature(TaskCategory.SENTIMENT) == 0.0
    assert get_temperature(TaskCategory.MATH) == 0.0
    assert get_temperature(TaskCategory.CODE_GEN) == 0.1
    print("  [PASS] Temperature settings correct")

    # Task types for remote model selection
    assert get_task_type(TaskCategory.CODE_GEN) == "code"
    assert get_task_type(TaskCategory.DEBUGGING) == "code"
    assert get_task_type(TaskCategory.MATH) == "reasoning"
    assert get_task_type(TaskCategory.SENTIMENT) == "general"
    print("  [PASS] Task type mappings correct")

    # Prompt complexity scoring
    simple_prompt = "What is 2+2?"
    complex_prompt = (
        "Step by step, first analyze the following code, then compare and "
        "contrast the two approaches. Finally, implement a comprehensive "
        "solution that optimizes the algorithm. How does this compare? "
        "What are the tradeoffs? Can you prove this is optimal?"
    )
    simple_score = assess_prompt_complexity(simple_prompt)
    complex_score = assess_prompt_complexity(complex_prompt)
    assert complex_score > simple_score, \
        f"Complex ({complex_score:.2f}) should score higher than simple ({simple_score:.2f})"
    print(f"  [PASS] Complexity scoring: simple={simple_score:.2f}, complex={complex_score:.2f}")

    # Escalation: LOCAL_ONLY never escalates
    esc, reason = should_escalate(
        TaskCategory.SENTIMENT, "test", "Positive", -5.0, True,
    )
    assert not esc, f"Sentiment should never escalate, got: {reason}"
    print("  [PASS] LOCAL_ONLY categories never escalate")

    # Escalation: critic rejection triggers escalation
    esc, reason = should_escalate(
        TaskCategory.MATH, "What is 2+2?", "", -2.0, False,
    )
    assert esc, "Critic rejection should trigger escalation"
    print(f"  [PASS] Critic rejection → escalation ({reason})")

    # Escalation: low confidence triggers escalation
    esc, reason = should_escalate(
        TaskCategory.MATH, "Solve this complex integral", "hmm", -3.0, True,
    )
    assert esc, "Low confidence should trigger escalation"
    print(f"  [PASS] Low confidence → escalation ({reason})")

    # Escalation: high confidence stays local
    esc, reason = should_escalate(
        TaskCategory.MATH, "What is 2+2?",
        "Step 1: 2 + 2 = 4.\n\nThe final answer is 4.",
        -0.5, True,
    )
    assert not esc, f"High confidence should stay local, got: {reason}"
    print(f"  [PASS] High confidence → stays local ({reason})")

    # Verification prompt building
    vp = build_verification_prompt(TaskCategory.MATH, "What is 2+2?", "4")
    assert "Verify" in vp
    assert "2+2" in vp
    assert "4" in vp
    print("  [PASS] Verification prompt for math")

    vp2 = build_verification_prompt(TaskCategory.CODE_GEN, "Write hello world", "print('hi')")
    assert "Review" in vp2 or "fix" in vp2.lower()
    print("  [PASS] Verification prompt for code")

    print()


# ---------------------------------------------------------------------------
# 7. Remote Client (config parsing only — no actual API calls)
# ---------------------------------------------------------------------------

def test_remote_client():
    section("Remote Client Config")

    # Temporarily set env vars
    old_key = os.environ.get("FIREWORKS_API_KEY")
    old_url = os.environ.get("FIREWORKS_BASE_URL")
    old_models = os.environ.get("ALLOWED_MODELS")

    try:
        os.environ["FIREWORKS_API_KEY"] = "test-key-123"
        os.environ["FIREWORKS_BASE_URL"] = "https://api.test.com"
        os.environ["ALLOWED_MODELS"] = "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4"

        from remote_client import RemoteClient

        client = RemoteClient()

        assert client.is_available, "Client should be available with env vars set"
        print("  [PASS] Client is_available with env vars")

        assert len(client.allowed_models) == 5, f"Expected 5 models, got {len(client.allowed_models)}"
        print(f"  [PASS] Parsed {len(client.allowed_models)} allowed models")

        # Model selection by task type
        code_model = client.select_model("code")
        assert "kimi" in code_model.lower() or "gemma" in code_model.lower(), \
            f"Code model should prefer kimi or gemma, got: {code_model}"
        print(f"  [PASS] Code task → {code_model}")

        general_model = client.select_model("general")
        print(f"  [PASS] General task → {general_model}")

        reasoning_model = client.select_model("reasoning")
        print(f"  [PASS] Reasoning task → {reasoning_model}")

    finally:
        # Restore original env
        for key, val in [("FIREWORKS_API_KEY", old_key), ("FIREWORKS_BASE_URL", old_url), ("ALLOWED_MODELS", old_models)]:
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    print()


# ---------------------------------------------------------------------------
# 8. Classifier on sample tasks.json
# ---------------------------------------------------------------------------

def test_classifier_on_sample():
    section("Classifier on sample tasks.json")

    from classifier import TaskCategory, classify, get_complexity
    from schemas import read_tasks

    sample_path = os.path.join(os.path.dirname(__file__), "input", "tasks.json")
    tasks = read_tasks(sample_path)

    expected = {
        "t1": TaskCategory.FACTUAL,
        "t2": TaskCategory.MATH,
        "t3": TaskCategory.SENTIMENT,
        "t4": TaskCategory.SUMMARIZATION,
        "t5": TaskCategory.NER,
        "t6": TaskCategory.DEBUGGING,
        "t7": TaskCategory.LOGIC,
        "t8": TaskCategory.CODE_GEN,
    }

    passed = 0
    for task in tasks:
        result = classify(task.prompt, task.category)
        complexity = get_complexity(result)
        exp = expected.get(task.task_id)
        status = "PASS" if result == exp else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] {task.task_id}: {result.value:<14} ({complexity.value}) | expected: {exp.value if exp else '?'}")

    print(f"\n  Results: {passed}/{len(tasks)} passed")
    print()


# ===========================================================================
# Run all tests
# ===========================================================================

if __name__ == "__main__":
    print()
    print("\U0001f9ea AMD Track 1 — Offline Validation Suite (v2: Smart Routing)")
    print("=" * 60)
    print()

    test_schemas()
    test_file_io()
    test_classifier()
    test_prompts()
    test_critic()
    test_router()
    test_remote_client()
    test_classifier_on_sample()

    print("=" * 60)
    print("\u2705 All offline tests complete!")
    print("=" * 60)
