"""
Central orchestrator — the "Smart Traffic Cop".

Execution flow:
  1. INSTANT BOOT  → Read /input/tasks.json immediately (pass 60s SLA).
  2. CLASSIFY      → Assign each task a category, complexity tier, and routing tier.
  3. SORT          → Process easy tasks first (build accuracy cushion).
  4. LAZY LOAD     → Load local Gemma-2-2B weights on first inference call.
  5. SMART ROUTE   → 3-tier routing:
                     - LOCAL_ONLY: sentiment, NER → zero-shot, no escalation.
                     - LOCAL_FIRST: factual, summary, math, logic, debug, code →
                       try local with confidence check, escalate if uncertain.
                     - REMOTE_PREFERRED: extremely complex prompts → remote first.
  6. CRITIC        → Validate each output; retry once on failure.
  7. ESCALATE      → If critic rejects AND confidence is low → remote API.
  8. PANIC SWITCH  → At 9 minutes, dump remaining tasks to Fireworks API.
  9. WRITE OUTPUT  → Serialize via Pydantic → /output/results.json.
  10. EXIT 0.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

from classifier import TaskCategory, ComplexityTier, classify, get_complexity
from critic import validate_output
from local_engine import LocalEngine
from prompts import build_messages
from router import (
    RoutingTier,
    get_routing_tier,
    get_temperature,
    get_task_type,
    should_escalate,
    build_verification_prompt,
    select_local_model,
)
from schemas import TaskInput, TaskOutput, read_tasks, write_results
from validator import validate_and_correct

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------
PANIC_THRESHOLD_SECS = 9 * 60       # Abandon local processing at 9 minutes
PER_TASK_TIMEOUT_SECS = 25          # Per-task ceiling (under 30s SLA)

# ---------------------------------------------------------------------------
# Max-token budgets per category (tuned for conciseness)
# ---------------------------------------------------------------------------
_MAX_TOKENS: Dict[TaskCategory, int] = {
    TaskCategory.SENTIMENT: 64,
    TaskCategory.NER: 256,
    TaskCategory.SUMMARIZATION: 300,
    TaskCategory.FACTUAL: 256,
    TaskCategory.MATH: 384,
    TaskCategory.LOGIC: 512,
    TaskCategory.DEBUGGING: 768,
    TaskCategory.CODE_GEN: 768,
}

# Processing order: easy categories first to build accuracy cushion
_PROCESSING_ORDER: Dict[TaskCategory, int] = {
    TaskCategory.SENTIMENT: 0,
    TaskCategory.NER: 1,
    TaskCategory.FACTUAL: 2,
    TaskCategory.SUMMARIZATION: 3,
    TaskCategory.MATH: 4,
    TaskCategory.LOGIC: 5,
    TaskCategory.DEBUGGING: 6,
    TaskCategory.CODE_GEN: 7,
}


def _get_max_tokens(category: TaskCategory) -> int:
    return _MAX_TOKENS.get(category, 512)


def _sort_key(item: Dict[str, Any]) -> tuple:
    """Sort tasks by model_type (to minimize swapping), then processing order."""
    model_type = select_local_model(item["task"].prompt)
    # Process Gemma first (faster), then DeepSeek
    model_priority = 0 if model_type == "gemma" else 1
    return (model_priority, _PROCESSING_ORDER.get(item["category"], 99))


def main() -> None:
    start = time.monotonic()

    def elapsed() -> float:
        return time.monotonic() - start

    print("[BOOT] Agent starting...", flush=True)

    # ------------------------------------------------------------------
    # 1. Read tasks IMMEDIATELY (pass 60s boot SLA)
    # ------------------------------------------------------------------
    input_path = os.environ.get("INPUT_PATH", "/input/tasks.json")
    output_path = os.environ.get("OUTPUT_PATH", "/output/results.json")

    tasks: List[TaskInput] = read_tasks(input_path)
    print(f"[INIT] Loaded {len(tasks)} tasks ({elapsed():.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # 2. Classify every task and assign routing tier
    # ------------------------------------------------------------------
    classified: List[Dict[str, Any]] = []
    for task in tasks:
        category = classify(task.prompt, task.category)
        complexity = get_complexity(category)
        tier = get_routing_tier(category)
        classified.append({
            "task": task,
            "category": category,
            "complexity": complexity,
            "tier": tier,
        })
    print(f"[CLASSIFY] Done ({elapsed():.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # 3. Sort: easy tasks first (sentiment, NER, factual, summary)
    # ------------------------------------------------------------------
    classified.sort(key=_sort_key)
    print(
        f"[SORT] Order: {', '.join(item['category'].value for item in classified)}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 4. Prepare the local engine (lazy — loads on first .generate())
    # ------------------------------------------------------------------
    engine = LocalEngine()

    # ------------------------------------------------------------------
    # 5. Smart routing loop
    # ------------------------------------------------------------------
    results: List[TaskOutput] = []
    pending_remote: List[Dict[str, Any]] = []  # For panic or batch escalation
    escalation_queue: List[Dict[str, Any]] = []  # Confidence-based escalations
    tokens_used = 0  # Track remote token usage

    total = len(classified)
    for idx, item in enumerate(classified):

        # ---- PANIC CHECK ----
        if elapsed() >= PANIC_THRESHOLD_SECS:
            print(
                f"[PANIC] {elapsed():.0f}s elapsed — escalating "
                f"{total - idx} remaining tasks to remote API.",
                flush=True,
            )
            for remaining in classified[idx:]:
                cat = remaining["category"]
                cplx = remaining["complexity"]
                pending_remote.append({
                    "task_id": remaining["task"].task_id,
                    "messages": build_messages(
                        cat.value,
                        remaining["task"].prompt,
                        use_few_shot=(cplx == ComplexityTier.HIGH),
                    ),
                    "max_tokens": _get_max_tokens(cat),
                    "task_type": get_task_type(cat),
                })
            break

        task: TaskInput = item["task"]
        category: TaskCategory = item["category"]
        complexity: ComplexityTier = item["complexity"]
        tier: str = item["tier"]
        use_few_shot = complexity == ComplexityTier.HIGH
        max_tokens = _get_max_tokens(category)
        temperature = get_temperature(category)

        messages = build_messages(
            category.value, task.prompt, use_few_shot=use_few_shot
        )

        try:
            # --- LOCAL INFERENCE with confidence ---
            model_type = select_local_model(task.prompt)
            
            # DeepSeek needs more tokens to accommodate the <think> reasoning block
            actual_max_tokens = max(max_tokens, 768) if model_type == "deepseek" else max_tokens
            
            answer, confidence = engine.generate_with_confidence(
                messages,
                model_type=model_type,
                max_tokens=actual_max_tokens,
                temperature=temperature,
            )

            # --- VALIDATOR (Auto-correction) ---
            validator_passed, answer = validate_and_correct(task.prompt, answer)

            # --- CRITIC VALIDATION ---
            is_valid, reason = validate_output(category, task.prompt, answer)

            if not is_valid:
                print(
                    f"[CRITIC] Task {task.task_id} rejected ({reason}). "
                    f"Retrying with temp=0.3...",
                    flush=True,
                )
                # Retry with slightly higher temperature
                answer, confidence = engine.generate_with_confidence(
                    messages,
                    model_type=model_type,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                validator_passed, answer = validate_and_correct(task.prompt, answer)
                is_valid, reason = validate_output(category, task.prompt, answer)

            # --- SMART ESCALATION DECISION ---
            should_esc, esc_reason = should_escalate(
                category, task.prompt, answer, confidence, is_valid, validator_passed
            )

            if should_esc and tier != RoutingTier.LOCAL_ONLY:
                print(
                    f"[ROUTE] Task {task.task_id} ({category.value}) → "
                    f"ESCALATE ({esc_reason})",
                    flush=True,
                )
                # Build a compressed verification prompt to save remote tokens
                verification_prompt = build_verification_prompt(
                    category, task.prompt, answer,
                )
                escalation_queue.append({
                    "task_id": task.task_id,
                    "messages": [{"role": "user", "content": verification_prompt}],
                    "max_tokens": max_tokens,
                    "task_type": get_task_type(category),
                    "local_answer": answer,  # Keep as fallback
                })
            else:
                # Accept local answer
                results.append(TaskOutput(task_id=task.task_id, answer=answer))
                status = "LOCAL" if not should_esc else "LOCAL_ONLY"
                print(
                    f"[OK] {idx + 1}/{total} | {category.value:<14} | "
                    f"{status} | conf={confidence:.2f} | {elapsed():.1f}s",
                    flush=True,
                )

        except Exception as exc:
            print(
                f"[ERROR] Task {task.task_id}: {exc}. Queued for remote.",
                flush=True,
            )
            pending_remote.append({
                "task_id": task.task_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "task_type": get_task_type(category),
            })

    # ------------------------------------------------------------------
    # 6. Process escalation queue (confidence-based remote calls)
    # ------------------------------------------------------------------
    if escalation_queue:
        print(
            f"[ESCALATE] Sending {len(escalation_queue)} tasks to remote API "
            f"for verification...",
            flush=True,
        )
        try:
            from remote_client import RemoteClient

            client = RemoteClient()
            if client.is_available:
                remote_results = asyncio.run(
                    client.generate_batch(escalation_queue)
                )
                for r in remote_results:
                    answer = r["answer"]
                    # If remote returned empty, use local answer as fallback
                    if not answer or answer == "Unable to process.":
                        # Find the local fallback
                        for eq in escalation_queue:
                            if eq["task_id"] == r["task_id"]:
                                answer = eq.get("local_answer", answer)
                                break
                    results.append(
                        TaskOutput(task_id=r["task_id"], answer=answer)
                    )
                    print(
                        f"[OK] {r['task_id']} | REMOTE_VERIFIED | {elapsed():.1f}s",
                        flush=True,
                    )
            else:
                print(
                    "[ESCALATE] Remote not available. Using local answers.",
                    flush=True,
                )
                for eq in escalation_queue:
                    results.append(
                        TaskOutput(
                            task_id=eq["task_id"],
                            answer=eq.get("local_answer", "Unable to process."),
                        )
                    )
        except Exception as exc:
            print(f"[ESCALATE] Batch failed: {exc}. Using local answers.", flush=True)
            for eq in escalation_queue:
                results.append(
                    TaskOutput(
                        task_id=eq["task_id"],
                        answer=eq.get("local_answer", "Unable to process."),
                    )
                )

    # ------------------------------------------------------------------
    # 7. Emergency remote escalation (panic timer or errors)
    # ------------------------------------------------------------------
    if pending_remote:
        print(
            f"[REMOTE] Escalating {len(pending_remote)} tasks to Fireworks API...",
            flush=True,
        )
        try:
            from remote_client import RemoteClient

            client = RemoteClient()
            if client.is_available:
                remote_results = asyncio.run(client.generate_batch(pending_remote))
                for r in remote_results:
                    results.append(
                        TaskOutput(task_id=r["task_id"], answer=r["answer"])
                    )
            else:
                print("[REMOTE] Client not configured. Using fallback.", flush=True)
                existing_ids = {r.task_id for r in results}
                for pt in pending_remote:
                    if pt["task_id"] not in existing_ids:
                        results.append(
                            TaskOutput(
                                task_id=pt["task_id"],
                                answer="Unable to process.",
                            )
                        )
        except Exception as exc:
            print(f"[REMOTE] Batch failed: {exc}. Filling fallbacks.", flush=True)
            existing_ids = {r.task_id for r in results}
            for pt in pending_remote:
                if pt["task_id"] not in existing_ids:
                    results.append(
                        TaskOutput(
                            task_id=pt["task_id"],
                            answer="Unable to process.",
                        )
                    )

    # ------------------------------------------------------------------
    # 8. Write output — validated through Pydantic
    # ------------------------------------------------------------------
    write_results(results, output_path)

    # Summary stats
    local_count = total - len(escalation_queue) - len(pending_remote)
    print(
        f"[DONE] Wrote {len(results)} results in {elapsed():.1f}s. "
        f"Local: {local_count}, Escalated: {len(escalation_queue)}, "
        f"Panic: {len(pending_remote)}. Exiting.",
        flush=True,
    )

    # If 100% of answers are "Unable to process.", crash loudly
    if results and all(r.answer == "Unable to process." for r in results):
        print("[FATAL] 100% of tasks failed. Exiting with code 1.", flush=True)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
