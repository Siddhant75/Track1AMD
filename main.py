"""
Central orchestrator — the "Smart Traffic Cop".

Execution flow:
  1. INSTANT BOOT  → Read /input/tasks.json immediately (pass 60s SLA).
  2. CLASSIFY      → Assign each task a category, complexity tier, and routing tier.
  3. BUDGET SKIP   → Skip the hardest 10% of tasks (save time for hard tasks).
  4. SORT          → Process easy tasks first (build accuracy cushion).
  5. LAZY LOAD     → Load local Gemma-2-2B weights on first inference call.
  6. SMART ROUTE   → 3-tier routing:
                     - LOCAL_ONLY: sentiment, NER → zero-shot, no escalation.
                     - LOCAL_FIRST: factual, summary → try local, escalate if uncertain.
                     - REMOTE_PREFERRED: math, logic, debug, code → remote first.
  7. CRITIC        → Validate each output; retry once on failure.
  8. ESCALATE      → If critic rejects AND confidence is low → remote API.
  9. PANIC SWITCH  → At 8 minutes, dump remaining tasks to Fireworks API.
  10. WRITE OUTPUT  → Serialize via Pydantic → /output/results.json.
  11. EXIT 0.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

from classifier import TaskCategory, ComplexityTier, classify, get_complexity
from critic import validate_output
from deterministic_solver import solve_deterministic
from extractor import extract_answer
from local_engine import LocalEngine
from prompts import build_messages
from router import (
    RoutingTier,
    get_routing_tier,
    get_temperature,
    get_task_type,
    should_escalate,
    select_local_model,
    assess_prompt_complexity,
)
from schemas import TaskInput, TaskOutput, read_tasks, write_results
from validator import validate_and_correct, get_constraint_hint

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------
PANIC_THRESHOLD_SECS = int(5.5 * 60)  # 330 seconds (Hard Constraint: 5.5 minutes)
PER_TASK_TIMEOUT_SECS = 25          # Per-task ceiling (under 30s SLA)
RETRY_DEADLINE_SECS = 330           # Limit for retries

# ---------------------------------------------------------------------------
# Max-token budgets per category (tuned for conciseness)
# ---------------------------------------------------------------------------
_MAX_TOKENS: Dict[TaskCategory, int] = {
    TaskCategory.SENTIMENT: 32,
    TaskCategory.NER: 64,
    TaskCategory.FACTUAL: 64,
    TaskCategory.SUMMARIZATION: 128,
    TaskCategory.LOGIC: 128,
    TaskCategory.MATH: 384,
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


def _sort_key(item: dict) -> tuple:
    """Two-phase sort: Gemma tasks first, DeepSeek tasks second.

    Within each phase, sort by processing order (easy categories first).
    This minimises model swaps — Gemma loads once, runs all its tasks,
    then DeepSeek loads once, runs all its tasks.
    """
    from router import select_local_model, RoutingTier
    tier = item["tier"]
    # REMOTE_PREFERRED tasks go last (they'll be escalated anyway)
    if tier == RoutingTier.REMOTE_PREFERRED:
        model_priority = 2
    else:
        model_type = select_local_model(item["task"].prompt)
        model_priority = 0 if model_type == "gemma" else 1  # Gemma=0, DeepSeek=1
    return (model_priority, _PROCESSING_ORDER.get(item["category"], 99))


def _build_remote_messages(prompt: str) -> List[Dict[str, str]]:
    """Build zero-shot messages for the remote Fireworks API.

    CRITICAL: Do NOT use {"role": "system"} — some Fireworks models (e.g.
    Gemma variants) reject system role and return HTTP 400.  Prepend the
    instruction into the user message instead.
    """
    system_instruction = (
        "You are a precise AI assistant. Follow the user's instructions exactly, "
        "paying strict attention to any length, format, or constraint requirements. "
        "Be concise and direct. Do not add introductory or concluding remarks."
    )
    return [
        {
            "role": "user",
            "content": f"{system_instruction}\n\n{prompt}",
        }
    ]


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
    # 2. Classify every task and assign routing tier + complexity score
    # ------------------------------------------------------------------
    classified: List[Dict[str, Any]] = []
    for task in tasks:
        category = classify(task.prompt, task.category)
        complexity = get_complexity(category)
        tier = get_routing_tier(category)
        complexity_score = assess_prompt_complexity(task.prompt)
        classified.append({
            "task": task,
            "category": category,
            "complexity": complexity,
            "complexity_score": complexity_score,
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
    # 5. Prepare the local engine (lazy — loads on first .generate())
    # ------------------------------------------------------------------
    engine = LocalEngine()

    # ------------------------------------------------------------------
    # 6. Smart routing loop
    # ------------------------------------------------------------------
    results: List[TaskOutput] = []
    pending_remote: List[Dict[str, Any]] = []  # For panic or batch escalation
    escalation_queue: List[Dict[str, Any]] = []  # Confidence-based escalations

    active_total = len(classified)
    for idx, item in enumerate(classified):

        # ---- PANIC CHECK ----
        if elapsed() >= PANIC_THRESHOLD_SECS:
            print(
                f"[PANIC] {elapsed():.0f}s elapsed — escalating "
                f"{active_total - idx} remaining tasks to remote API.",
                flush=True,
            )
            for remaining in classified[idx:]:
                cat = remaining["category"]
                pending_remote.append({
                    "task_id": remaining["task"].task_id,
                    "messages": _build_remote_messages(remaining["task"].prompt),
                    "max_tokens": _get_max_tokens(cat),
                    "task_type": get_task_type(cat),
                    "local_answer": "",
                    "original_prompt": remaining["task"].prompt,
                    "category": cat,
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

        # --- DETERMINISTIC FAST-PASS ---
        fast_answer = solve_deterministic(task.prompt)
        if fast_answer is not None:
            results.append(TaskOutput(task_id=task.task_id, answer=fast_answer))
            print(
                f"[FAST] {idx + 1}/{active_total} | {category.value:<14} | "
                f"DETERMINISTIC | {elapsed():.1f}s",
                flush=True,
            )
            continue

        try:
            if tier == RoutingTier.REMOTE_PREFERRED:
                should_esc = True
                esc_reason = "remote_preferred_tier"
                answer = ""
                confidence = 0.0
                validator_passed = True
                is_valid = True
            else:
                # --- LOCAL INFERENCE with confidence ---
                model_type = select_local_model(task.prompt)

                # DeepSeek needs more tokens to accommodate the <think> reasoning block
                # Capped at 512 to prevent rambling from causing a timeout
                actual_max_tokens = min(max_tokens, 512) if model_type == "deepseek" else max_tokens

                answer, confidence = engine.generate_with_confidence(
                    messages,
                    model_type=model_type,
                    max_tokens=actual_max_tokens,
                    temperature=temperature,
                )

                # --- EXTRACTOR: pull core answer from verbose output ---
                answer = extract_answer(category, answer)

                # --- VALIDATOR (Auto-correction pass 1) ---
                validator_passed, answer = validate_and_correct(task.prompt, answer)

                # --- EARLY EXIT: fast path for LOCAL_ONLY if already valid ---
                if tier == RoutingTier.LOCAL_ONLY and validator_passed:
                    results.append(TaskOutput(task_id=task.task_id, answer=answer))
                    print(
                        f"[FAST] {idx + 1}/{active_total} | {category.value:<14} | "
                        f"LOCAL_FAST | {elapsed():.1f}s",
                        flush=True,
                    )
                    continue

                # --- CRITIC VALIDATION ---
                is_valid, reason = validate_output(category, task.prompt, answer)

                # --- CONSTRAINT RETRY ---
                if not is_valid:
                    # Time-Aware Retry: If we are running out of time, skip retry and escalate
                    if elapsed() > RETRY_DEADLINE_SECS:
                        print(f"[RETRY-SKIP] Task {task.task_id} failed constraints, but time > {RETRY_DEADLINE_SECS}s. Escalating.", flush=True)
                        should_esc = True
                        esc_reason = "timeout_prevent_retry"
                    else:
                        print(f"[RETRY] Task {task.task_id} failed constraints: {reason}. Retrying...", flush=True)
                        hint = get_constraint_hint(category, reason)
                        retry_messages = messages + [
                            {"role": "assistant", "content": answer},
                            {"role": "user", "content": f"Your previous answer was rejected because: {reason}\n\n{hint}"}
                        ]
                        
                        answer, confidence = engine.generate_with_confidence(
                            retry_messages,
                            model_type=model_type,
                            max_tokens=actual_max_tokens,
                            temperature=0.1,  # Keep deterministic — hint guides it
                        )
                        answer = extract_answer(category, answer)
                        validator_passed, answer = validate_and_correct(task.prompt, answer)
                        is_valid, reason = validate_output(category, task.prompt, answer)
                        
                        if not is_valid:
                            should_esc = True
                            esc_reason = "failed_constraint_retry"
                else:
                    should_esc = False

                # --- SMART ESCALATION DECISION ---
                if not should_esc:
                    should_esc, esc_reason = should_escalate(
                        category, task.prompt, answer, confidence, is_valid, validator_passed
                    )

            if should_esc and tier != RoutingTier.LOCAL_ONLY:
                print(
                    f"[ROUTE] Task {task.task_id} ({category.value}) -> "
                    f"ESCALATE ({esc_reason})",
                    flush=True,
                )
                escalation_queue.append({
                    "task_id": task.task_id,
                    # Zero-shot, no system role — avoids HTTP 400 on Gemma-based models
                    "messages": _build_remote_messages(task.prompt),
                    "max_tokens": max_tokens,
                    "task_type": get_task_type(category),
                    "local_answer": answer,  # Keep as fallback
                    "original_prompt": task.prompt,
                    "category": category,
                })
            else:
                # Accept local answer
                results.append(TaskOutput(task_id=task.task_id, answer=answer))
                status = "LOCAL" if not should_esc else "LOCAL_ONLY"
                print(
                    f"[OK] {idx + 1}/{active_total} | {category.value:<14} | "
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
                "messages": _build_remote_messages(task.prompt),
                "max_tokens": max_tokens,
                "task_type": get_task_type(category),
                "local_answer": "",
                "original_prompt": task.prompt,
                "category": category,
            })

    # ------------------------------------------------------------------
    # 7. Process escalation queue (confidence-based remote calls)
    # ------------------------------------------------------------------
    if escalation_queue:
        if elapsed() > 540:
            print("[TIMEOUT] 9 minutes exceeded. Skipping remote escalation to exit safely.", flush=True)
            for eq in escalation_queue:
                results.append(TaskOutput(task_id=eq["task_id"], answer="Unable to process."))
            escalation_queue.clear()
            
    if escalation_queue:
        print(
            f"[ESCALATE] Sending {len(escalation_queue)} tasks to remote API...",
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
                    orig_prompt = ""
                    local_fallback = ""
                    cat = None
                    for eq in escalation_queue:
                        if eq["task_id"] == r["task_id"]:
                            orig_prompt = eq.get("original_prompt", "")
                            local_fallback = eq.get("local_answer", "")
                            cat = eq.get("category")
                            break

                    if not answer or answer == "Unable to process.":
                        # Remote failed — try local DeepSeek as emergency fallback
                        if local_fallback:
                            answer = local_fallback
                            print(
                                f"[FALLBACK] Task {r['task_id']} — remote empty, "
                                f"using local answer.",
                                flush=True,
                            )
                        else:
                            # Last resort: run DeepSeek locally right now
                            # BUT ONLY if we aren't in a panic state or out of time!
                            if elapsed() > 540 or elapsed() > RETRY_DEADLINE_SECS:
                                print(f"[FALLBACK] Task {r['task_id']} — SKIPPING local emergency due to deadline or panic.", flush=True)
                                answer = "Unable to process."
                            else:
                                try:
                                    fb_messages = build_messages(
                                        cat.value if cat else "factual",
                                        orig_prompt,
                                        use_few_shot=True,
                                    )
                                    answer, _ = engine.generate_with_confidence(
                                        fb_messages,
                                        model_type="deepseek",
                                        max_tokens=512,
                                        temperature=0.1,
                                    )
                                    _, answer = validate_and_correct(orig_prompt, answer)
                                    print(
                                        f"[FALLBACK] Task {r['task_id']} — DeepSeek local emergency.",
                                        flush=True,
                                    )
                                except Exception as fb_exc:
                                    print(f"[FALLBACK] DeepSeek failed: {fb_exc}", flush=True)
                                    answer = "Unable to process."
                    else:
                        # Validate and clean up the remote answer with Python
                        _, answer = validate_and_correct(orig_prompt, answer)

                    results.append(
                        TaskOutput(task_id=r["task_id"], answer=answer)
                    )
                    print(
                        f"[OK] {r['task_id']} | REMOTE_VERIFIED | {elapsed():.1f}s",
                        flush=True,
                    )
            else:
                print(
                    "[ESCALATE] Remote not available. Using local DeepSeek fallback.",
                    flush=True,
                )
                for eq in escalation_queue:
                    fallback = eq.get("local_answer", "")
                    if not fallback:
                        try:
                            fb_messages = build_messages(
                                eq["category"].value if eq.get("category") else "factual",
                                eq.get("original_prompt", ""),
                                use_few_shot=True,
                            )
                            fallback, _ = engine.generate_with_confidence(
                                fb_messages,
                                model_type="deepseek",
                                max_tokens=768,
                                temperature=0.1,
                            )
                            _, fallback = validate_and_correct(
                                eq.get("original_prompt", ""), fallback
                            )
                        except Exception:
                            fallback = "Unable to process."
                    results.append(
                        TaskOutput(task_id=eq["task_id"], answer=fallback)
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
    # 8. Emergency remote escalation (panic timer or deferred/skipped tasks)
    # ------------------------------------------------------------------
    if pending_remote:
        if elapsed() > 540:
            print("[TIMEOUT] 9 minutes exceeded. Skipping panic batch to exit safely.", flush=True)
            existing_ids = {r.task_id for r in results}
            for pt in pending_remote:
                if pt["task_id"] not in existing_ids:
                    results.append(TaskOutput(task_id=pt["task_id"], answer="Unable to process."))
            pending_remote.clear()
            
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
                    answer = r["answer"]
                    orig_prompt = ""
                    local_fallback = ""
                    cat = None
                    for pt in pending_remote:
                        if pt["task_id"] == r["task_id"]:
                            orig_prompt = pt.get("original_prompt", "")
                            local_fallback = pt.get("local_answer", "")
                            cat = pt.get("category")
                            break

                    if not answer or answer == "Unable to process.":
                        # Try local DeepSeek as last resort
                        if local_fallback:
                            answer = local_fallback
                        else:
                            try:
                                fb_messages = build_messages(
                                    cat.value if cat else "factual",
                                    orig_prompt,
                                    use_few_shot=True,
                                )
                                answer, _ = engine.generate_with_confidence(
                                    fb_messages,
                                    model_type="deepseek",
                                    max_tokens=768,
                                    temperature=0.1,
                                )
                                _, answer = validate_and_correct(orig_prompt, answer)
                            except Exception:
                                answer = "Unable to process."
                    else:
                        _, answer = validate_and_correct(orig_prompt, answer)

                    results.append(
                        TaskOutput(task_id=r["task_id"], answer=answer)
                    )
            else:
                print("[REMOTE] Client not configured. Using DeepSeek fallback.", flush=True)
                existing_ids = {r.task_id for r in results}
                for pt in pending_remote:
                    if pt["task_id"] not in existing_ids:
                        fallback = pt.get("local_answer", "")
                        if not fallback:
                            try:
                                fb_messages = build_messages(
                                    pt["category"].value if pt.get("category") else "factual",
                                    pt.get("original_prompt", ""),
                                    use_few_shot=True,
                                )
                                fallback, _ = engine.generate_with_confidence(
                                    fb_messages,
                                    model_type="deepseek",
                                    max_tokens=768,
                                    temperature=0.1,
                                )
                                _, fallback = validate_and_correct(
                                    pt.get("original_prompt", ""), fallback
                                )
                            except Exception:
                                fallback = "Unable to process."
                        results.append(
                            TaskOutput(task_id=pt["task_id"], answer=fallback)
                        )
        except Exception as exc:
            print(f"[REMOTE] Batch failed: {exc}. Filling fallbacks.", flush=True)
            existing_ids = {r.task_id for r in results}
            for pt in pending_remote:
                if pt["task_id"] not in existing_ids:
                    results.append(
                        TaskOutput(
                            task_id=pt["task_id"],
                            answer=pt.get("local_answer", "Unable to process."),
                        )
                    )

    # ------------------------------------------------------------------
    # 9. Write output — validated through Pydantic
    # ------------------------------------------------------------------
    write_results(results, output_path)

    # Summary stats
    local_count = active_total - len(escalation_queue) - len(pending_remote)
    print(
        f"[DONE] Wrote {len(results)} results in {elapsed():.1f}s. "
        f"Local: {local_count}, Escalated: {len(escalation_queue)}, "
        f"Deferred/Panic: {len(pending_remote)}. Exiting.",
        flush=True,
    )

    # If 100% of answers are "Unable to process.", crash loudly
    if results and all(r.answer == "Unable to process." for r in results):
        print("[FATAL] 100% of tasks failed. Exiting with code 1.", flush=True)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
