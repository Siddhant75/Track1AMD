"""
Smart remote client for the Fireworks AI API.

Used for two purposes:
  1. Confidence-based escalation — when the local model's logprob
     confidence is below the threshold on hard tasks.
  2. Emergency panic — when the 9-minute timer fires.

All API calls MUST go through FIREWORKS_BASE_URL (injected by the harness).
The FIREWORKS_API_KEY is also injected — never hardcoded.
Model IDs are read from ALLOWED_MODELS at runtime — never hardcoded.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import aiohttp

# Model preference for different task types
# Higher priority = tried first
_MODEL_PRIORITY = {
    "code": ["kimi-k2p7-code", "gemma-4-31b-it", "gemma-4-26b-a4b-it"],
    "general": ["gemma-4-31b-it", "gemma-4-26b-a4b-it", "minimax-m3", "kimi-k2p7-code"],
    "reasoning": ["gemma-4-31b-it", "minimax-m3", "gemma-4-26b-a4b-it"],
}


class RemoteClient:
    """Async client for the Fireworks AI API (smart escalation)."""

    def __init__(self) -> None:
        self.api_key: str = os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url: str = os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/")

        allowed_raw = os.environ.get("ALLOWED_MODELS", "")
        self.allowed_models: List[str] = [
            m.strip() for m in allowed_raw.split(",") if m.strip()
        ]

        # Build model lookup for fast matching
        self._allowed_set = set(self.allowed_models)

    def select_model(self, task_type: str = "general") -> str:
        """Select the best available model for a given task type.

        Reads from ALLOWED_MODELS (injected at runtime) and picks the
        highest-priority model that is actually available.

        Args:
            task_type: One of 'code', 'general', or 'reasoning'.

        Returns:
            A model ID string, or empty string if none available.
        """
        if not self.allowed_models:
            return ""

        preference = _MODEL_PRIORITY.get(task_type, _MODEL_PRIORITY["general"])

        # Return the first model in our preference list that is allowed
        for model in preference:
            # Check exact match or substring match (model IDs may have prefixes)
            for allowed in self.allowed_models:
                if model in allowed or allowed in model:
                    return allowed

        # Fallback: first available model
        return self.allowed_models[0]

    @property
    def is_available(self) -> bool:
        """Check if the remote client is properly configured."""
        return bool(self.api_key and self.base_url and self.allowed_models)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.1,
        task_type: str = "general",
    ) -> str:
        """Send a single chat-completion request to the Fireworks API.

        Args:
            messages: Standard chat-completion message list.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            task_type: Hint for model selection ('code', 'general', 'reasoning').

        Returns:
            The model's response text, or an empty string on failure.
        """
        if not self.is_available:
            return ""

        model = self.select_model(task_type)
        if not model:
            return ""

        base = self.base_url
        if base.endswith("/v1"):
            base = base[:-3]
        url = f"{base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=28)  # Under 30s per-request SLA
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return content.strip() if content else ""
                    else:
                        body = await resp.text()
                        print(
                            f"[REMOTE] HTTP {resp.status} | model={model} | "
                            f"url={url} | body={body[:500]}",
                            flush=True,
                        )
                        return ""
        except asyncio.TimeoutError:
            print("[REMOTE] Request timed out (28s)", flush=True)
            return ""
        except Exception as exc:
            print(f"[REMOTE] Error: {exc}", flush=True)
            return ""

    async def generate_batch(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 8,
    ) -> List[Dict[str, str]]:
        """Fire off multiple tasks concurrently to the remote API.

        Each task dict must contain:
          - "task_id": str
          - "messages": List[Dict[str, str]]
          - (optional) "max_tokens": int
          - (optional) "task_type": str

        Returns:
            A list of {"task_id": ..., "answer": ...} dicts.
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _process_one(task: Dict[str, Any]) -> Dict[str, str]:
            async with sem:
                answer = await self.generate(
                    task["messages"],
                    max_tokens=task.get("max_tokens", 512),
                    task_type=task.get("task_type", "general"),
                )
                return {"task_id": task["task_id"], "answer": answer or "Unable to process."}

        results = await asyncio.gather(
            *[_process_one(t) for t in tasks],
            return_exceptions=True,
        )

        processed: List[Dict[str, str]] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                tid = tasks[i]["task_id"] if i < len(tasks) else "unknown"
                print(f"[REMOTE] Batch error for {tid}: {r}", flush=True)
                processed.append({"task_id": tid, "answer": "Unable to process."})
            else:
                processed.append(r)

        return processed
