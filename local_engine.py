"""
Local inference engine wrapping llama-cpp-python.

Configured for the grading sandbox constraints:
  - CPU-only (n_gpu_layers=0)
  - 2 threads (matches 2 vCPU)
  - 2048 default context window
  - Lazy model loading (passes 60s boot SLA)
  - Logprob-based confidence scoring for smart routing
"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Optional, Tuple

from llama_cpp import Llama

# Default path inside the Docker container
MODEL_PATH = os.environ.get(
    "LOCAL_MODEL_PATH",
    "/app/models/gemma-2-2b-it-Q4_K_M.gguf",
)

# Inference defaults tuned for accuracy on a 2B model
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_TEMPERATURE = 0.1  # Low temperature for deterministic answers
_DEFAULT_TOP_P = 0.9
_DEFAULT_REPEAT_PENALTY = 1.1

# Confidence threshold — below this, escalate to remote
CONFIDENCE_THRESHOLD = -1.5  # Average logprob; tuned empirically


class LocalEngine:
    """Lazy-loading wrapper around llama-cpp-python for CPU-only inference."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        n_ctx: int = 2048,
        n_threads: int = 2,
        n_batch: int = 512,
    ):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._n_batch = n_batch
        self._model: Optional[Llama] = None

    def _load_model(self) -> None:
        """Load model weights into RAM on first inference call (lazy loading)."""
        if self._model is not None:
            return

        self._model = Llama(
            model_path=self._model_path,
            n_gpu_layers=0,          # CPU only — no GPU in grading sandbox
            n_ctx=self._n_ctx,       # Context window
            n_threads=self._n_threads,  # Match 2 vCPU ceiling
            n_batch=self._n_batch,   # CPU-optimized batch size
            logits_all=False,        # Only need last-token logits
            verbose=False,           # Suppress llama.cpp logs
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        top_p: float = _DEFAULT_TOP_P,
        repeat_penalty: float = _DEFAULT_REPEAT_PENALTY,
    ) -> str:
        """Generate a response from the local Gemma model.

        Args:
            messages: Chat-completion format messages list.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_p: Nucleus sampling threshold.
            repeat_penalty: Penalty for repeated tokens.

        Returns:
            The model's response text, stripped of whitespace.
        """
        self._load_model()
        assert self._model is not None

        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            logprobs=True,
            top_logprobs=1,
        )

        # Extract the assistant's reply
        content = response["choices"][0]["message"]["content"]
        return content.strip() if content else ""

    def generate_with_confidence(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        top_p: float = _DEFAULT_TOP_P,
        repeat_penalty: float = _DEFAULT_REPEAT_PENALTY,
    ) -> Tuple[str, float]:
        """Generate a response and return a confidence score.

        The confidence score is the average log-probability of the
        generated tokens. Higher (closer to 0) = more confident.
        Lower (more negative) = less confident.

        Args:
            messages: Chat-completion format messages list.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            repeat_penalty: Penalty for repeated tokens.

        Returns:
            Tuple of (response_text, average_logprob).
        """
        self._load_model()
        assert self._model is not None

        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            logprobs=True,
            top_logprobs=1,
        )

        content = response["choices"][0]["message"]["content"]
        text = content.strip() if content else ""

        # Extract logprobs for confidence scoring
        avg_logprob = _extract_avg_logprob(response)

        return text, avg_logprob

    @property
    def is_loaded(self) -> bool:
        """Check whether the model weights are currently in memory."""
        return self._model is not None


def _extract_avg_logprob(response: dict) -> float:
    """Extract the average log-probability from a chat completion response.

    Returns a very negative value (-10.0) if logprobs are not available,
    which signals low confidence and triggers remote escalation.
    """
    try:
        logprobs_data = response["choices"][0].get("logprobs")
        if not logprobs_data:
            return -10.0

        content_logprobs = logprobs_data.get("content", [])
        if not content_logprobs:
            return -10.0

        total = 0.0
        count = 0
        for token_info in content_logprobs:
            lp = token_info.get("logprob")
            if lp is not None and math.isfinite(lp):
                total += lp
                count += 1

        if count == 0:
            return -10.0

        return total / count

    except (KeyError, IndexError, TypeError):
        return -10.0
