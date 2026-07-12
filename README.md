# AMD Hackathon Track 1 - Multi-Agent Hybrid Orchestrator

This repository contains our submission for Track 1 of the AMD Hackathon. The goal of this track is to build an intelligent, resource-efficient AI agent capable of passing an 80% accuracy threshold on a hidden dataset while minimizing the use of remote API tokens.

## Architecture Overview

Our system utilizes a **Multi-Agent Memory Swapping Architecture**. To achieve maximum token efficiency on severely constrained hardware (2 vCPUs, 0 GPUs, limited RAM), we run multiple specialized agents locally, safely swapping them in and out of memory.

1. **Gemma Agent (`gemma-2-2b-it`)**: A fast, local agent designed to handle simpler, zero-shot tasks (Sentiment Analysis, Factual Q&A, NER) with 0 token cost.
2. **DeepSeek Agent (`DeepSeek-R1-Distill`)**: A specialized reasoning agent invoked locally for highly complex tasks (Math, Logic, Coding).
3. **Multi-Agent Memory Swapping**: The engine safely unloads and loads agent weights on-the-fly, preventing RAM overflow while allowing us to run multiple models on a single lightweight container.
4. **Smart Router**: Analyzes prompt constraints, complexity, and categories to dynamically route to the ideal local agent. 
5. **Validator & Critic**: A fast Python layer that automatically validates local agent outputs against constraints, triggering re-tries or escalations.
6. **Panic Switch & Remote Escalation**: If the local agents fail the Critic's strict formatting checks, or the 9-minute panic threshold is triggered, remaining tasks are escalated to the Fireworks API. This guarantees a >80% accuracy rate while satisfying SLA time constraints.

## Project Structure / Where to Look

To save you time, here are the most critical files driving our architecture:
- `local_engine.py`: Contains the **Multi-Agent Memory Swapping** logic that safely loads/unloads models to prevent RAM overflow on 2 vCPUs.
- `router.py`: The intelligence layer. Calculates complexity and routes tasks to Gemma, DeepSeek, or the Remote API based on category and token constraints.
- `validator.py` & `critic.py`: The strict fallback layer that parses generated outputs and forcefully enforces task constraints before allowing a submission.
- `main.py`: The core orchestrator managing the event loop, thread concurrency, and the 9-minute panic switch.

## Run Instructions

Build the Docker image:
```bash
docker build -t amd-track1-agent:latest .
```

Run the container (the judging harness will inject the input/output volumes and environment variables):
```bash
docker run --rm \
  -e FIREWORKS_API_KEY="your_api_key" \
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
  -e ALLOWED_MODELS="gemma-4-31b-it,kimi-k2p7-code" \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  amd-track1-agent:latest
```
