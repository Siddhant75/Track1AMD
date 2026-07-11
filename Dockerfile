# ===========================================================================
# Multi-stage Dockerfile for AMD Hackathon Track 1
# Zero-Token All-Local Gemma Agent
#
# Target:  linux/amd64
# Build:   docker buildx build --platform linux/amd64 -t agent:latest .
# Runtime: 4 GB RAM / 2 vCPU / NO GPU
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build — compile llama-cpp-python for CPU-only execution
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv for blazing-fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency spec and install with CPU-only flags
COPY pyproject.toml .
RUN CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF" \
    uv pip install --system .

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean final image
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages

# Copy application code
COPY schemas.py classifier.py prompts.py local_engine.py critic.py \
     router.py remote_client.py main.py ./

# Copy the pre-downloaded quantized model weights
# IMPORTANT: Place the GGUF file in models/ before building:
#   mkdir -p models
#   huggingface-cli download bartowski/gemma-2-2b-it-GGUF \
#       --include "gemma-2-2b-it-Q4_K_M.gguf" \
#       --local-dir models
COPY models/ /app/models/

# Create the output directory expected by the grading harness
RUN mkdir -p /output

# Environment configuration
ENV LOCAL_MODEL_PATH=/app/models/gemma-2-2b-it-Q4_K_M.gguf
ENV PYTHONUNBUFFERED=1

# Entrypoint
CMD ["python", "main.py"]
