# ===========================================================================
# Dockerfile for AMD Hackathon Track 1
# Zero-Token All-Local Gemma Agent
# Target:  linux/amd64
# Runtime: 4 GB RAM / 2 vCPU / NO GPU
# ===========================================================================

FROM python:3.11-slim

# 1. Install necessary system libraries for C++ / ML execution
# python:slim lacks libstdc++ and libgomp, which llama.cpp needs to run!
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy dependency spec
COPY pyproject.toml .

# 3. Install dependencies using pre-compiled PyPI wheels.
# We do NOT compile from source here to prevent the GitHub Actions runner
# (which has AVX512) from optimizing the binary for its own CPU, which 
# causes "Illegal instruction (core dumped)" crashes on the grading server.
RUN pip install --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu .

# 4. Copy all Python code
COPY schemas.py classifier.py prompts.py local_engine.py critic.py \
     router.py remote_client.py validator.py extractor.py deterministic_solver.py main.py ./

# 5. Copy the pre-downloaded quantized model weights (downloaded in CI/CD)
COPY models/ /app/models/

# 6. Create the output directory expected by the grading harness
RUN mkdir -p /output

# 7. Environment configuration
ENV LOCAL_MODEL_PATH=/app/models/gemma-2-2b-it-Q4_K_M.gguf
ENV LOCAL_REASONING_MODEL_PATH=/app/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
ENV PYTHONUNBUFFERED=1

# 8. Entrypoint
CMD ["python", "main.py"]
