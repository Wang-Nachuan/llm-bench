#!/usr/bin/env bash
set -euo pipefail

# Output directory inside the container (user can copy it out later).
export RESULTS_ROOT="/workspace/bench_out"
mkdir -p "${RESULTS_ROOT}"

export QUERIES_PATH="/workspace/queries"

# vLLM long context guardrails (existing behavior)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Debugging
export VLLM_LOGGING_LEVEL=INFO
export VLLM_LOG_STATS_INTERVAL=10

python3 /workspace/run.py