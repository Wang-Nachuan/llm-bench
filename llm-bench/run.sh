#!/usr/bin/env bash
set -euo pipefail

# Output directory inside the container (user can copy it out later).
export RESULTS_ROOT="/workspace/bench_out"
mkdir -p "${RESULTS_ROOT}"

export QUERIES_PATH="/workspace/queries"

# vLLM long context guardrails (existing behavior)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Power sampling knobs
export POWER_SAMPLE_PERIOD_S=1.0
export POWER_AGG_PERIOD_S=10.0

# Debugging
export VLLM_LOGGING_LEVEL=INFO
export VLLM_LOG_STATS_INTERVAL=10

# ===== Select one configuration =====

# --- TEST MODE ---
# export TEST_MODE=1
# export GPU_TYPE=h100
# export MODEL_SIZE=7b
# export TP_SIZE=1
# export PP_SIZE=1
# export DP_SIZE=1

# --- 8 x H100 ---
# export TEST_MODE=0
# export GPU_TYPE=h100
# export MODEL_SIZE=70b
# export TP_SIZE=1
# export PP_SIZE=4
# export DP_SIZE=2

# --- 16 x A100 ---
export TEST_MODE=0
export GPU_TYPE=a100
export MODEL_SIZE=70b
export TP_SIZE=1
export PP_SIZE=4
export DP_SIZE=4

# --- 8 x A100 ---
# export TEST_MODE=0
# export GPU_TYPE=a100
# export MODEL_SIZE=70b
# export TP_SIZE=1
# export PP_SIZE=4
# export DP_SIZE=2

# ====================================

python3 /workspace/run.py