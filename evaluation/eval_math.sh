#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to an HF model ID or checkpoint directory}"
OUTPUT_DIR=${OUTPUT_DIR:-results/math}
TP_SIZE=${TP_SIZE:-1}
TASKS=${TASKS:-weave_aime24,weave_aime25,weave_hmmt25}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Repetition is part of each task YAML, not a nonstandard --repeats CLI flag.
python -m lm_eval \
    --model vllm \
    --model_args "pretrained=${MODEL},tensor_parallel_size=${TP_SIZE},max_model_len=40960,gpu_memory_utilization=0.85,seed=42${MODEL_ARGS:+,${MODEL_ARGS}}" \
    --include_path "${ROOT}/evaluation/math_tasks" \
    --tasks "${TASKS}" \
    --batch_size auto \
    --apply_chat_template \
    --seed 42 \
    --log_samples \
    --output_path "${OUTPUT_DIR}"
