#!/usr/bin/env bash
# AIME25 checkpoint evaluation for the Modal Training Gym math run.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "${PROJECT_ROOT}"
exec uv run --no-project --python 3.12 --script "${PROJECT_ROOT}/configs/math_grpo/eval_aime25.py" "$@"
