#!/usr/bin/env bash
# Online Qwen3-4B GRPO on Lightning Weave's Skywork math task.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec uv run --no-project --python 3.12 --script "${PROJECT_ROOT}/configs/math_grpo/train.py" "$@"
