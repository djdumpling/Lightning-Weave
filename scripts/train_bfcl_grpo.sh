#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-alex-dev-2}"
exec uv run --no-project --python 3.12 --script "${PROJECT_ROOT}/configs/bfcl_grpo/train.py" "$@"
