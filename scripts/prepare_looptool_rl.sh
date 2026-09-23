#!/usr/bin/env bash
# CPU-only LoopTool-23k RL preprocessing; see docs/looptool_rl_preprocessing.md.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec uv run --no-project --python 3.12 --script "${PROJECT_ROOT}/data_curation/prepare_looptool_rl.py" "$@"
