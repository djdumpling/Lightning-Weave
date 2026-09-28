#!/usr/bin/env bash
# Launch explicit stages of the LoopTool Offline Direct-OPD Modal pipeline.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
ACTION=${1:-plan}
if [[ $# -gt 0 ]]; then
    shift
fi

case "${ACTION}" in
    plan)
        exec python3 "${PROJECT_ROOT}/configs/looptool_opd/config.py"
        ;;
    prepare|cache|convert|train|export|all)
        export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-alex-dev-2}
        exec uv run --no-project --python 3.12 --with modal==1.5.5 \
            modal run "${PROJECT_ROOT}/configs/looptool_opd/modal_pipeline.py" \
            --action "${ACTION}" "$@"
        ;;
    *)
        echo "Usage: bash scripts/run_looptool_opd.sh plan|prepare|cache|convert|train|export|all [Modal entrypoint options]" >&2
        exit 2
        ;;
esac
