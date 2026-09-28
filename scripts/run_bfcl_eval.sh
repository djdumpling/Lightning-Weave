#!/usr/bin/env bash
# BFCL evaluation of the LoopTool OPD student and its base model on Modal.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
ACTION=${1:-plan}
if [[ $# -gt 0 ]]; then
    shift
fi

MODAL=(uv run --no-project --python 3.12 --with modal==1.5.5 modal run)
ENTRYPOINT="${PROJECT_ROOT}/configs/bfcl_eval/modal_eval.py"

case "${ACTION}" in
    plan)
        exec python3 "${PROJECT_ROOT}/configs/bfcl_eval/config.py"
        ;;
    smoke)
        export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-alex-dev-2}
        exec "${MODAL[@]}" "${ENTRYPOINT}" --smoke-samples "${SMOKE_SAMPLES:-3}" "$@"
        ;;
    run)
        export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-alex-dev-2}
        exec "${MODAL[@]}" --detach "${ENTRYPOINT}" "$@"
        ;;
    *)
        echo "Usage: bash scripts/run_bfcl_eval.sh plan|smoke|run [--models base,opd] [--categories a,b]" >&2
        exit 2
        ;;
esac
