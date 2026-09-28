#!/usr/bin/env bash
# tau-bench (TAU1) and tau2-bench (TAU2) evaluation of the LoopTool OPD student and its base model on Modal.
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
ACTION=${1:-plan}
if [[ $# -gt 0 ]]; then
    shift
fi

MODAL=(uv run --no-project --python 3.12 --with modal==1.5.5 modal)
ENTRYPOINT="${PROJECT_ROOT}/configs/tau_bench_eval/modal_eval.py"
SECRET=prime-secret

require_secret() {
    if ! "${MODAL[@]}" secret list --json | grep -q "\"${SECRET}\""; then
        echo "Modal secret '${SECRET}' is missing in environment ${MODAL_ENVIRONMENT}; the user simulators need it:" >&2
        echo "  MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT} ${MODAL[*]} secret create ${SECRET} PRIME_API_KEY=..." >&2
        exit 2
    fi
}

case "${ACTION}" in
    plan)
        exec python3 "${PROJECT_ROOT}/configs/tau_bench_eval/config.py"
        ;;
    smoke)
        # Detached like `run`; print its comparison later with `compare --smoke-samples 3`.
        export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-alex-dev-2}
        require_secret
        exec "${MODAL[@]}" run --detach "${ENTRYPOINT}" --smoke-samples "${SMOKE_SAMPLES:-3}" "$@"
        ;;
    run)
        export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-alex-dev-2}
        require_secret
        exec "${MODAL[@]}" run --detach "${ENTRYPOINT}" "$@"
        ;;
    compare)
        export MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT:-alex-dev-2}
        exec "${MODAL[@]}" run "${ENTRYPOINT}" --compare-only "$@"
        ;;
    *)
        echo "Usage: [TAU_PROFILE=prime|user30b] bash scripts/run_tau_eval.sh plan|smoke|run|compare [--models base,opd,thinking2507] [--domains tau2_airline,...] [--trials N]" >&2
        exit 2
        ;;
esac
