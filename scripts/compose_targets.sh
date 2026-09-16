#!/usr/bin/env bash
# Compose cached anchor shifts; numerical target construction stays in Python.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
    echo "TASK=math RUN_DIR=outputs/math bash scripts/compose_targets.sh"
    echo "Defaults: Klear weight=1, DECS weight=1, 12,800 rows repeated twice."
    echo "For arbitrary anchors, pass the Python composer's full CLI after this script."
    exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/_pipeline.sh"
if (($#)); then
    exec "${PYTHON}" data_curation/build_direct_opd_composed_target.py "$@"
fi

"${PYTHON}" data_curation/build_direct_opd_composed_target.py \
    --anchor-name klear \
    --anchor-manifest "${KLEAR_MANIFEST:-${RUN_DIR}/anchors/klear/final/manifest.json}" \
    --anchor-weight "${KLEAR_WEIGHT:-1.0}" \
    --anchor-name decs \
    --anchor-manifest "${DECS_MANIFEST:-${RUN_DIR}/anchors/decs/final/manifest.json}" \
    --anchor-weight "${DECS_WEIGHT:-1.0}" \
    --output-dir "${DATA_DIR:-${RUN_DIR}/composed}" \
    --rows "${COMPOSE_ROWS:-${EXPECTED_ROWS}}" --repeat "${REPEAT:-2}" \
    --rows-per-output-shard 128
