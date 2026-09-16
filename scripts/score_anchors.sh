#!/usr/bin/env bash
# Score one anchor pair on the shared rollout set and seal its training data.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
    echo "ANCHOR_NAME=klear|decs STUDENT_MODEL=... bash scripts/score_anchors.sh"
    echo "Use KLEAR_{PRE,POST}_MODEL / DECS_{PRE,POST}_MODEL, or PRE_MODEL / POST_MODEL."
    echo "Each invocation loads post, pre, and reference models sequentially on one GPU."
    exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/_pipeline.sh"
configure_anchor
ensure_asset_lock

score() {
    local model=$1 revision=$2 field=$3 input_dir=$4 output_dir=$5
    "${PYTHON}" data_curation/precompute_direct_opd_scores.py \
        --model "${model}" --model-revision "${revision}" \
        --asset-lock "${ASSET_LOCK}" --score-field "${field}" \
        --input "${input_dir}" --output-dir "${output_dir}" \
        --dtype "${DTYPE}" --device "${SCORE_DEVICE:-cuda:0}" \
        --row-batch-size "${SCORE_BATCH_SIZE:-8}" \
        --chunk-size "${SCORE_CHUNK_SIZE:-4096}" \
        --rank 0 --world-size 1
}

score "${POST_MODEL}" "${POST_REVISION}" post_teacher_log_probs \
    "${ROLLOUT_DIR}" "${ANCHOR_DIR}/post"
score "${PRE_MODEL}" "${PRE_REVISION}" pre_teacher_log_probs \
    "${ANCHOR_DIR}/post" "${ANCHOR_DIR}/pre"
score "${STUDENT_MODEL}" "${STUDENT_REVISION}" student_ref_sampled_log_probs \
    "${ANCHOR_DIR}/pre" "${ANCHOR_DIR}/final"

"${PYTHON}" data_curation/prepare_direct_opd_manifest.py \
    --input "${ANCHOR_DIR}/final" --source-dataset "${SOURCE_DATASET}" \
    --asset-lock "${ASSET_LOCK}" \
    --manifest-out "${ANCHOR_DIR}/final/manifest.json"
