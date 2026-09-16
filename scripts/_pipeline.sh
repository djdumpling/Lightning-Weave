#!/usr/bin/env bash
# Shared paths for the offline data pipeline.

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON=${PYTHON:-python}
TASK=${TASK:-math}
case "${TASK}" in
    math) DEFAULT_PROMPT_LENGTH=1024; DEFAULT_LABEL_KEY=reward_model ;;
    code) DEFAULT_PROMPT_LENGTH=4096; DEFAULT_LABEL_KEY=label ;;
    *) echo "TASK must be math or code" >&2; exit 2 ;;
esac
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/outputs/${TASK}}
SOURCE_DATASET=${SOURCE_DATASET:-${RUN_DIR}/prompts.parquet}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
NUM_PROMPTS=${NUM_PROMPTS:-3200}
RESPONSES_PER_PROMPT=${RESPONSES_PER_PROMPT:-4}
EXPECTED_ROWS=$((NUM_PROMPTS * RESPONSES_PER_PROMPT))
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-${DEFAULT_PROMPT_LENGTH}}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
DTYPE=${DTYPE:-float32}

configure_anchor() {
    : "${STUDENT_MODEL:?Set STUDENT_MODEL to a downloaded Hugging Face model}"
    STUDENT_REVISION=${STUDENT_REVISION:-${STUDENT_MODEL##*/}}
    ANCHOR_NAME=${ANCHOR_NAME:-klear}
    case "${ANCHOR_NAME}" in
        klear)
            POST_MODEL=${POST_MODEL:-${KLEAR_POST_MODEL:-}}
            PRE_MODEL=${PRE_MODEL:-${KLEAR_PRE_MODEL:-}}
            ;;
        decs)
            POST_MODEL=${POST_MODEL:-${DECS_POST_MODEL:-}}
            PRE_MODEL=${PRE_MODEL:-${DECS_PRE_MODEL:-}}
            ;;
    esac
    : "${POST_MODEL:?Set POST_MODEL or the corresponding anchor POST_MODEL variable}"
    : "${PRE_MODEL:?Set PRE_MODEL or the corresponding anchor PRE_MODEL variable}"
    POST_REVISION=${POST_REVISION:-${POST_MODEL##*/}}
    PRE_REVISION=${PRE_REVISION:-${PRE_MODEL##*/}}
    ANCHOR_DIR=${ANCHOR_DIR:-${RUN_DIR}/anchors/${ANCHOR_NAME}}
    ASSET_LOCK=${ASSET_LOCK:-${ANCHOR_DIR}/assets.json}
}

ensure_asset_lock() {
    if [[ ! -f "${ASSET_LOCK}" ]]; then
        "${PYTHON}" data_curation/prepare_direct_opd_assets.py \
            --student "${STUDENT_MODEL}" \
            --student-revision "${STUDENT_REVISION}" \
            --post-teacher "${POST_MODEL}" \
            --post-teacher-revision "${POST_REVISION}" \
            --pre-teacher "${PRE_MODEL}" \
            --pre-teacher-revision "${PRE_REVISION}" \
            --output "${ASSET_LOCK}"
    fi
}
