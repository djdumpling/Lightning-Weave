# Offline data pipeline

The pipeline is **prepare prompts → collect once → score each anchor pair →
compose → train**. An anchor is a post-trained checkpoint paired with its
pre-training-stage reference. The scripts call the existing data and loss
implementation; they do not implement a second version of the algorithm.

## Models

The default example uses:

| Role | Hugging Face checkpoint |
| --- | --- |
| Student | `Qwen/Qwen3-4B` |
| Klear post-anchor | `Kwai-Klear/Klear-Reasoner-8B` |
| Klear pre-anchor | `Qwen/Qwen3-8B-Base` |
| DECS post-anchor | `pixas/DECS_1.5B` |
| DECS pre-anchor | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |

Download models before running the pipeline. The standard Hugging Face cache
uses an immutable commit SHA as the snapshot directory name:

```bash
export STUDENT_MODEL="$(hf download Qwen/Qwen3-4B)"
export KLEAR_POST_MODEL="$(hf download Kwai-Klear/Klear-Reasoner-8B)"
export KLEAR_PRE_MODEL="$(hf download Qwen/Qwen3-8B-Base)"
export DECS_POST_MODEL="$(hf download pixas/DECS_1.5B)"
export DECS_PRE_MODEL="$(hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)"
```

The wrappers default `STUDENT_REVISION`, `POST_REVISION`, and `PRE_REVISION`
to the corresponding model directory's basename. These are optional metadata
labels; set them to commit SHAs or another useful identifier if desired. They
do not choose a Hub revision or verify the loaded checkpoint.

## Math prompts

```bash
export TASK=math
export RUN_DIR="$PWD/outputs/qwen3-4b-math"
export NUM_PROMPTS=3200
hf download Skywork/Skywork-OR1-RL-Data \
  data/math-00000-of-00001.parquet \
  --repo-type dataset --local-dir data/skywork-or1
SOURCE_INPUT="$PWD/data/skywork-or1/data/math-00000-of-00001.parquet" \
  bash scripts/prepare_data.sh
```

The math converter wraps each question in the DAPO-style instruction and
extracts its answer from the source JSON. It accepts any number of rows.
Collection selects the first `NUM_PROMPTS` prompts meeting the prompt-length
limit.

## Code prompts

Code uses `Kwai-Klear/KlearReasoner-CodeSub-15K`. The preparer reads Parquet
shards in filename order, filters by student-tokenizer prompt length, removes
exact and high-overlap benchmark matches, and deduplicates normalized prompts.
It then shuffles eligible prompts with `SEED` (default 42) and selects
`NUM_PROMPTS`.

The output contains only `prompt`, `label`, and `source_prompt_id`. The label
is `sha256:` followed by the SHA256 of the original ground-truth JSON string.
Executable test bundles are neither parsed nor retained, and training does
not execute them. The preparer does not enforce a source revision, shard hash
allowlist, or fixed row count.

```bash
export TASK=code
export RUN_DIR="$PWD/outputs/qwen3-4b-code"
export NUM_PROMPTS=3200
hf download Kwai-Klear/KlearReasoner-CodeSub-15K \
  --repo-type dataset \
  --revision f6c935197e097ab144d5517a62cc498e0209e060 \
  --include 'default/train/*.parquet' --local-dir data/codesub15k
export SOURCE_INPUT="$PWD/data/codesub15k/default/train"
bash scripts/prepare_data.sh
```

By default, the wrapper downloads the LiveCodeBench raw JSONL files and
creates the benchmark-cache configuration used for decontamination. It calls
`evaluation/livecodebench_v6/lock_dataset.py` with a February 1, 2025 start
date and the 131-problem v6 split; it does not execute Hugging Face dataset
scripts. To use already downloaded raw files, set `LCB_RAW_DIR` to their
directory. Set `LCB_LOCK` to reuse an existing configuration. Code preparation
reads only its `start_date` and the `path` values in `source_files`; benchmark
questions dated on or after that date participate in decontamination. The
preparer does not check the configuration's hashes, schema version, or recorded
question IDs.

## Collect and score

After preparing either task, keep the same `TASK`, `RUN_DIR`, and model
variables for every command:

```bash
bash scripts/collect_rollouts.sh
ANCHOR_NAME=klear bash scripts/score_anchors.sh
ANCHOR_NAME=decs bash scripts/score_anchors.sh
bash scripts/compose_targets.sh
```

Collection runs once. Both scoring invocations read **the same cached
responses, prefixes, candidate IDs, and behavior log-probabilities**.
Do not generate a separate rollout dataset for each anchor.

Defaults are four responses per prompt, a 2,048-token response limit,
temperature 1.0, top-p 1.0, Top-K support of 16, and float32 collection/scoring.
The prompt limit is 1,024 for math and 4,096 for code. These are training
rollout limits, not evaluation budgets.

Each scoring command loads the post-anchor, pre-anchor, and frozen student
reference **sequentially**, then writes the dataset manifest consumed by
training. It uses one GPU by default. `CUDA_VISIBLE_DEVICES` can select that
GPU; `SCORE_BATCH_SIZE` and `SCORE_CHUNK_SIZE` control scoring memory use.
Collection supports `TP_SIZE`; for multiple independent collectors, assign
different `RANK` values and the same `WORLD_SIZE`, using identical settings
and the same output directory. Wait for every collector before scoring.

The wrappers create `assets.json` with model paths, revision labels, and the
tokenizer action spaces needed for scoring. Despite the historical
`ASSET_LOCK`/`--asset-lock` name, this file is scoring metadata, not an audit
gate. Equal token-to-ID mappings use one shared action space; different
mappings select byte-level exact-token projection. Positions unsupported by
an anchor are masked, and composition intersects the per-anchor masks.

## Weights and training hand-off

The composer accepts **raw, nonnegative coefficients**, not probabilities:

```text
delta = KLEAR_WEIGHT * (log p_klear_post - log p_klear_pre)
      + DECS_WEIGHT  * (log p_decs_post  - log p_decs_pre)
```

Default weights are `1.0 + 1.0`. The Qwen3-4B joint recipe uses `ALPHA=1.25`
with these raw weights. For the five-point sweep, DECS fractions
`0.25, 0.375, 0.50, 0.625, 0.75` correspond to:

| DECS fraction | Klear coefficient | DECS coefficient |
| --- | --- | --- |
| 0.25 | 1.50 | 0.50 |
| 0.375 | 1.25 | 0.75 |
| 0.50 | 1.00 | 1.00 |
| 0.625 | 0.75 | 1.25 |
| 0.75 | 0.50 | 1.50 |

Changing to coefficients that sum to one without adjusting alpha changes the
target. A zero coefficient also does not remove that anchor's validity mask;
for a true single-anchor run, use its own scored dataset.

The default composition uses 12,800 unique trajectories and repeats them
twice, yielding 25,600 dataset rows. This matches 100 **rollout iterations**
at 256 rows per iteration. The number of optimizer updates additionally
depends on global batch size; 100 rollout iterations are not 100 optimizer
steps. See the training instructions for that distinction.

```bash
OUTPUT_DIR="$RUN_DIR/student-torch-dist" \
  bash scripts/convert_checkpoint.sh to-megatron
LOAD_DIR="$RUN_DIR/student-torch-dist" DATA_DIR="$RUN_DIR/composed" \
  ALPHA=1.25 bash scripts/train.sh
```

For another anchor pair, use `ANCHOR_NAME=my_anchor POST_MODEL=...
PRE_MODEL=...` when scoring. To compose three or more anchors, pass repeated
`--anchor-manifest`, `--anchor-name`, and `--anchor-weight` arguments directly:

```bash
bash scripts/compose_targets.sh \
  --anchor-manifest "$RUN_DIR/anchors/klear/final/manifest.json" \
  --anchor-name klear --anchor-weight 1.0 \
  --anchor-manifest "$RUN_DIR/anchors/decs/final/manifest.json" \
  --anchor-name decs --anchor-weight 1.0 \
  --anchor-manifest "$RUN_DIR/anchors/my_anchor/final/manifest.json" \
  --anchor-name my_anchor --anchor-weight 1.0 \
  --output-dir "$RUN_DIR/composed-three" --rows 12800 --repeat 2
```

Use a distinct output directory for each coefficient setting and a fresh run
directory when collecting or scoring again; the pipeline does not resume
partial runs. All paths are configurable, and no Slurm scheduler or
organization-specific account is required.
