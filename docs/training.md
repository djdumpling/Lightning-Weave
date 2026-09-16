# Training and checkpoint export

Lightning Weave trains a student on previously collected trajectories and cached
anchor scores. No live anchor model or rollout engine is allocated during this
stage. Prepare and compose the cache before launching training.

## Development status and tests

This is the development repository for the open-source release. The numerical
implementation is extracted from the research code; GPU end-to-end reproduction
with the portable launchers has not yet been validated.

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 python -m pytest -q
```

CPU tests cover target composition, loss behavior, token projection, sealed
data, and launcher settings. Optional distributed tests require their respective
training dependencies and explicitly enabled execution.

The launcher retains the experiment implementation in
`slime/rollout/offline_direct_opd.py`, including the cached Top-K support and
remaining-vocabulary bucket. It does not replace the tested tilted-target loss
with a different KL loss. The runtime names `offline-direct-opd` remain unchanged
for compatibility with the data format.

## Qwen3

Use a GPU environment with the Slime, Megatron-LM, Transformer Engine, SGLang,
and Ray dependencies installed. Set `MEGATRON_PATH` if Megatron-LM is installed
from a source checkout rather than into the active Python environment.

```bash
export STUDENT_MODEL="$PWD/models/Qwen3-4B"
export MODEL_TYPE=qwen3-4B
export MEGATRON_PATH=/path/to/Megatron-LM
export NUM_GPUS=8

OUTPUT_DIR="$PWD/checkpoints/qwen3-4b-initial" \
    bash scripts/convert_checkpoint.sh to-megatron

export DATA_DIR="$PWD/data/qwen3-4b/klear-decs"
export LOAD_DIR="$PWD/checkpoints/qwen3-4b-initial"
export SAVE_DIR="$PWD/checkpoints/qwen3-4b-klear-decs"
export ALPHA=1.25

bash scripts/train.sh --num-rollout 100
```

`DATA_DIR` must contain the sealed `manifest.json` and its Parquet shards. The
manifest locks the student checkpoint and generation configuration, so `--student`
must identify the same checkpoint used to collect the cache. Nondefault collection
settings should also be passed to training (`--max-prompt-length`,
`--max-response-length`, `--temperature`, `--top-p`, `--top-k`, and
`--responses-per-prompt`). As in data preparation, `TASK=code` sets a 4,096-token
prompt limit instead of the 1,024-token math default. Both use a 2,048-token
response limit; these training limits are separate from evaluation budgets.
The Megatron sequence/position limit covers both prompt and response
(`max(4096, prompt_limit + response_limit)`): this is 6,144 for the code
defaults, without changing the 2,048-token response generation budget.

The example is a configurable starting point, not a claim that one hyperparameter
setting produced every paper result. `ALPHA` controls the magnitude of the combined
shift through `delta / alpha`. The composer uses the supplied raw coefficients,
so scaling all anchor weights without scaling `ALPHA` changes the target.
The original Qwen3-4B single-anchor recipe used `ALPHA=2.0`; raw-sum two-anchor
experiments also used `ALPHA=1.25`.

Supported launcher names are:

| `MODEL_TYPE` | Training path |
|---|---|
| `qwen3-1.7B` | Megatron |
| `qwen3-4B` | Megatron |
| `qwen3-4B-Thinking-2507` | Megatron, rotary base 5,000,000 |
| `qwen3.5-4B` | Megatron with the included Qwen3.5 text-model adapter |
| `olmo3-7b-think` | Hugging Face model with FSDP |

The architectures and loss paths come from the research implementation. The
packaged launcher itself has CPU configuration tests; those tests are not a new
end-to-end GPU reproduction. Qwen3.5 requires compatible FLA, Transformers,
Megatron-LM and Transformer Engine builds; it should not be assumed to work in
the older Qwen3 container without those dependencies. The recorded Qwen3.5
runtime used Megatron Core 0.14 with `fla-core==0.4.2`,
`flash-linear-attention==0.4.2`, and `transformers==5.5.3`. Those Python
dependencies were added to the existing CUDA/PyTorch environment rather than
replacing its compiled packages. The converter also requires the `mbridge`
module (distinct from `megatron.bridge`).

Both backends use BF16 training by default: the Megatron argument adapter sets
`bf16 = not fp16`, and FSDP uses BF16 parameters with FP32 reduction. The launcher
does not pass a shared `--bf16` flag because the FSDP parser exposes `--fp16`
instead. The loss calculations retain their original FP32 casts.

## Cached batches versus optimizer updates

`--num-rollout` counts cached batches, not newly generated trajectories.
Each batch contains `ROLLOUT_BATCH_SIZE` rows. With one cached response per row,

```text
optimizer updates per cached batch = ROLLOUT_BATCH_SIZE / GLOBAL_BATCH_SIZE
total optimizer updates = num_rollout × updates per cached batch
```

The default batch sizes are 256 and 64, respectively: 25 cached batches perform
100 optimizer updates, whereas 100 cached batches perform 400 updates. Choose
either `--optimizer-steps` or `--num-rollout`; the launcher prints both counts.
The default is 100 cached batches (400 optimizer updates with these batch
sizes), retaining the original training horizon. `--save-interval` counts cached batches.
The cache must provide at least `num_rollout × ROLLOUT_BATCH_SIZE` rows;
replaying rows is a data-preparation decision, not an extra hidden training loop.

## OLMo

The OLMo recipe loads its native Hugging Face model to preserve its sliding/full
attention and rotary-embedding configuration. Input conversion is unnecessary.

```bash
STUDENT_MODEL="$PWD/models/Olmo-3-7B-Think" \
MODEL_TYPE=olmo3-7b-think \
DATA_DIR="$PWD/data/olmo3-7b/klear-decs" \
SAVE_DIR="$PWD/checkpoints/olmo3-7b-klear-decs" \
    bash scripts/train.sh --num-rollout 100
```

Unset `LOAD_DIR` for a fresh FSDP run. For either backend, resume explicitly by
setting `LOAD_DIR` to the existing training checkpoint root. Do not use a
different student's cached data when resuming.

## Ray and command preview

The launcher starts a local Ray head unless `RAY_JOB_ADDRESS` points to an existing
Ray dashboard, for example `http://127.0.0.1:8265`. The launcher does not stop Ray
or kill unrelated processes. A local Ray head started here remains available
after the job; stop that dedicated head yourself when finished.

The local head and dashboard bind to `127.0.0.1`. Jobs use absolute paths on the
same machine. No Ray `working_dir` or `py_modules` upload is configured, so the
launcher does not package the repository, model weights, caches, or `.git`
directory for transfer. If using an existing Ray service, its workers must
already have access to these paths and the same installed environment.

```bash
bash scripts/train.sh --num-rollout 100 --dry-run
```

## Export to Hugging Face

Select a completed `iter_XXXXXXX` directory from the training output. The
Megatron output uses the zero-based cached-batch index; its 100-batch example
ends at `iter_0000099`. FSDP uses a one-based directory suffix, so its equivalent
output is `iter_0000100`. The directory name is not an optimizer-update count.

```bash
INPUT_DIR="$SAVE_DIR/iter_0000099" \
OUTPUT_DIR="$PWD/checkpoints/qwen3-4b-klear-decs-hf" \
    bash scripts/convert_checkpoint.sh to-hf
```

For OLMo, set `MODEL_TYPE=olmo3-7b-think`; the wrapper uses the FSDP converter
with strict tensor-name matching. Export to a new output directory. Only load
checkpoints from sources you trust: distributed checkpoints include serialized
Python metadata.
