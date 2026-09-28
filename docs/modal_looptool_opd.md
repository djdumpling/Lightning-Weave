# LoopTool single-anchor Offline Direct-OPD on Modal

This is the agentic-task **single-teacher baseline**, not multi-anchor Lightning
Weave and not GRPO. Modal only schedules CPUs and GPUs; the data schema, anchor
scoring, tilted target, and loss are the repository's native Offline Direct-OPD
implementation.

Training Gym is intentionally not used. Its managed API is designed around an
online rollout/reward/training loop. Offline Direct-OPD instead needs a sealed,
immutable cache built in this order:

1. sample frozen-student responses;
2. score their cached Top-K support with the post-trained anchor;
3. score the same tokens with the pre-trained anchor;
4. cache the frozen student's sampled-token likelihoods;
5. train the student without a rollout engine or live teacher.

Direct Modal functions are smaller than adapting Training Gym around those
offline stages. The GPU runtime is the digest-pinned `tonyhao96/jetmoe:v0.2`
environment published by the upstream Lightning-OPD authors. Rollout generation
uses a separate digest-pinned official vLLM 0.11.0 image, matching the upstream
separation between curation and Megatron training environments.

## Model roles

The arrows describe an anchor delta, not a teacher generating target text:

```text
behavior policy and trainable student: Qwen/Qwen3-4B
pre anchor:                           Qwen/Qwen3-4B-Base
post anchor:                          Qwen/Qwen3-4B-Thinking-2507

delta(a|s) = log p_post(a|s) - log p_pre(a|s)
q(a|s)     proportional to p_frozen_student(a|s) * exp(delta(a|s) / alpha)
```

All three Hugging Face revisions and both container digests are locked in
[`config.py`](../configs/looptool_opd/config.py). The Base and Thinking models
never generate demonstrations and are never loaded during the final training
stage. They only assign token likelihoods to responses already sampled from the
frozen Qwen3-4B behavior policy.

At the pinned revisions, all three tokenizers have the same 151,669-entry
token-to-ID map (canonical SHA-256
`f488fa45d324a8bc64c84f0e27b47223872d550f2a6900564f77dfa67ca5ff4d`).
Anchor scoring therefore uses the behavior tokens directly; no cross-tokenizer
projection or lossy text retokenization is involved.

## Locked recipe

Run this locally to inspect every resolved value without contacting Modal:

```bash
bash scripts/run_looptool_opd.sh plan
```

| Setting | Value |
| --- | ---: |
| learning rate | `1e-6`, constant |
| Direct-OPD alpha | `2.0` |
| global batch size | `64` trajectories |
| cached replay batch | `256` trajectories |
| replay rounds | `50` |
| optimizer updates per round | `4` |
| total optimizer updates | `200` |
| maximum prompt | `8,192` tokens |
| maximum sampled response | `2,048` tokens |
| training sequence limit | `10,240` tokens |
| frozen responses per prompt | `4` |
| selected prompts | `3,200` |
| sealed trajectory rows | `12,800` |
| cached support | Top-K `16` plus the remaining-vocabulary bucket |
| sampling | temperature `1.0`, top-p `1.0` |
| training hardware | one Modal node with `8 × H100` |

The remaining actor settings follow the native Qwen3-4B recipe: Adam with
betas `(0.9, 0.999)`, weight decay `0.01`, gradient clipping at `1.0`, BF16
parameters with FP32 reductions and attention softmax, zero attention/hidden
dropout, TP=1/CP=1/DP=8, full activation recomputation, dynamic batching capped
at 10,240 tokens per GPU, and checkpoints every five replay rounds. W&B is off
by default; Modal logs and persistent checkpoints remain available.

Here `alpha=2.0` is not a GRPO KL-loss coefficient. It divides the anchor log
likelihood shift in the tilted target. The implementation recomputes the current
student distribution over the cached Top-K support plus an aggregate
remaining-vocabulary bucket.

“50 steps” is represented as 50 **cached replay rounds**, which is the meaning
of `--num-rollout` in this Slime launcher. At replay batch 256 and global batch
64, it performs 200 optimizer updates. The launcher prints both quantities. Do
not replace this with `--optimizer-steps 50` unless the experiment definition is
explicitly changed.

## How the 22.8K dataset is used

The CPU stage reruns the pinned LoopTool canonicalization and BFCL contamination
audit. It renders every canonical prompt with the frozen student's own chat
template and its structured tools, then applies the recipe's 8,192-token prompt
limit. Reference answers are not copied into the DOPD prompt file.

The 50-round cache has exactly 12,800 trajectories. With four responses per
prompt, that means 3,200 prompts. They are selected by a stable seeded hash from
the entire eligible canonical pool, rather than taking an accidental source-file
prefix. The summary records the full eligible-pool count and both the eligible
and selected single-turn/multi-turn and target-kind distributions. Caching all
roughly 22.8K prompts would be substantially more expensive while the locked
50-round run could consume only 12,800 trajectory rows; that is deliberately not
the default recipe.

The earlier 22,849 figure used a 10,240-token prompt cap and the Thinking-2507
template. This recipe must remeasure with the actual Qwen3-4B behavior tokenizer
and the mentor-provided 8,192-token cap, so the eligible-pool count will be lower.
The CPU summary makes that difference explicit rather than silently claiming all
22,849 rows fit.

## Modal stages

The wrapper defaults to the existing `alex-dev-2` Modal environment; override it
with `MODAL_ENVIRONMENT` if needed.

```bash
# CPU/tokenizer-and-config stage only. No model weights or GPU are needed.
bash scripts/run_looptool_opd.sh prepare

# Eight one-H100 rollout workers, followed by post/pre/reference scoring.
bash scripts/run_looptool_opd.sh cache --workers 8

# One-time student conversion on 8 H100s.
bash scripts/run_looptool_opd.sh convert

# Actor-only Offline Direct-OPD on 8 H100s. No rollout or teacher servers.
bash scripts/run_looptool_opd.sh train

# CPU export of the latest completed Megatron iteration to Hugging Face format.
bash scripts/run_looptool_opd.sh export
```

`all` executes the full prepare/cache/convert/train/export pipeline in order. It
is provided for later reproduction, but the explicit commands are preferable for
the first run because
they make it easy to inspect the prompt summary and sealed manifest before
spending on training:

```bash
bash scripts/run_looptool_opd.sh all --workers 8
```

No stage is launched by importing the config or by running `plan`.

## Persistent artifacts and restart behavior

The pipeline uses three named Modal Volumes:

- `lightning-weave-hf-models`: immutable Hugging Face snapshots;
- `lightning-weave-looptool-opd-data`: canonical prompts, rollouts, all score
  stages, asset lock, and sealed manifest;
- `lightning-weave-checkpoints`: converted initial weights and training
  checkpoints.

Each rollout/scoring worker owns rank-specific files. Completed files are reused;
the pipeline never overwrites them implicitly. The final manifest locks every
Parquet checksum, source prompt checksum, generation configuration, model
revision, tokenizer hash, and row count. A partial conversion or mismatched
recipe lock fails closed.

Training also refuses an existing output directory by default. Resume only after
checking the checkpoint contents:

```bash
bash scripts/run_looptool_opd.sh train --resume
```

## Checks before the first full job

After `prepare`, inspect `prompts_summary.json` in the data Volume. Confirm the
eligible count, length percentiles, and selected distribution. After `cache`, the
sealed manifest must report exactly 12,800 rows and four responses per prompt.
Only then run conversion and training.

This setup intentionally does not include BFCL training, reference-answer
rewards, SFT, GRPO, multi-anchor composition, or a live tool environment. BFCL
remains an external evaluation target; its exact prompt/schema overlaps are
removed during LoopTool canonicalization.
