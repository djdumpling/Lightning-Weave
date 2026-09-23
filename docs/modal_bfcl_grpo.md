# BFCL GRPO on Modal

This recipe trains `Qwen/Qwen3-4B-Thinking-2507` with GRPO on BFCL v4's
executable `multi_turn_base` environment. Each sample is one user turn; prior
turns retain calls/results but no thinking, while the active turn can use up to
20 assistant/tool steps. BFCL's official state and response checkers supply the
terminal verdict, augmented by the shaped reward from Training Gym's BFCL
tutorial.

```bash
bash scripts/train_bfcl_grpo.sh --dry-run
bash scripts/train_bfcl_grpo.sh --smoke-test --wait
# Run the full job only after reviewing the smoke run:
bash scripts/train_bfcl_grpo.sh
# Override only the cumulative prompt/response/tool-observation budget:
bash scripts/train_bfcl_grpo.sh --max-model-tokens 65536
```

The wrapper defaults to `MODAL_ENVIRONMENT=alex-dev-2`, requests one `H100:8`
node, launches detached unless `--wait` is supplied, and logs through the
existing `wandb-secret`.

Key limits are 8,192 prompt tokens, 4,096 generated tokens per assistant step,
32,768 total model tokens, and 20 assistant steps. Sampling follows Qwen's
recommendation: temperature 0.6, top-p 0.95, top-k 20. Qwen's official
`rope_theta=5,000,000` is used with no rope scaling. A response that hits a cap
keeps its environment reward for diagnostics/GRPO normalization but its entire
trajectory is masked out of the policy loss.

The custom rollout forwards SGLang's exact sampled token IDs, log-probabilities,
and top-p replay metadata into Slime. Tool observations are retained in context
but loss-masked, including empty top-p spans so all response metadata stays
token-aligned.

## Review before a full run

- BFCL is an evaluation benchmark, not an official training corpus. The recipe
  reserves the final 30 tasks but training on the other 170 contaminates BFCL as
  an external benchmark; use a separate held-out task source for publishable
  evaluation.
- This first adapter intentionally covers only executable `multi_turn_base`.
  `miss_func`, `miss_param`, long-context, live, and V4 agentic categories need
  their category-specific tool mutation, context, credentials, or graders.
- `Qwen3-4B-Thinking-2507` is not a named Training Gym model class. The recipe
  reuses its compatible Qwen3-4B architecture and overrides the official model
  path and RoPE base; checkpoint conversion still needs smoke validation.
- A 32,768-token training sequence is a deliberately demanding H100 setting.
  The smoke run must confirm memory headroom before the 100-rollout job.
- The shaped reward is useful for RL but is not the official BFCL leaderboard
  score. Report a separate clean BFCL evaluation.

Pinned versions: Training Gym `8899342e709189e2a09a6f9bdadc1e36a28dae79`
and `bfcl-eval==2026.3.23`.
