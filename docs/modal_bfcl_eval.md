# BFCL evaluation on Modal

This evaluates the LoopTool Offline Direct-OPD student
(`/checkpoints/looptool-offline-dopd-qwen3-4b-v1/hf`), its exact starting point
(`Qwen/Qwen3-4B` at `1cfa9a72`), their inference-time baselines and the
agent-efficiency students on **BFCL v3** (17 categories) under one protocol.
Deltas against a reference model are the result; see
[Comparability](#comparability).

```bash
bash scripts/run_bfcl_eval.sh plan              # resolved protocol, no Modal calls
bash scripts/run_bfcl_eval.sh smoke             # 3 entries per category, base and opd
bash scripts/run_bfcl_eval.sh run               # all 17 v3 categories, base and opd, detached
bash scripts/run_bfcl_eval.sh run --models opd --categories multi_turn_base,live_simple
bash scripts/run_bfcl_eval.sh run --models ae.joint.acc-legacy+decs --seed 1
bash scripts/run_bfcl_eval.sh run --wait-for-checkpoints --max-wait-hours 10 --models ae.joint.paper-acc-legacy
```

`run` is detached: the laptop can disconnect once the app is launched, and the
results land in the Volume either way. With `--wait-for-checkpoints`, a
server-side function starts each agent-efficiency model once its export is
complete. In that mode a model whose results directory already holds a claim or
a manifest is skipped, so a restarted launch never evaluates it twice.

## Models

`--models` takes comma-separated tags (default `base,opd`):

| Tag | Model |
| --- | --- |
| `base`, `opd` | Qwen3-4B and the LoopTool OPD student (V0) |
| `base-nothink`, `opd-nothink` | thinking disabled through the chat template's `enable_thinking` switch |
| `base-cap4k`, `opd-cap4k` | a 4,096-token per-step cap |
| `base-concise`, `opd-concise` | a system instruction to reason only as much as the task requires |
| `opd-r20` | V0 after 20 of its 50 replay rounds (80 of 200 updates) |
| `thinking2507` | Qwen3-4B-Thinking-2507, natively long-context (no YaRN), `deepseek_r1` reasoning parser |
| `ae.<arm>.<variant>[.s<seed>][.concise]` | an agent-efficiency student at `/checkpoints/agent-eff/<arm>/<variant>/seed<seed>/hf` (training seed 1234 by default) |

The cap and the system instruction are applied by the usage proxy (below). Each
manifest records non-default serving settings as `serving_overrides`, and an
agent-efficiency student's training provenance (variant, target revision, α,
passes, seed and final iteration) as `training_provenance`.

## Protocol

| Setting | Value |
| --- | --- |
| Benchmark | BFCL v3, agentic-eval's 17 categories (no memory, no web search) |
| Generation budget | 32,768 tokens per assistant step |
| Serving window | 65,536 tokens (prompt + response), static YaRN `factor` 2.0 over 32,768 original positions |
| Sampling | temperature 0.6, top-p 0.95 per request; top-k 20 from `generation_config.json` |
| Tool calls | native `tool_calls` (`++use_client_parsing=False`), hermes parser |
| Reasoning | split from `content` by vLLM's qwen3 reasoning parser |
| Earlier reasoning | kept for earlier steps of the current turn (interleaved thinking); dropped for earlier user turns |
| Decoding seed | 0; `--seed N` runs a replicate in its own results tree |

This matches the Qwen3 technical report, which sets a 32,768-token max output
and evaluates BFCL multi-turn with YaRN at a 64k context
([arXiv:2505.09388](https://arxiv.org/abs/2505.09388), Section 4.6), and the
mentor's runs (65,536 total, 32,768 response). YaRN applies to every category,
as in the mentor's runs; the report enabled it for multi-turn. The Qwen3-4B
model card recommends `factor` 2.0 for a 64k context and notes that static YaRN
may slightly affect short texts, which applies equally to every model served
with YaRN.
agentic-eval's v3 recipe allows 131,072 generated tokens; the recipe value is
recorded as `recipe_tokens_to_generate` in every manifest.

Prompt lengths at the start of each task (Qwen3-4B chat template, messages plus
tool schemas): single-turn categories have medians of 258–980 tokens (maximum
6,548); the four multi-turn categories start at a median of about 6,000 and a
maximum of 8,005, because each task carries roughly 6k tokens of tool schemas,
and then grow with every assistant turn and tool result. NeMo-Skills sends the
same `max_completion_tokens` with every request, so vLLM rejects any request
whose prompt exceeds 32,768 tokens (65,536 - 32,768); NeMo-Skills scores such a
task as `_ran_out_of_context_` (a failure). With the native 40,960 window the
limit would be 8,192, which a third of multi-turn smoke tasks exceeded. Each lane
reports its `out_of_context` count, and the server refuses to start unless vLLM
reports the 65,536 window and rejects an overflowing probe request with an error
NeMo-Skills recognizes.

**Interleaved thinking (from 2026-10-01).** The model sees its reasoning from
earlier steps of the current turn, between tool calls, as the Qwen3 chat
template intends. Earlier user turns' reasoning is still dropped, by design.
NeMo-Skills always sent that reasoning back in `reasoning_content`, but vLLM
0.11.0 drops the field before templating; 0.11.1 and later pass it through. The
usage proxy therefore moves it into the message text as `<think>...</think>`,
which the template renders byte-identically (`tests/test_bfcl_efficiency.py`).
`--no-interleaved-thinking` reproduces the original protocol. Results from the
two protocols cannot be mixed, and the LoopTool training caches also lack
earlier reasoning in their histories.

**Decoding seed.** NeMo-Skills sends `++inference.random_seed` as every
request's sampling `seed`, so `--seed` passes it there as well as to vLLM's
`--seed`; the engine seed alone would not change sampling.

v4 remains available by setting `Protocol.bfcl_version = "v4"` and
`tokens_to_generate = 8192`. Its two `web_search` categories are never run:
keyless DuckDuckGo search rate-limits every IP (about one query per 15 seconds
here), and no paid search key is configured.

## What is reused, and from where

The benchmark itself is consumed at pinned commits and not modified:

| Piece | Source | Pin |
| --- | --- | --- |
| Category lists, generation budgets, lane planner, leaderboard aggregators | `djdumpling/agentic-eval` `benchmarks/bfcl/` | `36a183a0` |
| BFCL generation driver (`nemo_skills.inference.eval.bfcl`) and inline scoring | NVIDIA-NeMo/Skills | `8979a15f` |
| `bfcl_eval`, test data, official checker | ShishirPatil/gorilla | `86d0374d` |

The NeMo-Skills invocation matches agentic-eval's lane driver apart from the
budget above and scheduling. agentic-eval is a private repository, so
`benchmarks/bfcl/` is copied from a local checkout (`~/agentic-eval`, or
`AGENTIC_EVAL_DIR`); every launch refuses to start unless that checkout is at
the pin with no local changes there.

The lane image is built from the pinned sources rather than agentic-eval's
`nemo-skills-2603.sqsh`, whose registry source is unrecorded. It follows
NeMo-Skills' `Dockerfile.nemo-skills` for the BFCL-relevant steps (including
`ffmpeg`, which `torchcodec` links against) and applies agentic-eval's container
fixes (drop the py2 `pathlib` backport, add `soundfile`). Three resolution rules
keep it equivalent to the pinned code:

- Unpinned dependencies resolve as of 2026-06-02, the day after the NeMo-Skills
  pin (`uv --exclude-newer`). Without this, `mcp` 2.0 (July 2026) breaks the
  driver's imports.
- gorilla's own `==` pins (for example `tree_sitter==0.21.3`, which its Java and
  JavaScript checkers need) are constraints on every later install.
- NeMo-Skills' `compute-eval` requirement is dropped. It is the evaluator for an
  unrelated CUDA benchmark, imported only for `eval_type=compute-eval`, and it
  requires `tree-sitter>=0.25.2`, which would break the BFCL checker.

The build fails unless:

- NeMo-Skills and gorilla are at their pins and the pinned NeMo-Skills is the
  active install;
- every installed package matches gorilla's `==` pins;
- every v3 and v4 category's `test.jsonl` has the same entry ids as gorilla@pin's
  data. Upstream `prepare.py` shallow-clones gorilla **HEAD** while the checker
  grades against the pinned install (5 upstream data commits since the pin at
  the time of writing), so the build points both `prepare.py` scripts at the
  pinned checkout;
- `bfcl_eval.constants.model_config` (the first import of `bfcl_eval
  evaluate`), the driver, and `memory_vector` (with its baked `all-MiniLM-L6-v2`
  encoder) import.

## Serving and scheduling

```text
local entrypoint ─┬─ serve_and_evaluate(model 1) : 4×H100, vLLM DP=4 ─ usage proxy ─ modal.forward ─┐
                  └─ serve_and_evaluate(model N) : 4×H100, vLLM DP=4 ─ usage proxy ─ modal.forward ─┤
                                                                                                    │
                     run_category × 17 per model (CPU, one category each) ◄─────────────────────────┘
```

- **Four H100s per model, every model of a launch concurrently**, up to
  `Serving.max_concurrent_models` (24) servers; further models queue. Each launch
  is its own app, so concurrent launches add their caps.
  Modal falls back to four H200s when H100s are unavailable; the GPU type is
  recorded in `summary.json` and is not part of the protocol.
  One vLLM server per model runs `--data-parallel-size 4`, so each GPU holds a
  full copy of the 4B model; tensor parallelism would only add communication.
  Multi-turn chains are latency-bound (about 110 tokens/s per sequence), so the
  extra GPUs do not speed up a single task; they give the KV-cache room to run
  every task at once.
- **vLLM 0.11.0**, the same digest-pinned image that generated the LoopTool
  rollouts: `--enable-auto-tool-choice --tool-call-parser hermes
  --reasoning-parser qwen3`, `--max-model-len 65536` with the YaRN `rope_scaling`
  passed through `--hf-overrides`, `--max-num-seqs 256`,
  `--max-num-batched-tokens 16384` with V1 chunked prefill, prefix caching
  (multi-turn requests resend history and identical tool schemas), CUDA graphs
  on, and a shared `vllm-cache` compile cache.
- **top-k 20 comes from `generation_config.json`.** NeMo-Skills' OpenAI client
  rejects `top_k`, so the server applies the checkpoint defaults through
  `--generation-config auto`. Startup fails unless vLLM logs defaults of
  temperature 0.6, top-p 0.95, top-k 20.
- **Serving contract probe before any lane starts:** an unauthenticated request
  must be refused (the tunnel is public, so vLLM runs with a random per-run
  `--api-key`), a context overflow must return an error NeMo-Skills recognizes,
  reasoning must be split out of `content`, and a tool request must return
  native `tool_calls`.
- **Usage proxy.** Lanes call `/lane/<category>/v1` on a proxy in front of
  vLLM. It applies the interleaved-thinking rewrite and a baseline's cap or
  system instruction, and logs every completion to `usage.jsonl`: the request's
  user-message digests, tool-schema digest, body digest and history reasoning, as
  the harness sent it, with prompt, completion and re-tokenized reasoning tokens,
  the finish reason and the number of tool calls.
- **One CPU container per category.** `bfcl_eval` hardcodes its result and score
  trees under `/opt/gorilla`, so isolation must be per container; one category
  per container also removes agentic-eval's category queueing inside a lane.
- **Request concurrency** is 200 for each multi-turn category (every task at
  once), 256 for the three largest single-turn categories, and 64 otherwise. agentic-eval used 16 because each of its backends
  served one sequence at a time. Continuous batching means a long thinking
  generation no longer blocks the requests queued behind it, which was the
  head-of-line problem agentic-eval's README describes; the multi-turn
  categories, whose steps are serial, set the total run time.
- **GPU time equals run time.** The server exists only inside
  `serve_and_evaluate`, which exits when its lanes finish.

The server writes one metrics line per minute to the log and to
`vllm_metrics.jsonl`: running and waiting requests, KV-cache usage,
preemptions, prefix-cache hit rate, and prompt/generation tokens per second.
Waiting stuck at 0 with low throughput means the lanes are the bottleneck
(raise `Lanes.concurrency`); a growing queue or preemptions mean the GPU is
(add a replica). Concurrency does not change scores, so it is excluded from the
run id and can be retuned between runs.

## Results and resume

Everything lands in the `lightning-weave-bfcl-eval` Volume under
`<run_id>/<model>/`:

- `manifest.json`: model path, model identity hash, resolved protocol, and any
  serving overrides or training provenance. A rerun with a different model or
  protocol under the same run id fails.
- `bfcl_v3.<category>/output.jsonl`: NeMo-Skills generations, merged with
  per-entry `is_correct` after scoring.
- `scores/BFCL_v4_<category>_score.json`: the `bfcl_eval` score file (it uses the
  `v4` prefix for both versions); its first line is
  `{accuracy, correct_count, total_count}`.
- `aggregate.json` and `summary.json`: the leaderboard aggregate from
  agentic-eval's vendored v3 scorer (the unweighted mean of non-live, live, and
  multi-turn), and per-lane status including `out_of_context`.
- `usage.jsonl` (the proxy's log) and `vllm_metrics.jsonl`.

The run id is `bfcl-v3-<protocol hash>-full` (or `-smoke<N>`), so smoke results
never mix with full ones. The decoding seed and the thinking protocol are part
of the hash:

| decoding seed | interleaved thinking | original protocol (every run before 2026-10-01) |
| --- | --- | --- |
| 0 | `bfcl-v3-22b3e917da76-full` | `bfcl-v3-3e6e955a00df-full` |
| 1 | `bfcl-v3-a9dc3055c864-full` | `bfcl-v3-8b2ad2de30cd-full` |
| 2 | `bfcl-v3-3905dd869e16-full` | `bfcl-v3-4743b22b979b-full` |

Rerunning resumes: a category with a score file is
skipped, and a partially generated category continues from its
`output.jsonl-async`, committed every two minutes.

## Cost analysis

[`bfcl_efficiency.py`](../evaluation/bfcl_efficiency.py) takes a reference run
directory and the runs to compare with it. It computes per-entry cost vectors
(generated, reasoning and prompt tokens, steps, tool calls, duplicate calls,
runaways, overflows), leaderboard-weighted aggregates, and paired,
category-stratified bootstrap comparisons. These include cost on entries both
models got right, outcome-transition strata, the multi-turn common horizon, and
cost per success. Medians and quantiles use the leaderboard category weights.

- **The join.** Logged requests are assigned to entries by their user-message
  sequence, plus the tool schemas for single-turn entries, then by exact
  per-step completion-token counts, which also separates retried attempts. Every
  entry must reconcile exactly, its logged completion tokens equaling
  `num_generated_tokens_list`, or the analysis fails (`--allow-partial-usage`
  overrides).
- Base and OPD in the original tree predate the proxy, so they have no usage
  log; `--tokenizer` gives such runs a reasoning/visible split.

[`bfcl_pooled.py`](../evaluation/bfcl_pooled.py) pools paired comparisons over
replicate pairs (training and decoding seeds) and applies pre-registered decision
rules. [`bfcl_multiturn_failures.py`](../evaluation/bfcl_multiturn_failures.py)
classifies where multi-turn entries first fail. Commands are in the
[agent-efficiency synthesis](agent_efficiency_synthesis.md#running).

## Comparability

- The 32,768-token response and 65,536-token window match the Qwen3 report and
  the mentor's Qwen3-4B runs, not agentic-eval's v3 recipe (131,072 / 131,072),
  so compare against those rather than agentic-eval's reference numbers.
- The LoopTool canonicalization removed exact BFCL prompt/schema overlaps from
  the training data; see [LoopTool preprocessing](looptool_rl_preprocessing.md).
