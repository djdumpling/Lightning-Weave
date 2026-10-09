# tau-bench evaluation on Modal

This evaluates the LoopTool Offline Direct-OPD student
(`/checkpoints/looptool-offline-dopd-qwen3-4b-v1/hf`) and its exact starting
point (`Qwen/Qwen3-4B` at `1cfa9a72`) on the five tau-bench rows Qwen reports
for Qwen3-4B: TAU1-Retail and TAU1-Airline (tau-bench), TAU2-Retail,
TAU2-Airline, and TAU2-Telecom (tau2-bench). The OPD-minus-base delta is the
result; the Qwen numbers are the reproduction target for `base`.

```bash
MODAL_ENVIRONMENT=alex-dev-2 uv run --no-project --python 3.12 --with modal==1.5.5 \
    modal secret create prime-secret PRIME_API_KEY=...   # once: the user simulators
bash scripts/run_tau_eval.sh plan                 # resolved protocol, no Modal calls
bash scripts/run_tau_eval.sh smoke                # 3 tasks per domain, 1 trial, both models, detached
bash scripts/run_tau_eval.sh compare --smoke-samples 3   # the smoke run's comparison
bash scripts/run_tau_eval.sh run                  # all 443 tasks x 4 trials, detached
bash scripts/run_tau_eval.sh run --models opd --domains tau2_airline,tau2_retail --trials 1
bash scripts/run_tau_eval.sh compare              # re-print the comparison from the Volume
```

`smoke` and `run` are detached: the laptop can disconnect once the app is
launched, and the results land in the Volume either way. The cross-model comparison is printed by
the local entrypoint, so after a disconnect print it with `compare`.

### Self-hosted user simulator (`TAU_PROFILE=user30b`)

```bash
TAU_PROFILE=user30b bash scripts/run_tau_eval.sh run --models thinking2507 --trials 5
```

TAU2 only, with Qwen3-30B-A3B-Thinking-2507 (`144afc2f`) as the user instead
of gpt-4.1, so no API credits are used. One 8-GPU node serves both models: the
agent on GPUs 0-3 (data parallel) and the user on GPUs 4-7 (tensor parallel 4),
each behind its own key-protected tunnel, each with its own startup checks,
probe and watchdog. The user runs at its recommended sampling (temperature 0.6,
top-p 0.95, top-k 20) rather than tau2's 0.0, and both servers use vLLM's
`deepseek_r1` reasoning parser, because the 2507 thinking templates open
`<think>` in the prompt and vLLM 0.11's `qwen3` parser would then leave the
reasoning in the message. The agent window is 65,536 (Thinking-2507 has 262,144
native positions), so the base Qwen3-4B cannot run under this profile without
YaRN. An empty user turn is scored 0 as `user_error`, not as the agent's
failure. The profile has its own run id, and its scores are not comparable with
gpt-4.1-user numbers.

### Qwen3-4B agents with the 235B user (`TAU_PROFILE=user235b-4b`)

```bash
TAU_PROFILE=user235b-4b bash scripts/run_tau_eval.sh run --models base,ae.joint.acc-legacy --domains tau2_airline,tau2_retail --trials 4
TAU_PROFILE=user235b-4b bash scripts/run_tau_eval.sh run --models ae.fresh.acc-legacy --trials 4 --wait-hours 20
```

prime's agent settings (native 40,960 window, `qwen3` parser, thinking-mode
sampling) with the self-hosted Qwen3-235B-A22B-Instruct-2507-FP8 user at tau2's
temperature 0, so base, the LoopTool student and the agent-efficiency students
run without API credits (run id `tau-98179d00b25d-full`). `ae.<arm>.<variant>[.s<seed>]`
names a student exported under `/checkpoints/agent-eff/`, as in the BFCL eval.
`--wait-hours` waits on a CPU for each checkpoint (an `ae.*` export is complete
once its provenance file exists), then starts its evaluation.

### Collecting training states (`collect`)

```bash
TAU_PROFILE=user235b-4b bash scripts/run_tau_eval.sh collect --plan plan.json
```

Collector models play AReaL's tau2 *training* tasks (`inclusionAI/AReaL-tau2-data`,
pinned in `config.py`, each task on its own database; audit them against the
evaluation tasks with `data_curation/areal_tau2_tasks.py`). The plan is
`{"run_id": "collect-...", "assignments": {tag: {domain: [task ids]}}}`. Each
episode is one record under `<run_id>/<tag>/<domain>/` holding every agent request
as served: its messages and tools, the response, and the server's own prompt token
ids (from vLLM's `/tokenize`). Nothing is graded except an action-match
diagnostic. `data_curation/fresh_states.py` turns the records into training
prompts (see the `tau2-fresh` state pool in `configs/looptool_opd/config.py`).

## What Qwen published

| Domain | Tasks | Qwen3-4B (thinking) | Qwen3-4B-Thinking-2507 |
| --- | ---: | ---: | ---: |
| TAU1-Retail | 115 | 33.9 | 66.1 |
| TAU1-Airline | 50 | 32.0 | 48.0 |
| TAU2-Retail | 114 | 38.6 | 53.5 |
| TAU2-Airline | 50 | 28.0 | 58.0 |
| TAU2-Telecom | 114 | 17.5 | 27.2 |

Source: the [Qwen3-4B-Thinking-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507),
published 2025-08-06. The reported percentages are compatible with single-trial counts
(32.0 = 16/50, 33.9 ≈ 39/115, 38.6 ≈ 44/114, 17.5 ≈ 20/114), but this does
not identify the number of trials: repeated trials can produce the same
percentages. The local test checks numerical compatibility, not Qwen’s
undocumented evaluation repeat count.

Beyond that, the model cards say only "we set the output length to 32,768"
for non-reasoning tasks, and a [request for the tau settings](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/discussions/8)
(2025-08-20) is unanswered. The closest first-party statement is Qwen's own
tau2-bench leaderboard submissions for Qwen3-Max-Thinking (evaluated
[2025-10-30](https://github.com/sierra-research/tau2-bench/tree/main/web/leaderboard/public/submissions/qwen3-max_qwen_2025-10-30)
and 2026-01-23): tau2-bench v0.1.3, user simulator `gpt-4.1-2025-04-14`,
"standard tau2-bench protocol", unmodified prompts, 4 trials. For TAU1 there
is no Qwen statement; Alibaba's EvalScope tells users to keep tau-bench's
default gpt-4o to match its leaderboard. The protocol below is therefore
each harness's defaults at the version Qwen ran, plus Qwen's documented
generation settings for Qwen3-4B.

## Protocol

| Setting | Value | Source |
| --- | --- | --- |
| TAU1 harness | `sierra-research/tau-bench` @ `4754e6b` (2025-08-27), `tool-calling` agent, `test` split, 30 steps | Tasks unchanged since 2025-01-22; the pin adds only `response_cost or 0`, which a self-hosted model needs |
| TAU2 harness | `sierra-research/tau2-bench` v0.1.3 @ `5ba9e3e` (2025-08-26), `llm_agent`, 200 steps, 10 errors, seed 300 | Tasks and graders identical to v0.1.0-v0.1.2; no commits between 2025-07-15 and 2025-08-26 |
| User simulator | TAU1 `gpt-4o-2024-08-06` (harness sends no temperature, so 1.0); TAU2 `gpt-4.1-2025-04-14` at 0.0 | Harness defaults, resolved to their mid-2025 snapshots; Qwen's tau2 submissions and tau2-bench's reference runs use gpt-4.1-2025-04-14. Served through [Prime Intellect Inference](https://docs.primeintellect.ai/inference/overview) as `openai/gpt-4o` and `openai/gpt-4.1` |
| Agent sampling | temperature 0.6, top-p 0.95, top-k 20, thinking on | Qwen3-4B card: "DO NOT use greedy decoding" in thinking mode; same as the BFCL eval |
| Turn budget | 32,768 tokens per agent turn (server-wide), clipped to the remaining window | Qwen's output length for non-reasoning tasks |
| Window | 40,960 native, no YaRN | Qwen3-4B card: enable YaRN only for contexts past 32,768 |
| History | final answers and tool calls only; reasoning is never resent | Qwen3-4B card's multi-turn guidance; both harnesses, via vLLM 0.11 |
| Trials | 4 (pass^1 is the mean over trials) | tau2-bench's reference runs; Qwen's repeat count is not established |
| Tool calls | native `tool_calls`, hermes parser; tool schemas exactly as each harness builds them | as the BFCL eval |

`v1.0.0` of tau2-bench (2026-03) rewrote 75+ airline and retail tasks, so
current `main` is not comparable with Qwen's numbers; both harnesses are pinned
at their mid-2025 state, and dependencies resolve as of 2025-08-28
(`uv --exclude-newer`; litellm 1.76.0, openai 1.102.0).

### Budget and window: why not BFCL's 32k + 32k YaRN window

Agent prompts, rendered with the Qwen3-4B chat template (policy, tool schemas,
dialogue, tool results) along the published reference trajectories:

| Domain | First turn | Final prompt, median | p99 | Max |
| --- | ---: | ---: | ---: | ---: |
| TAU1 airline (gpt-4o, Sonnet 3.5) | 3,852 | 6,251-7,611 | 12,240-15,960 | 23,067 |
| TAU1 retail (gpt-4o, Sonnet 3.5) | 4,259 | 7,686-8,964 | 12,590-13,959 | 22,773 |
| TAU2 airline (gpt-4.1) | 4,831 | 8,119 | 15,269 | 15,688 |
| TAU2 retail (gpt-4.1) | 4,770 | 8,997 | 13,171 | 13,574 |
| TAU2 telecom (gpt-4.1) | 7,137 | 9,910 | 14,294 | 24,131 |

Per-turn generation from this model on BFCL v3 multi-turn (last night's run,
same serving stack): base median 360 tokens, p99 3,717, p99.9 8,017, with one
runaway of 1,876 turns at the 32,768 cap; OPD maximum 8,280.

So a fixed 32,768-token `max_tokens` inside the native 40,960 window would
overflow on most conversations (any prompt over 8,192), but the 64k YaRN window
is not needed either. The harnesses send no `max_tokens`; vLLM's
`--override-generation-config '{"max_new_tokens": 32768}'` makes every turn's
budget `min(32768, 40960 - prompt)`. The largest measured prompt still leaves a
16.8k-token budget, twice the longest non-runaway turn. The model then runs as
released, without static YaRN, which the Qwen3-4B card warns can degrade
shorter texts and which the OPD student never saw in training (at most 10,240
tokens).

Shortening either limit would not make the run cheaper. vLLM's KV cache is
paged, so the window reserves no memory, and the turn budget is a cap: only
turns that would exceed it change, and those are exactly the ones a shorter
cap would score differently.

The BFCL-style window remains available by setting `max_model_len=65_536` and
the YaRN `rope_scaling` in `Protocol`; that starts a new run id.

## Scoring

- **Success** is reward 1 (DB state and required outputs, as each harness
  grades them). The headline is **pass^1**, the mean success over tasks and
  trials, reported per domain and as unweighted means over TAU1, TAU2, and all
  five. **pass^k** for k up to the trial count uses tau-bench's estimator
  (mean over tasks of C(c, k) / C(n, k)).
- **OPD minus base** per domain comes with a 95% bootstrap interval that
  resamples tasks, paired by task.
- **Failures** follow tau2-bench v1.0's taxonomy. The model's failures score 0
  and are counted by termination reason: `context_window_exceeded` (a prompt
  past 40,960), `agent_error` (for example reasoning that ends with neither a
  message nor a tool call), `bad_request`. A turn that hits the 32,768 cap
  mid-thought has no `</think>`, so vLLM 0.11's qwen3 parser returns the raw
  thinking as the message; the harness passes it on as it would from any
  server, and the next prompt usually overflows. Such turns are counted as
  `truncated_turns`. Infrastructure
  failures (429, 5xx, dropped tunnel) are retried per call with backoff for
  about 4 minutes, then the conversation restarts (3 attempts). A conversation
  that still fails is reported missing, never scored, and `complete` is false.
  A bad key or unknown model stops the lane.
- **Efficiency**, relevant to Lightning Weave's thesis: completion tokens per
  agent turn (mean and percentiles), per conversation, turns per conversation,
  and truncated turns.

## Serving and scheduling

```text
local entrypoint ─ check_user_models (CPU) ─┬─ serve_and_evaluate(base) : 4×H100, vLLM DP=4 ─ modal.forward ─┐
                                            └─ serve_and_evaluate(opd)  : 4×H100, vLLM DP=4 ─ modal.forward ─┤
                                                                                                      │
                          run_lane × 5 per model (CPU, one domain each, OpenAI user simulator) ◄──────┘
```

- **The serving stack is the BFCL eval's**: the digest-pinned vLLM 0.11.0 image,
  hermes tool parser, qwen3 reasoning parser, prefix caching, 4 GPUs per model
  (H200 fallback), a random per-run `--api-key` on the public tunnel. top-k 20
  comes from `generation_config.json` via `--generation-config auto`.
- **Cascade attention is off** (`--disable-cascade-attn`). vLLM 0.11 uses it
  when 8+ running requests share a 256+ token prefix, which every tau lane's
  requests do (the policy prompt). In the first TAU1 run both servers died
  within 22 s of each other with a CUDA illegal memory access (Xid 31) once a
  lane's batch became uniform. It only changes speed, not outputs, so it is not
  part of the protocol hash.
- **A watchdog restarts vLLM** (up to 3 times) if it exits; vLLM shuts down
  when its engine dies. A warm restart takes about 45 s, inside the ~4 minutes
  each agent call is retried, so conversations continue. `server_restarts` is
  recorded in `summary.json`.
- **Startup checks** fail the run unless vLLM reports the 40,960 window and
  default sampling `{temperature 0.6, top_k 20, top_p 0.95, max_tokens 32768}`,
  and unless a probe shows that a prompt 8 tokens short of the window gets at
  most 8 completion tokens (clipping, not an error), a prompt past the window
  gets the error litellm maps to `ContextWindowExceededError`, reasoning is
  split from `content`, and tools return native `tool_calls`.
- **Before any GPU starts**, `check_user_models` makes one tool-calling request
  per user model through Prime (well under a cent): it must return a native
  tool call (telecom's customer calls tools) and report the protocol's
  snapshot or the bare alias; any other snapshot, or a bad key, stops the run.
  The served name is recorded in `summary.json` and on every user call.
- **One CPU container per domain** runs its conversations on a thread pool,
  64 at a time (128 for telecom, whose conversations are about twice as long).
  Each conversation goes through the harness's own entry point
  (`ToolCallingAgent.solve`, `run_task`). The lane only wraps the harness's
  `completion`: agent calls go to the tunnel, user calls go to Prime under the
  snapshot's alias with `PRIME_API_KEY` (plus `X-Prime-Team-ID` if the secret
  has `PRIME_TEAM_ID`), any other model is refused, and every call's usage is
  recorded. Concurrency does not change scores; lower it with
  `--concurrency N` if Prime rate-limits.

## Results and resume

Everything lands in the `lightning-weave-tau-eval` Volume under
`<run_id>/<model>/`:

- `manifest.json`: model path, model identity hash, and resolved protocol. A
  rerun with a different model or protocol under the same run id fails.
- `<domain>/trial<k>/<task>.json`: one conversation: reward, termination,
  per-call agent usage (prompt and completion tokens, finish reason), user
  simulator usage and cost, and the harness's native record (tau-bench's
  `EnvRunResult` or tau2-bench's `SimulationRun`, including the agent's
  reasoning in `raw_data`/`reasoning_content`).
- `<domain>/failures.json`: conversations that are still missing after all attempts.
- `invocations/<time>-<domains>.json`: one per launch: its lanes, probe, GPU
  type, server restarts, user models served; `.vllm_metrics.jsonl` beside it.
- `summary.json`: per-lane metrics and the aggregate, rebuilt from every
  conversation record on disk by `summarize_run` whenever a launch finishes
  or `compare` runs. Launches of different domains (even concurrent ones)
  therefore never overwrite each other's results. Each lane's `user_cost_usd`
  is the user simulator's spend, from each call's tokens at OpenAI's list
  prices (`USER_PRICES`); a self-hosted user costs nothing.
- `<run_id>/comparison.json`: pass^k per domain per model, and paired deltas.

The run id is `tau-<protocol hash>-full` (or `-smoke<N>`). Rerunning resumes
at the conversation level. The trial count is not part of the run id: trial i
uses the same tau2 seed at any trial count, so `--trials 8` later adds trials
4-7 to the same tree.

## Comparability

- Qwen's repeat count and full evaluation configuration are not established.
  Numerical proximity to a published score is not proof of reproduction.
  Compare checkpoints under one pinned protocol, report task-cluster paired
  uncertainty, and treat the published column as contextual reference.
- The user simulators and agent sampling are reconstructions (see above). If
  the mentor has Qwen's actual settings, they are one `Protocol` edit each.
- LoopTool-23k (`b6c572d4`, all 23,040 rows) contains none of the tau domains'
  distinctive tool names (for example `update_reservation_flights`,
  `exchange_delivered_order_items`, `toggle_airplane_mode`,
  `transfer_to_human_agents`) or policy text ("As an airline agent, you can
  help users", "2024-05-15 15:00:00 EST", `###STOP###`). This string audit
  found no direct domain overlap; it does not rule out semantic task overlap
  or independently establish uncontaminated evaluation.
