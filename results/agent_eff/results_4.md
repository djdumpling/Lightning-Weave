# Fresh-state, fixed-anchor DOPD pilot: does training on multi-turn tau2 states help?

Question: the agent-efficiency arms all trained on LoopTool states (12.4% after a tool result; BFCL has about 63%).
Does the same Direct-OPD recipe trained on multi-turn states that trained students actually visit improve multi-turn
accuracy, and does it shrink DECS's incremental multi-turn cost? Only the state pool changes: the anchor and the
response sampler are the clean Qwen3-4B (b0), every student starts from b0, and the targets keep their functions
(acc-legacy; acc-legacy + DECS at the cache's calibrated coefficient 3.1135845716525035, not recalibrated).

Design (pre-registered 2026-10-08; configs/agent_eff/fresh_comparisons.json, evaluation/tau_pooled.py):

| training states | acc-legacy | acc-legacy + DECS mid |
|---|---|---|
| LoopTool cache (existing students) | ae.joint.acc-legacy[.s5678] | ae.joint.acc-legacy+decs[.s5678] |
| fresh tau2 states (new) | ae.fresh.acc-legacy[.s5678] | ae.fresh.acc-legacy+decs-fixed[.s5678] |

- Primary: fresh acc-legacy − cache acc-legacy, on tau2 pass^1 (airline + retail, 4 trials, task bootstrap) and BFCL
  multi-turn (2 training seeds x 3 decoding seeds).
- Secondary: the interaction (fresh DECS − fresh acc) − (cache DECS − cache acc), read with realized token savings.
- tau2 protocol `user235b-4b` (run tau-98179d00b25d-full): Qwen3-4B agents at their native 40,960 window (qwen3
  parser, T 0.6 / top-p 0.95 / top-k 20), self-hosted Qwen3-235B-A22B-Instruct-2507-FP8 user at T 0.

## 1. State collection

Tasks: inclusionAI/AReaL-tau2-data@86971dc0 (training tasks; data_curation/areal_tau2_tasks.py). Excluded: every
airline task naming a tau2 evaluation task's user or reservation (210; the official airline database is reused by
543 AReaL tasks, and one more AReaL database is a near copy); no retail task overlaps; no task's text overlaps an
evaluation task's (word 5-gram Jaccard < 0.2). Eligible: airline 938, retail 563.

Collection (run collect-98179d00b25d-v1, 2026-10-08 03:08–03:41 EDT): one episode per task, collector chosen by
task hash, evaluation decoding, the 235B user at T 0. All 1,163 episodes finished (0 infrastructure failures).

| collector | domain | episodes | gold-action match (diagnostic) | agent turns / episode | context overflows |
|---|---|---|---|---|---|
| acc-legacy s1234 | airline | 300 | 4.8% | 9.2 | 6 |
| acc-legacy s1234 | retail | 282 | 14.2% | 12.5 | 0 |
| acc-legacy+decs s1234 | airline | 300 | 4.7% | 8.7 | 5 |
| acc-legacy+decs s1234 | retail | 281 | 10.3% | 12.9 | 0 |

Gold-action match is tau2's strict action check against AReaL's reference actions; it describes the tasks' difficulty,
not a success rate comparable with evaluation pass^1.

States (data_curation/fresh_states.py): every logged agent request is a candidate. The prompt is the decoding of the
server's own prompt token ids and must re-encode to them exactly: 12,478 requests, 0 capture errors, 0 count or
round-trip mismatches; 123 airline states (1%) exceed 16,384 tokens and are dropped. At most 4 states per episode
(by hash), then 800 per domain x collector cell (pools of 1,119–1,165).

| 3,200 selected states | share | | LoopTool cache |
|---|---|---|---|
| after a tool result | 54.3% | | 12.4% |
| first user turn / later user turns | 11.5% / 34.3% | | |
| collector's action there: text / read / write / transfer | 45.5% / 33.2% / 17.0% / 4.3% | | |
| prompt tokens p50 / p90 / max | 5,694 / 8,490 / 16,345 | | p50 4,548 / p90 6,801 / max 8,192 |

Behavior responses (b0 = clean Qwen3-4B, K 4, T 1, top-p 1, Top-16, 2,048-token cap), whole caches (external audit,
research/followup/artifacts/fresh_completion_full_cache_audit.json):

| 12,800 rows each | LoopTool cache | fresh tau2 states |
|---|---|---|
| trainable response tokens | 5.18M | 9.86M (1.90x) |
| tokens per response: mean / p50 / p90 | 405 / 314 / 725 | 770 / 618 / 1,527 |
| responses cut at the cap (rows / tokens) | 0.41% / 2.10% | 3.95% / 10.49% |
| responses with unfinished thinking (rows / tokens) | 0.29% / 1.46% | 3.24% / 8.62% |

Truncated responses are trained on, as in the original recipe; on fresh states far more supervision falls on reasoning
that never reaches an action.

## 2. Targets on the new states

Same functions, composed on the sealed fresh cache (12,800 rows): acc-legacy (raw agent_acc) and acc-legacy+decs-fixed
(DECS at 3.1135845716525035). Per-token strength, with q built exactly as the tilted loss builds it:

| per-token KL | LoopTool cache | fresh tau2 states | ratio |
|---|---|---|---|
| accuracy target vs behavior, KL(q_acc ‖ b) (whole cache) | 0.0502 | 0.0798 | 1.59 |
| accuracy + DECS target vs behavior (whole cache) | 0.0439 | 0.0691 | 1.57 |
| DECS increment, KL(q_acc+decs ‖ q_acc) (800-row samples) | 0.0145 | 0.0200 | 1.38 |

Whole-cache values: research/followup/artifacts/fresh_training_audit.json. The fixed coefficient does not
fix strength: on multi-turn states both shifts push ~1.4–1.6x harder per token, on ~1.9x the tokens. Fresh-vs-cache
contrasts therefore change the states and, with them, exposure and realized strength; the DECS contrast is read against
its realized token savings.

## 3. tau2 reference: base and the LoopTool-cache students (protocol user235b-4b)

Airline (50 tasks) + retail (114) x 4 trials per model; training seeds pooled; 95% intervals resample tasks within
each domain. Run tau-98179d00b25d-full (99 airline conversations of acc-legacy+decs.s5678 were lost to tunnel
connection errors and rerun).

| model (seeds pooled) | airline pass^1 | retail pass^1 | tokens / conversation (airline / retail) |
|---|---|---|---|
| base | 30.5 | 32.9 | 7,896 / 8,298 |
| cache acc-legacy | 30.5 | 34.1 | 8,395 / 9,083 |
| cache acc-legacy + DECS | 28.2 | 33.4 | 5,598 / 7,903 |

| contrast | all tasks | airline | retail |
|---|---|---|---|
| cache acc − base | +0.8 [−2.4, +4.2] | +0.0 [−6.2, +6.2] | +1.2 [−2.6, +5.2] |
| cache DECS − cache acc | −1.1 [−4.1, +1.8] | −2.2 [−7.0, +2.8] | −0.7 [−4.3, +3.0] |

On tau2 the LoopTool-trained student is indistinguishable from base, and DECS's shortening (−33% airline, −13% retail
tokens) costs no detectable accuracy. (Earlier, with gpt-4.1 users and one trial, V0 vs base was +3.3 [−2.9, +9.5].)

## 4. tau2: fresh-state students (pre-registered readouts)

Same protocol as section 3; 9 models x 656 conversations, all complete (two airline conversations of
ae.fresh.acc-legacy were rerun after tunnel errors; its retail failures.json is stale: every record exists).

| model (2 training seeds pooled) | airline pass^1 | retail pass^1 | tokens / conversation (airline / retail) |
|---|---|---|---|
| base | 30.5 | 32.9 | 7,896 / 8,298 |
| cache acc-legacy | 30.5 | 34.1 | 8,395 / 9,083 |
| cache acc-legacy + DECS | 28.2 | 33.4 | 5,598 / 7,903 |
| fresh acc-legacy | 32.2 | 37.5 | 8,951 / 8,718 |
| fresh acc-legacy + DECS (fixed coef) | 30.5 | 35.5 | 7,456 / 7,690 |

| contrast (points, 95% task bootstrap) | all 164 tasks | airline | retail |
|---|---|---|---|
| PRIMARY fresh acc − cache acc | **+2.9 [+0.0, +5.7]** | +1.8 [−2.5, +6.2] | +3.4 [−0.3, +6.9] |
| fresh DECS − cache DECS | +2.1 [−1.4, +5.6] | +2.2 [−4.5, +9.0] | +2.1 [−1.8, +6.0] |
| DECS cost on fresh states | −1.9 [−4.4, +0.7] | −1.8 [−6.8, +3.0] | −2.0 [−4.9, +1.0] |
| DECS cost on the cache | −1.1 [−4.1, +1.8] | −2.2 [−7.0, +2.8] | −0.7 [−4.3, +3.0] |
| SECONDARY interaction | −0.8 [−4.8, +3.4] | +0.5 [−7.0, +7.8] | −1.3 [−6.2, +3.7] |
| fresh acc − base | +3.7 [+0.2, +7.3] | +1.8 [−5.0, +8.5] | +4.6 [+0.3, +8.9] |

By training seed (all tasks): fresh − cache is +4.1 (acc, s1234), +1.7 (acc, s5678), +2.7 (DECS, s1234), +1.5 (DECS,
s5678): positive in 4 of 4. But the same-arm training-seed difference is as large as the effect (cache acc s5678 −
s1234 = +2.3; fresh acc −0.2), and the bootstrap interval covers task sampling only. Reading: a modest in-domain tau2
gain from training on tau2-like states, consistent in sign, not established beyond training-seed noise. DECS still
saves tokens on tau2 after fresh training (−17% airline, −12% retail vs fresh acc, smaller than on the cache's
students), and its tau2 cost is not resolvable either way.

What changed in behavior (per conversation, both seeds pooled):

| model | airline: transfer to human / agent turns / tool calls | retail: transfer to human / agent turns / tool calls |
|---|---|---|
| base | 48.5% / 8.2 / 4.58 | 34.0% / 11.2 / 5.90 |
| cache acc-legacy | 59.5% / 7.9 / 4.33 | 37.3% / 11.9 / 6.35 |
| cache acc-legacy + DECS | 59.2% / 6.4 / 4.50 | 36.8% / 11.9 / 6.42 |
| fresh acc-legacy | 49.5% / 8.7 / 4.70 | 27.7% / 11.8 / 6.34 |
| fresh acc-legacy + DECS | 41.8% / 9.0 / 4.90 | 27.7% / 12.2 / 6.51 |

The LoopTool-trained students hand conversations to a human more often than base; training on tau2 states cuts
escalation by about 10 points in both domains (and lengthens airline conversations). In a paired analysis (external
review, research/followup/artifacts/fresh_run_audit.json), 37 of the 38 net additional successes are
in episode pairs whose transfer behavior changes. This is an association: transfers change along with everything else.
The cause is open. AReaL's prohibition on transfers is not a sufficient explanation: it appears in 84 of the 600
collected airline tasks and in none of the 563 retail tasks, yet retail escalation fell as much. None of the collected
tasks' gold actions includes a transfer, while some tau2 tasks need one, so the useful next step separates necessary
from unnecessary transfers rather than lowering the rate.

## 5. BFCL v3: fresh-state students (pre-registered readouts)

Interleaved-thinking trees, decoding seeds 0/1/2 (22b3e917da76, a9dc3055c864, 3905dd869e16) x training seeds
1234/5678 = 6 pairs per comparison; the cache cells are the existing runs. Joint entry bootstrap
(evaluation/bfcl_pooled.py; a four-run item is a difference of paired changes). Four of the 12 fresh runs stalled on
lost tunnel responses and finished after the client timeout.

| comparison | Δacc | Δacc multi-turn | Δacc single-turn | Δtotal tokens |
|---|---|---|---|---|
| PRIMARY fresh acc − cache acc | **−0.55 [−1.06, −0.07]** | **−1.40 [−2.77, −0.13]** | −0.12 [−0.47, +0.23] | −4.0% |
| fresh DECS − cache DECS | −0.21 [−0.69, +0.26] | −1.13 [−2.50, +0.08] | +0.25 [−0.08, +0.58] | −3.4% |
| DECS cost on fresh states | −0.48 [−0.97, −0.03] | −2.10 [−3.35, −0.87] | +0.34 [−0.01, +0.68] | −14.3% |
| DECS cost on the cache (reproduces results_2) | −0.82 [−1.30, −0.34] | −2.37 [−3.67, −1.12] | −0.04 [−0.38, +0.30] | −14.8% |
| SECONDARY interaction (fresh − cache DECS cost) | +0.34 [−0.31, +0.97] | **+0.27 [−1.48, +1.96]** | +0.37 [−0.09, +0.82] | +0.5% |

Primary multi-turn per pair: −1.9, +1.1, −2.5, −1.8, −1.1, −2.3 (5 of 6 negative); by category: long context −2.8 (6 of 6
negative), base −1.9, missing function −1.9, missing parameter +1.0; no overflow difference. The interaction per
pair: +3.1, −2.8, +1.9, +2.6, −1.5, −1.8.

## Summary

The pools differ in exposure as well as states (section 2): 1.9x the trainable tokens, a 1.6x stronger per-token
target, and 5x the share of tokens in truncated responses. With global token normalization, 1.9x the tokens is not
1.9x the gradient. This compares two training recipes; it does not isolate state coverage.

| readout | result | verdict |
|---|---|---|
| tau2 (in domain), fresh acc − cache acc | +2.9 [+0.0, +5.7]; 4 of 4 seed contrasts positive | positive, modest; suggestive (training-seed spread is as large) |
| BFCL multi-turn (out of domain), fresh acc − cache acc | −1.40 [−2.77, −0.13]; 5 of 6 pairs negative | negative, small |
| DECS's incremental cost, fresh vs cache states | BFCL multi-turn +0.27 [−1.48, +1.96]; tau2 −0.8 [−4.8, +3.4] | inconclusive: no sign of a rescue, but a change of ~1.5 points either way is not excluded |

What this supports:
1. This tau2 refresh recipe gives a suggestive in-domain tau2 gain and a small BFCL multi-turn loss, most in long
   context. Domain, visitation, response length, truncation and target strength change together; after-tool coverage
   itself was not tested.
2. Fresh states did not reduce DECS's BFCL cost in the point estimates (−2.1 vs −2.4 multi-turn at the same −14%
   tokens), but the interaction is inconclusive; it neither establishes invariance nor an unavoidable cost of shortening.
3. On tau2 the LoopTool students equal base (+0.8). DECS's tau2 effects (−1.1 [−4.1, +1.8] on the cache, −1.9
   [−4.4, +0.7] on fresh states) are compatible with a BFCL-sized loss; not detecting it on tau2 does not show a
   different effect.
4. Fixed coefficients do not fix strength: the same functions are ~1.5x stronger per token on multi-turn states.
5. The most interesting signal is behavioral: escalation to a human drops ~10 points, and the extra tau2 successes sit
   in episodes where that decision changes.

Audit fixes after the run (no effect on these results): the overlap audit now also reads training tasks' gold actions
(one more task, airline_39, is excluded; it contributed 1 of 3,200 states); evaluation/tau_pooled.py now refuses
incomplete trial sets or unshared tasks (this run passes).

## 6. Escalation diagnosis (CPU, existing tau2 records)

Tool: evaluation/tau_escalation.py. A transfer is necessary when the task's gold actions include
transfer_to_human_agents (tau2 v0.1.3: 1 airline task, 4 retail tasks); every other transfer is unnecessary by that
standard. tau2 v0.1.3 grades the final database (and communicated info); its "agent should refuse" NL assertions are
not graded.

Success by task type and transfer decision (acc students, seeds pooled):

| | airline no-write tasks (20) | airline write tasks (30) | retail write tasks (105) |
|---|---|---|---|
| base: transfer / stay | 84% / 24% | 2% / 14% | 10% / 40% |
| cache acc: transfer / stay | 83% / 9% | 3% / 22% | 10% / 44% |
| fresh acc: transfer / stay | 90% / 10% | 2% / 20% | 13% / 43% |

- On airline, a transfer is the reliable way to pass a refusal task: the database stays unchanged. About 70% of every
  model's airline successes are transfers on no-write tasks (base 70%, cache acc 73%, fresh acc 71%). The airline
  score mostly measures "escalate when unsure", not task competence.
- On retail, a transfer almost always fails a write task; staying succeeds about 43% of the time for both students.

Paired fresh vs cache acc (same seed, task and trial), net extra successes:

| transfer change | airline | retail | both |
|---|---|---|---|
| fresh drops an unneeded transfer | −14 | +65 | +51 |
| fresh adds an unneeded transfer | +17 | −28 | −11 |
| transfer unchanged | +5 | −5 | 0 |
| necessary-transfer tasks (5 tasks, any change) | −1 | −1 | −2 |
| total | +7 | +31 | +38 |

Reading: the tau2 gain is a decision-policy change, not a competence change. The fresh students escalate less on
retail write tasks (288 to 209 transfers of 840) and, when they stay, succeed at the same rate (43–44%); that is a real
policy improvement for retail (the policy allows a transfer only when a request is out of scope). On airline the same
shift is neutral to slightly harmful, because unneeded transfers are rewarded on refusal tasks. Necessary transfers
(5 tasks, small n) fall slightly (cache 88% / 78% to fresh 63% / 63%, airline / retail). Unneeded transfers are mostly
the agent's own decision (the user asked for a human in only 7–11% of them).
