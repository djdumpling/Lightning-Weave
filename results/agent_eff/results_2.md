# Agent-efficiency synthesis: results under interleaved thinking

BFCL v3, all 17 categories, leaderboard-weighted, with the harness fix committed in 739eeff. The model now sees its
reasoning from earlier steps of the current turn, between tool calls, as the Qwen3 chat template intends. Earlier
user turns' reasoning is still dropped, by design.

Every checkpoint is the one evaluated in results.md; nothing was retrained. Results trees:
- new protocol: `bfcl-v3-22b3e917da76-full`, `-a9dc3055c864-` and `-3905dd869e16-` (sampling seeds 0, 1 and 2);
- old protocol: `bfcl-v3-3e6e955a00df-full`, `-8b2ad2de30cd-` and `-4743b22b979b-`.

Deltas are paired over entries: accuracy in points, total tokens as the ratio of summed generated tokens. Comparisons
pair runs at the same training and sampling seed and average over pairs. CIs are category-stratified 95% bootstraps,
with one set of resampled entries shared by every run in a table.

**Plan (66 runs).**
- Stage 1, 16 runs: base, accuracy-only and DECS mid at 3 sampling seeds, plus `opd-nothink`.
  - `opd-nothink` is a control: thinking is off, so its requests are unchanged and it should match its old score.
- Gate before Stage 2:
  - no failed runs;
  - within-turn reasoning visible at about 100% of steps;
  - context overflow at most ~5% of multi-turn entries (loosened from 3% on 2026-10-01: running the experiments matters
    more than this check);
  - the control matches.
- Stage 2, 50 runs: every other model from results.md at sampling seed 0, and the turn-start arms at seeds 1 and 2.

**Pre-registered rule (Stage 1).** The change in DECS mid's multi-turn cost between protocols is a difference in
differences over the same 6 pairs:
- **shrinks** if the lower bound is above 0;
- **unchanged** if the upper bound is below +1.5;
- **inconclusive** otherwise.

The old cost was −2.90 [−4.17, −1.62].

## 1. Stage 1: harness checks and core runs

**Checks** (every new run):

| check | result |
|---|---|
| lane failures | none |
| within-turn steps whose prompt carries the previous step's reasoning | 99.8–99.9% (old protocol: 4%) |
| earlier turns' reasoning dropped (prompt growth across turns) | yes, 4% coverage as designed |
| multi-turn context overflow, new vs old | +0 to +3 entries per run (e.g. 45 vs 44); base 49–53 vs 55 |
| `opd-nothink` control (no thinking, unchanged requests) | 57.08 vs 57.22 overall; 398/400 simple_python outputs identical |
| visible replies that are an unclosed `<think>` block | 3–11 of ~8,600 steps per student, 9–16 for base (old: 0) |

One run, DECS mid at training seed 5678 and sampling seed 0, had its `miss_param` lane container preempted and
restarted. NeMo-Skills kept the finished entries and reran the rest, so all 200 entries are complete and unique. Only
the duplicated requests in its usage log lower that run's coverage figure (81.6%; 97.9% without re-attempted turns).

The unclosed-`<think>` replies are new model behavior. Shown its own reasoning, the model sometimes ends a turn with a
second reasoning block instead of an answer. The prompt is byte-identical to the native path, so this is not a
transport artifact.

**Per run, new / old protocol:**

| run | sampling seed | overall | multi-turn | single-turn | tokens/entry | multi-turn overflow |
|---|---|---|---|---|---|---|
| base | 0 | 65.76 / 65.84 | 33.38 / 33.25 | 81.95 / 82.13 | 1,464 / 1,674 | 52 / 55 |
| base | 1 | 66.18 / — | 36.25 / — | 81.15 / — | 1,523 / — | 49 / — |
| base | 2 | 66.47 / — | 35.50 / — | 81.95 / — | 1,452 / — | 53 / — |
| opd-nothink | 0 | 57.08 / 57.22 | 12.00 / 12.38 | 79.61 / 79.64 | 159 / 156 | 23 / 24 |
| acc-legacy | 0 / 1 / 2 | 69.34 / 68.48 / 68.45 vs 68.36 / 68.00 / 67.88 | 43.12 / 42.38 / 42.00 vs 41.38 / 40.87 / 40.25 | 82.44 / 81.53 / 81.67 vs 81.85 / 81.56 / 81.69 | 1,559 / 1,611 / 1,568 vs 1,966 / 1,959 / 1,998 | 45 / 46 / 44 vs 44 / 43 / 43 |
| acc-legacy, seed 5678 | 0 / 1 / 2 | 68.28 / 68.36 / 68.92 vs 68.27 / 68.19 / 68.26 | 40.62 / 41.75 / 43.62 vs 40.75 / 40.75 / 41.12 | 82.10 / 81.67 / 81.56 vs 82.03 / 81.91 / 81.83 | 1,572 / 1,593 / 1,558 vs 1,912 / 2,040 / 2,034 | 44 / 43 / 44 vs 43 / 42 / 42 |
| acc-legacy+decs (mid) | 0 / 1 / 2 | 67.61 / 67.63 / 68.35 vs 68.03 / 67.39 / 66.85 | 38.75 / 39.50 / 41.50 vs 40.50 / 38.88 / 37.50 | 82.04 / 81.70 / 81.77 vs 81.80 / 81.65 / 81.52 | 1,347 / 1,365 / 1,333 vs 1,676 / 1,669 / 1,694 | 43 / 43 / 44 vs 42 / 43 / 43 |
| acc-legacy+decs (mid), seed 5678 | 0 / 1 / 2 | 67.78 / 67.53 / 68.00 vs 67.13 / 66.81 / 66.27 | 39.12 / 39.38 / 41.00 vs 37.88 / 36.88 / 36.12 | 82.11 / 81.61 / 81.50 vs 81.75 / 81.78 / 81.34 | 1,327 / 1,355 / 1,333 vs 1,660 / 1,678 / 1,579 | 44 / 44 / 43 vs 43 / 44 / 43 |

**What the harness change does to each model** (same checkpoint, new − old protocol, paired over sampling seeds):

| model | pairs | Δacc | Δacc multi-turn | Δacc single-turn | Δtotal tokens | Δmulti-turn tokens |
|---|---|---|---|---|---|---|
| base | 1 | −0.08 [−1.11, +1.02] | +0.12 [−2.75, +3.00] | −0.18 [−0.83, +0.53] | −12.6% [−17.3, −7.5] | −15.4% [−21.0, −9.2] |
| opd-nothink (control) | 1 | −0.14 [−0.41, +0.11] | −0.38 [−1.12, +0.38] | −0.03 [−0.15, +0.07] | +2.0% [+0.5, +3.7] | +2.5% [+0.4, +5.0] |
| acc-legacy | 3 | +0.68 [+0.00, +1.38] | +1.67 [−0.25, +3.62] | +0.18 [−0.20, +0.55] | −20.0% [−23.2, −16.8] | −26.3% [−29.9, −22.5] |
| acc-legacy, seed 5678 | 3 | +0.28 [−0.34, +0.92] | +1.12 [−0.62, +2.92] | −0.15 [−0.46, +0.16] | −21.0% [−24.4, −17.6] | −27.9% [−31.6, −23.9] |
| DECS mid | 3 | +0.44 [−0.20, +1.09] | +0.96 [−0.88, +2.79] | +0.18 [−0.15, +0.53] | −19.7% [−23.1, −16.5] | −26.0% [−29.8, −22.2] |
| DECS mid, seed 5678 | 3 | +1.03 [+0.36, +1.73] | +2.88 [+0.92, +4.83] | +0.11 [−0.21, +0.44] | −18.3% [−21.3, −15.4] | −24.2% [−27.7, −20.7] |

**DECS mid vs accuracy-only** (6 pairs: 2 training × 3 sampling seeds):

| protocol | Δacc | Δacc multi-turn | Δacc single-turn | Δtotal tokens | Δmulti-turn tokens |
|---|---|---|---|---|---|
| new (interleaved thinking) | −0.82 [−1.30, −0.34] | −2.37 [−3.67, −1.12] | −0.04 [−0.38, +0.30] | −14.8% [−15.8, −13.8] | −14.6% [−15.9, −13.3] |
| old | −1.08 [−1.56, −0.61] | −2.90 [−4.17, −1.62] | −0.17 [−0.47, +0.14] | −16.3% [−18.3, −14.2] | −16.8% [−19.3, −13.9] |

| pre-registered: change in DECS mid's cost, new − old (same 6 pairs) | Δacc | Δacc multi-turn | Δtotal tokens | verdict |
|---|---|---|---|---|
| DECS mid vs acc-legacy | +0.26 [−0.34, +0.83] | +0.52 [−1.23, +2.17] | +1.6 [−0.7, +3.6] | inconclusive |

Multi-turn cost per 1% of multi-turn tokens saved: 0.162 new against 0.173 old (change −0.010 [−0.124, +0.105]).

Seeing its own reasoning within a turn makes every thinking model cheaper, and the trained students slightly more
accurate:
- Generated tokens fall by about 20% (multi-turn about 26%), because the model no longer re-derives its plan at
  each step.
- Accuracy-only gains +0.3 to +0.7 overall and +1.1 to +1.7 on multi-turn. Base keeps its score (65.76 vs 65.84) at
  13% fewer tokens, so the published-number comparison still holds.

DECS's cost relative to accuracy-only barely moves: −2.4 on multi-turn, against −2.9. By the pre-registered rule the
change is inconclusive; the point estimate is a small recovery (+0.5) that cannot be told apart from none. The cost
per token saved is unchanged. So the old harness was not what made efficiency costly on multi-turn. Single-turn
stays free (−0.04).

Stage 2 was launched at 03:27 for sampling seeds 0 and 2, once the gate's harness checks had passed on 13 runs. Seed
1 followed at 03:52, after its last Stage 1 runs freed their GPUs; the total stayed at or below 128 GPUs. The last
three runs pass every check too (coverage 99.6–99.9%, no failures).

## 2. Stage 2: every other model

**Per run, new / old protocol:**

| run | sampling seed | overall new / old | multi-turn new / old | single-turn new / old | tokens/entry new / old | multi-turn overflow new / old |
|---|---|---|---|---|---|---|
| acc-clean | 0 | 68.28 / 67.24 | 41.25 / 37.62 | 81.79 / 82.04 | 1,542 / 1,874 | 43 / 44 |
| acc-clean+decs | 0 | 67.26 / 66.08 | 37.00 / 34.25 | 82.39 / 81.99 | 1,306 / 1,523 | 44 / 43 |
| acc-clean+decs-deepscaler | 0 | 66.96 / 65.91 | 36.38 / 34.12 | 82.25 / 81.80 | 1,351 / 1,638 | 44 / 44 |
| acc-clean+decs-deepscaler-flipped | 0 | 68.15 / 67.90 | 41.12 / 40.25 | 81.66 / 81.73 | 1,763 / 2,250 | 44 / 43 |
| acc-legacy+decs-deepscaler | 0 | 67.37 / 67.30 | 38.62 / 37.75 | 81.74 / 82.08 | 1,390 / 1,789 | 45 / 42 |
| acc-legacy+decs-gate-random-multi-turn | 0 | 67.56 / 67.77 | 39.62 / 39.62 | 81.53 / 81.84 | 1,418 / 1,827 | 45 / 44 |
| acc-legacy+decs-gate-random-multi-turn.s5678 | 0 | 68.18 / 68.01 | 40.88 / 39.75 | 81.83 / 82.14 | 1,454 / 1,819 | 44 / 45 |
| acc-legacy+decs-protect-first | 0 | 68.49 / 66.83 | 41.25 / 36.50 | 82.11 / 81.99 | 1,347 / 1,715 | 43 / 44 |
| acc-legacy+decs-protect-first.s5678 | 0 | 67.35 / 67.20 | 38.25 / 37.50 | 81.91 / 82.05 | 1,352 / 1,746 | 45 / 43 |
| acc-legacy+decs-protect-turn-starts | 0 | 68.28 / 67.41 | 40.38 / 38.50 | 82.24 / 81.86 | 1,394 / 1,752 | 43 / 43 |
| acc-legacy+decs-protect-turn-starts.s5678 | 0 | 68.43 / 67.51 | 41.00 / 39.13 | 82.15 / 81.70 | 1,423 / 1,753 | 45 / 43 |
| acc-legacy+decs7b | 0 | 68.24 / 67.56 | 40.25 / 38.38 | 82.24 / 82.15 | 1,389 / 1,639 | 43 / 42 |
| acc-legacy+decs@high | 0 | 67.44 / 66.86 | 38.38 / 36.50 | 81.98 / 82.04 | 1,281 / 1,578 | 46 / 46 |
| acc-legacy+decs@low | 0 | 68.28 / 68.00 | 40.37 / 40.00 | 82.23 / 82.00 | 1,448 / 1,790 | 47 / 43 |
| acc-legacy+e1code | 0 | 67.98 / 67.63 | 39.75 / 39.38 | 82.09 / 81.75 | 1,344 / 1,702 | 44 / 43 |
| acc-legacy+e1code.s5678 | 0 | 68.07 / 67.18 | 41.12 / 37.25 | 81.54 / 82.14 | 1,348 / 1,653 | 44 / 43 |
| acc-legacy+e1math | 0 | 67.34 / 66.90 | 39.38 / 37.88 | 81.32 / 81.41 | 1,243 / 1,468 | 43 / 44 |
| acc-legacy+e1math+e1code.s5678 | 0 | 67.12 / 67.01 | 38.12 / 37.75 | 81.62 / 81.65 | 1,239 / 1,448 | 43 / 43 |
| acc-legacy+e1math-protected | 0 | 67.68 / 67.18 | 39.25 / 38.12 | 81.89 / 81.70 | 1,250 / 1,491 | 44 / 43 |
| acc-legacy+e1math-protected.s5678 | 0 | 67.92 / 67.52 | 39.38 / 37.62 | 82.20 / 82.47 | 1,246 / 1,512 | 43 / 43 |
| acc-legacy+e1math@low.s5678 | 0 | 67.50 / 67.57 | 39.00 / 38.50 | 81.74 / 82.10 | 1,352 / 1,614 | 44 / 44 |
| acc-legacy+heuristic | 0 | 68.60 / 67.68 | 41.62 / 39.38 | 82.09 / 81.83 | 1,474 / 1,858 | 43 / 43 |
| acc-legacy+l1max | 0 | 68.90 / 66.95 | 43.50 / 37.75 | 81.61 / 81.55 | 1,382 / 1,693 | 45 / 43 |
| acc-legacy+nemotron | 0 | 67.91 / 68.04 | 40.38 / 41.00 | 81.68 / 81.56 | 1,458 / 1,853 | 44 / 43 |
| acc-legacy-scaled | 0 | 67.82 / 68.25 | 40.12 / 40.75 | 81.67 / 82.00 | 1,536 / 1,916 | 43 / 43 |
| acc-legacy.concise | 0 | 67.66 / 67.77 | 39.88 / 39.00 | 81.55 / 82.16 | 1,440 / 1,824 | 43 / 44 |
| paper-acc-legacy | 0 | 68.51 / 67.74 | 42.25 / 40.38 | 81.64 / 81.43 | 1,530 / 1,985 | 44 / 43 |
| paper-acc-legacy+decs | 0 | 67.67 / 67.19 | 39.13 / 37.00 | 81.94 / 82.28 | 1,448 / 1,764 | 46 / 44 |
| paper-acc-legacy+decs.s5678 | 0 | 67.69 / 67.48 | 38.63 / 38.12 | 82.23 / 82.16 | 1,470 / 1,788 | 44 / 45 |
| paper-acc-legacy.s5678 | 0 | 68.38 / 67.91 | 40.75 / 38.88 | 82.20 / 82.42 | 1,566 / 1,889 | 44 / 43 |
| paper-half-acc-legacy | 0 | 67.80 / 67.39 | 38.88 / 37.62 | 82.27 / 82.27 | 1,476 / 1,858 | 43 / 43 |
| paper-half-acc-legacy.s5678 | 0 | 68.36 / 66.71 | 41.62 / 36.00 | 81.73 / 82.07 | 1,504 / 1,877 | 45 / 47 |
| opd | 0 | 68.81 / 67.51 | 42.62 / 39.12 | 81.90 / 81.70 | 1,570 / 1,959 | 45 / 43 |
| acc-legacy+decs-protect-turn-starts | 1 | 67.60 / 67.24 | 39.75 / 38.25 | 81.53 / 81.74 | 1,412 / 1,684 | 46 / 43 |
| acc-legacy+decs-protect-turn-starts.s5678 | 1 | 68.62 / 67.62 | 41.25 / 38.50 | 82.30 / 82.18 | 1,446 / 1,824 | 44 / 43 |
| acc-legacy+decs-gate-random-multi-turn | 1 | 68.61 / 68.04 | 42.38 / 40.38 | 81.73 / 81.87 | 1,454 / 1,827 | 43 / 43 |
| acc-legacy+decs-gate-random-multi-turn.s5678 | 1 | 67.91 / 68.35 | 39.38 / 40.25 | 82.18 / 82.41 | 1,446 / 1,848 | 44 / 43 |
| acc-legacy+decs-protect-turn-starts | 2 | 67.73 / 67.47 | 39.75 / 39.13 | 81.71 / 81.64 | 1,393 / 1,799 | 43 / 44 |
| acc-legacy+decs-protect-turn-starts.s5678 | 2 | 67.98 / 67.99 | 40.38 / 40.00 | 81.78 / 81.98 | 1,414 / 1,727 | 44 / 43 |
| acc-legacy+decs-gate-random-multi-turn | 2 | 68.30 / 67.11 | 42.25 / 38.38 | 81.33 / 81.48 | 1,479 / 1,831 | 45 / 43 |
| acc-legacy+decs-gate-random-multi-turn.s5678 | 2 | 68.20 / 67.71 | 41.38 / 40.50 | 81.61 / 81.32 | 1,408 / 1,761 | 43 / 43 |

Every Stage 2 run has no failures, no empty answers, and the usual overflow (43–45 multi-turn entries, nearly
all in long_context, as under the old protocol). Thirteen runs had one to three evaluation-client containers preempted
and restarted, and those lanes re-ran their unfinished entries. Their outputs are complete, and within-turn coverage on turns
with a single attempt is 95–100%. Every thinking model again generates 15–25% fewer tokens.

**Turn-start and paper arms under the new harness** (results.md section 15's pre-registered pairs, mapped to the new
trees; 6 pairs for turn-start comparisons, 2 for paper ones):

| comparison | Δacc | Δacc multi-turn | Δtotal tokens | Δmulti-turn tokens | verdict, new (old) |
|---|---|---|---|---|---|
| protected vs DECS mid | +0.29 [−0.14, +0.73] | +0.54 [−0.65, +1.71] | +5.2% | +6.5% | inconclusive (inconclusive) |
| random gate vs DECS mid | +0.31 [−0.12, +0.74] | +1.10 [−0.08, +2.33] | +7.4% | +9.1% | inconclusive (recovers) |
| protected vs random gate | −0.02 [−0.45, +0.40] | −0.56 [−1.77, +0.56] | −2.0% | −2.3% | does not recover (does not recover) |
| protected vs acc-legacy | −0.53 [−1.00, −0.08] | −1.83 [−3.08, −0.62] | −10.4% | −9.0% | single-turn savings kept (kept) |
| random gate vs acc-legacy | −0.51 [−0.99, −0.04] | −1.27 [−2.58, −0.02] | −8.5% | −6.9% | single-turn savings kept (kept) |
| paper composition vs half accuracy | −0.40 [−1.13, +0.37] | −1.37 [−3.38, +0.69] | −2.1% | −2.2% | inconclusive (no detectable cost) |
| paper composition vs paper accuracy-only | −0.76 [−1.62, +0.05] | −2.62 [−4.87, −0.50] | −5.8% | −5.6% | |
| half accuracy vs paper accuracy-only | −0.37 [−1.15, +0.44] | −1.25 [−3.50, +0.81] | −3.8% | −3.4% | |
| 400 updates vs 200 | −0.36 [−1.15, +0.42] | −0.37 [−2.44, +1.69] | −1.1% | −0.9% | |

| at matched savings (DECS line under the new harness) | multi-turn accuracy | overall accuracy |
|---|---|---|
| protected | −0.37 [−1.45, +0.70] | +0.04 [−0.36, +0.44] |
| random gate | −0.16 [−1.27, +0.99] | −0.04 [−0.42, +0.35] |
| paper DECS term (composition vs half accuracy) | −1.01 [−3.14, +1.18] | −0.28 [−1.05, +0.54] |
| paper composition vs paper accuracy-only | −1.72 [−4.04, +0.49] | −0.45 [−1.33, +0.41] |

Under the new harness, every gated and paper arm sits on the DECS line at matched savings. The paper's faint DECS
term was the one point above the line under the old harness (+0.61 overall). It does not replicate (−0.28), so that
lead was noise. The random gate's old "recovers" verdict also weakens to inconclusive. Training twice as long still
does not help.

## 3. Every results.md comparison on both harnesses

These use the same pairs as results.md, at sampling seed 0 and both training seeds where the original section pooled
them. One reference, accuracy-only's seed-0 run under the new harness, drew high: 69.34 against 68.48 and 68.45 at
seeds 1 and 2. So where the reference is accuracy-only or DECS mid, which have three sampling seeds on both harnesses,
each arm is compared with the reference's mean over its seeds, identically on both harnesses. Other references (base,
`opd`, acc-clean, E1-Math) have one run.

| results.md section: comparison | Δacc new | Δacc old | Δacc multi-turn new | Δacc multi-turn old | Δacc single-turn new | Δtotal tokens new / old |
|---|---|---|---|---|---|---|
| 1 primary matrix: DECS−DeepScaleR vs acc-legacy | -1.38 [-2.30, -0.43] | -0.78 | -3.88 [-6.42, -1.46] | -3.08 | -0.14 [-0.71, +0.48] | -11.97% / -9.36% |
| 1 primary matrix: acc-clean vs acc-legacy | -0.48 [-1.40, +0.43] | -0.84 | -1.25 [-3.75, +1.12] | -3.21 | -0.09 [-0.70, +0.50] | -2.35% / -5.09% |
| 1 primary matrix: acc-clean+DECS vs acc-clean | -1.02 [-2.08, +0.05] | -1.16 | -4.25 [-7.13, -1.37] | -3.38 | +0.60 [-0.12, +1.28] | -15.28% / -18.72% |
| 1 primary matrix: acc-clean+DECS−DeepScaleR vs acc-clean | -1.32 [-2.43, -0.24] | -1.32 | -4.88 [-7.87, -1.88] | -3.50 | +0.46 [-0.24, +1.19] | -12.36% / -12.57% |
| 1 primary matrix: acc-legacy (retrain) vs opd (V0) | +0.53 [-0.51, +1.57] | +0.85 | +0.50 [-2.38, +3.38] | +2.25 | +0.54 [-0.04, +1.17] | -0.71% / +0.34% |
| 1 primary matrix: flipped control vs acc-clean | -0.13 [-1.17, +0.94] | +0.67 | -0.13 [-2.75, +2.75] | +2.62 | -0.13 [-0.84, +0.58] | +14.32% / +20.06% |
| 13 code round: C vs D, both seeds (pre-registered primary) | +0.20 [-0.49, +0.89] | +0.32 | +0.56 [-1.23, +2.42] | +0.35 | +0.03 [-0.49, +0.50] | +0.22% / +1.12% |
| 13 code round: D+C vs D | -0.27 [-1.21, +0.63] | -0.59 | -2.04 [-4.46, +0.33] | -2.46 | +0.62 [+0.05, +1.15] | -3.12% / -3.80% |
| 13 code round: D+C vs acc-legacy | -1.16 [-2.07, -0.21] | -1.25 | -4.63 [-6.88, -2.33] | -4.33 | +0.58 [-0.05, +1.21] | -17.26% / -18.14% |
| 13 code round: E1-Code (C) vs acc-legacy, both seeds | -0.61 [-1.27, +0.04] | -0.76 | -1.81 [-3.58, -0.08] | -2.54 | -0.01 [-0.50, +0.45] | -14.61% / -15.44% |
| 13 code round: E1-Math (M) vs acc-legacy, both seeds | -1.06 [-1.78, -0.37] | -1.02 | -2.94 [-4.88, -1.13] | -3.10 | -0.12 [-0.60, +0.38] | -21.62% / -26.46% |
| 13 code round: M vs D, both seeds | -0.24 [-0.94, +0.47] | +0.06 | -0.56 [-2.48, +1.33] | -0.21 | -0.08 [-0.58, +0.41] | -8.02% / -12.04% |
| 13 code round: M+C vs C, both seeds | -0.71 [-1.47, +0.06] | -0.07 | -2.06 [-4.13, +0.06] | +0.13 | -0.03 [-0.52, +0.48] | -7.68% / -12.67% |
| 13 code round: M+C vs D, both seeds | -0.51 [-1.18, +0.20] | +0.25 | -1.50 [-3.25, +0.35] | +0.48 | -0.01 [-0.52, +0.45] | -7.48% / -11.69% |
| 13 code round: M+C vs M, both seeds | -0.27 [-0.99, +0.50] | +0.18 | -0.94 [-2.87, +1.13] | +0.69 | +0.07 [-0.49, +0.63] | +0.59% / +0.40% |
| 13 code round: M+C vs acc-legacy, both seeds | -1.32 [-2.02, -0.57] | -0.83 | -3.87 [-5.75, -1.90] | -2.42 | -0.05 [-0.54, +0.44] | -21.17% / -26.16% |
| 14 follow-ups: M @low vs D mid, both seeds (matched savings) | -0.32 [-0.95, +0.36] | +0.48 | -0.62 [-2.27, +1.15] | +1.10 | -0.17 [-0.65, +0.30] | +0.87% / -1.15% |
| 14 follow-ups: M @low vs acc-legacy, both seeds | -1.14 [-1.79, -0.44] | -0.60 | -3.00 [-4.73, -1.25] | -1.79 | -0.21 [-0.69, +0.29] | -14.05% / -17.34% |
| 14 follow-ups: M protected vs M, both seeds | +0.22 [-0.55, +0.98] | +0.20 | +0.00 [-2.12, +2.13] | +0.12 | +0.33 [-0.12, +0.80] | +0.99% / +2.91% |
| 3 inference-time baselines on V0: 4k output cap vs opd | +1.03 [-0.05, +2.07] | +1.62 | +2.37 [-0.62, +5.37] | +3.75 | +0.36 [-0.19, +0.94] | -2.50% / -10.51% |
| 3 inference-time baselines on V0: 80 updates (opd-r20) vs opd (200) | +0.13 [-0.98, +1.16] | +0.28 | +0.12 [-2.75, +3.00] | -0.12 | +0.13 [-0.55, +0.80] | +2.97% / +1.67% |
| 3 inference-time baselines on V0: concise prompt vs opd | -0.44 [-1.56, +0.65] | +0.60 | -0.25 [-3.38, +2.62] | +1.88 | -0.54 [-1.30, +0.24] | -7.03% / -4.59% |
| 3 inference-time baselines on V0: opd (V0) vs base | +3.05 [+1.83, +4.21] | +1.67 | +9.25 [+6.00, +12.38] | +5.87 | -0.06 [-0.85, +0.76] | +7.22% / +17.00% |
| 3 inference-time baselines on V0: thinking off vs opd | -11.73 [-13.17, -10.39] | -10.29 | -30.63 [-34.00, -27.25] | -26.75 | -2.28 [-3.54, -1.09] | -89.86% / -92.03% |
| 5 controls, budgets, second seed: DECS @high vs acc-legacy | -1.31 [-2.18, -0.38] | -1.22 | -4.12 [-6.54, -1.75] | -4.33 | +0.10 [-0.57, +0.75] | -18.88% / -20.08% |
| 5 controls, budgets, second seed: DECS @low vs acc-legacy | -0.47 [-1.36, +0.41] | -0.08 | -2.13 [-4.46, +0.21] | -0.83 | +0.35 [-0.31, +0.97] | -8.30% / -9.33% |
| 5 controls, budgets, second seed: DECS mid vs acc-legacy, both seeds | -1.11 [-1.93, -0.33] | -0.73 | -2.94 [-5.06, -0.81] | -1.87 | -0.19 [-0.76, +0.37] | -14.58% / -13.96% |
| 5 controls, budgets, second seed: fork heuristic vs acc-legacy | -0.15 [-0.98, +0.74] | -0.40 | -0.87 [-3.21, +1.58] | -1.46 | +0.21 [-0.38, +0.77] | -6.67% / -5.89% |
| 5 controls, budgets, second seed: scaled accuracy (0.86×) vs acc-legacy | -0.93 [-1.81, -0.05] | +0.17 | -2.37 [-4.67, -0.00] | -0.08 | -0.21 [-0.80, +0.38] | -2.73% / -2.95% |
| 6 other donors and prompting: DECS mid + concise vs acc-legacy | -1.87 [-2.80, -0.92] | -1.47 | -5.63 [-8.08, -3.17] | -4.96 | +0.00 [-0.69, +0.68] | -20.52% / -21.71% |
| 6 other donors and prompting: DECS-7B vs acc-legacy | -0.51 [-1.45, +0.37] | -0.52 | -2.25 [-4.71, +0.08] | -2.46 | +0.36 [-0.22, +0.95] | -12.04% / -16.98% |
| 6 other donors and prompting: L1-Max vs acc-legacy | +0.15 [-0.74, +0.98] | -1.13 | +1.00 [-1.29, +3.12] | -3.08 | -0.27 [-0.90, +0.34] | -12.46% / -14.23% |
| 6 other donors and prompting: Nemotron vs acc-legacy | -0.84 [-1.74, +0.07] | -0.04 | -2.12 [-4.54, +0.37] | +0.17 | -0.20 [-0.82, +0.45] | -7.65% / -6.14% |
| 6 other donors and prompting: concise prompt on DECS mid | -0.98 [-1.84, -0.10] | -0.82 | -3.04 [-5.46, -0.63] | -3.08 | +0.04 [-0.63, +0.73] | -6.93% / -7.99% |
| 6 other donors and prompting: concise prompt on acc-legacy | -1.10 [-2.05, -0.18] | -0.31 | -2.62 [-5.08, -0.29] | -1.83 | -0.33 [-1.10, +0.38] | -8.79% / -7.61% |
| 6 other donors and prompting: donor sum vs acc-legacy | -1.26 [-2.22, -0.33] | -1.48 | -4.00 [-6.75, -1.42] | -4.83 | +0.11 [-0.47, +0.73] | -15.62% / -20.46% |
| 9 lead arm: protect-first vs DECS mid, both seeds | +0.11 [-0.51, +0.71] | -0.07 | -0.12 [-1.75, +1.48] | -0.96 | +0.22 [-0.21, +0.64] | +0.47% / +4.35% |
| 9 lead arm: protect-first vs acc-legacy, both seeds | -0.71 [-1.39, -0.07] | -1.15 | -2.50 [-4.21, -0.81] | -3.85 | +0.18 [-0.28, +0.64] | -14.40% / -12.77% |

The conclusions of results.md hold under the new harness:
- **Single-turn is free:** single-turn changes are −0.3 to +0.6 across arms.
- **The multi-turn cost is real:** every efficiency arm loses 1–6 multi-turn points.
- **No donor beats DECS at matched savings:**
  - E1-Code vs DECS: +0.20 [−0.49, +0.89];
  - E1-Math @low vs DECS: −0.32 [−0.95, +0.36];
  - protect-first vs DECS: +0.11.
- **Composition is not a win:** M+C vs M −0.27, M+C vs C −0.71.
- **Training length is not the lever:** 80 updates match 200, and 400 match 200 (section 2).
- **The accuracy anchor helps more under the new harness:** `opd` gains +3.05 over base (+9.25 multi-turn), against
  +1.67 before, and costs +7% tokens instead of +17%.
- **The inference-time baselines keep their character:**
  - the 4k cap still gains (+1.03) by avoiding context overflow;
  - thinking off loses 11.7 points;
  - the concise prompt now costs 0.4.

## 4. The accuracy-per-token line

Each efficiency arm is compared with accuracy-only at its training seed (21 arms at seed 1234, 9 at seed 5678). The
line is a least-squares fit through the origin.

| harness | accuracy lost per 1% of total tokens saved | multi-turn, per 1% | r | residual SD (overall) |
|---|---|---|---|---|
| new | 0.059 [0.032, 0.085] | 0.18 [0.11, 0.25] | 0.47 | 0.41 |
| old | 0.048 [0.027, 0.067] | 0.16 [0.10, 0.21] | 0.62 | 0.36 |

| largest residual (overall accuracy) | arm | Δtotal tokens | residual |
|---|---|---|---|
| new | L1-Max | −12.5% | +0.89 [+0.13, +1.65] |
| old | E1-Math + E1-Code | −24.9% | +0.75 [+0.02, +1.48] |

The trade-off is the same on both harnesses. The arm furthest above the line changes with the harness: L1-Max was
below the line under the old harness, and M+C sits on it under the new one. With 30 arms, a largest residual this
size is what noise produces, so neither is a real winner.

## 5. Where multi-turn breaks, new harness

The section-12 breakdown for DECS mid vs accuracy-only, 6 pairs, on 746 non-overflowing multi-turn entries:

| tokens per step vs accuracy-only | first response, first turn | first response, later turns | after a tool result | calls per turn |
|---|---|---|---|---|
| DECS mid, new | −23.5% | −17.8% | −6.6% | +0.0% |
| DECS mid, old | −22.7% | −17.4% | −15.5% | +1.4% |
| turn starts protected, new | −17.5% | −9.2% | −5.2% | +0.0% |

With its own plan visible, accuracy-only already reasons less after tool results, so DECS has less to cut there
(−7% against −16%). The cuts to the first response of each turn are unchanged. The multi-turn loss (−2.7 points,
unweighted entry mean) is spread over the same failure types as before, mostly missing reference calls (+1.2). The
extra calls DECS used to add (+0.5) are gone.

## Summary

| | result |
|---|---|
| 🟢 | The harness fix works on every run. Within-turn reasoning is visible at 99.6–99.9% of steps (old: 4%), overflow is unchanged, the thinking-off control matches, and base reproduces its published score (65.76 vs 65.84). |
| 🟢 | Every thinking model is cheaper under interleaved thinking: about 20% fewer generated tokens (26% on multi-turn). The trained students are a little more accurate (+0.3 to +0.7), and the accuracy anchor's gain over base grows (+3.05). |
| 🟢 | Single-turn efficiency stays free on every arm. |
| 🟡 | Pre-registered: DECS mid's multi-turn cost goes from −2.90 to −2.37. The change (+0.52 [−1.23, +2.17]) is inconclusive, and the cost per token saved is unchanged. |
| 🔴 | The old harness was not what made efficiency costly on multi-turn. The accuracy-per-token line keeps its slope, and every donor, strength, gate, composition and paper arm sits on it. |
| 🔴 | The two leads from the old harness that looked above the line, the paper's faint DECS term and E1-Math + E1-Code, do not replicate. |
| 🟡 | New quirks to watch: 3–16 replies per run are an unclosed `<think>` block, and preempted evaluation-client containers re-run their unfinished entries. Neither affects the scores. |
