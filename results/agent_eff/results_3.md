# Agent-efficiency synthesis: the decision-preserving projection

Pre-registration and method: docs/agent_efficiency_synthesis.md, "Decision-preserving projection". Three arms train
on the same samples of the recipient (acc-legacy, training seed 1234) and differ only in their weights: uniform,
ordinary tilt, and projected tilt (the ordinary tilt with each decision's probability held fixed).

## 1. The recipient's samples (2026-10-02)

8 responses for each of the 3,200 LoopTool prompts. The decoder matches BFCL's (T 0.6, top-p 0.95, top-k 20), with
no YaRN and at most 2,048 response tokens. Run: chain fc-01M3XH2740GG8G1HZZQ7BX3KPY, 8 H100s, about 25 minutes.
Output: data volume `looptool-qwen3-4b-v1/projection/rollouts`.

| quantity | value |
|---|---|
| rows / prompts | 25,600 / 3,200 (8 each, 32 shards) |
| ended by the stop token / at the length limit | 25,377 / 223 (0.9%) |
| tool calls / text replies / unclosed reasoning | 22,214 / 3,191 / 195 |
| response tokens: mean / median / p90 | 436 / 329 / 784 |

A decision is the server's exact visible content and finish reason. Base-shard reference: frozen Qwen3-4B, 4 samples
for each of 200 prompts.

| decision structure | recipient, 8 samples | base shard, 4 samples |
|---|---|---|
| samples sharing a decision with another sample | 83.9% | 75.4% |
| prompts where every sample makes the same decision | 62.3% (1,992) | 60.5% (121) |
| prompts where every sample decides differently | 10.6% (339) | 18.0% (36) |
| most a projected target can cut (each decision's shortest sample) | 16.7% | 12.4% |
| most an ordinary target can cut (each prompt's shortest sample) | 27.8% | 26.0% |

The pre-registered 15% target is below the projection's 16.7% bound, but at 90% of it. Reaching 15% means putting
nearly all of each decision group's weight on one sample, and only works if DECS's scores rank responses by length
within a decision. The calibration needs the donor scores (the next stage).

## 2. Donor scores, weights and training (2026-10-02)

**Reasoning-only DOPD scores** (DECS against R1-Distill-1.5B, float32, 8 H100s, about 70 minutes). Each score covers a
response's reasoning through `</think>`.

| quantity | value |
|---|---|
| scored responses | 25,600 (all finite; 99.2% close their reasoning) |
| score: mean / sd | −27.2 / 14.7 nats (−0.085 nats per reasoning token: DECS finds this reasoning less likely than R1-Distill does) |
| correlation with reasoning length | −0.83 |
| within-decision Spearman with length | median −0.62; negative in 89% of groups |

DECS prefers shorter reasoning within the same decision. Its score is close to a per-token length penalty: the
per-token sd, 0.030, is a third of the mean.

**Calibration.** The pre-registered 15% is out of reach: the projected target peaks at 11.35% implied savings near
α = 0.16, because DECS's favorite response is not always the shortest. The weights stage therefore used the approved
fallback, 10%, reached at α = 2.880. All arms share that α.

| at α = 2.880 | uniform | ordinary | projected |
|---|---|---|---|
| implied cut in response tokens | 0 | 18.1% | 10.0% |
| effective sample size per prompt (fraction of 8): median / p10 | 1 / 1 | 0.39 / 0.18 | 0.48 / 0.23 |
| change in decision frequencies (total variation per prompt): mean / p90 | 0 | 0.142 / 0.54 | 0 (largest group-mass error 3e−15) |

**Training.** Each of the 6 students (3 arms × training seeds 1234 and 5678) starts from the recipient. Each makes one
pass over its arm's 25,600 weighted rows: 100 batches of 256, 400 updates, learning rate 1e−6, about 30 minutes on
8 H100s. Every run ended at iteration 99 on its arm's sealed target. Seed 1234 read the rows in stored order; seed 5678
read them shuffled.

## 3. BFCL: the pre-registered comparisons (2026-10-02)

18 full BFCL v3 runs: 3 arms × training seeds 1234 and 5678 × decoding seeds 0, 1, 2, with request logging. Each pair
shares its training seed and decoding seed; the recipient's own runs at decoding seeds 0–2 (results_2.md) are the
reference. Every run has all 17 lanes scored, and 43–47 multi-turn context overflows (the recipient's range). One lane
(ordinary, seed 1234, decoding seed 1, `multi_turn_miss_param`) failed on a transient DNS error and was rerun alone.
Two lanes waited out lost tunnel responses and recovered by themselves.

| comparison (6 pairs) | Δacc | Δacc multi-turn | Δacc single-turn | Δtotal tokens | Δmulti-turn tokens | Δsingle-turn tokens | pre-registered verdict |
|---|---|---|---|---|---|---|---|
| uniform vs recipient | −0.13 [−0.66, +0.40] | −0.56 [−2.04, +0.87] | +0.08 [−0.29, +0.44] | +0.0% [−1.4, +1.4] | −1.0% [−2.9, +0.8] | +2.4% [+1.1, +3.8] | non-inferior: inconclusive; tokens: not reduced |
| ordinary vs recipient | −0.49 [−1.04, +0.07] | −1.00 [−2.44, +0.35] | −0.24 [−0.61, +0.13] | −5.0% [−6.3, −3.6] | −5.2% [−7.0, −3.3] | −4.4% [−5.7, −3.1] | non-inferior: inconclusive; tokens: reduced |
| projected vs recipient | −0.39 [−0.95, +0.14] | −0.50 [−1.98, +0.94] | −0.34 [−0.70, +0.03] | −3.3% [−4.7, −2.0] | −4.1% [−6.0, −2.3] | −1.5% [−2.8, −0.2] | non-inferior: inconclusive; tokens: reduced |
| projected vs ordinary | +0.10 [−0.32, +0.53] | +0.50 [−0.69, +1.62] | −0.10 [−0.37, +0.16] | +1.7% [+0.6, +2.9] | +1.2% [−0.4, +2.7] | +3.0% [+1.9, +4.1] | recovery: inconclusive |
| projected vs uniform | −0.26 [−0.69, +0.15] | +0.06 [−1.08, +1.21] | −0.42 [−0.72, −0.11] | −3.4% [−4.3, −2.4] | −3.2% [−4.4, −1.9] | −3.8% [−4.9, −2.8] | (no rule) |
| ordinary vs uniform | −0.36 [−0.81, +0.07] | −0.44 [−1.63, +0.73] | −0.32 [−0.62, −0.02] | −5.0% [−6.0, −3.9] | −4.3% [−5.7, −2.7] | −6.7% [−7.8, −5.5] | (no rule) |

Matched savings (pre-registered, 3-point tolerance): projected saves 3.3% and ordinary 5.0% of total tokens against
the recipient, a 1.6-point difference, so "projected vs ordinary" is at matched savings.

| against the DECS line (0.059 overall, 0.18 multi-turn per 1% of total tokens) | saved | Δacc | line | residual | Δacc multi-turn | line | residual |
|---|---|---|---|---|---|---|---|
| uniform | 0.0% | −0.13 | 0.00 | −0.13 | −0.56 | 0.00 | −0.57 |
| ordinary | 5.0% | −0.49 | −0.29 | −0.20 | −1.00 | −0.90 | −0.10 |
| projected | 3.3% | −0.39 | −0.20 | −0.20 | −0.50 | −0.60 | +0.10 |

**Exploratory: environment-rejected calls.** Replaying every call through BFCL's executor, the recipient's call steps
are rejected 16.3–17.4% of the time. Mean change over the 6 pairs: uniform +0.26 points, ordinary +0.63 (positive in
6 of 6 pairs), projected +0.33 (4 of 6). Per 1% of tokens saved, the ordinary arm's rise (~0.13) matches DECS mid's
(+1.6 points at 14.8%), and the projected arm's is no larger than the uniform control's. The per-pair noise is about
±0.6 points, so this is a hint only.

Every pre-registered verdict is inconclusive. The realized BFCL dose is small, 3–5% of total tokens. (The targets'
10% and 18% are reweighted lengths of cached LoopTool responses, a different quantity, so their ratio is not a
measure of fitting.) At that dose the DECS line predicts a multi-turn cost of only 0.6–0.9 points, below what 6
pairs resolve (about ±1.5 points; even the uniform arm, which saves nothing, cannot be shown non-inferior). Both
tilted arms sit on the line. Projected vs ordinary, +0.50 [−0.69, +1.62] multi-turn, is not evidence of a projection
advantage. At training seed 1234 the two arms saved the same (−4.11% vs −4.14%) and projected was −0.17 multi-turn
points against ordinary. The pooled difference comes from seed 5678, where projected also shortened less (−2.6% vs
−5.8%). With the historical slope, the dose difference accounts for about 0.29 of the 0.50 points.
The rejected-call hint is dose-confounded in the same way: the savings difference predicts about 0.2 of the 0.30-point
gap between ordinary and projected.

## 4. Decision drift on frozen histories (exploratory)

The histories are 1,000 BFCL multi-turn requests (500 turn starts, 500 after a tool result), frozen (sha256
88e782f9…) from a logged accuracy-only run at BFCL decoding seed 3. That run scored 42.0 on multi-turn and is kept out
of the accuracy comparisons. Each model gave 4 responses per history with BFCL's serving settings.

| view | recipient vs its own second draw: total variation / match rate | saturated (the two draws never agree) | share of histories with a repeated decision |
|---|---|---|---|
| byte-exact | 0.406 / 0.548 | 27.4% | 68–71% |
| call-level | 0.107 / 0.856 | 1.3% | 97–99% |

| student | mean tokens (recipient 598) | excess total variation, byte-exact | excess total variation, call-level | match deficit, call-level |
|---|---|---|---|---|
| uniform, 1234 / 5678 | 594 / 583 | −0.004 / −0.009 | +0.002 / −0.013 | −0.000 / −0.011 |
| ordinary, 1234 / 5678 | 577 / 556 | +0.000 / −0.001 | −0.001 / −0.001 | −0.001 / −0.001 |
| projected, 1234 / 5678 | 576 / 579 | +0.000 / −0.003 | −0.001 / −0.005 | −0.003 / −0.003 |

Bootstrap 95% intervals are about ±0.01 for every entry. The result is inconclusive, not a demonstrated null: low
saturation removes one failure mode but does not make the statistic sensitive. With 4 samples, a shift from (0.5, 0.5)
to (0.6, 0.4) between two calls, a true total variation of 0.10, gives an expected excess of only about 0.009, and
signed excesses can cancel across histories. The replay also gives every model the recipient's earlier reasoning, so it
measures conditional behavior at fixed states, not what follows a student's own shortened plan. An unbiased squared
distance (pooled recipient draws, whole tasks resampled) gives projected minus ordinary, call-level, 0.00073
[−0.00466, +0.00642]: unresolved.

## 5. The probe of the trained students (2026-10-02)

Pre-registered in `PROJECTION_PROBE` and the docs. Prompts: 400 training prompts (by a fixed hash) and the 600 held-out
LoopTool prompts of the reasoning-value probe. Each model gave 8 responses with the collection settings (T 0.6,
top-p 0.95, top-k 20, at most 2,048 tokens). The recipient sampled twice in disjoint seed blocks (pooled as the
reference, 16 responses), and every student sampled once. Teacher forcing scored the 3,200 cached responses of the
training prompts. Run: chain fc-01M3ZAW2VHGB17PC5W1T5CQ2JM. The first launch wrote every sample, then ran out of GPU
memory in teacher forcing: vLLM's full-vocabulary float32 prompt log-probabilities are outside its memory profile.
Teacher forcing now runs in its own process with smaller steps. The samples were kept.

Recipient mean response tokens: 445 on training prompts, 420 held-out. Recipient against its second draw, call-level
squared distance: +0.0021 on training prompts, −0.0015 held-out.

**Pre-registered readings.** The realization is relative to the recipient. "Within margin" means below 0.0258, half
the ordinary target's own call-level squared distance on the training prompts (0.0515).

| student | savings, training prompts | realization (target 19.1% / 11.0%) | savings, held-out | generalization | call-level squared distance, training / held-out | flags (movement, margin), training / held-out |
|---|---|---|---|---|---|---|
| uniform 1234 | 2.5% [0.8, 4.4] | — | −0.2% [−1.5, 1.2] | — | 0.0145 / 0.0017 | yes, yes / no, yes |
| uniform 5678 | 1.0% [−0.8, 2.8] | — | 0.1% [−1.3, 1.4] | — | 0.0050 / 0.0033 | no, yes / no, yes |
| ordinary 1234 | 7.6% [5.8, 9.4] | 0.40 [0.30, 0.49] inconclusive | 4.2% [2.7, 5.6] | 0.56, inconclusive | 0.0141 / 0.0046 | yes, yes / yes, yes |
| ordinary 5678 | 8.6% [6.7, 10.5] | 0.45 [0.35, 0.55] inconclusive | 4.5% [3.2, 5.9] | 0.53, inconclusive | 0.0155 / 0.0049 | yes, yes / no, yes |
| projected 1234 | 6.1% [4.4, 7.9] | 0.56 [0.39, 0.71] inconclusive | 2.7% [1.4, 4.0] | 0.45, inconclusive | 0.0142 / 0.0017 | yes, yes / no, yes |
| projected 5678 | 5.0% [3.3, 6.7] | 0.46 [0.30, 0.61] inconclusive | 2.2% [0.8, 3.6] | 0.44, inconclusive | 0.0138 / 0.0019 | yes, yes / no, yes |

| teacher-forced fit (raw T = 1, descriptive) | uniform | ordinary | projected |
|---|---|---|---|
| slope of log-likelihood change on centered log-weight, 1234 / 5678 | −0.46 / −0.51 (on the ordinary weights) | 0.80 / 0.70 | 0.74 / 0.69 |
| R² of that regression, 1234 / 5678 | | 0.25 / 0.23 | 0.20 / 0.20 |
| slope after subtracting the uniform student's changes, 1234 / 5678 (exploratory) | | 1.27 / 1.21 | 1.07 / 1.05 |
| mean log-likelihood change of the cached responses (nats), 1234 / 5678 | +2.2 / +3.1 | −1.2 / −0.5 | +0.0 / +0.7 |

Held-out exact-call accuracy: recipient 75.8%, every student 75.5–75.9%.

Every realization and generalization reading is inconclusive. Under the pre-registered rule, the fit slope then
decides whether optimization comes first, but the slope cannot carry that decision. It is not a fraction fitted: it
explains a fifth to a quarter of the variance, and subtracting the uniform student's changes moves it from 0.69–0.80
to 1.05–1.27. The uniform student, which fits no weights, has a slope of about −0.5. Training raises a response's
summed log-likelihood roughly in proportion to its length, and the log-weights fall with length, so the regression
mixes fitting with a length effect. It also sees only the cached strings, at T = 1, while training fits T = 0.6. With
no usable slope, the default is to examine optimization first. Decision movement on the training prompts is detectable for every arm, including the
uniform control, and every arm is within the margin. Fitting the recipient's own 8 samples moves decisions by itself.
The pre-registered distance therefore cannot separate the tilt's decision shift from this generic movement.

**Exploratory (not pre-registered): the uniform control and the direction of decision change.**

*Savings against the uniform student of the same training seed.* This removes the generic effect of fitting the
recipient's own samples. That effect shortens training-prompt responses (uniform 1234: 2.5%) but does not carry to
held-out prompts.

| student | training prompts | realization against uniform | held-out | held-out / training | BFCL total tokens against uniform (section 3, both seeds) |
|---|---|---|---|---|---|
| ordinary 1234 / 5678 | 5.2% [2.8, 7.4] / 7.6% [5.5, 9.8] | 0.27 [0.15, 0.39] / 0.40 [0.29, 0.51] | 4.4% / 4.5% | 0.84 / 0.59 | 5.0% [3.9, 6.0] |
| projected 1234 / 5678 | 3.7% [1.8, 5.6] / 4.0% [2.2, 5.8] | 0.34 [0.16, 0.51] / 0.37 [0.20, 0.53] | 2.9% / 2.1% | 0.78 / 0.53 | 3.4% [2.4, 4.3] |

*Movement along the ordinary target's decision shift.* On each training prompt, t = q_ordinary − p̂ is the change in
call-level decision frequencies that the ordinary target asks for (over the cached responses; nonzero on 24.5% of
prompts). For each student, μ = Σ ⟨f_student − f_recipient, t⟩ / Σ |t|². Here f are frequencies of fresh samples,
which are independent of the cached ones, so the numerator is unbiased. μ = 1 means the student moved its decisions as
far as the ordinary target asked. Prompts are resampled; for pooled entries, the two seeds are averaged per prompt.

| student | μ, seed 1234 | μ, seed 5678 | μ, pooled |
|---|---|---|---|
| uniform | −0.01 [−0.14, +0.14] | −0.07 [−0.17, +0.03] | −0.04 [−0.14, +0.06] |
| ordinary | +0.18 [+0.05, +0.32] | +0.24 [+0.11, +0.37] | +0.21 [+0.10, +0.34] |
| projected | −0.00 [−0.12, +0.10] | +0.07 [−0.05, +0.20] | +0.04 [−0.06, +0.13] |
| recipient's second draw (control) | | | +0.06 [−0.06, +0.18] |
| ordinary − projected | | | **+0.17 [+0.09, +0.27]** |
| ordinary − uniform | | | +0.25 [+0.16, +0.34] |
| projected − uniform | | | +0.08 [−0.00, +0.16] |

Held-out, call-level squared distance to the recipient, seeds averaged per prompt: ordinary − projected +0.0030
[−0.0009, +0.0069], ordinary − uniform +0.0022 [−0.0017, +0.0062], projected − uniform −0.0007 [−0.0046, +0.0033].

**What this shows.**

- **The projection suppresses the ordinary target's requested decision shift on training prompts.** The ordinary
  students moved their decisions about a fifth of the way along that shift. The projected students did not move along
  it more than the uniform control (the +0.08 against uniform is at the edge of zero). The ordinary − projected
  contrast, +0.17 [+0.09, +0.27], is clear in both seeds. This is exploratory, and narrower than "decisions are
  preserved". It measures one direction, and the total decision distances of the two arms are similar (about 0.014 on
  training prompts). Held-out prompts point the same way but are not resolved. Nothing shows that the suppressed shift
  is the one that costs accuracy.
- **Much of the dose is lost before BFCL.** Against the uniform control, about a third of the target's implied
  savings reaches fresh samples on the training prompts themselves (0.27–0.40). New LoopTool prompts keep 0.53–0.84 of
  that, a further loss. BFCL savings are similar to held-out LoopTool savings (ordinary 5.0% against 4.4–4.5%,
  projected 3.4% against 2.1–2.9%), but they are different measures, so this is reassuring, not proof that BFCL adds no
  loss. The cause of the first loss is not identified. Candidates: too little optimization, the 8-response
  approximation, interference through shared parameters, and the gap between raising cached-sequence likelihoods and
  changing stopping behavior on fresh prefixes.
- **This one-pass configuration produced too little savings for the BFCL comparison to resolve the intended
  benefit.** At 3–5% the DECS line predicts −0.6 to −0.9 multi-turn points, below the ±1.5 that 6 pairs resolve. The
  one-third realization is an observation from this run, not a property of the method. More training, more samples or
  another target could change it. The effective sample size (0.39–0.48 of 8) shows concentrated weights, not a
  plateau. A larger dose is a power heuristic, not a guarantee: the DECS line predicts the total cost, not how much of
  it the projection prevents.
- **No single-step accuracy cost** at this dose: held-out exact-call accuracy moved by at most 0.3 points.

## 6. Does the projection reach the costly decision at BFCL? (exploratory, CPU, 2026-10-02)

Independent reviews before deciding on more training. The analyses reuse the 18 projection BFCL runs, the recipient's
and DECS mid's runs at decoding seeds 0–2, and the cached LoopTool samples. Scripts are in the session scratchpad
(`redteam_A`, `redteam_D`). The sample is 745 multi-turn entries that overflow in no run. Each interval resamples
entries.

**The costly decision at BFCL is ending the first turn early.** A premature stop is a text reply after at least one
call but before the reference number of calls. In the recipient, every entry whose first turn makes too few calls
fails (P(fail) = 1.00, against 0.53 otherwise). Within an entry, those first turns have 31% shorter first-step
reasoning (ratio 0.685 [0.61, 0.77]).

| turn 0 | premature stop (recipient 9.7%) | too few calls (recipient 7.6%) | failures with turn 0 short of calls |
|---|---|---|---|
| DECS mid vs recipient (about 15% saved) | +1.58 [+0.21, +3.20] | +1.53 [+0.59, +2.62] | +1.19 [+0.45, +2.04] of DECS's +2.55 more failures |
| recipient seed 5678 vs 1234 (null) | −0.15 [−2.0, +1.4] | +0.46 [−0.67, +1.66] | +0.36 [−0.58, +1.25] |
| ordinary vs recipient (5.0% saved) | +1.14 [−0.20, +2.52] | | +0.65 [−0.04, +1.39] |
| projected vs recipient (3.3% saved) | +0.81 [−0.46, +2.22] | | +0.49 [−0.16, +1.21] |
| ordinary vs projected | +0.32 [−0.78, +1.48] | | +0.16 [−0.45, +0.78] |

Per 1% of tokens saved, the premature-stop rate rises about equally in both tilted arms (ordinary 0.23, projected
0.25, against the recipient). Turn by turn, ordinary and projected behave the same (turns missing a reference
function: −0.01 [−0.48, +0.45]).

**What the ordinary target's decision term does on the cache.** At α = 2.88, the Z(h, a) reweighting raises
exact-call matches on the cached samples by +0.49 points [+0.10, +0.92]. It also cuts unfinished or malformed
responses from 0.93% to 0.17%. Each unfinished response is its own decision, so the projection keeps those. The one
harmful push lands on multi-turn states right after a tool result that still need a call (93 prompts, 3% of the
training prompts; about 63% of BFCL steps follow a tool result). There, text replies rise by +4.0 points
[−0.1, +8.2].

**Reading.** The projection removes the target's explicit choice of decisions at training states. It does not
remove what costs accuracy at BFCL. The model writes reasoning, then decides, and in the recipient itself shorter
first-step reasoning goes with ending the turn early. A student that learns "reason less" carries that to BFCL states
it never trained on, and its decisions there follow its shorter reasoning. The explicit decision term the projection
removes was mostly harmless at single steps. This matches the 30-arm line in results_2.md: the cost follows the
reasoning removed at test time, not the structure of the training target. The plan-length link is correlational; a
plan-swap replay (the recipient's first-step reasoning given to DECS mid at turn 0) would test whether it is causal.

**Power** (redteam_B). Run-to-run multi-turn SD is 0.99 points, so a pair has SD 1.40. With 2 training seeds, no
number of decoding seeds pushes the minimum detectable projected − ordinary difference below about 1.0 point. Showing
that the projection prevents half of the line's cost needs about 50–75 pairs at 5–6% savings (550–700 H100-hours).
The pre-registered recovery margin (1.5 points) is larger than the line's whole predicted cost below 8.3% savings,
and the 3-point matching tolerance allows a confound of 0.54 points.

## Summary

| | result |
|---|---|
| 🟢 | The pipeline ran end to end. DECS prefers shorter reasoning within identical decisions (Spearman −0.62), and the projected target keeps every decision's frequency exactly, while the ordinary target moves 14% of decision probability. |
| 🟡 | 15% was out of reach (the projection peaks at 11.4%), so the approved 10% fallback set α = 2.88. |
| 🔴 | The realized BFCL dose is small (3–5% of total tokens), and the drift measurement could not resolve decision changes; whether the students fit their targets is not yet measured. |
| 🟡 | Every pre-registered verdict is inconclusive; both tilted arms sit on the DECS line. Projected vs ordinary, +0.50 [−0.69, +1.62] multi-turn, is dose-confounded: at the seed where the two saved equally, projected was −0.17. |
| 🟡 | Exploratory, dose-confounded: the ordinary arm adds environment-rejected calls at DECS's per-token rate; the projected arm does not exceed the uniform control. |
| 🟡 | Probe, pre-registered: every realization (0.40–0.56) and generalization reading is inconclusive; the fit slope cannot decide whether optimization limits realization; every arm, including uniform, moves training-prompt decisions detectably and within the margin. |
| 🟢 | Probe, exploratory: the projection suppresses the ordinary target's requested decision shift on training prompts. Ordinary students move along it (μ +0.21), projected students no more than uniform (ordinary − projected +0.17 [+0.09, +0.27]). One direction only; its link to accuracy is not shown. |
| 🔴 | Probe, exploratory: about a third of the implied savings reaches fresh samples on the training prompts, for a cause not yet identified. This one-pass configuration gave too little savings for BFCL to resolve the multi-turn question. |
| 🔴 | BFCL, exploratory: the costly decision is ending the first turn before all reference calls (DECS +1.58 points; about half of its extra failures). Both tilted arms raise it at the same rate per 1% saved, so the projection does not reach it. Shorter first-step reasoning goes with these early stops in the recipient itself. |
