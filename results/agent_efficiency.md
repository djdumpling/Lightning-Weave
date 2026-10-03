# Agent-efficiency transfer: results tracker

Plan: `docs/agent_efficiency_synthesis.md`. BFCL v3 (32,768-token steps in a 65,536 YaRN window,
T 0.6 / top-p 0.95 / top-k 20).
All runs share `bfcl-v3-3e6e955a00df-full`.

## E0: accuracy-trained student vs base (existing runs, no new GPU)

`python evaluation/bfcl_efficiency.py base opd`.
- Accuracy and means use the leaderboard category weights, and accuracy matches `aggregate.json`.
- Generated tokens are per entry and include thinking.
- Deltas are OPD − base, with category-stratified paired-bootstrap 95% CIs over entries.
- These are single decoding runs with no training-seed replicate, so the intervals omit both sources of
  variance.

| group | base acc | OPD acc | Δacc | base gen | OPD gen | Δgen | Δgen, both correct |
|---|---|---|---|---|---|---|---|
| overall | 65.84 | 67.51 | +1.67 [+0.46, +2.89] | 2,647 | 3,141 | +493 [+297, +711] | +229 [+132, +347] |
| non-live | 84.03 | 83.52 | −0.52 [−1.73, +0.70] | 517 | 582 | +65 [+30, +103] | +35 [+13, +59] |
| live | 80.23 | 79.88 | −0.36 [−1.42, +0.71] | 578 | 614 | +36 [−20, +82] | −1 [−50, +34] |
| multi-turn | 33.25 | 39.12 | +5.88 [+2.75, +9.00] | 6,847 | 8,225 | +1,378 [+789, +1,988] | +652 [+356, +1,006] |

**Primary cost estimate: the paired log-ratio of generated tokens.** This is the leaderboard-weighted mean
per-entry log(1 + OPD) − log(1 + base), shown as a relative change. It is about 4–5× more precise than
the mean-token difference, which runaways dominate:

| group | tokens, all entries | tokens, both correct |
|---|---|---|
| overall | +11.8% [+10.1, +13.5] | +8.6% [+6.4, +10.8] |
| non-live | +8.3% [+5.5, +11.1] | +6.9% [+4.0, +9.9] |
| live | +10.5% [+8.4, +12.7] (the mean difference cannot resolve this) | +7.3% [+5.3, +9.1] |
| multi-turn | +16.7% [+12.9, +20.6] | +11.8% [+6.1, +18.0] |

- **What the data establish:** the accuracy-trained student is more accurate and produces more tokens
  per entry. The increase is also present on entries both models solve. It is not only a side effect of
  solving harder entries.
- **Turns:** BFCL multi-turn gives both models the same user turns (97–99% of entries differ by 0 turns;
  the rest are context overflows). The extra multi-turn cost is +1.0 step per entry [+0.8, +1.3] and more
  tokens per step. Over the common horizon: +1,331 [+679, +1,997].
- **Outcome strata** (leaderboard-weighted):

  | | both correct | OPD only | base only | neither |
  |---|---|---|---|---|
  | overall | 61.4% | 6.1% | 4.4% | 28.1% |
  | multi-turn | 25.6% | 13.5% | 7.6% | 53.2% |

- **Distribution and cost per success:**
  - weighted median generated tokens: 517 → 582;
  - runaway steps (32,768 cap): 0.64% → 0;
  - generated tokens per success: 4,021 → 4,652.
- **Irrelevance accuracy** (irrelevance + live_irrelevance) drops by 1.7 points [−3.0, −0.4]. This fits
  the untrained special-token artifact below.
- **Decision rule** (non-inferior accuracy, −5% tokens, no guard regressions): OPD does not pass. There
  is no token reduction, and irrelevance is not non-inferior. AES (descriptive only) is +0.029.

## E0: training length (V0 after 80 vs 200 updates)

V0's round-20 checkpoint (80 of 200 updates, where its training loss flattens) was exported and run on
the full BFCL v3 (tag `opd-r20`). Deltas are paired over entries.

| | overall | non-live | live | multi-turn | gen tokens | runaway |
|---|---|---|---|---|---|---|
| base | 65.84 | 84.03 | 80.23 | 33.25 | 2,647 | 0.64% |
| V0, 80 updates | 67.78 | 84.30 | 80.05 | 39.00 | 3,201 | 0.04% |
| V0, 200 updates | 67.51 | 83.52 | 79.88 | 39.12 | 3,141 | 0.00% |

- **Final vs round 20:** Δacc −0.28 [−1.35, +0.81], tokens −0.1% [−1.4, +1.2], Δsteps +0.06 [−0.01, +0.12].
  Updates 80 → 200 changed nothing measurable.
- **Conclusion:** the accuracy anchor's whole effect (+1.9 accuracy, +11.9% tokens, +0.9 multi-turn steps)
  is in place by 80 updates. Training length is not the lever, so the 400-update pilot is skipped. The
  modest gain is limited by the target and data, not by steps.
- **Noise yardstick:** the two checkpoints of the same run differ by 0.3 points overall. This is not a
  training-seed replicate.

**Cost decomposition (first run with the usage log).** Reasoning (`reasoning_content`, re-tokenized) is
about 90% of generated tokens in every group: 2,891 of 3,201 overall, and 7,590 of 8,386 per multi-turn
entry. Cumulative prompt tokens per multi-turn entry are about 71k, because each step re-reads the growing
context. So reasoning, the part the math donors have evidence on, is most of the generation cost. Input
cost, by contrast, scales with steps, which the donors cannot address. All 4,441 entries reconcile exactly
with the log.

## E0: mechanism in the cached targets

Centered agent-acc shift (nats) at the student's own LoopTool states. The first two rows are from one
shard (800 trajectories); the last two are from a 200-row smoke with prompt-clustered 95% CIs. These are
heuristic state labels, not yet validated against manual annotation.

| fork | contrast | legacy agent-acc | clean agent-acc (untrained rows carry no shift) |
|---|---|---|---|
| stop-thinking (single-newline end vs paragraph break) | stop − continue | −2.67 (n = 90; 17% favor stopping) | unchanged (no special tokens) |
| paragraph-start reflection | Wait / Hmm vs So / Alternatively | +3.5 / +3.1 vs −1.1 / −2.2 | unchanged |
| act vs talk | `<tool_call>` − rest | +14.2 [+13.8, +14.7] | +1.1 [+0.9, +1.3] |
| call boundary | `<|im_end|>` − call again | +19.7 [+17.8, +21.2] | +2.9 [+1.6, +3.9] |

At act-vs-talk states the legacy and clean directions have Fisher cosine −0.28. The Qwen3-4B-Base
untrained-row artifact dominates the legacy accuracy target exactly where the model decides whether to
call a tool.

## Donor evidence

- **DECS/DeepScaleR.** The alias map raises mapped behavior mass to about 100% at call-boundary and
  end-of-message states. But whole-position coverage stays at about 7.5% and 67% there, and at 0% at
  act-vs-talk. `<tool_call>` is untrained in both R1-family anchors.
- **Mapped vs usable.** Mapped mass is not usable evidence. `verify` now reports mapped AND trained
  coverage per state type.
- **Rescoring.** The DECS and DeepScaleR shards were scored with an earlier, non-injective alias map and
  are being rescored. Klear's shards are unaffected.

## E1: core geometry on the full cache (analysis/core.json, 12,800 trajectories, 5.2M positions)

All three core donors (Klear, DECS, DeepScaleR) are scored with evidence masking and verified. The run
took 10 minutes on one CPU container.

**Behavioral probes.** Values are the centered shift (nats) at the student's own forks, with
prompt-clustered 95% CIs. A positive value favors the first option of the contrast.

| probe (forks / prompts) | agent acc (clean) | DECS | DECS − DeepScaleR | DeepScaleR | Klear | DECS − Klear |
|---|---|---|---|---|---|---|
| stop − continue (1,238 / 543) | −1.88 [−2.12, −1.62] | +0.33 | **+0.24 [+0.20, +0.28]** | +0.09 | +1.97 | −1.64 |
| reflect − conclude (8,959 / 1,784) | +6.56 [+6.33, +6.81] | −2.44 | **−1.66 [−1.70, −1.62]** | −0.78 | +1.24 | −3.68 |
| end message − rest (1,852 / 464) | +0.04 [−0.19, +0.29] | +0.74 | +0.86 [+0.84, +0.88] | −0.12 | +18.02 | −17.28 |
| end − call again (8,423 / 2,380) | +1.85 | +0.11 | +0.21 | −0.10 | +17.16 | −17.04 |
| tool call − rest (11,192 / 2,846) | +1.05 | +0.12 | +0.10 (no evidence) | +0.02 | +0.58 | −0.46 |

- **The quasi-matched efficiency contrast opposes the accuracy anchor exactly where efficiency should
  act.** DECS − DeepScaleR favors stopping and disfavors reflection markers; agent acc does the
  opposite. Residualization barely changes this (stop +0.24, reflect −1.54).
- **The analogy as first stated (DECS − Klear) points the wrong way.** Klear itself favors stopping
  (+1.97), so subtracting it makes DECS − Klear push against stopping (−1.64). Its message-boundary
  terms (±17 nats) are Klear's base-to-chat-format change (Qwen3-8B-Base does not end chat messages),
  not efficiency.

**Fisher cosine with agent acc.** Placebo 95% interval: about ±0.02 overall, ±0.03 at reflection
forks, ±0.06 at stop forks.

| slice | DECS − DeepScaleR | DECS | Klear | DECS − Klear |
|---|---|---|---|---|
| all | −0.26 [−0.27, −0.26] | −0.27 | +0.34 | −0.38 |
| reflection forks | −0.53 [−0.55, −0.52] | −0.64 | +0.22 | −0.57 |
| stop forks | −0.05 [−0.10, −0.01] | +0.15 | +0.28 | −0.26 |
| supported / unsupported | −0.26 / −0.26 | | | |
| single- / multi-turn | −0.28 / −0.25 | | | |

The two accuracy shifts (agent acc, Klear) agree with each other; the efficiency contrast is
anti-aligned with both.

**Where the energy is.**
- **Localization.** About 91% of every direction's Fisher energy is in ordinary reasoning tokens. DECS
  puts 18% at reflection forks, against 4% for agent acc (DECS − DeepScaleR: 7%). This fits DECS
  targeting redundancy.
- **Scale.** The efficiency contrast is small: agent acc is 5.3× larger in RMS. At stop forks the
  efficiency push (+0.24) is about 8× weaker than the accuracy anchor's anti-stopping push (−1.88).

**Caveats.**
- The state labels are heuristic and not yet validated by annotation.
- DECS and DeepScaleR differ in more than the length reward.
- None of this shows BFCL efficiency yet; that needs the primary matrix.

## E2: the seven primary-arm targets (composed; not yet trained)

Efficiency terms are calibrated to 0.0135 nats/token of added KL on a 25% prompt-stratified sample.

| arm | efficiency coef | target KL to student | stop − continue | reflect − conclude | tool call − rest | end message − rest |
|---|---|---|---|---|---|---|
| acc-legacy | — | 0.0502 | −1.88 | +6.56 | +14.04 | +13.55 |
| acc-clean | — | 0.0498 | −1.88 | +6.56 | +1.05 | +0.04 |
| acc-legacy+decs | +3.11 | 0.0439 | −0.85 | −1.04 | +14.40 | +15.85 |
| acc-legacy+decs-deepscaler | +2.70 | 0.0439 | −1.23 | +2.09 | +14.31 | +15.88 |
| acc-clean+decs | +3.11 | 0.0434 | −0.85 | −1.03 | +1.42 | +2.34 |
| acc-clean+decs-deepscaler | +2.70 | 0.0435 | −1.23 | +2.09 | +1.33 | +2.37 |
| acc-clean+decs-deepscaler-flipped | −2.34 | 0.0812 | −2.44 | +10.45 | +0.81 | −1.99 |

Values are the centered target shift (nats) at the student's forks; prompt-clustered CIs are about ±0.25
at stop forks and ±0.2 at reflection forks.

- **Stop and reflection forks.** The efficiency terms cut the accuracy target's anti-stopping push by
  35–55%, without reversing it. They cut its pull toward reflection markers by 68% (the contrast) or reverse
  it (raw DECS). The flipped control amplifies both.
- **Tool calls.** The efficiency terms leave the call decision essentially unchanged (no donor evidence
  there), and put no energy at tool-name, argument, or call-boundary states.
- **Where the added energy goes.** 91% is in ordinary reasoning tokens at about average intensity. 7% is at
  reflection forks, at 28× the average intensity (raw DECS: 18%, at 70×).
- **Distance from the student.** Adding efficiency moves each target closer to the student (0.050 → 0.044),
  because the terms partly cancel the accuracy shift; the flipped arm moves it further away (0.081).
- **Per-state KL.** The efficiency term's KL is concentrated: median about 0, p99 about 0.27, max about 4–5
  nats.

## E3: primary matrix on BFCL v3 (7 arms, one training seed, 200 updates, efficiency budget 0.0135)

Every arm ran all 17 categories with no failures, and every usage log reconciles 4,441 of 4,441 entries.

| arm | overall | non-live | live | multi-turn | gen/entry | multi-turn steps |
|---|---|---|---|---|---|---|
| base | 65.84 | 84.03 | 80.23 | 33.25 | 2,647 | 9.65 |
| V0 (existing OPD) | 67.51 | 83.52 | 79.88 | 39.12 | 3,141 | 10.70 |
| acc-legacy | 68.36 | 83.87 | 79.83 | 41.38 | 3,146 | 10.70 |
| acc-clean | 67.24 | 84.52 | 79.56 | 37.62 | 2,977 | 10.19 |
| acc-legacy+decs | 68.03 | 83.82 | 79.79 | 40.50 | 2,686 | 10.74 |
| acc-legacy+decs-deepscaler | 67.30 | 84.37 | 79.79 | 37.75 | 2,887 | 10.65 |
| acc-clean+decs | 66.08 | 83.62 | 80.36 | 34.25 | 2,405 | 9.89 |
| acc-clean+decs-deepscaler | 65.91 | 83.33 | 80.28 | 34.12 | 2,605 | 10.00 |
| acc-clean+decs-deepscaler-flipped | 67.90 | 83.93 | 79.52 | 40.25 | 3,586 | 10.28 |

**Each efficiency arm vs its accuracy-only arm** (paired; tokens are the paired log-ratio):

| arm | Δacc overall | Δacc multi-turn | tokens | tokens, both correct | reasoning tokens | visible tokens |
|---|---|---|---|---|---|---|
| acc-legacy+decs | −0.32 [−1.49, +0.79] | −0.88 [−4.12, +2.38] | **−9.1% [−10.3, −8.0]** | −6.7% | −10.1% | +0.3% |
| acc-legacy+decs-deepscaler | −1.06 [−2.19, +0.01] | −3.63 [−7.00, −0.50] | −7.4% [−8.7, −6.0] | −5.0% | −8.3% | −0.1% |
| acc-clean+decs | −1.16 [−2.25, −0.06] | −3.38 [−6.25, −0.00] | −10.0% [−11.2, −8.8] | −7.6% | −11.2% | +0.1% |
| acc-clean+decs-deepscaler | −1.32 [−2.40, −0.26] | −3.50 [−6.38, −0.50] | −8.8% [−10.0, −7.6] | −6.6% | −10.0% | +0.0% |
| acc-clean+decs-deepscaler-flipped | +0.67 [−0.39, +1.75] | +2.62 [−0.25, +5.62] | +10.8% [+9.3, +12.4] | +10.1% | +12.1% | −0.3% |

**Findings.**
- **Transfer works on cost.** A math-only efficiency shift, composed into the agent target, cuts BFCL
  generated tokens by 7–10% (both-correct 5–8%). All of the savings are reasoning tokens (−8 to −11%);
  visible output and prompt tokens are unchanged. That matches the evidence map: the donors act only on
  reasoning states.
- **It is directional.** The sign-flipped control adds tokens (+10.8%; contrast vs flipped −17.7%).
- **It moves along an accuracy–cost trade-off.** Less reasoning costs about 1 point (≈3 on multi-turn) in
  most arms; more reasoning (flipped) gains a little. The best arm is acc-legacy+decs: vs base it keeps
  +2.20 of acc-legacy's +2.52 accuracy at +2.5% tokens instead of +12.8%. It still fails the
  pre-registered rule, narrowly: its accuracy lower bound is −1.49 and irrelevance is not non-inferior.
- **The quasi-controlled contrast did not beat raw DECS downstream.** At equal KL, raw DECS saved more
  tokens at an equal or smaller accuracy cost. The geometry preference for DECS − DeepScaleR did not
  translate.
- **Cleaning the accuracy target hurt instead of helping** (acc-clean − acc-legacy): Δacc −1.12
  [−2.22, −0.11] overall and −3.75 [−6.5, −1.0] on multi-turn, while irrelevance is unchanged (−0.19). The
  +14-nat untrained-token push was not the cause of the irrelevance drop, and appears to help multi-turn.
- **Training noise is not small.** acc-legacy retrains V0's target; it lands +0.85 [−0.20, +1.95] overall
  (+2.25 multi-turn) above V0, with the same seed and tokens within +0.9%. Accuracy differences of about 1
  point between single-seed arms are therefore at the training-noise level. Token differences are not:
  the replicate is +0.9%, against 7–10% effects.

**Surprise 1, raw DECS vs the contrast, paired directly** (contrast − raw):

| accuracy base | overall accuracy | overall tokens | multi-turn tokens |
|---|---|---|---|
| legacy | −0.73 [−1.93, +0.41] | +1.9% [+0.6, +3.2] | +3.9% [+0.8, +7.1] |
| clean | −0.17 [−1.23, +0.91] | +1.3% [+0.2, +2.4] | +4.4% [+1.6, +7.1] |

The contrast is not better. It saves slightly fewer tokens at the same KL, and the accuracy difference is
within noise. DeepScaleR itself reflects less than R1-Distill (reflect −0.78 at our forks), so subtracting
it also removes efficiency signal. Per unit KL, raw DECS concentrates more of its push on reflection forks
(18% of energy vs 7%).

**Surprise 2, multi-turn behavior** (per-entry means, 800 entries):

| model | steps | tool calls | call steps | calls per call step | text steps | duplicate calls |
|---|---|---|---|---|---|---|
| base | 9.65 | 7.35 | 5.67 | 1.30 | 3.98 | 1.15 |
| V0 | 10.70 | 7.12 | 6.67 | 1.07 | 4.03 | 0.69 |
| acc-legacy | 10.70 | 7.07 | 6.67 | 1.06 | 4.03 | 0.67 |
| acc-clean | 10.19 | 7.03 | 6.15 | 1.14 | 4.04 | 0.80 |

- **What the accuracy anchor changes in multi-turn:** the same number of calls, split one per message.
  Each call's result is observed before the next call, and duplicates drop.
- **What cleaning does:** it undoes about half of that. Steps fall by 0.51 [−0.71, −0.32]; calls
  (−0.04) and text steps (+0.01) are unchanged. For scale, retraining the same target changes steps by
  −0.00 [−0.16, +0.15].
- **Why:** the legacy target's +19-nat push toward `<|im_end|>` at call boundaries came from Base's
  untrained row. It has arbitrary size but the right sign: end the message after a call. Cleaning gives
  `<|im_end|>` δ = 0 while the competing "call again" tokens keep their shifts. Masking one side of a
  binary decision is not neutral; it biases the decision.

