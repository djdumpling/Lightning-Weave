# Agent-efficiency synthesis: results

BFCL v3, all 17 categories, leaderboard-weighted. Deltas are paired over entries: accuracy in points,
tokens as the paired log-ratio of generated tokens per entry (typical entries), and total tokens as the ratio
of summed tokens (what a run costs, dominated by multi-turn, which holds 76% of V0's tokens). All CIs are
category-stratified 95% bootstraps.
Every trained arm uses the LoopTool Offline Direct-OPD recipe (α 2.0, 200 updates) from Qwen3-4B. Efficiency
terms are scaled to a mean per-token KL of 0.0135 against their accuracy-only target unless noted.

**Code for earlier sections (2026-10-01).** The cleanup commit f95f213 removed tools that only completed analyses
used. To rerun them, check out snapshot commit 543e6d0. That applies to:
- section 4: residualization and the efficiency basis in `data_curation/shift_geometry.py`;
- section 7: `data_curation/stop_margins.py` and `evaluation/bfcl_breakdown.py`;
- section 8: `data_curation/reflection_probe.py`;
- section 10: `data_curation/prompt_difficulty.py` weights, `evaluation/bfcl_difficulty_strata.py`, and
  `evaluation/bfcl_shared_factor.py`;
- section 11: `data_curation/reasoning_value_probe.py`.

543e6d0 also registers the 23 exploratory arms that were never trained. Every trained arm is still registered,
unchanged.

**Note on direction (2026-10-01): stop adjusting DECS, and stop running redundant experiments.** From section 5
on, about 15 arms varied the donor, its strength, the composition, or a gate. Every one landed on the same
accuracy-per-token line. That is one finding confirmed many times, not many findings. Section 15 is the last
round of that kind: its turn-start arm tests the one open lead (efficiency is free on single-turn and costly on
multi-turn). After it, the next experiment must ask a different question rather than change the same settings.
Examples:
- a real agent-efficiency anchor: a GRPO pair on LoopTool tasks with a token or call penalty. This tests the
  project's central claim directly: does a direction synthesized from math and code match one learned on agent
  tasks, and does either beat the line?
- a second benchmark (tau2): does the line hold outside BFCL?

Before launching a new arm, say what it would teach that earlier arms could not.

## 1. Primary matrix (one training seed)

| arm | trains on | overall | multi-turn | tokens/entry | vs | Δacc | Δtokens |
|---|---|---|---|---|---|---|---|
| base | — (Qwen3-4B) | 65.84 | 33.25 | 2,647 | | | |
| V0 | LoopTool OPD student (existing) | 67.51 | 39.12 | 3,141 | base | +1.67 [+0.46, +2.89] | +11.8% [+10.1, +13.5] |
| V0 @ 80 updates | its round-20 checkpoint | 67.78 | 39.00 | 3,201 | V0 | +0.28 [−0.81, +1.35] | +0.1% [−1.2, +1.4] |
| acc-legacy | V0's target, retrained | 68.36 | 41.38 | 3,146 | V0 | +0.85 [−0.20, +1.95] | +0.9% [−0.2, +2.2] |
| acc-clean | agent-acc, untrained special tokens zeroed | 67.24 | 37.62 | 2,977 | acc-legacy | −1.12 [−2.22, −0.11] | −1.6% [−2.7, −0.4] |
| acc-legacy+decs | + DECS shift | 68.03 | 40.50 | 2,686 | acc-legacy | −0.32 [−1.49, +0.79] | **−9.1% [−10.3, −8.0]** |
| acc-legacy+decs-deepscaler | + DECS − DeepScaleR | 67.30 | 37.75 | 2,887 | acc-legacy | −1.06 [−2.19, +0.01] | −7.4% [−8.7, −6.0] |
| acc-clean+decs | + DECS shift | 66.08 | 34.25 | 2,405 | acc-clean | −1.16 [−2.25, −0.06] | −10.0% [−11.2, −8.8] |
| acc-clean+decs-deepscaler | + DECS − DeepScaleR | 65.91 | 34.12 | 2,605 | acc-clean | −1.32 [−2.40, −0.26] | −8.8% [−10.0, −7.6] |
| acc-clean+decs-deepscaler-flipped | − (DECS − DeepScaleR) | 67.90 | 40.25 | 3,586 | acc-clean | +0.67 [−0.39, +1.75] | +10.8% [+9.3, +12.4] |

A math-only efficiency shift, composed into the agent target, cuts BFCL tokens by 7–10%. All of the cut is
reasoning, and the sign-flipped control adds a matching +10.8%, so the effect follows the direction. Accuracy
costs 0.3–1.3 points, at the level of training noise: retraining V0's exact target moved accuracy by +0.85.
The best arm, acc-legacy+decs, keeps +2.2 of acc-legacy's +2.5-point gain over base, at +2.5% tokens over
base instead of +12.8%.
Neither refinement helped: the matched DECS − DeepScaleR contrast did worse than raw DECS, and cleaning the
accuracy target hurt multi-turn.

## 2. Where the accuracy goes (entry flips, CPU re-analysis of section 1)

Points are leaderboard-weighted. Token change is the median over entries in each flip class.

| comparison | net non-live | net live | net multi-turn | multi-turn lost / gained | token change: lost / gained / both right |
|---|---|---|---|---|---|
| V0 → acc-legacy (same target) | +0.35 | −0.04 | +2.25 | 7.75 / 10.00 | +2.8% / −4.2% / +0.0% |
| acc-legacy → acc-clean | +0.65 | −0.27 | −3.75 | 10.25 / 6.50 | −0.1% / −8.6% / +0.0% |
| acc-legacy → +decs | −0.05 | −0.04 | −0.88 | 10.50 / 9.62 | −8.3% / −9.7% / −1.2% |
| acc-legacy → +decs-deepscaler | +0.50 | −0.04 | −3.62 | 11.88 / 8.25 | −7.8% / −14.2% / −0.3% |
| acc-clean → +decs | −0.90 | +0.80 | −3.38 | 11.25 / 7.88 | −10.8% / −13.5% / −1.8% |
| acc-clean → +decs-deepscaler | −1.18 | +0.71 | −3.50 | 10.75 / 7.25 | −10.4% / −18.6% / −1.3% |

On the legacy base, single-turn accuracy is unchanged, so the whole cost is multi-turn. There, even
retraining the same target flips about 18 points of entries in each direction. Single-turn losses have the
replicate's error mix: mostly wrong arguments and false calls on irrelevance, with no rise in missing calls.
Lost entries were not cut more than gained ones. So there is no sign of a new underthinking failure, and the
second training seed decides whether the roughly 1-point multi-turn cost is real.

## 3. Inference-time baselines on V0

| V0 decoding | overall | multi-turn | tokens/entry | Δacc vs V0 | Δtokens | Δtotal tokens |
|---|---|---|---|---|---|---|
| thinking, 32k tokens per step (V0) | 67.51 | 39.12 | 3,141 | | | |
| no thinking | 57.22 | 12.38 | 246 | −10.29 [−11.67, −8.92] | −89.6% [−89.9, −89.3] | −92.0% [−92.7, −91.3] |
| 4k tokens per step | 69.13 | 42.88 | 2,780 | +1.62 [+0.50, +2.75] | +0.6% [−0.8, +1.9] | −10.5% [−15.2, −5.8] |
| concise system prompt | 68.11 | 41.00 | 3,001 | +0.60 [−0.46, +1.68] | −7.2% [−8.6, −5.9] | −4.6% [−8.9, +0.3] |
| *trained: acc-legacy → acc-legacy+decs* | | | | *−0.32 [−1.49, +0.79]* | *−9.1% [−10.3, −8.0]* | *−14.7% [−19.1, −10.1]* |

Turning thinking off is not a usable efficiency baseline. The 4k cap's accuracy gain is a harness artifact.
V0 fails 43 of 200 long-context entries because the prompt plus a 32k completion budget overflows the 65k
serving window, and under the cap only 2 fail. Excluding overflow entries, the cap's multi-turn effect is +0.50
[−2.40, +3.28]. It truncates 1.1% of steps, cutting the long tail but not typical entries. Every trained arm has
the same overflow count (42–46), so the trained comparisons are unaffected. A concise instruction trims typical
entries (−7.2%) but barely moves the total (−4.6%). The trained transfer cuts the long multi-turn entries that
dominate cost, which makes it three times as effective on the total.

## 4. Census geometry (10 donor pairs at the LoopTool student's states, no training)

Stop probe > 0 favors ending the thought at stop forks. Reflect probe > 0 favors "Wait"-type markers over
concluding at reflection forks. Prompt-bootstrap CIs are ±0.01 on cosines and at most ±0.22 on probes. The
shuffled-state placebo cosine is at most 0.02.

| direction | kind | Fisher energy/token | cos with agent-acc | stop probe | reflect probe |
|---|---|---|---|---|---|
| agent-acc (Qwen3-4B-Base → Thinking-2507) | accuracy | 0.472 | +1 | −1.88 | +6.56 |
| Klear (Qwen3-8B-Base → Klear-8B) | accuracy | 0.338 | +0.35 | +1.97 | +1.24 |
| DeepCoder (R1D-1.5B → DeepCoder) | accuracy | 0.019 | +0.20 | −0.20 | −0.85 |
| DECS (R1D-1.5B → DECS) | efficiency | 0.013 | −0.26 | +0.33 | −2.44 |
| DECS − DeepScaleR | efficiency | 0.017 | −0.26 | +0.24 | −1.66 |
| DLER − DeepScaleR | efficiency | 0.018 | +0.02 | +1.65 | −1.51 |
| AdaptThink − DeepScaleR | efficiency | 0.008 | −0.12 | −1.03 | +0.60 |
| L1-Max (DeepScaleR → L1) | efficiency | 0.059 | −0.22 | +0.19 | −2.66 |
| Nemotron v1 → v2 | efficiency | 0.103 | −0.04 | +1.42 | +0.48 |
| DECS-7B − Skywork-OR1-7B | efficiency | 0.033 | −0.09 | +1.65 | −2.77 |

The sign structure holds across all ten pairs: both accuracy donors align with the agent-accuracy shift, and
every efficiency direction is anti-aligned with it or orthogonal to it. At the forks, the efficiency
directions mostly agree with each other and oppose agent-acc: 6 of 7 push toward stopping and 5 of 7 away from
reflection. AdaptThink, which learns to skip thinking rather than to stop sooner, is the consistent exception.
Beyond that sign they share little. Efficiency contrasts with no model in common have a mean Fisher cosine of
−0.04 overall and +0.18 at reflection forks. The first component of the efficiency basis (37%) is just the
shared DeepScaleR control, and residualization changes none of this. What transfers is therefore a coarse
push at a few fork types, not a shared token-level direction, which is what the heuristic control tests.

## 5. Controls, budget sweep, and a second training seed

| arm | overall | multi-turn | tokens/entry | vs | Δacc | Δacc multi-turn | Δtokens | Δtotal tokens |
|---|---|---|---|---|---|---|---|---|
| acc-legacy | 68.36 | 41.38 | 3,146 | | | | | |
| acc-legacy-scaled (0.86 × agent-acc) | 68.25 | 40.75 | 3,065 | acc-legacy | −0.11 [−1.19, +1.01] | −0.62 [−3.75, +2.38] | −2.4% [−3.7, −1.1] | −2.5% [−7.4, +2.4] |
| acc-legacy+heuristic (fork rule) | 67.68 | 39.38 | 2,970 | acc-legacy | −0.68 [−1.79, +0.41] | −2.00 [−4.88, +1.00] | −3.8% [−5.0, −2.6] | −5.5% [−9.4, −1.3] |
| acc-legacy+decs @low (KL 0.005) | 68.00 | 40.00 | 2,880 | acc-legacy | −0.36 [−1.50, +0.77] | −1.38 [−4.50, +1.62] | −5.8% [−7.0, −4.6] | −8.9% [−13.6, −4.4] |
| acc-legacy+decs @mid (KL 0.0135) | 68.03 | 40.50 | 2,686 | acc-legacy | −0.32 [−1.49, +0.79] | −0.88 [−4.12, +2.38] | −9.1% [−10.3, −8.0] | −14.7% [−19.1, −10.1] |
| acc-legacy+decs @high (KL 0.027) | 66.86 | 36.50 | 2,527 | acc-legacy | −1.50 [−2.61, −0.30] | −4.88 [−7.75, −1.87] | −12.9% [−14.1, −11.6] | −19.7% [−23.5, −15.9] |
| acc-legacy, seed 5678 | 68.27 | 40.75 | 3,054 | acc-legacy | −0.08 [−1.16, +0.97] | −0.62 [−3.75, +2.37] | −0.8% [−2.1, +0.4] | −2.7% [−7.5, +1.9] |
| acc-legacy+decs @mid, seed 5678 | 67.13 | 37.88 | 2,663 | acc-legacy, seed 5678 | −1.15 [−2.26, −0.06] | −2.88 [−5.75, −0.13] | −8.7% [−10.0, −7.5] | −13.2% [−17.1, −8.9] |
| acc-legacy+decs @mid, both seeds | | | | acc-legacy, both seeds | −0.73 [−1.57, +0.05] | −1.88 [−4.06, +0.38] | −8.9% [−9.8, −8.0] | −14.0% [−16.8, −10.9] |

Seed 5678 trains on the same targets in a shuffled order. Both-seed rows average each entry over the two
seeds, so their CIs reflect entry sampling only.

The token cut replicates across seeds (−14.7% and −13.2% of total tokens), and the accuracy target itself
reproduces (−0.08). The accuracy cost is small but probably real: −0.73 pooled over both seeds, mostly on
multi-turn (−1.9); seed 1234 was the favorable draw. Relative to base, the transfer removes the accuracy
anchor's whole verbosity tax. acc-legacy costs +17.4% total tokens over base, while acc-legacy+decs costs
−0.4% [−4.6, +4.3] and keeps +1.7 of the +2.5-point gain. The effect is not just a weaker accuracy target,
since shrinking agent-acc to its projected share saves 2.5%. It is not just the fork rule either: matching
DECS's push at stop and reflection forks by hand saves 5.5% for a similar accuracy cost. The remainder comes
from DECS's diffuse shift in the think body, which independent donors do not share (section 4). Savings grow
with the budget (−8.9, −14.7, −19.7%), and accuracy holds through mid but breaks at high (−4.9 on
multi-turn), so mid is the knee.

## 6. Round 1: other donors, the donor sum, and prompting on top

Every arm is compared with acc-legacy at the mid budget (one training seed). The pre-registered rule was that a
donor transfers if it saves at least 8% of total tokens at an accuracy change of at least −1.0. The prompting rows
answer a different question (do prompting and training stack?), so they have no rule. "vs line" is the accuracy
residual from the line fitted below; positive is better than the line.

| arm | overall | multi-turn | tokens/entry | Δacc | Δacc multi-turn | Δtokens | Δtotal tokens | rule | vs line |
|---|---|---|---|---|---|---|---|---|---|
| + DECS (R1D-1.5B → DECS) | 68.03 | 40.50 | 2,686 | −0.32 [−1.49, +0.79] | −0.88 [−4.12, +2.38] | −9.1% [−10.3, −8.0] | −14.7% [−19.1, −10.1] | pass | +0.75 |
| + DECS-7B (R1D-7B → DECS-7B) | 67.56 | 38.38 | 2,614 | −0.80 [−1.90, +0.26] | −3.00 [−6.00, +0.25] | −9.5% [−10.6, −8.2] | −16.6% [−20.8, −12.7] | pass | +0.41 |
| + L1-Max (DeepScaleR → L1-Max) | 66.95 | 37.75 | 2,716 | −1.41 [−2.51, −0.29] | −3.63 [−6.62, −0.50] | −6.4% [−7.6, −5.2] | −13.9% [−17.3, −10.2] | fail (accuracy) | −0.40 |
| + Nemotron (v1 → v2) | 68.04 | 41.00 | 2,968 | −0.32 [−1.43, +0.81] | −0.38 [−3.50, +2.75] | −4.1% [−5.4, −2.9] | −5.7% [−10.1, −0.9] | fail (tokens) | +0.10 |
| + donor sum (equal Fisher weight, all four) | 66.60 | 36.00 | 2,506 | −1.76 [−2.81, −0.71] | −5.38 [−8.25, −2.50] | −12.5% [−13.5, −11.3] | −20.1% [−23.9, −16.3] | fail (accuracy) | −0.29 |
| concise system prompt | 67.77 | 39.00 | 2,929 | −0.59 [−1.71, +0.46] | −2.38 [−5.25, +0.62] | −8.1% [−9.4, −6.6] | −7.2% [−12.3, −2.1] | | −0.06 |
| + DECS and concise system prompt | 66.60 | 35.88 | 2,470 | −1.75 [−2.95, −0.57] | −5.50 [−8.38, −2.38] | −15.3% [−16.5, −14.0] | −21.4% [−25.1, −17.5] | | −0.19 |

Every donor cuts tokens, and the size of the cut follows the census reflect probe. DECS and L1-Max push away from
reflection at forks (−2.4 and −2.7) and save 14–15%; Nemotron pushes slightly toward it (+0.5) and saves 5.7%. At
the same KL, the equal-weight donor sum saves more than any single donor (−20.1%), so the donors' nearly
orthogonal shifts do add. Its accuracy cost grows with the saving, however, and the same holds for every other
lever. Across all twelve comparisons against acc-legacy in sections 5–6, accuracy falls by about 0.073 points
per 1% of total tokens saved (r = 0.83). The scatter around that line (SD 0.33) is below single-seed training
noise, so no donor, sum, budget or prompt beats the line by more than noise. The line is a descriptive fit,
not an established frontier: the comparisons share a baseline and evaluation entries, and two training seeds
say little about training variance, so finalists should be compared at matched realized savings on held-out
entries. All of them beat not distilling:
base is 14.8% cheaper than acc-legacy but 2.5 points worse, 1.4 points below the line.

The rule therefore mostly measured how far along the line an arm went, not whether it beat the line. L1-Max and
the donor sum fail it by going as far as DECS or further (−13.9% and −20.1%), not by trading clearly worse; Nemotron fails it by
being a weak efficiency donor, which the reflect probe predicted. Prompting and training are substitutes on the
same line rather than complements. The transfer's advantages are that it beats not distilling and reaches savings
(14–20%) that prompting does not (about 7%). Future arms should be compared at matched savings, or by their
residual from the line.

## 7. Step 0: where a selective stop target could act (CPU diagnostics)

**Stop states in the training cache** (12,800 responses, 4.3M thinking positions). Stop probabilities are under
the behavior policy b, the accuracy-only target q_acc, and the acc-legacy+decs target q_new. The overall added
KL of q_new over q_acc is 0.0135, which reproduces the calibrated budget.

| state | positions | share of thinking | stop prob. b / q_acc / q_new | share of added KL |
|---|---|---|---|---|
| `</think>` is the top candidate (the thought's actual end) | 12,763 | 0.30% | 1.00 / 1.00 / 1.00 | 0.00% |
| `</think>` a candidate within 2 nats of the top | 1 | <0.001% | — | 0.00% |
| `</think>` a candidate further away | 31 | 0.001% | 0.01 / 0.09 / 0.08 | 0.00% |
| `</think>` absent, certified more than 2 nats below the top | 4,294,856 | 99.70% | 0 | 97.6% |
| `</think>` absent, unresolved at 2 nats | 118 | 0.003% | 0 | 0.00% |
| newline stop fork (single newline vs paragraph break) | 1,238 | 0.03% | 0.32 / 0.31 / 0.36 | 0.09% |

**Where each efficiency term's push lands** (composed at the mid budget). Reflection forks are split into the
first, second and later ones in a response (5,932 / 2,323 / 5,031 positions).

| donor | stop fork: q_acc → q_new | stop fork share of added KL | energy: ordinary reasoning | energy: reflection forks (first / second / later) |
|---|---|---|---|---|
| DECS | 0.31 → 0.36 | 0.09% | 79.6% | 17.9% (10.0 / 3.0 / 4.8) |
| DECS-7B | 0.31 → 0.39 | 0.12% | 87.2% | 9.4% (4.6 / 1.5 / 3.4) |
| L1-Max | 0.31 → 0.35 | 0.17% | 92.2% | 4.8% (1.4 / 0.6 / 2.8) |
| Nemotron | 0.31 → 0.35 | 0.21% | 92.8% | 3.3% (1.1 / 0.6 / 1.6) |

**Accuracy change by category** (points vs acc-legacy of the same seed; per-category CIs are about ±6).

| arm | base | miss_func | miss_param | long_context | irrelevance | live_irrelevance |
|---|---|---|---|---|---|---|
| + DECS, seed 1234 | −3.5 | −3.0 | +1.5 | +1.5 | −2.9 | +1.7 |
| + DECS, seed 5678 | −3.5 | −6.0 | −2.0 | +0.0 | −0.8 | −0.2 |
| + DECS-7B | −3.5 | −4.5 | +0.0 | −4.0 | −0.4 | +0.9 |
| + L1-Max | −4.0 | +1.0 | −7.0 | −4.5 | −1.7 | −0.3 |
| + donor sum | −3.5 | −7.0 | −5.0 | −6.0 | −0.8 | +0.5 |
| + DECS @high | −5.5 | −8.5 | +0.0 | −5.5 | −2.1 | +1.9 |
| + fork rule (heuristic) | +0.0 | −4.5 | −3.5 | +0.0 | −0.4 | +0.6 |
| concise prompt | +0.5 | −2.5 | −2.5 | −5.0 | −0.8 | −0.3 |

The student never hovers near stopping. `</think>` is a cached candidate only where the thought actually ends;
everywhere else it is certified more than 2 nats below the top token. A ThinkBrake-style gate is therefore empty
on this cache, at every position and at sentence boundaries. The stop decision appears only at the newline fork,
in 6% of responses. Every efficiency term raises stopping there, but those forks receive 0.1–0.2% of the added
KL. The push is spread over ordinary reasoning instead (80–93%), and for DECS also over reflection forks (18%),
most of all the first reflection after a new observation.

The multi-turn cost concentrates in multi_turn_base (−3.5 on both DECS seeds) and miss_func (negative in 7 of 8
arms), where the model must notice that a function it needs is missing. These are exploratory per-category
estimates. Irrelevance shows no consistent rise in false tool calls.

The model does not see its earlier reasoning. Within a user turn, 96% of consecutive steps grow the prompt by
less than the previous step's reasoning (median 61 vs about 550 tokens), and no prompt ever shrinks. So every step
reasons from scratch, identically for all arms.

*Corrected 2026-10-01 from the proxy logs of section 15:* the harness does send the reasoning back. NeMo-Skills
returns the API's assistant message, and its `reasoning_content` field is present in 100% of multi-turn requests
with history. The loss happens in serving:
- Reasoning from earlier user turns is dropped by the Qwen3 chat template, by design.
- Reasoning from earlier steps of the same turn would be kept by the template (checked by rendering it). But the
  rendered prompt still lacks it: median growth is 60 tokens per step, against 383 tokens of reasoning. So this
  vLLM version drops the field before the template sees it.

## 8. What a reflection is worth: forcing "Wait" vs "So" at cached reflection forks

At 900 reflection forks where both markers are live (median behavior probability 0.31 for the reflection marker
and 0.30 for the conclusion marker), the acc-legacy student continued 8 times after each forced marker to the
end of the message. Decoding used BFCL settings with an 8k-token budget. Call match means the message's tool
calls equal the LoopTool reference. Intervals resample prompts. No continuation hit the budget.

| stratum | forks (prompts) | call match: reflect / conclude | Δmatch, reflect − conclude | Δtokens |
|---|---|---|---|---|
| multi-turn, first reflection | 300 (219) | 0.714 / 0.714 | −0.000 [−0.015, +0.013] | +40 [+20, +58] |
| multi-turn, later reflections | 300 (142) | 0.510 / 0.512 | −0.003 [−0.019, +0.014] | +78 [+63, +93] |
| single-turn, first reflection | 150 (96) | 0.578 / 0.583 | −0.004 [−0.023, +0.014] | +91 [+57, +125] |
| single-turn, later reflections | 150 (57) | 0.398 / 0.416 | −0.018 [−0.047, +0.006] | +90 [+58, +123] |
| multi-turn, first − later | | | +0.002 [−0.019, +0.023] | −38 [−63, −14] |

Forcing a reflection at a fork does not change whether the next tool call matches the reference, in any
stratum. It only adds 40–90 tokens. At 78% of forks all 16 continuations reach the same outcome. Where outcomes
vary, reflecting helps (93 forks) about as often as it hurts (90). The first reflection after a new observation
is not worth more than later ones, so the premise of the protect-first arm does not hold at the message level.
Suppressing reflection at forks looks nearly free for the next tool call. DECS's multi-turn cost more likely
comes from its diffuse shift over ordinary reasoning (80–93% of its push), or from effects across turns that a
message-level check cannot see. This check scores call structure against the LoopTool reference on LoopTool
prompts; it is not BFCL multi-turn success.

## 9. Lead arm: DECS with the first reflection protected (two training seeds)

The acc-legacy+decs target, with DECS's push removed at each response's first reflection fork. DECS's coefficient
is fixed at its calibrated value (3.11), so the gate is the only change. Every row is compared with acc-legacy of
the same seed. "vs line" is the accuracy residual from the section 6 line; positive is better than the line.

| arm | seed | overall | multi-turn | tokens/entry | Δacc | Δacc multi-turn | Δtotal tokens | vs line |
|---|---|---|---|---|---|---|---|---|
| + DECS | 1234 | 68.03 | 40.50 | 2,686 | −0.32 [−1.49, +0.79] | −0.88 [−4.12, +2.38] | −14.7% [−19.1, −10.1] | +0.75 |
| + DECS | 5678 | 67.13 | 37.88 | 2,663 | −1.15 [−2.26, −0.06] | −2.88 [−5.75, −0.13] | −13.2% [−17.1, −8.9] | −0.19 |
| + DECS, first reflection protected | 1234 | 66.83 | 36.50 | 2,750 | −1.53 [−2.63, −0.47] | −4.88 [−7.63, −2.00] | −12.8% [−17.5, −8.1] | −0.60 |
| + DECS, first reflection protected | 5678 | 67.20 | 37.50 | 2,813 | −1.07 [−2.18, +0.06] | −3.25 [−6.37, −0.13] | −8.7% [−13.3, −3.8] | −0.44 |
| + DECS | both | | | | −0.73 [−1.57, +0.05] | −1.88 [−4.06, +0.38] | −14.0% [−16.8, −10.9] | +0.29 |
| + DECS, first reflection protected | both | | | | −1.30 [−2.07, −0.49] | −4.06 [−6.25, −1.88] | −10.8% [−14.1, −7.2] | −0.52 |

Against plain DECS over both seeds, protecting the first reflection changes accuracy by −0.57 [−1.34, +0.17], and
multi-turn by −2.19 [−4.31, +0.06]. It also gives back 3.7 points of total-token savings [−0.2, +7.7].

Protecting the first reflection gives back about a quarter of DECS's savings and recovers no accuracy. It lands
below the line, where plain DECS sits slightly above it. This is what the reflection probe (section 8) predicted:
the first check is not worth more than later ones, so sparing it only spends tokens. A one-hour probe forecast a
two-seed training result, which makes it a usable screen for future gates. Neither closeness to stopping (section
7) nor reflection order separates useful reasoning from waste on this cache, so beating the line needs another
signal.

## 10. Is there a basis for a difficulty gate? (CPU re-analyses, after external review)

**Verified difficulty of the training cache.** Each of the 4 cached responses per prompt is scored against the
LoopTool reference. Tool calls must match exactly; text references cannot be verified.

| stratum | prompts | response match rate | verified solved (all 4 match) | share of cache tokens on solved prompts |
|---|---|---|---|---|
| all | 3,200 | 0.68 | 1,943 (61%) | 47% |
| multi-turn, single call | 1,537 | 0.81 | 74% | 62% |
| multi-turn, parallel calls | 464 | 0.71 | 58% | 49% |
| single-turn, single call | 512 | 0.81 | 75% | 65% |
| single-turn, parallel calls | 303 | 0.61 | 51% | 40% |
| text references (multi / single) | 330 / 54 | not verifiable | excluded | — |

**Where BFCL accuracy changes, by how reliably the accuracy-only student solves an entry.** Entries are sorted
by the accuracy-only student's correctness in two other accuracy-only runs, which are disjoint from the
compared pair. Values are Δ accuracy in points (entry means) against the same-seed reference, with entry-bootstrap
intervals.

| arm | both right | split | both wrong | multi-turn: both right | multi-turn: split | multi-turn: both wrong |
|---|---|---|---|---|---|---|
| + DECS, seed 1234 | −0.7 [−1.4, −0.1] | +0.4 [−6.4, +7.8] | +1.6 [−0.5, +3.6] | −6.7 [−12.1, −1.7] | +4.5 [−5.2, +14.9] | +0.6 [−3.6, +4.7] |
| + DECS, seed 5678 | −0.8 [−1.4, −0.2] | −2.6 [−9.5, +4.4] | −0.4 [−2.3, +1.5] | −1.6 [−6.0, +3.6] | −5.7 [−17.7, +5.0] | −3.3 [−6.9, +0.3] |
| + DECS-7B | −0.7 [−1.3, −0.0] | +2.5 [−5.0, +9.6] | +0.2 [−1.8, +2.3] | −6.7 [−12.1, −1.2] | +0.0 [−9.7, +9.7] | −2.5 [−6.4, +1.4] |
| + donor sum | −0.8 [−1.4, −0.2] | −5.3 [−12.8, +2.1] | −0.4 [−2.5, +1.7] | −5.4 [−10.4, −0.4] | −8.4 [−18.8, +1.3] | −4.7 [−8.7, −1.4] |
| concise prompt (on acc-legacy) | −1.0 [−1.7, −0.4] | +2.1 [−5.7, +9.2] | +0.7 [−1.2, +2.7] | −5.0 [−10.4, +0.4] | +0.6 [−9.8, +9.7] | −2.0 [−5.6, +1.4] |
| entries (seed 1234) | 3,173 | 282 | 938 | 240 | 154 | 358 |

Pooling DECS over both seeds and stratifying by V0 alone gives −0.8 [−1.4, −0.3] on entries V0 solves and
+0.5 [−0.9, +2.2] on entries it fails. On multi-turn the figures are −3.1 [−6.6, +0.6] and −1.4 [−4.5, +1.5].

**Do different levers cut the same entries?** Reliability is the correlation between a lever's two repeats.
The latent correlation divides the cross-lever correlation by those reliabilities, bootstrapped jointly. Latent
correlations are identified only for levers with repeats.

| lever pair | reliabilities | latent correlation of per-entry token changes |
|---|---|---|
| DECS ~ DECS with first reflection protected | 0.21 / 0.22 | 0.96 [0.89, 1.04] |
| DECS ~ concise prompt | 0.21 / 0.40 | 0.30 [0.19, 0.39] |
| DECS with first reflection protected ~ concise prompt | 0.22 / 0.40 | 0.26 [0.16, 0.36] |

The verified-solved label covers 61% of training prompts. On BFCL, however, shortening loses accuracy on
entries the accuracy-only student reliably solves: −0.7 to −1.0 points in every arm, and up to −6.7 on
multi-turn at one seed. The "split" stratum is too small to show whether uncertain entries lose more. So on
BFCL, being solved does not make shortening safe, which is the premise a solved-prompt gate relies on. DECS and
the concise prompt shorten largely different entries (latent correlation 0.30), yet trade accuracy for tokens
at the same aggregate rate. The shared slope is therefore not a shared mechanism. How deeply an entry is cut
does not predict whether it loses accuracy (correlation −0.04 to +0.01 for arms measured against other
references). Two caveats: the strata come from only two runs, and 8 of 15 null checks exclude zero, so the
accuracy-only runs are not fully exchangeable.

## 11. What the reasoning DECS removes is worth: held-out LoopTool prompts

The probe used 600 held-out LoopTool prompts with tool-call references (400 multi-turn, 200 single-turn), none
used in training. acc-legacy ("full") and acc-legacy+decs ("short") each sampled 8 complete messages with BFCL
decoding and an 8k budget. Success means the message's tool calls equal the reference exactly. The base
policy's 4 samples at the cache's decoding give the label a gate would use: "verified solved" means all 4
match. Intervals resample prompts. No message hit the budget.

| group | prompts | success: full / short | shortening effect (short − full) | total tokens |
|---|---|---|---|---|
| all | 600 | 0.760 / 0.764 | +0.004 [−0.003, +0.011] | −9.6% |
| verified solved by base | 416 | 0.983 / 0.985 | +0.002 [−0.003, +0.007] | −5.9% |
| not solved by base | 184 | 0.255 / 0.264 | +0.009 [−0.011, +0.029] | −13.9% |
| multi-turn | 400 | 0.793 / 0.798 | +0.005 [−0.003, +0.013] | −7.4% |
| single-turn | 200 | 0.693 / 0.695 | +0.003 [−0.011, +0.016] | −13.3% |

Solved minus not-solved effect: −0.007 [−0.027, +0.015].

On the training distribution, DECS's shortening costs no measurable next-call accuracy, whether or not the
base policy solves the prompt (±1 point), while saving 10% of tokens. It saves more where the base policy
fails (−14%) than where it succeeds (−6%). The verified-solved label strongly predicts success (0.98 vs 0.25)
but not the shortening effect, which is about zero everywhere, so there is no unsafe region for a difficulty
gate to avoid. The accuracy cost measured on BFCL (sections 5–6 and 10) therefore does not come from a
measurable loss in single-step tool calls on LoopTool-like prompts. It arises in BFCL's multi-turn interaction, or from
BFCL-specific behavior, which a per-prompt gate on training prompts cannot target. This probe scores single
messages against LoopTool references; it is not multi-turn task success.

## 12. Where multi-turn breaks (CPU, from finished BFCL runs)

BFCL grades a multi-turn entry turn by turn and stops at the first failing turn. Each failure gets one type, read
from that turn by comparing the model's calls with the reference's function names. Across types, the changes
in the share of entries failing add up to the multi-turn accuracy loss. The sample is 755 multi-turn entries (45
overflowing entries dropped), as unweighted entry means in points. Intervals resample entries. DECS pools both
seeds, each against its same-seed reference, and the null compares the accuracy-only student's two training seeds.

| type of first failure | accuracy-only (% of entries) | + DECS, both seeds | + DECS, first reflection protected | null: seed 5678 − 1234 |
|---|---|---|---|---|
| loop hit the 20-step cap | 0.5 | +0.46 [+0.07, +0.93] | +0.46 [−0.07, +1.06] | −0.13 [−0.79, +0.53] |
| acted on an unanswerable turn (missing function or parameter) | 11.1 | +0.00 [−1.39, +1.39] | +0.20 [−1.13, +1.59] | −0.93 [−2.91, +1.06] |
| no call where one was needed | 7.4 | −1.06 [−2.38, +0.26] | −0.86 [−2.25, +0.53] | +0.13 [−1.72, +1.99] |
| missing reference calls | 13.9 | +0.40 [−1.26, +2.19] | +1.32 [−0.40, +3.05] | +2.91 [+0.66, +5.30] |
| different functions | 8.5 | −0.07 [−1.46, +1.46] | +0.99 [−0.53, +2.45] | −1.32 [−3.31, +0.66] |
| a reference call repeated | 2.4 | +0.00 [−0.99, +0.93] | −0.33 [−1.13, +0.46] | −0.26 [−1.46, +0.93] |
| calls to other functions besides the reference | 3.2 | +1.59 [+0.40, +2.78] | +1.46 [+0.33, +2.65] | −0.40 [−1.72, +0.93] |
| right functions, wrong arguments | 9.8 | +0.73 [−0.86, +2.32] | +1.13 [−0.40, +2.65] | +0.79 [−1.32, +2.91] |
| all failures (= −Δaccuracy) | 56.7 | +2.05 [−0.13, +4.24] | +4.37 [+2.12, +6.56] | +0.79 [−2.38, +3.97] |

| behavior per turn | accuracy-only | + DECS, both seeds | change | null change |
|---|---|---|---|---|
| needed a call, made none (turns before the first failure) | 4.4% | 4.2% | −0.2 [−0.8, +0.4] | −0.2 [−1.0, +0.7] |
| made calls but missed a reference call | 12.3% | 12.3% | −0.0 [−0.9, +0.9] | +0.3 [−1.0, +1.5] |
| called beyond the reference | 24.7% | 25.0% | +0.4 [−0.7, +1.4] | −2.0 [−3.3, −0.6] |
| called on an unanswerable turn | 57.0% | 57.2% | +0.2 [−2.5, +2.8] | −2.4 [−5.8, +1.1] |
| calls per turn (all turns) | 1.71 | 1.78 | +3.6% [+0.8, +6.8] | −2.8% [−5.6, −0.0] |
| tokens, first step of a turn (answering the user) | 784 | 633 | −19.2% [−20.9, −17.5] | +0.9% [−2.2, +4.0] |
| tokens, each later step (after a tool result) | 736 | 657 | −10.8% [−16.3, −4.7] | −3.4% [−11.6, +5.8] |

DECS's multi-turn loss is not a new kind of failure. It spreads over the failure types the accuracy-only
student already makes. Until its first failure, DECS acts turn by turn as the accuracy-only student does
(within a point). In particular, it acts on unanswerable turns no more often, so missing an absent function or
parameter is not the cost; the accuracy-only student already acts on 57% of those turns. Two small shifts hold
on both seeds. Loops that hit the step cap add +0.46; they double a rare failure and account for most of the
rise in identical repeated calls. Failing turns that also call functions outside the reference add +1.6; this
is positive in every efficiency arm and −0.4 in the null. Failures do not move to later turns. DECS saves most
on the first response to each user message (−19%), about twice its cut after tool results (−11%).

Solved entries take 10.6 model steps on average, so a slip of about 0.4% per step compounds to the 2-point
loss. That is at the edge of what the single-step probe ruled out (section 11, lower bound −0.3%). A small,
diffuse slip per step, compounded over a conversation, fits both results, so there is no single reasoning
function to protect. Lost entries show the same picture, though at this noise level most are flips (each seed
loses about 83 entries and gains 58–76). They include stopping after a tool error and handing back to the
user, guessing a value instead of looking it up, and repeating a completed action. Failure types compare
function names only, and the reference is one valid path.

## 13. Code-efficiency round (pre-registered): compression directions from math and from code

Each arm adds one term to acc-legacy at the mid KL budget:

- C = E1-Code-14B − DeepCoder-14B;
- M = E1-Math-1.5B − DeepScaleR;
- D = DECS − R1-Distill.

Each E1 model was trained from its domain's accuracy model with correctness-only RL under a 1K-token thinking
budget. So C and M isolate that budgeted training, which compresses reasoning but also continues training and
teaches answering from interrupted reasoning. They are empirical compression directions, not an isolated cost
reward.

Both pairs passed the pre-training checks:

- support at reasoning states (median / p10): C 1.00 / 0.91, M 0.97 / 0.61;
- no evidence on the untrained `<tool_call>` token, like DECS (this does not make the call decision neutral;
  see the mechanism notes);
- the float32 re-score, arithmetic only, 64 rows: Fisher correlation 0.94 and 0.89 against DECS's 0.98, with
  relative RMS disagreement 34% and 46% against 18%.

Their shifts are about half (C) and 40% (M) of DECS's size, calibrated to coefficients of 6.0 and 7.9 against
DECS's 3.1. Both sums kept raw 1:1 weights (size ratios 1.35 and 1.93). The math arms' seed 5678 was added after
seed 1234, as the pre-registration requires for arms selected on results. "Total tokens" are generated tokens.

| arm | seed | overall | multi-turn | tokens/entry | Δacc | Δacc multi-turn | Δtokens | Δtotal tokens | vs line |
|---|---|---|---|---|---|---|---|---|---|
| + C | 1234 | 67.63 | 39.38 | 2,727 | −0.73 [−1.84, +0.36] | −2.00 [−4.88, +1.00] | −9.4% [−10.7, −8.2] | −13.4% [−17.6, −9.1] | +0.25 |
| + C | 5678 | 67.18 | 37.25 | 2,648 | −1.10 [−2.20, +0.03] | −3.50 [−6.50, −0.62] | −8.7% [−9.9, −7.6] | −13.6% [−17.5, −9.4] | −0.11 |
| + M | 1234 | 66.90 | 37.88 | 2,334 | −1.46 [−2.65, −0.31] | −3.50 [−6.62, −0.38] | −14.6% [−15.8, −13.4] | −25.3% [−28.9, −21.8] | +0.39 |
| + M | 5678 | 67.39 | 37.62 | 2,303 | −0.88 [−1.95, +0.17] | −3.12 [−5.88, −0.25] | −14.9% [−16.0, −13.8] | −24.2% [−27.4, −20.9] | +0.89 |
| + (M + C) | 1234 | 67.64 | 39.12 | 2,367 | −0.71 [−1.82, +0.33] | −2.25 [−5.25, +0.88] | −15.8% [−16.8, −14.7] | −24.6% [−27.5, −21.7] | +1.08 |
| + (M + C) | 5678 | 67.01 | 37.75 | 2,303 | −1.26 [−2.30, −0.23] | −3.00 [−5.88, +0.25] | −15.1% [−16.1, −14.0] | −24.3% [−27.7, −20.7] | +0.51 |
| + (D + C) | 1234 | 66.83 | 36.50 | 2,592 | −1.53 [−2.70, −0.46] | −4.88 [−7.88, −1.88] | −11.5% [−12.6, −10.3] | −17.8% [−22.4, −13.2] | −0.23 |
| + D (reference) | 1234 | 68.03 | 40.50 | 2,686 | −0.32 [−1.49, +0.79] | −0.88 [−4.12, +2.38] | −9.1% [−10.3, −8.0] | −14.7% [−19.1, −10.1] | +0.75 |
| + D (reference) | 5678 | 67.13 | 37.88 | 2,663 | −1.15 [−2.26, −0.06] | −2.88 [−5.75, −0.13] | −8.7% [−10.0, −7.5] | −13.2% [−17.1, −8.9] | −0.18 |

| paired comparison, same seed | Δacc | Δacc multi-turn | total tokens |
|---|---|---|---|
| C vs D, both seeds pooled (pre-registered primary) | −0.20 [−1.03, +0.64] | −0.90 [−3.19, +1.37] | +1.5% / −0.4% |
| M vs D, both seeds pooled | −0.44 [−1.22, +0.39] | −1.43 [−3.50, +0.69] | −12.4% / −12.7% |
| M + C vs D, both seeds pooled | −0.27 [−1.11, +0.61] | −0.78 [−3.06, +1.50] | −11.6% / −12.8% |
| M + C vs M (1234 / 5678) | +0.75 [−0.42, +1.89] / −0.38 [−1.51, +0.70] | +1.25 / +0.13 | +0.9% / −0.1% |
| M + C vs C (1234 / 5678) | +0.01 [−1.03, +1.11] / −0.16 [−1.29, +0.98] | −0.25 / +0.50 | −12.9% / −12.4% |
| D + C vs D / vs C (1234) | −1.20 [−2.38, +0.02] / −0.80 [−1.82, +0.32] | −4.00 / −2.88 | −3.6% / −5.0% |

Counting prompt tokens changes the size of the savings but not the ranking. At a quarter of the generated-token
price, the cost savings shrink to 10–12% for M and M + C, and to 4–5% for C and DECS. Prompt tokens are about 7×
generated tokens in these conversations.

**Code vs DECS.** No clear difference was detected between the pre-registered code arm and DECS: pooled −0.20
[−1.03, +0.64] at the same cost, on both seeds. That interval does not establish equivalence. Pooled
entry-bootstrap intervals also do not capture variation across training runs.

**Same-method math.** M saves 24–25% of generated tokens on both seeds at the same KL, against 13–15% for DECS
and code, at an accuracy pooled −0.44 [−1.22, +0.39] relative to DECS. That is further than DECS was measured to
go: DECS's high budget reached −19.7% at −1.50, one seed. A matched-savings comparison needs M at a lower strength
(see the follow-ups). Extrapolating DECS's curve gives +1.34 [−0.99, +8.32], which does not establish that M is
better.

**Composition.** At fixed total KL, M + C keeps M's savings and roughly its accuracy. Its seed-1234 edge over M
(+0.75) did not replicate (−0.38 at seed 5678). It clearly beats C (the same accuracy at 12–13% fewer tokens) but
not M. At that fixed KL, adding C also lowers M's coefficient from 7.95 to 4.45. So this tests one mixture, not
C's marginal value at a fixed M strength. DECS + code is worse than both of its parts.

**Against base and V0.** Relative to the base Qwen3-4B, M + C is more accurate and cheaper on both seeds:
+1.80 [+0.66, +2.89] and +1.18 [+0.08, +2.31] points at −11.5% and −13.5% generated tokens. It is within noise
of V0 (+0.14 / −0.49) at 24–26% fewer generated tokens.

**Mechanism.** The E1 shifts save differently from DECS:

- At stop forks they favor continuing, where DECS favors stopping (−0.7 vs +1.0). E1's training forced
  `</think>` rather than rewarding it.
- At reflection forks E1-Code pushes against reflection half as hard (−3.7 vs −7.6).
- M compresses every step: −27% of first-step tokens and −21% after tool results, against −19% and −11% for DECS.
- After a tool call, M pushes toward continuing rather than ending the message: −10 nats before dividing by α,
  where DECS gives +0.4. The donors barely support the student's candidates there (median support: E1-Math
  0.47%, E1-Code 5.9%, DECS 0.95%), so this push rests on unreliable scores. On the re-scored rows it is also
  consistent across arithmetic, so it is not rounding.
  - M's calls per turn rise 9% and extra-call failures +2.5 points. Adding C cuts the rise to +3.6%.
- Masking `<tool_call>` did not keep any donor out of the call decision. Pushes on competing candidates moved
  the call-versus-rest contrast toward calling by +0.4 to +0.7 in every arm, so only removing the term at those
  states preserves the decision.
- M's shift is nearly orthogonal to the accuracy shift (cosine −0.05, against −0.22 to −0.28 for DECS, E1-Code
  and L1-Max). This does not predict the cost per point saved across donors, so it remains a hypothesis.

**Caveats.**

- Two seeds per arm.
- On the final targets (64 rows), scoring disagreement amounts to 4% of the added KL for DECS, 11% for E1-Code
  and 21% for E1-Math (24% in reasoning states). Checkpoint rounding was not tested.
- About 41–46% of the added KL sits in 1% of positions, similar to DECS (50% on the same rows), with per-position
  maxima of 8–9 nats. A mean KL of 0.0135 does not mean individual decisions are barely moved.
- The training cache holds frozen trajectories, so donor pushes are fitted open-loop. BFCL multi-turn is
  closed-loop, but it is our development benchmark, so the math arms need a second benchmark before any claim.
- How closely each student fits its target, by state type, was not measured.
- DECS-7B, Skywork-7B and Nemotron were prepared before the near-duplicate untrained-row rule. Their
  `<tool_call>` log-ratios therefore reached the DECS-7B, Nemotron and donor-sum targets (small in KL).
- Before 2026-09-30, the efficiency decision rule's guards passed whenever an interval included zero. They now
  require non-inferiority. No reported result used that rule's verdict.

## 14. Follow-ups after external review: protocol-state protection and matched savings

Both arms change one thing from acc-legacy+e1math (M), at seeds 1234 and 5678:

- **Protected:** M at its own coefficient (7.945, not recalibrated), with the term removed at the call-or-reply and
  just-after-a-call states, where donor support is 0–6%.
- **@low:** M at the low KL budget (0.005, coefficient 4.78), which lands at DECS mid's savings.

| arm | seed | overall | multi-turn | tokens/entry | Δacc | Δacc multi-turn | Δtokens | Δtotal tokens | vs line |
|---|---|---|---|---|---|---|---|---|---|
| + M, protected | 1234 | 67.18 | 38.12 | 2,383 | −1.18 [−2.24, −0.14] | −3.25 [−6.12, −0.38] | −14.9% [−16.0, −13.8] | −24.2% [−27.1, −21.0] | +0.58 |
| + M, protected | 5678 | 67.52 | 37.62 | 2,421 | −0.75 [−1.77, +0.32] | −3.12 [−6.12, −0.25] | −13.9% [−15.0, −12.8] | −20.9% [−25.2, −16.2] | +0.77 |
| + M @low | 1234 | 67.55 | 39.62 | 2,661 | −0.81 [−1.93, +0.28] | −1.75 [−4.88, +1.25] | −9.3% [−10.5, −8.1] | −15.3% [−19.2, −10.9] | +0.31 |
| + M @low | 5678 | 67.57 | 38.50 | 2,576 | −0.71 [−1.79, +0.33] | −2.25 [−5.00, +0.62] | −9.8% [−11.0, −8.7] | −15.6% [−19.4, −11.6] | +0.43 |

| comparison, pooled over both seeds | Δacc | Δacc multi-turn | Δtotal tokens | per seed Δacc |
|---|---|---|---|---|
| M protected vs M (mechanism) | +0.20 [−0.55, +0.99] | +0.12 [−1.94, +2.31] | +2.9% [−0.2, +6.1] | +0.28 / +0.13 |
| M @low vs D mid (matched savings) | −0.02 [−0.84, +0.79] | −0.13 [−2.37, +2.12] | −1.7% [−5.3, +1.8] | −0.48 / +0.44 |

**Protection fixes the calls, not the accuracy.** Removing E1-Math's push at the call decisions removes its extra
calls: calls per turn go from +9.3% to +1.1%, and extra-call failures from +2.5 to +0.9 points of multi-turn
entries. Accuracy does not recover (pooled +0.20, multi-turn +0.1). Failures move to other types instead
(different-function failures go from +1.1 to +3.4). So the push after tool calls on unsupported scores was real,
and it caused the extra calls, but it was not the cause of the accuracy cost.

**At matched savings, E1-Math and DECS are indistinguishable.** At the low budget, E1-Math saves 15.4% of generated
tokens (DECS mid: 14.0%), at an accuracy difference of −0.02 [−0.84, +0.79]. E1-Math is a stronger direction per
unit of KL, reaching 24–25% at the mid budget. On the accuracy-per-token trade, however, it sits on the same line
as DECS.

Counting prompt tokens at a quarter of the price, E1-Math @low saves 7% against DECS's 4–5%.

Across donors, strengths, compositions and now protection, the accuracy cost follows how much reasoning is
removed, not which donor removes it. That fits the diffuse per-step slip of section 12.


## 15. Paper recipe, turn-start allocation, and decoding replicates

Pre-registered in docs/agent_efficiency_synthesis.md ("Next"). The turn-start arms and their references are
decoded at BFCL sampling seeds 0, 1 and 2. Comparisons pair runs at the same training and decoding seed and
average over pairs (`evaluation/bfcl_pooled.py`).

**References at three decoding seeds** (seed 0 is the run used in every section above):

| run | decoding seed | overall | multi-turn | single-turn | tokens/entry |
|---|---|---|---|---|---|
| acc-legacy | 0 / 1 / 2 | 68.36 / 68.00 / 67.88 | 41.38 / 40.87 / 40.25 | 81.85 / 81.56 / 81.69 | 1,966 / 1,959 / 1,998 |
| acc-legacy, seed 5678 | 0 / 1 / 2 | 68.27 / 68.19 / 68.26 | 40.75 / 40.75 / 41.12 | 82.03 / 81.91 / 81.83 | 1,912 / 2,040 / 2,034 |
| acc-legacy+decs (mid) | 0 / 1 / 2 | 68.03 / 67.39 / 66.85 | 40.50 / 38.88 / 37.50 | 81.80 / 81.65 / 81.52 | 1,676 / 1,669 / 1,694 |
| acc-legacy+decs (mid), seed 5678 | 0 / 1 / 2 | 67.13 / 66.81 / 66.27 | 37.88 / 36.88 / 36.12 | 81.75 / 81.78 / 81.34 | 1,660 / 1,678 / 1,579 |

| comparison | pairs | Δacc | Δacc multi-turn | Δacc single-turn | Δtotal tokens |
|---|---|---|---|---|---|
| DECS mid vs acc-legacy | 6 | −1.08 [−1.56, −0.61] | −2.90 [−4.17, −1.62] | −0.17 [−0.47, +0.14] | −16.3% [−18.3, −14.2] |
| null: acc-legacy seed 5678 vs 1234 | 3 | +0.16 [−0.49, +0.80] | +0.04 [−1.71, +1.75] | +0.22 [−0.18, +0.62] | +1.1% [−1.6, +3.8] |
| null: acc-legacy decoding seeds 1, 2 vs 0 | 4 | −0.23 [−0.95, +0.44] | −0.31 [−2.19, +1.56] | −0.19 [−0.69, +0.32] | +3.6% [+0.1, +7.3] |

With six pairs, DECS mid's cost is clearly real: −1.1 overall and −2.9 on multi-turn, against −0.73 and −1.9 from
the two decoding-seed-0 pairs (section 5). Seed 0 drew the most favorable of the six pairs (−0.32). Single-turn
stays free (−0.17), at −15% single-turn tokens. The multi-turn interval narrows from about ±2.2 to ±1.3, and the
training-seed null stays at zero.

**Turn starts protected.** This is DECS mid with DECS removed from every training prompt that answers a new user
message mid-conversation: 1,934 of 3,200 prompts, 83% of the multi-turn ones.

| run | decoding seed | overall | multi-turn | single-turn | tokens/entry |
|---|---|---|---|---|---|
| protect-turn-starts | 0 / 1 / 2 | 67.41 / 67.24 / 67.47 | 38.50 / 38.25 / 39.13 | 81.86 / 81.74 / 81.64 | 1,752 / 1,684 / 1,799 |
| protect-turn-starts, seed 5678 | 0 / 1 / 2 | 67.51 / 67.62 / 67.99 | 39.13 / 38.50 / 40.00 | 81.70 / 82.18 / 81.98 | 1,753 / 1,824 / 1,727 |

| comparison (6 pairs) | Δacc | Δacc multi-turn | Δacc single-turn | Δtotal tokens | Δmulti-turn tokens | verdict |
|---|---|---|---|---|---|---|
| protected vs DECS mid | +0.46 [+0.01, +0.90] | +0.96 [−0.27, +2.19] | +0.21 [−0.07, +0.48] | +5.9% [+3.9, +8.1] | +6.9% [+4.2, +9.8] | multi-turn recovery: inconclusive |
| protected vs acc-legacy | −0.62 [−1.07, −0.17] | −1.94 [−3.17, −0.77] | +0.04 [−0.28, +0.36] | −11.5% [−13.3, −9.7] | −11.1% [−13.4, −8.7] | single-turn savings kept (−12.5%) |

| tokens per step vs acc-legacy (multi-turn) | first response, first turn | first response, later turns | after a tool result |
|---|---|---|---|
| DECS mid | −22.8% | −17.4% | −15.4% |
| protected | −16.7% | −8.4% | −10.7% |

Protection halves the cut on the messages it targets (−8% against −17%). But the student still shortens its
reasoning in multi-turn conversations: −11% of multi-turn tokens, against −17% for DECS. The shortening learned on
single-turn prompts and after tool results carries over to turn starts at test time. Multi-turn accuracy still
drops (−1.9). Per point of multi-turn tokens saved, the cost matches DECS's: 0.17 points of multi-turn accuracy
per 1% for both (point estimates). Protection moves along the accuracy-per-token line rather than off it. So the multi-turn cost
follows how much multi-turn reasoning the student removes at test time, not where DECS was trained.

| protected arm at matched savings (6 pairs, joint bootstrap) | protected | DECS line at the same savings | difference |
|---|---|---|---|
| multi-turn accuracy | −1.94 at −11.1% multi-turn tokens | −1.92 | −0.01 [−1.16, +1.07] |
| overall accuracy | −0.62 at −11.5% total tokens | −0.76 | +0.14 [−0.28, +0.54] |

The DECS line runs through zero and DECS mid's pooled point. At matched savings the protected arm is no better than
DECS. More seeds would tighten the interval around zero but would not turn the +0.96 against DECS mid into a gain.

**Random multi-turn gate (control).** This is DECS mid with DECS removed from the same number of multi-turn prompts
(1,934), chosen at random: 1,612 turn starts and 322 of the 397 messages after tool results.

| run | decoding seed | overall | multi-turn | single-turn | tokens/entry |
|---|---|---|---|---|---|
| gate-random-multi-turn | 0 / 1 / 2 | 67.77 / 68.04 / 67.11 | 39.62 / 40.38 / 38.38 | 81.84 / 81.87 / 81.48 | 1,827 / 1,827 / 1,831 |
| gate-random-multi-turn, seed 5678 | 0 / 1 / 2 | 68.01 / 68.35 / 67.71 | 39.75 / 40.25 / 40.50 | 82.14 / 82.41 / 81.32 | 1,819 / 1,848 / 1,761 |

| comparison (6 pairs) | Δacc | Δacc multi-turn | Δtotal tokens | Δmulti-turn tokens | verdict |
|---|---|---|---|---|---|
| random gate vs DECS mid | +0.75 [+0.31, +1.18] | +1.85 [+0.63, +3.02] | +9.6% [+7.4, +12.1] | +11.6% [+8.8, +14.8] | multi-turn recovery: recovers |
| random gate vs acc-legacy | −0.33 [−0.78, +0.12] | −1.04 [−2.29, +0.13] | −8.3% [−10.2, −6.3] | −7.1% [−9.5, −4.5] | single-turn savings kept (−11.9%) |
| protected vs random gate | −0.29 [−0.75, +0.14] | −0.90 [−2.10, +0.29] | −3.4% [−5.3, −1.5] | −4.2% [−6.8, −1.7] | multi-turn recovery: does not recover |

| at matched savings (DECS line) | multi-turn accuracy | overall accuracy |
|---|---|---|
| protected | −0.01 [−1.16, +1.07] | +0.14 [−0.28, +0.54] |
| random gate | +0.19 [−0.93, +1.27] | +0.22 [−0.18, +0.62] |

| tokens per step vs acc-legacy (multi-turn) | first response, first turn | first response, later turns | after a tool result |
|---|---|---|---|
| random gate | −17.1% | −7.5% | −4.6% |

The random gate "recovers" by the pre-registered rule. But that rule compares against DECS mid without matching
savings, and the gate keeps only 7% of multi-turn savings against DECS's 17%. At matched savings, both gated arms
sit on the DECS line. Choosing turn starts does not help: the protected arm keeps DECS after tool results, cuts
that reasoning by 11% instead of 5%, and loses 0.9 more points. That is the same trade again. Neither the
position nor the amount of multi-turn training data moves the student off the line. Single-turn prompts alone
still teach −12% single-turn tokens for free.

**Paper recipe** (α 2.0, weights summing to one, 400 updates; default decoding seed only, 2 pairs).

| run | training seed | overall | multi-turn | single-turn | tokens/entry |
|---|---|---|---|---|---|
| paper-acc-legacy (accuracy at 1.0) | 1234 / 5678 | 67.74 / 67.91 | 40.38 / 38.88 | 81.43 / 82.42 | 1,985 / 1,889 |
| paper-half-acc-legacy (accuracy at 0.5) | 1234 / 5678 | 67.39 / 66.71 | 37.62 / 36.00 | 82.27 / 82.07 | 1,858 / 1,877 |
| paper-acc-legacy+decs (0.5 + 0.5) | 1234 / 5678 | 67.19 / 67.48 | 37.00 / 38.12 | 82.28 / 82.16 | 1,764 / 1,788 |

| comparison (2 pairs) | Δacc | Δacc multi-turn | Δtotal tokens | at matched savings (overall / multi-turn) | verdict |
|---|---|---|---|---|---|
| 400 updates vs 200 (paper-acc-legacy vs acc-legacy) | −0.49 [−1.30, +0.26] | −1.44 [−3.63, +0.69] | −0.1% [−3.9, +4.1] | | |
| half accuracy vs paper accuracy-only | −0.78 [−1.58, −0.00] | −2.81 [−4.94, −0.62] | −3.5% [−7.9, +1.1] | −0.54 / −2.28 | |
| paper composition vs half accuracy (the DECS term) | +0.29 [−0.47, +1.05] | +0.75 [−1.38, +2.88] | −4.9% [−7.7, −2.1] | +0.61 [−0.21, +1.42] / +1.69 [−0.67, +3.96] | no detectable cost |
| paper composition vs paper accuracy-only (the paper's test) | −0.49 [−1.34, +0.31] | −2.06 [−4.31, +0.06] | −8.3% [−11.3, −5.0] | +0.05 / −0.62 | |

Training twice as long does not help the accuracy-only student: it ties on tokens and is no more accurate. So the
200-update recipe was not short of training.

The paper's composition costs about what its savings predict: −0.5 overall at −8%, on the DECS line. The two
changes it bundles differ:
- Halving the accuracy weight costs most of the multi-turn accuracy (−2.8) and saves little (−3.5%).
- The faint DECS term (0.0004 KL per token) saves 4.9% with no detectable cost. It is the one result this round
  above the line, but its interval includes the line.

That reading rests on 2 pairs and on the half-accuracy runs, which sit unusually far below the line themselves.
Against the paper's own accuracy-only student, the composition is on the line.

**Round summary.** Six pairs per comparison made the multi-turn effects measurable (±1.3 points). They show:
- DECS mid costs −1.1 overall and −2.9 multi-turn, more than the decoding-seed-0 estimate.
- Keeping DECS off turn starts, or off a random 83% of multi-turn prompts, keeps the free single-turn savings.
  It reduces the multi-turn cost only in proportion to the multi-turn savings it gives up. Both gated arms sit on
  the DECS line at matched savings.
- Shortened reasoning carries over from single-turn training prompts to multi-turn conversations at test time.
- Lightning Weave's literal weighting and 400 updates change nothing beyond this.
