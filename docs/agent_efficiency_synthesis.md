# Transferring an agent-efficiency shift from existing anchors

Agentic tool use has an accuracy anchor, Qwen3-4B-Base → Qwen3-4B-Thinking-2507
(the [LoopTool recipe](modal_looptool_opd.md)), but no efficiency anchor. These
experiments ask whether one can be transferred from public math/code anchors
without training it:

```text
r̂_agent,eff(a|s) = r_agent,acc(a|s) + λ · Σ_d β_d · ê_d(a|s),    ê_d = transform(r_d,eff − r_d,acc)
```

Every shift is evaluated at the frozen Qwen3-4B student's own LoopTool token
states: every donor pair scores the existing sealed cache
(`/opd/looptool-qwen3-4b-v1`). Variants train with the LoopTool recipe, and BFCL
v3 is the external evaluation. The results tracker is
`results/agent_eff/results.md` (local experiment artifacts).

The pre-cleanup source is preserved in commit `543e6d0e`. The maintained registry
keeps 27 arms, including the trained DECS low/high variants. Untrained exploratory
arms, residualization/PCA, completed reflection/value probes, and one-off diagnostics
were retired. The unrun Monte Carlo study remains available as the ground-truth test.
Historical tools can be recovered with `git show 543e6d0e:path/to/file.py`.

## What a shift is, and is not

KL-regularized RL motivates the construction: a converged policy's log-ratio to
its reference is a scaled advantage of its reward. The donors here, however, are
broad post-training runs from different organizations, recipes and sometimes
tokenizer families. So a policy shift is treated as an **empirical direction**,
not a recovered reward or a causal efficiency advantage. The Monte Carlo probes
and BFCL test whether it transfers.

| contrast | status |
| --- | --- |
| DECS − DeepScaleR | **Quasi-matched external contrast.** Same R1-Distill-1.5B base and closely related data, but different authors, steps and reward details. This is the primary efficiency direction. |
| DECS − Klear | The analogy as first stated. Klear differs in family, scale, tokenizer, data and recipe (SFT then RL), so subtracting it does not cleanly remove "accuracy". Exploratory only. |
| L1-Max − DeepScaleR, DLER / AdaptThink − DeepScaleR, Nemotron v1 → v2, DECS-7B − Skywork-OR1-7B | Length-aware policies that still differ in full training runs. Skywork is a weak control for DECS-7B, and Nemotron v2 also changes its algorithm and continues training. |
| Base → Thinking-2507 (agent accuracy) | A broad multi-stage post-training difference, not a single accuracy-reward update. |

**Cross-policy state mismatch.** Projection makes action IDs comparable, but it
does not make a Qwen3 tool-formatted prefix an in-distribution state for an R1
reasoning donor. Conclusions are therefore restricted to `supported` states,
where every source can score at least 90% of the student's candidate mass and
each of its anchors puts at least 0.5 of its own probability on those
candidates. Every analysis also reports the complementary `unsupported` slice.
For tool-call and message-boundary states the cleanest donors would be
same-family or in-house.

**In-house matched donor.** The strongest validation starts from one checkpoint
and one recipe and trains two versions: accuracy reward only, and accuracy plus
an explicit cost. That GRPO pair is required for the mechanistic claim; it is not
optional.

## Geometry conventions

The tilted-target loss works over K+1 buckets: the cached Top-16 plus one
remaining-vocabulary bucket fixed at δ = 0. On that space
`q(a|s) ∝ b(a|s)·exp(δ(a|s)/α)` is invariant to adding a per-state constant to
**all K+1 entries**, but not to shifting only the 16 candidates. So
[`shift_geometry.py`](../data_curation/shift_geometry.py) represents every
direction as `[δ_1..δ_16, 0]`, centered under the behavior bucket distribution
b. It compares directions with the Fisher metric of the tilt:
`⟨u,v⟩_s = Cov_b(u, v)`, where `KL(q_u‖q_v) ≈ Var_b(u − v)/(2α²)`.

**Evidence.** A source has no evidence at a candidate in two cases:

- **Unmapped:** the candidate has no exact or aliased counterpart in the donor's
  vocabulary, and its stored score is a placeholder.
- **Untrained:** either anchor's output row for that token is untrained.

Such a candidate gets δ = 0, the same "no change" value the loss gives every
unscored token. Its odds against the remaining vocabulary are therefore
untouched; a regression test pins this. Two quantities describe a source at each
state:

- **coverage:** the student's behavior mass on the candidates the source can score;
- **support:** the smaller of the two anchors' own mass on those candidates.
  Placeholders are excluded, and aliases must be injective.

Centering followed by re-leveling is the identity on raw pair shifts. So only
these operations change a composed target:

- evidence masking;
- gating.

**Stability.** If a synthesized shift has error `err` with `osc(err) ≤ 2ε` over
the support, then:

- every pairwise target log-odds moves by at most `2ε/α`;
- `KL(q*‖q̂) ≤ osc²/(8α²) ≤ ε²/(2α²)`.

This is a per-state statement: distillation error and state drift come on top of it.

**Uncertainty and slices.** Cosine intervals come from a prompt-level bootstrap.
The placebo is 200 draws of independent per-prompt, per-direction sign flips.
Probe intervals are prompt-clustered. Slices are:

- the token-state types;
- position deciles;
- supported and unsupported states;
- single-turn vs multi-turn conversations;
- the target kind (single call, parallel calls, text).

## Token-state taxonomy

[`shift_states.py`](../data_curation/shift_states.py) labels each position from
its prefix and the student's cached distribution, never from the sampled token.

| type | state |
| --- | --- |
| `think_stop_fork` | inside `<think>`, both a single-newline sentence end (which precedes `</think>`) and a paragraph break have mass. Qwen3 decides to stop here; `</think>` itself is nearly deterministic |
| `think_reflection_fork` | first token of a reasoning paragraph where Wait / Hmm / But / Alternatively has mass |
| `act_vs_talk` | outside reasoning and calls, `<tool_call>` competes with text |
| `tool_name`, `tool_args`, `tool_syntax` | inside `<tool_call>` JSON |
| `call_boundary` | after `</tool_call>`: end the message or call again |
| `end_of_message` | `<|im_end|>` has mass outside a call |

These rules are heuristics. Inspect representative states before building claims
on them, and rerun geometry at `--fork-mass` 0.01, 0.02 and 0.05 to check threshold
sensitivity.

## Donors and their evidence

[`configs/agent_eff/config.py`](../configs/agent_eff/config.py) pins every pair.

- **Core pairs:** Klear, DECS, and DeepScaleR.
- **Census pairs:** L1-Max, DLER, AdaptThink, Nemotron v1 → v2, DeepCoder, and
  DECS-7B with Skywork-OR1-7B. DeepCoder's pre-anchor is R1-Distill-1.5B, per its
  model card.

**Untrained special tokens.** Qwen3-4B-Base leaves all of its chat, tool and
think tokens untrained.

- In the legacy agent-acc target, `<tool_call>` is pushed at act-vs-talk states
  by about +14 nats, and `<|im_end|>` over another call by about +20.
- With those rows carrying no evidence, the act-vs-talk push is about +1.1 nats.
  The clean and legacy directions then have cosine −0.28 at those states (200-row
  smoke).

[`audit_donor_tokens.py`](../data_curation/audit_donor_tokens.py) flags a token
when either anchor's output row exactly duplicates another special row or is
near zero. The flag is positive evidence of an untrained row; its absence does
not prove the row was trained. The audit therefore also reports, for every
structural token, its row norm and its closest special-row cosine.
[`verify_donor_scores.py`](../data_curation/verify_donor_scores.py) checks the
scored states for special-token shifts that tie exactly. The flag uses the union
over the two anchors, because an untrained row in either one puts an arbitrary
term in the log-ratio.

**Unmappable candidates.** The R1-Distill vocabulary has `<think>` and
`</think>`, and has `<tool_call>` (untrained). It lacks `<|im_start|>`,
`<|im_end|>` and `<tool_response>`.

- Donor scoring passes `--per-candidate-validity`, so one unmappable candidate no
  longer masks its whole position.
- `--special-token-alias` maps `<|im_end|>` to R1's end-of-sentence token. Aliases
  must be injective, so no teacher token is counted twice.

**Mapped is not the same as usable.** On 400 audit rows, the aliases raise the
DECS/DeepScaleR mapped behavior mass to about 100% at call-boundary and
end-of-message states. Whole-position coverage stays much lower: about 7.5% at
call boundaries, about 67% at end-of-message states, and 0% at act-vs-talk. And
`<tool_call>` is untrained in both anchors, so at act-vs-talk states the donor's
usable evidence excludes the defining action entirely.

`verify` reports the evidence the composer actually uses, **mapped AND trained**,
per state type, as whole-position and behavior-mass coverage together with the
support distribution.

**Existing scores and aliases.** The DECS and DeepScaleR shards were first
scored with a non-injective alias map that also mapped `<|endoftext|>`; they are
rescored with the corrected map. Klear's shards are unaffected.

## Composition

[`build_synthetic_shift_target.py`](../data_curation/build_synthetic_shift_target.py)
composes a JSON spec of terms.

**Transforms:**

- `raw`: the legacy log-ratio, bit-identical to the reference composer, including
  untrained rows. It refuses unmapped placeholders.
- `evidence`: δ = 0 without evidence.
- `heuristic`: fixed pushes on token groups at the specified forks, used by the trained control.

**Gates:** exclude named state types, preserve the first reflection fork, or apply
per-prompt weights. The last two use `exclude_first_reflection` and `prompt_weights`;
protocol protection uses `exclude_state_types`.

**KL budget.** At most one term per spec may set a `kl_budget`. It is calibrated
on a prompt-stratified 25% sample and capped by `max_multiplier`. An unreachable
budget fails. The global and active-state KL quantiles (p50/p90/p95/p99/max) are
recorded, so a budget reached through a few extreme states is visible.

**Outputs.** Each variant is sealed once into `synthetic/<variant>/`, and its
manifest records the spec, source directories, untrained-token lists and
calibration. The Modal `compose` stage reuses an existing directory only if it
was composed from the same spec; otherwise it refuses and asks you to remove it.
The legacy target composed this way matches the trained V0 target to within
float32 rounding (max |Δδ| ≈ 1e-6).

## Experimental protocol

The work proceeds in stages, each building on the results of the one before.
Nothing later is launched until the earlier stage has been read.

1. **Donor scores.** Score the core pairs (Klear, DECS, DeepScaleR) and run
   `verify`, which reports each pair's evidence by state type.
2. **Core geometry.** A CPU pass over the scored cache: how the accuracy and
   efficiency directions relate, per state type, on supported states.
3. **First experiments: the primary matrix.** It is pre-registered, runs at one
   budget (0.0135 nats/token), and every arm is trained and evaluated on BFCL:

   | arm | accuracy term | efficiency term |
   | --- | --- | --- |
   | `acc-legacy` | legacy (the V0 target, retrained through this pipeline) | none |
   | `acc-clean` | untrained rows carry no shift | none |
   | `acc-legacy+decs`, `acc-clean+decs` | legacy / clean | `r_DECS` (naive cross-domain Weave) |
   | `acc-legacy+decs-deepscaler`, `acc-clean+decs-deepscaler` | legacy / clean | `r_DECS − r_DeepScaleR` |
   | `acc-clean+decs-deepscaler-flipped` | clean | the same contrast with its sign flipped, at matched KL |

   The 2×2 block separates cleaning the accuracy anchor from adding an
   efficiency term, and the flipped arm controls for "any perturbation at this
   KL". These are reported together; none is selected on BFCL.
4. **Building on it**, depending on what the matrix shows:
   - the inference-time baselines below;
   - the trained low/high DECS budgets, gated arms and census donors;
   - a second training seed and decoding seed for anything that looks like a
     result (BFCL `--seed` gives each decoding seed its own run tree).
5. **Last: mechanism validation.**
   - the Monte Carlo probes;
   - manual validation of the state taxonomy;
   - the in-house GRPO donor pair.

**Inference-time baselines**, which any learned efficiency must beat:

- thinking disabled (`base-nothink`, `opd-nothink`);
- a 4,096-token per-step cap (`*-cap4k`);
- a "reason only as much as needed" system prompt (`*-concise`).

A forced-stop policy that keeps tool output valid is not implemented.

**Decision rule** (from `bfcl_efficiency.py`). An arm passes when all of these hold:

- the accuracy change's lower CI bound is above −1 point;
- the upper CI bound of the paired log-ratio of generated tokens is below −5% (the primary cost estimate, about ±2% from one run);
- irrelevance accuracy is non-inferior;
- the guards are non-inferior: each upper CI bound stays within its tolerated increase (`GUARD_TOLERANCES`: +0.05
  duplicate calls per entry, and +0.5 points each for the runaway and overflow rates). Before 2026-09-30 a guard
  passed whenever its interval merely included zero.

AES is descriptive only.

## Code-efficiency round (pre-registered 2026-09-30)

The first author's construction applied to both domains:
agent efficiency ≈ agent accuracy + (math efficiency − math accuracy) + (code efficiency − code accuracy).
Elastic Reasoning (arXiv 2505.05315) trained each E1 model from its domain's accuracy RL model. That model is
therefore the pair's pre-anchor, and the shared R1-Distill root cancels.

| symbol | shift | pair (pinned in `config.py`) |
| --- | --- | --- |
| A | agent accuracy | Thinking-2507 − Qwen3-4B-Base (the `acc-legacy` term) |
| C | code efficiency − code accuracy | E1-Code-14B − DeepCoder-14B-Preview |
| M | math efficiency − math accuracy | E1-Math-1.5B − DeepScaleR-1.5B-Preview |
| D | Lightning Weave's efficiency anchor | DECS-1.5B − R1-Distill-1.5B |

**Verified before the round.**

- The pre-anchors are named in the paper and confirmed by weight slices: E1-Code is 2× closer to DeepCoder-14B
  than to R1-Distill-14B, and E1-Math is closer to DeepScaleR than to R1-Distill-1.5B.
- The four models share DECS's `tokenizer.json` byte for byte. Its 151,643 regular tokens equal Qwen3-4B's, so
  the same alias and projection apply.
- `<tool_call>` stays untrained in the R1-Distill lineage. In the 7B and 14B checkpoints, however, the untrained
  special rows differ by about one bfloat16 step: their cosine is within 1e-7 of 1, while trained special rows
  stay below 0.9. The audit's exact-duplicate rule therefore missed them. It now also flags near-duplicates
  (cosine ≥ 1 − 1e-4), so these donors carry no shift on `<tool_call>` itself, as DECS does not. A pair check
  enforces this.
  - This does not make a donor neutral about the call. Shifts on competing text candidates still move the call's
    probability after normalization, and donor support at the call-or-reply and call-boundary states is 0–6%.
    Only removing the term at those states entirely (the `exclude_state_types` gate) preserves those decisions.
  - Footnote: DECS-7B, Skywork-7B and Nemotron were prepared before this fix. Their empty untrained lists let
    their log-ratios at `<tool_call>` into the DECS-7B, Nemotron and donor-sum targets. The effect is small in
    KL terms (act-vs-talk mean KL about 0.001 vs 0.0001 for DECS on one shard) but is a known asymmetry.
- E1 trains with a 1K-token thinking budget: `</think>` is forced when the budget runs out, and the reward is
  correctness only. The cache's thinking averages about 350 tokens.
- So "E1 minus its starting model" is budget-constrained correctness RL. It compresses reasoning, but it also
  continues training and teaches answering from interrupted reasoning, so it is not an isolated token-cost reward.
  C and M are empirical compression directions, and a log-ratio is not the student's cost advantage.
- License: the E1 weights are CC-BY-NC-4.0.

**Arms.** Every arm keeps A at weight 1 and uses α = 2.0, 200 updates, and the evidence transform. It adds one
efficiency term, scaled to the primary KL budget (0.0135) with `max_multiplier` 50.

| arm | added term | training seeds |
| --- | --- | --- |
| `acc-legacy+e1code` | λ·C | 1234 and 5678, both run whatever the first shows |
| `acc-legacy+e1math` | λ·M | 1234 |
| `acc-legacy+e1math+e1code` | λ·(M + C) | 1234 |
| `acc-legacy+decs+e1code` | λ·(D + C) | 1234 |

Inside a sum (`equalize: "auto"`), the two shifts keep raw 1:1 weights when their Fisher RMS sizes are within
2×. Otherwise they are equalized to unit RMS. The sizes and the rule applied are recorded in the manifest.

**Checks, fixed in advance** (`PAIR_GATES`, `TAIL_GATE`). A failed pair holds every arm that reads it.

- At reasoning (`think_body`) states: evidence ≥ 0.9, support median ≥ 0.7, and support p10 ≥ 0.2. Every donor
  so far is at least 1.00 / 0.92 / 0.43.
- At act-vs-talk states, evidence ≤ 0.05 (DECS: 0.008). This checks that the untrained call rows were flagged
  (the pair has no evidence on the call token). It does not make the decision neutral; see above.
- Numerical robustness: 64 rows of the first shard are re-scored with the main run's bfloat16 weights, but with
  float32 arithmetic and SDPA attention, so only the numerics differ. Loading float32 checkpoints unrounded would
  change the pre-anchor by more than the whole E1 update. The Fisher-weighted shift correlation with the main
  bfloat16/FA2 scores must be ≥ 0.8. DECS is re-scored the same way as the reference. A re-score that cannot
  finish within 4 hours of launch is recorded as unmeasured and holds nothing. The check covers arithmetic only (not
  checkpoint rounding), on 64 rows of one shard, and a global correlation does not certify sensitive states.
- Each composed target's per-token KL p99 must be ≤ 4× that of `acc-legacy+decs`. With the mean fixed at the
  budget this is a weak check, since Markov's inequality already bounds p99 near 1.35. Calibration therefore
  also records where the KL lands, descriptively: per state type, and the share carried by the top 0.1%, 1% and
  10% of positions.
- A pair that fails to score holds only its own arms. A calibration that cannot reach the budget within the
  multiplier cap reads as "shift too small at this budget".

**Comparisons.**

- Primary: accuracy against DECS at matched realized total-token savings, using DECS's low/mid/high curve.
  Multi-turn is reported separately, and each seed's effect is reported alongside pooled intervals.
- A composition counts as a composition win only if it beats both of its constituents at comparable cost.
- The earlier accuracy-per-token line is an annotation, not the success criterion.
- BFCL is the development benchmark. Only `acc-legacy+e1code` is a pre-registered test. Anything selected after
  seeing results needs a fresh seed and a second benchmark.

The round-specific orchestrator has been retired. The reusable `onboard`, `precision`,
`compose`, `review`, and `build` stages remain; the pair/tail checks and seeds above
remain in the config as the record of this protocol.

The subsequent paper arms test Lightning Weave's equal normalized weights
(0.5 + 0.5 at α 2.0), without KL calibration; see below.

**Follow-ups after the external review** (`followup_specs()`, seeds 1234 and 5678). Each changes one thing from
`acc-legacy+e1math`.

- `acc-legacy+e1math-protected`: the same coefficient (7.945, not recalibrated), with the term removed at
  act-vs-talk and call-boundary states. This tests whether E1-Math's accuracy cost and extra calls come from its
  pushes at those unsupported protocol decisions.
- `acc-legacy+e1math@low`: the low KL budget (0.005), expected near DECS mid's 14% savings. It compares
  accuracy with DECS at matched realized savings instead of by extrapolating DECS's curve.

## Lightning Weave's published recipe and turn-start allocation (registered 2026-09-30)

Both ask why efficiency transfer is free on single-turn BFCL but costs about 2 points on multi-turn (results.md
sections 5 and 12–14), while Lightning Weave's composition costs about nothing against its accuracy-only student in
math and code. Every arm trains at seeds 1234 and 5678 (`NEXT_SEEDS`) and is compared with its same-seed reference.

**Paper arms** (`paper_specs()`, `TRAINING_PASSES`). Lightning Weave (arXiv 2609.14708, Sec. 4.1) trains every arm
at α = 2.0 with anchor weights that sum to one: Klear-only is accuracy at weight 1.0, and the equal composition is
0.5 accuracy + 0.5 DECS. Our arms instead keep accuracy at 1.0 and add DECS at 3.1× that weight (KL-calibrated),
over one pass. The paper does not state its number of updates; the released code defaults to two passes over the
cache (400 updates), so the paper arms train two passes on a byte-identical repeated copy of the target.

| arm | target | training |
| --- | --- | --- |
| `paper-acc-legacy` | the acc-legacy accuracy term at weight 1.0 (the Klear-only analog) | α 2.0, 2 passes |
| `paper-acc-legacy+decs` | accuracy at 0.5 + DECS at 0.5 (evidence transform) | α 2.0, 2 passes |
| `paper-half-acc-legacy` | accuracy at 0.5 alone | α 2.0, 2 passes |

- The paper's composition changes two things against Klear-only: it adds DECS and it halves the accuracy weight.
  Scaling the accuracy term down shortens responses here (0.86× saved 2.5% of total tokens, results.md section 5),
  since the accuracy anchor is what lengthens them (+17% over base). The half-accuracy arm separates the two.
- Per unit of shift, DECS pushes 0.5/2 = 0.25 here against 3.11/2 = 1.56 in our mid arm. On one shard (800 rows),
  DECS adds 0.0004 KL per token inside the composition, against 0.0145 for DECS mid. Halving accuracy moves the target
  much further (0.0134 from acc-legacy's). So the composition mostly tests the weaker accuracy weight, with a faint
  DECS push on top.
- The DECS term keeps the evidence transform of every DECS arm here. A raw composition would intersect the masks
  and delete the agent-accuracy signal at tool-call states.
- Comparisons:
  - composition vs paper accuracy-only: Lightning Weave's own test, with both changes at once;
  - composition vs half accuracy: what DECS adds at the same accuracy weight;
  - half accuracy vs paper accuracy-only: what halving the accuracy weight costs;
  - paper accuracy-only vs acc-legacy: what 400 updates instead of 200 do to the accuracy-only student.

**Turn-start allocation** (`turn_start_specs()`). Both arms keep DECS at its mid coefficient (3.1136, fixed, as in
the other gated DECS arms) and remove it from whole cached responses (reasoning, calls and text) on 1,934 of the
3,200 prompts. They differ only in which prompts.

- `acc-legacy+decs-protect-turn-starts` removes DECS on every prompt whose target answers a new user message after
  earlier assistant turns. These are 83% of the 2,331 multi-turn prompts; single-turn prompts (869) and messages
  after tool results (397) keep it.
- `acc-legacy+decs-gate-random-multi-turn` removes DECS on as many multi-turn prompts chosen at random (a fixed hash
  of the prompt id): 1,612 turn starts and 322 messages after tool results.
- So both arms mostly test an allocation: DECS kept off most multi-turn training prompts. The two gates share 1,612
  prompts and differ on 322 each, which makes their direct comparison weak by construction.
- Positions come from the canonical message roles (`data_curation/turn_positions.py`). They match a
  rendered-template classification on all 3,200 prompts.
- On one shard (800 rows), DECS adds 0.0078 KL per token in the protected target and 0.0068 in the random gate,
  against 0.0145 for DECS mid. On gated rows both targets equal acc-legacy's exactly, and elsewhere they equal DECS
  mid's. Target KL does not predict how much the student absorbs, so savings are measured, not predicted.

**Extra decoding seeds** (`EXTRA_DECODING_SEEDS`, `DECODING_REPLICATES`). With one decode per checkpoint, the pooled
95% interval of a multi-turn difference is about ±2.2 points, larger than DECS mid's whole multi-turn cost
(−1.88 [−4.06, +0.38]). The turn-start arms and their references (acc-legacy and DECS mid), at both training seeds,
are therefore also decoded at sampling seeds 1 and 2. Each seed is a full BFCL run in its own results tree. This
gives 3 decodes × 2 training seeds = 6 pairs per turn-start comparison. Decoding more cannot remove the noise
between training runs, so the interval narrows by an unknown amount, at most √3. The decoding-seed null measures it.

**Estimands.**

- Accuracy: the BFCL v3 leaderboard-weighted score overall, multi-turn (800 entries), and single-turn (the mean of
  the non-live and live groups).
- Cost, primary: total generated tokens summed over all entries. For the turn-start arms, the same sum over
  multi-turn entries is co-primary, and the sum over single-turn entries checks that the savings stay.
- Each pair compares an arm with its reference at the same training seed and the same decoding seed. A comparison is
  the mean over its pairs, with one category-stratified bootstrap over entries shared by every run
  (`evaluation/bfcl_pooled.py`). It reproduces results.md's DECS mid row (−0.73 [−1.57, +0.05]).

**Decision rules, fixed in advance.** They are written into `configs/agent_eff/next_comparisons.json` and applied by
`evaluation/bfcl_pooled.py`, from each metric's 95% interval [lo, hi]:

- Turn-start arms against DECS mid, multi-turn accuracy:
  - **recovers** if lo > 0 (**partial** if also hi < +1.0);
  - **does not recover** if hi < +1.0, which rules out recovering more than about half of DECS's cost;
  - **inconclusive** otherwise.
- The same rule applies to the protected arm against the random gate: does the choice of prompts matter beyond the
  allocation?
- Single-turn savings must stay: each turn-start arm's single-turn token change against acc-legacy needs hi < 0.
- Paper composition against half accuracy, overall accuracy:
  - **no detectable cost** if lo > −1.0 (**small cost within margin** if also hi < 0);
  - **costs** if hi < 0;
  - **inconclusive** otherwise.

  The multi-turn change is reported with its interval, without a rule. The paper arms use the default decode only.
- The same file feeds the section-12 failure breakdown (`evaluation/bfcl_multiturn_failures.py --comparisons`). That
  tool now splits first-step tokens between the conversation's first turn and later turns, the part the turn-start
  gate targets.

**Caveats.**

- Seed 5678 shuffles the rows. On a repeated copy the shuffle spans both copies, so each row is still read twice, but
  not once in each half of the run as at seed 1234.
- Gating whole prompts also removes DECS's push on their calls and text, not only their reasoning.
- BFCL decoding uses sampling seed 0 for every run, the same as all earlier runs. NeMo-Skills sends
  `inference.random_seed` as each request's seed, so a decoding-seed replicate now passes it explicitly
  (`configs/bfcl_eval/config.py`).
- A completed training run is reused only if its recorded target revision, α and number of passes match the
  current plan.

```bash
# training: 5 variants x 2 training seeds, server-side
bash scripts/run_agent_eff.sh build --seeds 1234,5678 \
  --variant paper-acc-legacy,paper-acc-legacy+decs,paper-half-acc-legacy,acc-legacy+decs-protect-turn-starts,acc-legacy+decs-gate-random-multi-turn
# BFCL at the default decoding seed: the 10 new students, each started once its checkpoint is exported
AGENTIC_EVAL_DIR=~/agentic-eval-36a183a0 bash scripts/run_bfcl_eval.sh run --wait-for-checkpoints --max-wait-hours 10 \
  --models "ae.joint.paper-acc-legacy,ae.joint.paper-acc-legacy.s5678,ae.joint.paper-acc-legacy+decs,ae.joint.paper-acc-legacy+decs.s5678,ae.joint.paper-half-acc-legacy,ae.joint.paper-half-acc-legacy.s5678,ae.joint.acc-legacy+decs-protect-turn-starts,ae.joint.acc-legacy+decs-protect-turn-starts.s5678,ae.joint.acc-legacy+decs-gate-random-multi-turn,ae.joint.acc-legacy+decs-gate-random-multi-turn.s5678"
# BFCL at decoding seeds 1 and 2: the 4 turn-start students and the 4 existing references
for seed in 1 2; do
  AGENTIC_EVAL_DIR=~/agentic-eval-36a183a0 bash scripts/run_bfcl_eval.sh run --wait-for-checkpoints --max-wait-hours 10 --seed "$seed" \
    --models "ae.joint.acc-legacy,ae.joint.acc-legacy.s5678,ae.joint.acc-legacy+decs,ae.joint.acc-legacy+decs.s5678,ae.joint.acc-legacy+decs-protect-turn-starts,ae.joint.acc-legacy+decs-protect-turn-starts.s5678,ae.joint.acc-legacy+decs-gate-random-multi-turn,ae.joint.acc-legacy+decs-gate-random-multi-turn.s5678"
done
# analysis (CPU), on a local copy of the results volume's run trees
python evaluation/bfcl_pooled.py --root RESULTS --comparisons configs/agent_eff/next_comparisons.json --output next_pooled.json
python evaluation/bfcl_multiturn_failures.py --root RESULTS --ground-truth DIR --output next_failures.json \
  --comparisons configs/agent_eff/next_comparisons.json
```

## Decision-preserving projection (pre-registered 2026-10-01)

Every donor, strength, gate and composition so far pays the same multi-turn price per token saved, while single-turn
savings are free. The working explanation is a property of the target, not of the donor. Write a response as
reasoning z followed by a decision a: its complete visible output, exactly as BFCL's server shows it (calls with
arguments, in order, and any visible text), with how it ended (its finish reason). The ordinary tilt
q ∝ p0(z, a | h) · e^{s(z)/α} changes the decision distribution too:

    q(a | h) ∝ p0(a | h) · Z(h, a),    Z(h, a) = E_{z ~ p0(· | h, a)}[e^{s(z)/α}],

so decisions whose reasoning is short gain probability. In a multi-turn task the environment executes each decision,
and a small shift per step compounds. The projection keeps the recipient's decision distribution and tilts only the
reasoning within each decision:

    q*(z, a | h) = p0(a | h) · q_B(z | h, a).

By the KL chain rule it is the closest target to the ordinary one with p0's decision marginals; it needs no accuracy
labels. If later histories depend only on earlier histories and decisions (P(h' | h, z, a) = P(h' | h, a)), matching
decisions at every step preserves the whole interaction, in a stochastic environment too. Under interleaved thinking
the reasoning stays visible later in the same turn, so there the projection is not exact.

**Arms** (`PROJECTION_*` in `configs/agent_eff/config.py`). All three start from the recipient, acc-legacy at
training seed 1234, and fit the same samples of it; only the weights differ (`data_curation/decision_projection.py`):

| arm | weight of sample i of a prompt with N samples | what it isolates |
| --- | --- | --- |
| `uniform` | 1 | training on its own samples, e.g. sharpening |
| `ordinary` | N · e^{s_i/α} / Σ_prompt e^{s_j/α} | the usual tilt in sequence form |
| `projected` | n_g · e^{s_i/α} / Σ_{j ∈ g} e^{s_j/α}, g = i's decision group | decision preservation |

- **Decisions** are the server's content, compared byte for byte: no whitespace, argument-order or call-format
  normalization. The split follows the pinned vLLM 0.11.0 `qwen3` reasoning parser (`served_split`): unless both
  `<think>` and `</think>` are present, the whole output is content, so an unclosed reasoning block is visible text
  (BFCL shows and keeps it in the history) and two different unclosed responses are two decisions. A length-limited
  response keeps what it showed, marked by its finish reason.
- **Samples**: 8 per LoopTool prompt (25,600) from the recipient with BFCL's decoder (T 0.6, top-p 0.95, top-k 20;
  `collect_direct_opd_rollouts.py --sampling-top-k`). They are drawn without BFCL's static YaRN override, because
  training computes the likelihood without it: the reference is the policy being fitted, and YaRN is added at
  evaluation alike for the recipient and every arm, as for every earlier student.
- **Score**: reasoning-only DOPD (`data_curation/score_reasoning.py`), s = log DECS(z | h) − log R1-Distill-1.5B(z | h).
  z is the reasoning through its closing `</think>` (all of an unclosed response, also when the prompt opened the
  block); h is BOS plus the recipient's exact prompt text and the response's opening `<think>`, encoded with the
  donors' shared tokenizer, without a chat template. Both donors load at their pinned snapshots, in float32. Decision
  tokens are not scored.
- **One α** for all arms, the weakest tilt at which the projected target's implied cut in response tokens reaches 15%
  (`calibrate_alpha` scans a fixed log grid from weak to strong tilt, then bisects the first bracket: savings need not
  grow as the tilt strengthens). If no grid point reaches 15%, the weights stage stops the round, with its
  diagnostics written, for a decision before training. The ordinary arm then implies more, because it can also move
  decisions; realized savings are measured.
- **Optional anchor** w_η = (1 − η) w + η for every arm, η = 0 unless the weight diagnostics call for it. It keeps each
  decision group's total weight.
- **Training** (`--loss-mode sequence_weighted`): weighted, summed log-likelihood over complete responses, including
  the decision and the stop token, with no token KL. The per-sample reducer is used and the trainer divides by the
  fixed global batch size, so a step fits Σ_i w_i log p(y_i) / 64; dividing by each batch's token count instead would
  weight a response by the inverse length of its batch (tested through slime's Megatron wrapper:
  `tests/test_sequence_weighted_loss.py`). Log-probabilities are temperature-scaled (T 0.6), as slime's
  generation-config lock requires. One pass over the 25,600 rows at the recipe's batch sizes and learning rate: 100
  batches of 256, 400 updates, the same budget for all three arms (their comparison is the control; the earlier 200
  vs 400 result does not establish equivalence under this loss). Training seeds 1234 and 5678 both start from the
  same recipient and samples (5678 also reorders the rows).

**Decision rules, fixed in advance** (`PROJECTION_RULES`, applied by `evaluation/bfcl_pooled.py`; each arm is paired
with the recipient at the same BFCL decoding seed 0, 1, 2):

- Each arm against the recipient, multi-turn accuracy: **no detectable cost** if lo > −1.5 points, **costs** if
  hi < 0, **inconclusive** otherwise. Total tokens: **reduced** if hi < 0.
- Projected against ordinary, multi-turn accuracy: **recovers** if lo > 0 (**partial** if also hi < +1.5), **does not
  recover** if hi < +1.5, **inconclusive** otherwise.
- "At matched savings" is claimed only if the two arms' realized total-token savings against the recipient (pooled
  point estimates) differ by at most 3 percentage points (`decision_projection.savings_matched`).
- Pre-registered hypothesis: the donor supplies useful shortening within identical decisions, and the projection gives
  less decision drift on fixed histories and better multi-turn accuracy than the ordinary tilt at comparable realized
  savings.

**Decision drift** (`evaluation/decision_drift.py`, `PROJECTION_DRIFT`; exploratory). Histories come from one new
recipient run at BFCL decoding seed 3, made with `--log-requests` (a run tree evaluated without logging cannot be
backfilled). That run serves only the drift histories and stays out of the accuracy comparisons at seeds 0-2. Up to
500 turn starts and 500 later steps of its multi-turn requests, exactly as the model saw them, are chosen by a fixed
hash and frozen (`histories.frozen.json` records their hash, which `compare` checks) before any arm is replayed.
Each history is replayed to the recipient twice and to every arm: 4 responses each, with BFCL's serving settings
(its decoder, 32,768 new tokens, the 65,536-token YaRN context), each reduced to its decision as the server shows it.
vLLM seeds the n samples of a request s, s + 1, ..., so the recipient's reference draw, its null draw and the arms
use disjoint blocks (0, 1,000, 2,000); the arms share one, for paired comparisons between them. Two views, byte-exact
and call-level (calls with arguments; any text reply counts as "reply"), each give per arm: the total variation from
the reference draw minus the null draw's, and the reference match rate against the null draw's, averaged over
histories with a bootstrap, overall and by kind. With 4 samples the empirical total variation saturates: if nearly
every response is unique, two draws of one policy and two different policies are all at distance 1, and the excess is
0 either way. Each view therefore reports how often each model repeats a decision and how often the recipient's own
two draws never agree; a small excess where these are high is no evidence that decisions were preserved.

**Diagnostics before training** (`decision_projection.py diagnose`): decision-group coverage, the
shortest-within-decision bound, effective sample size per prompt and per group, the target-weighted length curve, the
ordinary target's decision movement (total variation; the projection's is 0 by construction), and the within-group
score–length rank correlation (ties share their mean rank; a local check only).

**Caveats.**

- A text reply is a decision with its exact wording, so nearly every text reply is a singleton and gets no efficiency
  push. On the frozen-base shard (200 prompts × 4), 75% of samples share a decision with another sample; the
  recipient's 8 samples per prompt must be re-checked.
- The target preserves the recipient's empirical decision frequencies, not the trained student's; student drift is
  measured, not assumed. Fitting T-scaled likelihoods matches the decoder's temperature but not its top-k / top-p
  truncation, which the uniform arm shares and the drift measurement covers.
- Reasoning-only scoring differs from the complete-response log-ratio: the donors' likelihood of the decision tokens
  can vary with the reasoning before them, even within one decision group.

**The probe of the trained students (pre-registered 2026-10-02, before any probe result;** `PROJECTION_PROBE`,
`evaluation/projection_probe.py`**).** The first round realized 3–5% BFCL savings, and the BFCL comparisons were
inconclusive. Before any more training, one job measures what the existing students learned:

- **Prompts:** 400 training prompts, chosen by a fixed hash, and the 600 held-out LoopTool prompts of the
  reasoning-value probe (never trained on; tool-call references).
- **Sampling:** 8 responses each with the collection settings. The recipient samples twice, in disjoint seed blocks;
  every student samples once, in one shared block.
- **Teacher forcing:** the recipient and every student score the cached responses of the training prompts.
- **Realization:** training-prompt savings divided by the arm's target-implied savings on the same 400 prompts.
  "Substantially realized" if the lower bound is at least 0.7, "weakly realized" if the upper bound is at most 0.4,
  inconclusive otherwise. High realization shows the shortening was largely learned, not that the target was fitted
  completely or that more training cannot help transfer. Low realization motivates examining optimization (the
  teacher-forced fit slope first) without proving that more passes are the cure.
- **Generalization:** undefined unless the training-prompt savings are shown positive. Then "generalizes" if held-out
  savings are at least 0.7 × training-prompt savings with confidence, "training-specific" if at most 0.4 × with
  confidence, inconclusive otherwise. These are contrasts resampling both prompt sets, not a ratio of bootstrap draws.
- **Decision movement:** an unbiased estimate of Σ_a (p_a − q_a)² between each student's and the recipient's
  call-level decisions, per prompt, with a bootstrap over prompts. Two separate flags, which can both hold: detectable
  movement if the lower bound is above 0; within the margin if the upper bound is below half the ordinary target's own
  squared distance on the training prompts. The margin is a mechanism diagnostic, not a guarantee of preserved
  accuracy. Byte-exact distances and held-out exact-call accuracy are descriptive.
- **What follows:**
  - weak realization: inspect the fit slope, then continue one seed of all three arms for passes 2 and 3 at the same
    learning rate and re-probe;
  - substantial realization: the BFCL dose is lost in transfer, so the next experiment changes what is transferred
    (for example a length-scored projection, or training states closer to BFCL's), not the number of passes;
  - inconclusive: the fit slope decides whether optimization is examined first. The teacher-forced log-likelihoods
    are raw (T = 1) while training fits T = 0.6, so the slope is a descriptive proxy: a weak slope alone does not
    establish poor optimization of the actual objective.

  A length-only score tests whether conditional projection improves compression; it does not test DOPD capability
  transfer, so a donor-scored arm stays. Savings matched on held-out LoopTool prompts need not match on BFCL: there,
  dose and dose matching are measured outcomes. BFCL accuracy stays the primary outcome, rejected calls a secondary
  reliability outcome (resampling whole tasks, reporting both the rate and the rejected calls per task), and the
  decision distance a separate mechanism diagnostic.

**Running** (each stage reuses finished work; `tests/test_projection_pipeline.py` runs every stage's real command
locally on a tiny cache, with vLLM and the donors stubbed):

```bash
bash scripts/run_agent_eff.sh projection --stages sample --workers 8           # GPU: the recipient's samples
bash scripts/run_agent_eff.sh projection --stages score --workers 8            # GPU: reasoning-only DECS scores
bash scripts/run_agent_eff.sh projection --stages weights --workers 8          # CPU: diagnostics, alpha, sealed arms
bash scripts/run_agent_eff.sh projection --stages train --seeds 1234,5678      # GPU: convert, train, export
# BFCL: the six students with logging, and the recipient's logged multi-turn run for the drift histories
AGENTIC_EVAL_DIR=~/agentic-eval-36a183a0 bash scripts/run_bfcl_eval.sh run --wait-for-checkpoints --log-requests \
  --models "ae.projection.uniform,ae.projection.uniform.s5678,ae.projection.ordinary,ae.projection.ordinary.s5678,ae.projection.projected,ae.projection.projected.s5678"
# (repeat with --seed 1 and --seed 2)
AGENTIC_EVAL_DIR=~/agentic-eval-36a183a0 bash scripts/run_bfcl_eval.sh run --seed 3 --log-requests --models ae.joint.acc-legacy \
  --categories multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context
# freeze the histories as soon as that run finishes, before inspecting any arm's outputs
bash scripts/run_agent_eff.sh projection --stages histories --logged-run <seed-3 run id>/ae.joint.acc-legacy
bash scripts/run_agent_eff.sh projection --stages drift --seeds 1234,5678
bash scripts/run_agent_eff.sh projection --stages probe --seeds 1234,5678         # the students on LoopTool prompts
```

## Monte Carlo probes

[`mc_advantage_probes.py`](../data_curation/mc_advantage_probes.py) continues
each probed candidate to the end of the assistant message, 8 times.

- **Scope:** message-level token and call cost. There is no environment and no
  later turn, so it does not measure interaction efficiency.
- **`call_match`:** tool-call structure only. For text references it checks only
  that no call was made.
- **Candidates:** each state's defining alternatives are forced in (stop vs
  continue, reflect vs not, call vs text, end vs call again), and at least 70% of
  the behavior mass must be probed.
- **Censoring:** a continuation that hits the response cap is censored, and token
  outcomes use only fully finished states. Finish rates are reported per state
  type and candidate.
- **Inference:**
  - every candidate must have exactly the expected number of continuations;
  - correlation intervals use a prompt-level bootstrap;
  - coefficients are fitted on one half of the prompts and scored on the other,
    next to a split-half noise ceiling.

## BFCL evaluation

**Interleaved thinking (from 2026-10-01).** Runs now keep the model's reasoning
from earlier steps of the current turn, between tool calls, as the Qwen3 chat
template intends. Earlier user turns' reasoning is still dropped, by design.
- **What changed:** NeMo-Skills always sent that reasoning back in
  `reasoning_content`, but vLLM 0.11.0 drops the field before templating; 0.11.1
  and later pass it through. The usage proxy now moves it into the message text
  as `<think>...</think>`, which the template renders byte-identically
  (`tests/test_bfcl_efficiency.py`).
- **Where results land:** new runs go to a new tree (`bfcl-v3-22b3e917da76-full`).
  `--no-interleaved-thinking` reproduces the original protocol and its tree,
  `bfcl-v3-3e6e955a00df-full`, which holds every run before 2026-10-01: base, OPD
  and the agent-eff students.
- **Comparisons:** results from the two protocols cannot be mixed. The training
  caches also lack earlier reasoning in their histories.

- **Request logging.** Lanes call `/lane/<category>/v1`, and the proxy logs each
  request with its user-message digests, tool-schema digest, body digest, usage
  and finish reason. Base and OPD predate the proxy, so they have no usage log.
- **The join.** [`bfcl_efficiency.py`](../evaluation/bfcl_efficiency.py) assigns
  requests by the user-message sequence, plus tool schemas for single-turn
  entries. Identical bodies count as retries only beyond the number of entries
  that could have sent them. Every entry must reconcile exactly, its logged
  completion tokens equaling `num_generated_tokens_list`, or the analysis fails
  (`--allow-partial-usage` overrides).
- **Provenance.** An agent-eff student's results manifest also records its
  training provenance: variant, target revision, seed and final iteration.

**Reported comparisons.** On BFCL multi-turn the number of user turns is fixed,
and base and OPD differ in 97–99% of entries by zero turns. A more accurate model
does not simply get more turns there; its extra cost is steps and tokens per
turn. Comparisons are nevertheless reported:

- on entries both models got right;
- by outcome-transition strata;
- over the multi-turn common horizon;
- per success.

Medians and quantiles use the leaderboard category weights.

## Running

```bash
bash scripts/run_agent_eff.sh plan
bash scripts/run_agent_eff.sh download --donors klear,decs,deepscaler
bash scripts/run_agent_eff.sh prepare  --donors agent_acc,klear,decs,deepscaler
bash scripts/run_agent_eff.sh score    --donors decs,deepscaler --workers 8     # server-side: post -> pre -> verify
bash scripts/run_agent_eff.sh verify   --donors klear,decs,deepscaler
bash scripts/run_agent_eff.sh precision --donors decs                       # independent GPU re-score, then CPU comparison
bash scripts/run_agent_eff.sh analyze  --donors klear,decs,deepscaler --tag core
bash scripts/run_agent_eff.sh probe    --donors klear,decs,deepscaler --tag core --workers 4
bash scripts/run_agent_eff.sh compose  --variant acc-clean+decs-deepscaler
bash scripts/run_agent_eff.sh train    --variant acc-clean+decs-deepscaler --seed 1234
bash scripts/run_agent_eff.sh export   --variant acc-clean+decs-deepscaler --seed 1234
AGENTIC_EVAL_DIR=~/agentic-eval-36a183a0 bash scripts/run_bfcl_eval.sh run --models base,opd,ae.joint.acc-clean+decs-deepscaler --seed 0
```

- **Scoring and probes** run as server-side chains that skip finished shards, so
  the laptop may disconnect.
- **Training** changes only the target, the seed and the checkpoint cadence: it
  saves only the final checkpoint: iteration 49 for one pass, 99 for the paper
  arms' two passes. Completion and export require the exact final iteration and
  matching target/training provenance.
- **Artifacts:**
  - donor scores: `donors/<pair>/{post,pre}` plus `evidence.json`;
  - analyses: `analysis/`;
  - probes: `probes/<tag>/`;
  - students: `/checkpoints/agent-eff/joint/<variant>/seed<seed>/{train,hf}`.

**Not covered by this pipeline:**

- collecting a cache from another behavior policy (the sequential arm);
- cross-domain math/code caches;
- the LoopTool held-out development evaluation;
- the in-house GRPO donor pair.
