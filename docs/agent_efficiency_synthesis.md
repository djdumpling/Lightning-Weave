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
v3 is the external evaluation. Results:

- `results/agent_eff/results_0.md`: the first tracker, with the accuracy-trained student against base (E0), core
  geometry (E1), the composed primary targets (E2), and the primary matrix in full (E3);
- `results/agent_eff/results.md`: sections 1–15, from the primary matrix to the paper recipe and turn-start round;
- `results/agent_eff/results_2.md`: the same checkpoints re-evaluated under interleaved thinking;
- `results/agent_eff/results_3.md`: the decision-preserving projection round;
- `results/agent_eff/research_audit.md`: the October 2026 audit.

The registry keeps 27 arms, including the trained DECS low/high variants. Retired code can be recovered with
`git show <commit>:path/to/file.py`:

- commit `543e6d0e`: untrained exploratory arms, residualization/PCA, the reflection and value probes, and
  one-off diagnostics;
- commit `6f73aa2`: the geometry CLI (`shift_geometry.py` is now a library), the census-only donors (DLER,
  AdaptThink, DeepCoder-1.5B, Skywork-OR1-7B), the float32 precision re-score, the mixture and score-merge tools,
  and the Monte Carlo continuation study (`data_curation/mc_advantage_probes.py`), which was never run.

The decision-preserving projection round (2026-10-01/02) is shelved: every pre-registered verdict was inconclusive at
3–5% realized savings, and the costly BFCL decision (ending the first turn early) rises at the same rate per token
saved with or without the projection (`results/agent_eff/results_3.md`). Its pre-registration, pipeline and tests are
in commit `6f73aa2`.

## What a shift is, and is not

KL-regularized RL motivates the construction: a converged policy's log-ratio to
its reference is a scaled advantage of its reward. The donors here, however, are
broad post-training runs from different organizations, recipes and sometimes
tokenizer families. So a policy shift is treated as an **empirical direction**,
not a recovered reward or a causal efficiency advantage. BFCL tests whether it
transfers.

| contrast | status |
| --- | --- |
| DECS − DeepScaleR | **Quasi-matched external contrast.** Same R1-Distill-1.5B base and closely related data, but different authors, steps and reward details. This is the primary efficiency direction. |
| DECS − Klear | The analogy as first stated. Klear differs in family, scale, tokenizer, data and recipe (SFT then RL), so subtracting it does not cleanly remove "accuracy". Exploratory only. |
| L1-Max − DeepScaleR, Nemotron v1 → v2, DECS-7B − R1-Distill-7B | Length-aware policies that still differ in full training runs. Nemotron v2 also changes its algorithm and continues training. |
| Base → Thinking-2507 (agent accuracy) | A broad multi-stage post-training difference, not a single accuracy-reward update. |

**Cross-policy state mismatch.** Projection makes action IDs comparable, but it
does not make a Qwen3 tool-formatted prefix an in-distribution state for an R1
reasoning donor. `verify` therefore reports each pair's support per state type
(below). For tool-call and message-boundary states the cleanest donors would be
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
b. Directions are compared with the Fisher metric of the tilt:
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

These rules are heuristics, with one fork threshold (`FORK_MASS`, 0.02). Inspect
representative states before building claims on them.

## Donors and their evidence

[`configs/agent_eff/config.py`](../configs/agent_eff/config.py) pins every pair.

- **Core pairs:** Klear, DECS, and DeepScaleR.
- **Transfer donors** (`TRANSFER_DONORS`): DECS, DECS-7B, L1-Max, and Nemotron v1 → v2.
- **Code-round pairs:** E1-Code and E1-Math (below).

**Untrained special tokens.** Qwen3-4B-Base leaves all of its chat, tool and
think tokens untrained, and those rows dominate the legacy agent-acc target
where the model decides whether to call a tool (results_0.md, "E0: mechanism in
the cached targets").

[`audit_donor_tokens.py`](../data_curation/audit_donor_tokens.py) flags a token
when either anchor's output row duplicates another special row, exactly or
nearly (cosine ≥ 1 − 1e-4), or is near zero. The flag is positive evidence of an
untrained row; its absence does not prove the row was trained. The audit
therefore also reports, for every structural token, its row norm and its closest
special-row cosine.
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
  must be injective, so no teacher token is counted twice. DECS and DeepScaleR
  were rescored after an earlier, non-injective map.

**Mapped is not the same as usable.** Aliasing makes nearly all behavior mass
mappable at call-boundary and end-of-message states, but whole-position coverage
stays low there and is 0% at act-vs-talk (results_0.md, "Donor evidence").
`verify` reports the evidence the composer actually uses, **mapped AND trained**,
per state type, as whole-position and behavior-mass coverage together with the
support distribution.

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

The staged plan (donor scores, core geometry, then the primary matrix) has run.
The primary matrix is `primary_specs()`: seven arms at one efficiency budget
(0.0135 nats/token). The legacy and clean accuracy terms run alone and with DECS
or DECS − DeepScaleR added, and a sign-flipped contrast controls for any
perturbation at matched KL. Its results are in results.md section 1, with
per-group detail in results_0.md (E3).

**Inference-time baselines**, which any learned efficiency must beat (results.md
section 3):

- thinking disabled (`base-nothink`, `opd-nothink`);
- a 4,096-token per-step cap (`*-cap4k`);
- a "reason only as much as needed" system prompt (`*-concise`).

A forced-stop policy that keeps tool output valid is not implemented.

**Decision rule** (from `bfcl_efficiency.py`). An arm passes when all of these hold:

- the accuracy change's lower CI bound is above −1 point;
- the upper CI bound of the paired log-ratio of generated tokens is below −5% (the primary cost estimate, about ±2% from one run);
- irrelevance accuracy is non-inferior;
- the guards are non-inferior: each upper CI bound stays within its tolerated increase (`GUARD_TOLERANCES`: +0.05
  duplicate calls per entry, and +0.5 points each for the runaway and overflow rates).

AES is descriptive only.

## Code-efficiency round (2026-09-30)

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

The E1 models and their pre-anchors share DECS's `tokenizer.json`, so the same alias and projection apply. The E1
weights are CC-BY-NC-4.0.

**Near-duplicate untrained rows.** `<tool_call>` stays untrained in the R1-Distill lineage. In the 7B and 14B
checkpoints, however, the untrained special rows differ by about one bfloat16 step: their cosine is within 1e-7 of
1, while trained special rows stay below 0.9. The audit's exact-duplicate rule missed them, so it also flags
near-duplicates (cosine ≥ 1 − 1e-4). These donors then carry no shift on `<tool_call>` itself, as DECS does not.

- This does not make a donor neutral about the call. Shifts on competing text candidates still move the call's
  probability after normalization, and donor support at the call-or-reply and call-boundary states is 0–6%.
  Only removing the term at those states entirely (the `exclude_state_types` gate) preserves those decisions.
- DECS-7B and Nemotron were prepared before this fix. Their empty untrained lists let their log-ratios at
  `<tool_call>` into the DECS-7B, Nemotron and donor-sum targets. The effect is small in KL terms (act-vs-talk
  mean KL about 0.001 vs 0.0001 for DECS on one shard) but is a known asymmetry.

**Arms.** `code_round_specs()` adds one efficiency term to A at the primary KL budget (0.0135), with
`max_multiplier` 50: C, M, M + C, or D + C. Inside a sum (`equalize: "auto"`), the two shifts keep raw 1:1 weights
when their Fisher RMS sizes are within 2× (`EQUALIZE_MAX_RATIO`). Otherwise they are equalized to unit RMS. The
sizes and the rule applied are recorded in the manifest. `followup_specs()` adds M with the term removed at
`PROTOCOL_STATES` (at M's mid coefficient), and M at the low budget.

The pre-training checks, comparisons and results are in results.md sections 13–14. The full pre-registration is in
commit `6f73aa2`.

## Lightning Weave's published recipe and turn-start allocation

Lightning Weave (arXiv 2609.14708, Sec. 4.1) trains every arm at α = 2.0 with anchor weights that sum to one:
Klear-only is accuracy at weight 1.0, and the equal composition is 0.5 accuracy + 0.5 DECS (`PAPER_WEIGHTS`). The
other arms here instead keep accuracy at 1.0 and add a KL-calibrated efficiency term. `paper_specs()` registers the
accuracy-only, half-accuracy and 0.5 + 0.5 arms. They train two passes (`TRAINING_PASSES`, 400 updates, the
released code's default) on a repeated copy of the target (`repeat_target`). `turn_start_specs()` registers two arms
that keep DECS at its mid coefficient and remove it from whole responses: on turn-start prompts, or on as many
randomly chosen multi-turn prompts (`prompt_weights` files from `data_curation/turn_positions.py`). The pre-registered rules are in
`configs/agent_eff/next_comparisons.json`, applied by `evaluation/bfcl_pooled.py`. Results are in results.md
section 15 and, under interleaved thinking, results_2.md. The full pre-registration is in commit `6f73aa2`.

## Fresh-state pilot (2026-10-08)

The same recipe on another state pool: `OPD_DATA=tau2-fresh` selects the `tau2-fresh` pool in
`configs/looptool_opd/config.py` (its own data root, a 16,384-token prompt cap, and the tau2 collection its prompts
come from). The pool runs as its own Modal apps with the profile baked into their images, and its students live under
the `fresh` arm (`ae.fresh.<variant>`). `acc-legacy+decs-fixed` keeps DECS at the LoopTool cache's calibrated
coefficient rather than recalibrating it on the new states. Stages, after the collection
([modal_tau_eval.md](modal_tau_eval.md), `collect`):

```bash
OPD_DATA=tau2-fresh bash scripts/run_looptool_opd.sh cache-chain --max-wait-hours 20   # states, behavior responses, anchor scores, seal
OPD_DATA=tau2-fresh bash scripts/run_agent_eff.sh prepare --donors agent_acc,decs
OPD_DATA=tau2-fresh bash scripts/run_agent_eff.sh score   --donors decs
OPD_DATA=tau2-fresh bash scripts/run_agent_eff.sh build   --variant acc-legacy,acc-legacy+decs-fixed --seeds 1234,5678
python evaluation/bfcl_pooled.py --root RESULTS --comparisons configs/agent_eff/fresh_comparisons.json
python evaluation/tau_pooled.py --root TAU_RUN --spec configs/agent_eff/fresh_tau_comparisons.json --output tau.json
python evaluation/tau_escalation.py --root TAU_RUN --tau2-domains TAU2/data/tau2/domains \
  --spec configs/agent_eff/fresh_tau_comparisons.json --pairs ae.fresh.acc-legacy:ae.joint.acc-legacy
```

Results and caveats: results/agent_eff/results_4.md.

## BFCL evaluation

The harness (interleaved thinking, the usage proxy and its join, decoding seeds, `ae.*` tags) is described in
[modal_bfcl_eval.md](modal_bfcl_eval.md).

## Running

```bash
bash scripts/run_agent_eff.sh plan
bash scripts/run_agent_eff.sh download --donors klear,decs,deepscaler
bash scripts/run_agent_eff.sh prepare  --donors agent_acc,klear,decs,deepscaler
bash scripts/run_agent_eff.sh score    --donors decs,deepscaler --workers 8     # server-side: post -> pre -> verify
bash scripts/run_agent_eff.sh verify   --donors klear,decs,deepscaler
bash scripts/run_agent_eff.sh onboard  --donors l1max                          # server-side: download -> prepare -> score -> verify
bash scripts/run_agent_eff.sh compose  --variant acc-clean+decs-deepscaler
bash scripts/run_agent_eff.sh train    --variant acc-clean+decs-deepscaler --seed 1234
bash scripts/run_agent_eff.sh export   --variant acc-clean+decs-deepscaler --seed 1234
bash scripts/run_agent_eff.sh build    --variant paper-acc-legacy,acc-legacy+decs-protect-turn-starts --seeds 1234,5678
AGENTIC_EVAL_DIR=~/agentic-eval-36a183a0 bash scripts/run_bfcl_eval.sh run --models base,opd,ae.joint.acc-clean+decs-deepscaler --seed 0
# analysis (CPU), on a local copy of the results volume's run trees
python evaluation/bfcl_pooled.py --root RESULTS --comparisons configs/agent_eff/next_comparisons.json --output pooled.json
python evaluation/bfcl_multiturn_failures.py --root RESULTS --ground-truth DIR --output failures.json \
  --comparisons configs/agent_eff/next_comparisons.json
```

- **Scoring, onboarding, training and builds** run server-side, so the laptop may
  disconnect. Scoring skips finished shards. `build` composes each target (and
  repeats it for a multi-pass plan), then trains and exports every variant and
  seed, reusing finished stages.
- **Training** changes only the target, α, the number of passes, the seed and the
  checkpoint cadence. It saves only the final checkpoint: iteration 49 for one
  pass, 99 for the paper arms' two passes. Completion and export require the exact
  final iteration and matching target/training provenance.
- **Artifacts:**
  - donor scores: `donors/<pair>/{post,pre}` plus `evidence.json`;
  - targets: `synthetic/<variant>/`, and `synthetic/<variant>-x2/` for two passes;
  - turn positions and prompt weights: `analysis/turn_positions/`;
  - students: `/checkpoints/agent-eff/joint/<variant>/seed<seed>/{train,hf}`.

**Not covered by this pipeline:**

- collecting a cache from another behavior policy (the sequential arm; the fresh-state pool changes the states, not the behavior policy);
- cross-domain math/code caches;
- the LoopTool held-out development evaluation;
- the in-house GRPO donor pair.
