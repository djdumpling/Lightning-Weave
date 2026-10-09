# Revised priority after the Opus critique

October 8, 2026. This supersedes the next-experiment recommendation in README.md; the mathematical probes remain valid. No new GPU experiment was launched during this review. Sparse attention remains independent.

## Decision

Pause action-marginal estimation, hierarchical-target training, and the mid-conversation branch adapter. Run a small **fresh-state, fixed-anchor DOPD pilot** first. Use the mentor's complete-trajectory/prefix collection architecture with Lightning-Weave's existing targets. Include accuracy-only as the primary arm and accuracy+DECS as its paired control.

The previous recommendation over-weighted constructed counterexamples and under-weighted the actual failed projection experiment. Prescribing a better marginal on a finite cache does not solve generalization to unseen agent states. The new projection variant would need new evidence before it deserves its considerable estimator and infrastructure cost.

## What Opus gets right

- The q = Q_A times conditional-response construction extends the old decision-preserving projection; it is not a fundamentally new mathematical mechanism.
- The old projection demonstrably suppressed one requested training-state shift without establishing a held-out multi-turn benefit. [results_3.md](../../results/agent_eff/results_3.md), lines 189–216 and 239–260.
- The composition toy needs action-dependent incompatibility. Its existence establishes a possible failure, not its prevalence in our runs. The existing donor-sum results make it less compelling as the next explanation of the DECS tradeoff.
- Estimating full action probabilities through long private reasoning is a poor first investment, especially with a Base checkpoint. Format failures do not make the probabilities undefined, but can make native estimation impractically inefficient. Importance sampling has its own severe variance/support problems.
- Two continuations per branch can have a maximum standard error of 0.5 for an independent difference of Bernoulli success rates. That supports only pooled exploratory screening, not confident per-state classification.
- The proposed current-student outcome calibration creates the specific prerequisite-suppression hazard. Plain DOPD does not use those outcomes, so that hazard should not be presented as an existing DOPD defect.

## What the critique overstates

**The historical line is not an intrinsic frontier.** A failed refresh only rejects that particular intervention at its measured resolution. It cannot establish that every action-level objective will fail. Coverage mismatch and horizon amplification are compatible: inadequate coverage can cause precisely the small errors that compound. The source's approximately 0.4% per-step slip is a back-calculation consistent with aggregate losses, not a measured causal hazard. The original 0.33 residual SD is from an older fit; the corrected 30-arm fit reports 0.41. Neither is a sampling standard error or a universal decision threshold. The existing [research audit](../../results/agent_eff/research_audit.md), line 61, already warns against a fundamental-frontier interpretation.

**The projection evidence is narrower than “decisions were preserved.”** The +0.21 versus +0.04 values are projections onto one requested shift direction. Total decision distances were similar, and the report explicitly does not establish that the suppressed direction caused the accuracy loss. This weakens the priority of another projection experiment without proving a no-go theorem.

**Log-ratio telescoping does not telescope benchmark scores.** On matched prefixes/tokens,

```
(Agentic − Base) − (Agentic − Instruct) = Instruct − Base.
```

This supplies a useful score-level decomposition and reference-choice comparison. Two independently trained benchmark scores do not algebraically isolate “chat polish” versus agentic skill: target normalization, optimization and task success are nonlinear. For example, with recipient good-action mass .5, Agentic .8, Instruct .6 and Base .2, unit-temperature exact transfer gives Agentic/Base .94118 and Agentic/Instruct .72727 (gap .21390), whereas Instruct/Base gives .85714 (gain .35714). Use the identity on actual deltas; treat final arm-score differences as an intervention on reference choice.

**Hunt's correction sign has assumptions.** [Hunt et al.](https://proceedings.mlr.press/v97/hunt19a.html) derives a nonnegative divergence penalty for compatible optimal maximum-entropy policies and convex reward composition. Under an absorbing-terminal convention this can increase STOP odds relative to an overoptimistic uncorrected composition. Our arbitrary post/pre pairs, recipient anchor and mixture strengths need not meet those conditions. The general residual critic in our probes can have either sign and can favor loops. The reasoning-abstraction and continuation examples are related issues of composing and eliminating variables, but are not one established diagnosed error in these checkpoints.

## The minimal useful pilot

The central control is to separate **state collector**, **response sampler/target anchor**, and **trainable student**.

In the actual [tilted loss](../../slime/rollout/offline_direct_opd.py), the target uses cached behavior probabilities:

```
q_h(a) ∝ b(a|h) exp(D(h,a)/alpha).
```

If newly sampled behavior is the trained checkpoint, replacing b by that checkpoint reapplies the delta. In an exact-fit example with b0=(.5,.5), D=(log 2,0), alpha=1, the first target is (2/3,1/3); collecting from it and rebasing the same delta yields (4/5,1/5), even with no new states. Performance changes could therefore reflect stronger transfer rather than coverage.

Use this first design:

| Component | Fixed choice |
| --- | --- |
| State collection | Fresh complete training/development episodes from a prespecified trajectory-level mixture of the existing accuracy-only and accuracy+DECS students. A single stopping-damaged collector may never reach needed later states. |
| Prefix pool | One shared pool for both arms, including post-tool, clarification, confirmation and pre-STOP decisions; retain source task/episode/collector identity. Do not restrict it to WRITE/errors. |
| Current response generation | Original clean Qwen3-4B b0 on those prefixes, preserving the original response/support collection contract. |
| Target anchor/support | Original b0 probabilities and Top16 support; this avoids conflating new visitation with anchor changes. |
| Initialization | Both new students start from b0. |
| Arm A, primary | Existing accuracy-only target. |
| Arm B, paired control | Existing accuracy+DECS-mid target. |
| Training controls | Same donors, coefficients, alpha, tokenizer, loss and training context/response caps; matched trainable-token exposure and recorded optimizer budget. |
| Evaluation | Frozen native BFCL and τ2 protocols; paired task/seed comparisons, autonomous mean success, tokens, premature STOP, and existing-success retention. |

Use the relevant deployed history contract. The mentor's public-history collector drops previous private reasoning, whereas LW's BFCL path preserves it within a user turn. Its focused responses also allow 32,768 tokens versus the original LW 2,048. Do not silently introduce either change and call the result a state-only test. Record excluded long prefixes and length-censored responses; they affect whether the refreshed cache actually covers missing decisions.

The proposed general update is:

```
h ~ visitation of the current recipient (or the fixed collection mixture)
q_h(a) ∝ b0(a|h) exp(sum_j w_j delta_j(h,a)/alpha)
```

This is **refreshing visitation while keeping the transfer anchor fixed**. It combines the mentor's useful collection mechanism with Lightning-Weave's existing loss, without adding outcome branches or a new critic. Generic visitation refresh is established; do not claim it as a novel algorithm merely because it helps here.

For the first pilot, hold coefficients fixed and report target KL and realized student drift on a shared audit set. Forcing equal KL by retuning coefficients changes the intervention. If KL or realized token savings differ enough to explain the result, add a narrowly targeted strength-matched control before claiming a better frontier. A single pilot seed can screen; confirm any promising claim across independent training seeds.

Fresh τ2 data versus historical LoopTool data also changes domain, tool schemas and phase mix. Evaluation on both τ2 and BFCL assesses generalization, but does not isolate occupancy from domain transfer. Call this a practical data-source/visitation pilot. If it wins and causal attribution matters, add a same-domain original-policy versus trained-policy state-pool comparison afterward.

## Reading the outcome

| Observation | Supported conclusion |
| --- | --- |
| Fresh accuracy-only improves | The new state/data pool helps accuracy transfer under this recipe. |
| Both arms improve similarly; DECS's incremental loss persists at comparable savings | Baseline capability improved; the efficiency coupling remains. |
| Fresh DECS loses less relative to fresh accuracy-only at comparable savings | Evidence that the old data distribution contributed to the incremental DECS damage. |
| τ2 improves but BFCL does not | Benefit may be domain-specific; cross-domain multi-turn improvement is unproven. |
| Neither improves | This refresh recipe failed at this budget/resolution; no intrinsic-frontier conclusion. |

Do not select success solely by a +1 residual against the old regression. A practical minimum effect can be prespecified, but uncertainty and the paired fresh accuracy-only control determine what the result supports.

## The hypothesis worth retaining after this pilot

A donor can provide useful supervision for prerequisites before the student can exploit their outcomes. That is a plausible reason to keep dense donor guidance on READ/ASK states instead of pruning every branch whose current-student continuation fails.

It is not automatically pair-specific: teacher-only OPD also supplies competent-policy guidance. A later focused comparison should use the same fresh states for post-only OPD, pair-DOPD, and current-student outcome learning. The extra question is whether subtracting the pre checkpoint transfers useful increments without overwriting stronger recipient behavior. Arbitrary Base→Agentic deltas are relative likelihood changes, not certified action values.

Action-marginal composition returns to the queue only if we observe action-dependent composition harm on the new relevant states that simpler single-pair/mixture controls cannot explain. The earlier probes remain useful falsification examples, not the current reason to build a new training system.
