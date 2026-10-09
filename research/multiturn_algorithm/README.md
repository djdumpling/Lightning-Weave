# Multi-turn accuracy: algorithm candidates and the next decisive experiments

**Priority revised after the Opus critique:** the [updated decision and controlled pilot](REVISION_AFTER_OPUS.md) supersedes the next-experiment recommendation below. Fresh recipient states with a fixed DOPD anchor now come first; action-marginal estimation and branch infrastructure are deferred. The probes and limitations below remain part of the research record.

Research decision, October 8, 2026. This study concerns **Lightning-Weave multi-turn accuracy**, combining its model-pair/composition machinery with the mentor's fresh agentic-state collection. **Sparse attention is an independent project and is not part of this algorithm or its ablations.**

## Recommendation

There are principled fixes for specific structural failure mechanisms. We have constructed and run exact counterexamples; we have **not** established which mechanism causes the real accuracy plateau, or demonstrated a new BFCL/τ accuracy gain.

The most promising differentiated hypothesis is:

> Compose what post-training changed about executable decisions without requiring the donors to agree on the private reasoning used to reach them. Calibrate consequential decisions using continuations the student can actually execute.

This is more specific than alternative-action generation plus DOPD. Its distinguishing mechanism is that **marginalizing private reasoning and composing checkpoint ratios do not commute**. The next step is a small executed development-state panel, followed by a targeted test of that mechanism—not a large new training sweep or an immediate marginal-probability estimator project.

The practical fallback is verified teacher suffixes plus ordinary DOPD/repair training, refreshing states after the student learns the suffix. That has less novelty but may solve the accuracy problem more directly.

## What was actually done in this investigation

- Read the latest mentor pipeline and the current/historical Lightning-Weave objectives, extending the [earlier audit](../followup/README.md).
- Downloaded and analyzed the existing 25,600-response projection weights and 3,200 prompts from Modal `alex-dev-2`. These are historical efficiency-donor samples, useful for group/support structure; they are **not** new accuracy-donor results. Inputs have hashes in [cache_structure.json](artifacts/cache_structure.json).
- Built exact finite probability/MDP probes for action composition, constrained targets, continuation mismatch, information acquisition, and invalid reward/projection assumptions. These are scientific counterexamples, not learned language-model experiments.
- Reviewed close primary literature, including newer work that makes generic branch-and-distill novelty weak.
- No new GPU generation, GPU training, or official benchmark evaluation was launched. No production objective was replaced. The branch protocol below still needs a state-restoration adapter and a fresh collection manifest; the historical focused-response collector does not execute alternative branches.

### A real cache result that changes the plan

| Historical eight-response cache | All states | After-tool states |
| --- | ---: | ---: |
| States | 3,200 | 397 |
| Responses | 25,600 | 3,176 |
| States with eight distinct visible responses | 339 | 262 |
| Samples in singleton response groups | 16.11% | **78.34%** |
| Samples whose projected weight is exactly 1 | 16.12% | **78.34%** |

The previous projection preserved each empirical visible-response marginal. On singleton groups it has no ability to reweight reasoning. In addition, preserving the old action marginal cannot directly correct a bad action preference. These are limitations of that target, not proof that all these decisions were incorrect.

Canonicalizing JSON in pure-tool responses barely changes the after-tool result: 262 all-distinct states become 261. Most of those after-tool responses contain public text; broad paraphrase grouping would require a justified equivalence relation. This cannot be solved merely by sorting tool argument keys.

## 1. Primary algorithm: compose at the decision level, then constrain its outcome effect

Let a complete response be `y=(z,a)`, with private reasoning `z` and the externally consumed action/message bundle `a`. An action includes ordered calls, arguments, public prose and termination. Let `b` be the frozen student reference and let each donor pair be `(post_j, pre_j)`.

### The mechanism

For an idealized sequence tilt,

```
v(z,a|h) ∝ b(z,a|h) exp(Σ_j w_j log[p_post,j(z,a|h)/p_pre,j(z,a|h)] / α).
```

Define each donor's action-level innovation by marginalizing its private reasoning:

```
Δ_j^A(h,a) = log Σ_z p_post,j(z,a|h) − log Σ_z p_pre,j(z,a|h).
```

The marginal of the response tilt is

```
v_A(a|h) ∝ b_A(a|h) exp(Σ_j w_j Δ_j^A(h,a)/α) × C(h,a),
```

where `C(h,a)` is a partition function over the donors' conditional reasoning preferences. Even when all donors favor the same action, incompatible reasoning preferences can make `C` small and suppress it.

The current production target is token-local, not this globally normalized response tilt. Nevertheless, the companion probe also constructs an **exact autoregressive token-local witness** with only three first-step tokens and two second-step tokens; all support is represented. Donors agree on the good action, but compositional conflict in an earlier private token reduces its probability. Thus this mechanism can occur without truncation or a loss-sign bug. It is still a synthetic possibility, not a diagnosis of our trained model.

### Proposed target

There are two stages. In the practical pilot, use an explicitly defined candidate action prior `u_A` (for example, the frozen student’s empirical action frequencies on the fixed supported candidate set) and executed outcomes. The differentiated, experiment-B-gated extension replaces that prior with action-marginal delta composition:


```
u_A(a|h) ∝ b_A(a|h) exp(Σ_j w_j Δ_j^A(h,a)/α).
```

At selected consequential states, use executed student continuations to estimate `Q_student(h,a)` and calibrate `u_A`, for example through a prespecified KL-constrained outcome update. With an exact value, one simple target is

```
Q_A(a|h) ∝ u_A(a|h) exp(η Q_student(h,a)).
```

Outcome calibration itself is established prior art; finite noisy estimates are not an improvement guarantee. Use conservative, prespecified uncertainty handling and keep weak-evidence states close to their prior. Do not collapse to the maximum of two noisy branch outcomes.

Then fit a joint response target with exactly that prescribed action marginal:

```
q*(z,a|h) = v(z,a|h) × Q_A(a|h) / V_A(a|h)
          = Q_A(a|h) v(z|h,a),
V_A(a|h) = Σ_z v(z,a|h).
```

The `−log V_A(a)` correction is essential. Simply adding an action score to response log weights leaves the original reasoning partition in control. This constrained KL projection is standard mathematics; its role here is to stop conditional donor preferences from overturning the intended action target.

On represented candidates, train with an explicitly versioned sequence target/loss, or an action-loss plus conditional-response-loss whose normalization is specified. The current token Top16 tilted loss is not automatically this objective. Fractional token masks do not implement it. Keep ordinary DOPD on broad fresh states as a separate, matched auxiliary component if used.

### What is new relative to our old projection?

The old target fixed `Q_A` to the student's empirical decisions, including mistakes. The proposed target allows action mass to improve and uses the projection only to prevent conditional reasoning preferences from undoing that improvement. A further, more ambitious variant composes each donor's **action marginal before multiplying ratios**.

The probes also include a complementary-constraints example. Each donor and their arithmetic mixture chooses the action satisfying both constraints with probability 0.45; token-local composition gives 0.144437 because of conflicting rationales, while action-marginal composition gives 0.81. This identifies the desired benefit over a cheap mixture: combining decision constraints while allowing alternative valid reasoning paths. It remains a constructed example.

These are distinct claims:

1. Outcome-constrained response training may be useful without estimating full donor action marginals.
2. Marginalize-before-compose specifically tests the proposed capability-composition failure.
3. If an outcome-only target performs just as well, we have not shown that the pair or composition is necessary.

### The crucial feasibility limits

**Same call does not always mean same future state.** The mentor's public-history pipeline omits earlier private thoughts. Lightning-Weave retains some reasoning within a user turn. If that reasoning appears in the next assistant prompt, changing it can change continuation success despite identical calls. Restrict the first quotient experiment to verified memory-erased or terminal boundaries. Otherwise include retained reasoning in the effective action identity, or define action value averaged under an explicitly fixed conditional reasoning policy and reevaluate it when that policy changes. Response-specific continuations alone do not make a shared action value valid. A user-turn boundary is usable only after verifying that the actual renderer discards the old private reasoning. A history reset is a separately labeled protocol change.

**Full action marginals are expensive.** Begin with native pre/post generation counts for common action events and a clearly labeled finite candidate-set experiment. For rare exact actions, scoring one canonical JSON serialization is not the full semantic action probability: all equivalent serializations must also be marginalized or the approximation labeled. Only pursue a Monte Carlo full-marginal estimator if the mechanism is observed. An action-hinted rationale proposal may help, but its likelihood must be recorded and all donor likelihoods evaluated on the original unhinted prefix. Inspect denominator uncertainty, effective sample size, maximum importance weight, independent-proposal agreement and stability with increasing samples. High ESS alone does not prove coverage. Details and the correct `b/μ` proposal correction are in [hierarchical_target_notes.md](hierarchical_target_notes.md).

**Singleton groups remove the conditional method.** With one response per action, `q*=Q_A` and the conditional donor term cancels completely. Deliberately collect multiple genuinely equivalent realizations before testing a claim about learning donor reasoning within an action.

**Compare cheap alternatives.** A best single pair or a whole-response mixture of donors can avoid product conflicts without semantic marginalization. To justify composition, demonstrate complementary decision constraints: cases where combining skill increments beats each individual pair and their mixture. Merely repairing a bad product is insufficient evidence for a complex new method.

## 2. The multi-turn insight: action quality depends on the continuation policy

There are two opposite problems:

- A weak donor can discourage an action that the stronger student could complete.
- A strong donor can encourage an action whose required follow-up the student cannot perform.

The action-by-continuation-policy experiment distinguishes them. A donor's local likelihood or ratio is not a direct measurement of either student's terminal success.

Our exact two-step counterexample has an exactly KL-optimal donor and complete probabilities/support:

| Method | Task return |
| --- | ---: |
| Recipient baseline | 0.595000 |
| Local donor-delta transplant | 0.561827 |
| Sequentially corrected transplant | 0.748344 |

The same improved final-step policy appears in both transplant arms; the difference is whether the earlier decision accounts for recipient continuation competence.

There is a concrete experimental modification to the local target:

```
q0(a|h) ∝ b(a|h) exp(D(h,a)/α)
q1(a|h) ∝ b(a|h) exp((D(h,a) + E[F(h')|h,a])/α).
```

A residual critic can use the identity

```
E_π[D − α log(π/b)] = α log Z0(h) − α KL(π || q0).
```

This connects the current composed delta with future normalizer/KL returns, and may avoid requiring dense external rewards. Across stochastic tool/user transitions, use an expectation of the successor value; a global exponential tilt over sampled outcomes improperly rewards environmental luck.

**This is an optional research branch, not the default next training change.** The correction is classical soft policy iteration. It optimizes a specified ratio surrogate, whose relationship to accuracy must be established. In our 1,620-case synthetic sweep, it improved the surrogate but lowered raw task reward relative to local transfer in 470 settings. Another exact example rewards remaining in a high-normalizer loop. State-dependent reward offsets also change sequential preferences while leaving local normalized targets unchanged. See [continuation_notes.md](continuation_notes.md).

Only promote the critic if its predicted corrections improve held-out action ranking beyond local deltas after accounting for length, repetition, support mass and realized KL. A simpler executed-outcome correction is the initial comparator.

## 3. Information acquisition: teach the suffix before rejecting the prerequisite

A lookup, clarification, or confirmation often has no immediate payoff. More importantly, it can have low current-student value because the student cannot use the resulting observation yet.

In a two-hidden-world toy, the current policy gives READ probability 10%. Outcome calibration reduces it to 1.48% when the student succeeds after reading only 30% of the time. After that suffix skill improves to 90%, the same calibration raises READ to 85.85%. Thus blindly selecting actions using the current student's branches can eliminate prerequisites to skills we want to teach.

A concrete repair is:

1. Identify a READ/ASK decision with poor student continuation but good short teacher-assisted continuation.
2. Train the verified missing suffix skill on information the student will actually observe.
3. Recollect student trajectories and reevaluate the earlier READ/ASK action.
4. Move supervision earlier only when the student can use the resulting information.

This is a backward skill curriculum, with established antecedents. Its importance here is practical: the mentor currently resamples individual responses without executing them, so neither the missing prerequisite nor its dependent skill is measured.

There is a second information-set problem. A branch at one known hidden database state can label a lucky guessed WRITE as correct. That does not mean the student could know the required ID or choice. For a small controlled diagnostic, create valid training-world pairs with the **same public history** and different still-hidden facts. Before retrieval, the student must use one shared policy across those worlds; after observing the distinguishing result, it may branch. Only vary genuinely unrevealed facts and preserve all past tool outputs and user statements. Replaying future observations after changing earlier actions is invalid.

The [information-state probe](information_state_notes.md) also tests evidence-sensitive pre/post action odds across valid observation changes. That is a lower-priority algorithm variant: subtracting invariant bias can help, but can also destroy a useful prior. Information gain alone, semantic counterfactual sensitivity, and hindsight self-distillation already have close prior art.

## The experiments to run, in order

### Experiment A: small executed decision panel

Collect fresh development trajectories with an available frozen student; mentor scores/trajectories are unnecessary. Use 24 states across several independent development tasks, stratified across missing information, after-tool interpretation, WRITE/confirmation, recovery and STOP. Keep held-out official tasks outside this collection.

At each state, execute up to three unique responses: original student response, a fresh student retry, and a donor proposal. Use two student-continuation seeds per response: **at most 144 suffixes**. On eight informative states, repeat the same candidates under donor continuation: **up to 48 more suffixes**. A short teacher-assisted bridge can replace or supplement that latter comparison, but must be labeled separately from full donor continuation. Expand repeats only where uncertainty changes the next research decision.

This is a discovery panel, not a test of a three-percentage-point benchmark improvement. Two repeats cannot reliably rank every individual state. Cluster uncertainty by original task, retain paired branch seeds, and confirm promising mechanisms on fresh states.

Before executing alternatives, restoration must reproduce the database, pending simulator state, public messages, rendered student context and applicable randomness. Preserve native policy/history rules and use equal remaining budgets. Record censored and infrastructure outcomes separately. Complete continuations are needed on a subset to audit any short-horizon screen.

Score all these exact candidate responses with teacher likelihood, each individual delta, the composed delta, and any proposed corrected target. This reuses the same expensive outcome labels for many cheap comparisons.

| Observation | What it says | Next action |
| --- | --- | --- |
| Student retry is as good as teacher proposal | Exploration may be enough | Compare outcome-weighted self-training before adding more teachers |
| A better branch exists but delta ranks it below the original | Target alignment problem | Test action/response target corrections |
| Delta ranks useful alternatives correctly, but they are absent from ordinary samples/support | Coverage problem | Improve candidate/state coverage before changing algebra |
| Action works only with teacher continuation | Missing student suffix skill | Teach the suffix, refresh states, then reevaluate |
| Student can finish a branch the donor avoids | Recipient continuation mismatch | Test outcome correction; optionally evaluate residual critic |
| All attempted branches fail | No identified useful label | Distinguish irrecoverable state, bad proposals, insufficient budget and missing prerequisite; do not call the original action bad from this alone |

Include mentor-pair decomposition in this scoring pass. For shared-post pairs,

```
(Agentic − Base) − (Agentic − Instruct) = Instruct − Base.
```

This separates a broad interface/reasoning increment from the extra agentic increment. Two ratios with the same post model are not independent experts. Begin with same-tokenizer pairs and identical prefixes/support.

### Experiment B: test composition-versus-reasoning interference

Use 8–12 states from A where donors show relevant decision agreement but response-score disagreement. Compare a few genuinely equivalent rationale variants per action, including each donor's natural style, and measure whether the composition loses useful action mass.

Control for retained reasoning, length, malformed calls and differing public prose. Include best-single-pair and whole-response-mixture controls. Instability without a correctness consequence does not justify a new loss.

Only if this succeeds, estimate common action marginals with native generations from a same-tokenizer instruct→agentic pair. Direct counts avoid importance-density assumptions but may be inadequate for rare full-argument actions. For those rare events, try action-conditioned estimation on 12 states × two actions × 16 rationales (384 proposal samples). Increase uncertain cells to 64 and replicate on fresh states. Stop the estimator route if action rankings do not stabilize, denominator uncertainty dominates, or a cheap mixture works equally well. Candidate-set targets remain a separate, explicitly limited option.

### Experiment C: one matched-data training pilot

Use the same fresh states, candidates, branch outcomes, support expansion, student initialization, optimizer exposure and evaluation protocol in every arm:

1. Existing DOPD loss on the common candidate data (not necessarily the original on-policy collection recipe).
2. Outcome-weighted candidate training without donor deltas.
3. Flat outcome-plus-delta response target.
4. Hierarchical target with prescribed improved action marginals and conditional delta supervision.

Specify a common outcome estimate `R(a)`, calibration strength `η`, response prior `v`, and candidate set. Arm 3 uses `q_flat(y) ∝ v(y) exp(η R(a))`, whose action marginal is proportional to `V_A(a) exp(η R(a))`. Arm 4 prescribes `Q_A(a) ∝ u_A(a) exp(η R(a))` and uses `q_hier(y)=v(y)Q_A(a)/V_A(a)`. This explicitly tests replacing the donor-induced action prior `V_A` with `u_A`; their realized action marginals are intentionally different. First fix `u_A` independently of outcomes and donors. If B supports marginalize-before-compose, test its action prior as a separately identified extension. Report the realized marginals and do not attribute a changed action prior to the projection theorem alone.

First screen one training seed with preregistered evaluation; use independent seeds and fresh task-level evaluation for confirmation. Measure mean autonomous task success, policy violations, early STOP, lookup/clarification usefulness and retention of already successful tasks. Match generation/training budget and report actual KL/exposure rather than only steps. A 50-task repeated τ result cannot by itself establish a robust two-point gain across the task population.

For a paper, add genuine teacher-only OPD/SPOT-style outcome calibration, best single pair, whole-response donor mixture, repair-only SFT and the relevant backward-curriculum/teacher-bridge baseline. The first four-arm pilot is a mechanism screen, not a complete benchmark table.

## Screened alternatives

| Idea | Decision and reason |
| --- | --- |
| Balanced after-tool/READ/STOP states | High-priority control; substantial observed coverage mismatch; not novel by itself |
| Refresh states after updates | High-priority when visitation changes; fresh responses on old prefixes do not do this |
| Equal-turn instead of token-average loss | Useful controlled ablation; long thoughts currently contribute more terms |
| Targeted successful-response support expansion | Promote if useful action is outside ordinary candidate support; keep cache contracts explicit |
| Larger Top16 everywhere | Defer until support audit; does not fix wrong ranking or wrong states |
| Longer rollouts | Audit completed-action rate and censoring first; source length does not guarantee informative response targets |
| More responses at the same failure prefix | Control for exploration; cannot establish recovery without execution |
| First explicit tool-error focusing | Incomplete; silent bad writes, missed questions and early stops remain |
| Whole-trajectory success weight on every turn | Weak credit; use branch comparisons for consequential decisions |
| Add scalar reward to every vocabulary logit | Reject as action credit: common constants cancel in normalized tilts; Top16-only constants mostly manipulate OTHER |
| Role- or phase-specific donor gates | Conditional on measured complementary action ranking; avoid tuning another mixture blindly |
| Subtract generic style/evidence bias | Lower priority; can remove useful priors and overlaps prior work |
| Action-level marginalize-before-compose | Primary differentiated hypothesis if B validates the mechanism and estimation is tractable |
| Constrained action target with conditional DOPD | Primary practical prototype; exact target property, but classical projection mathematics |
| Normalizer-return critic | Optional; elegant surrogate correction with demonstrated accuracy/loop counterexamples |
| Short teacher bridges/backward suffix curriculum | Best fallback when missing follow-up competence is the bottleneck |
| Matched hidden-world READ/ASK supervision | Strong controlled stress test; requires valid world construction and shared public information |
| Outcome-calibrated single teacher | Mandatory close baseline; may suffice without pairs |
| Whole-response donor mixture | Cheap challenge to action-marginal estimation; require composition to beat it |
| Global response-probability exponential tilt | Not equivalent to current token-local training; across stochastic environments it can tilt luck |
| Sparse attention in the same algorithm | Excluded: independent project, separate experiments and claims |

## Novelty judgment

**Generic alternative-action execution plus DOPD is not a strong novelty claim.** [SPOT](https://arxiv.org/html/2608.04419v1) probes teacher candidates with verifier-scored student continuations and uses KL-regularized outcome targets. [FutureBridge-OPD](https://arxiv.org/html/2608.01953v2) executes teacher interventions and evaluates subsequent student behavior. [PivotOPD](https://arxiv.org/html/2609.40285v1) already covers pivot prevention/recovery. Merely applying these ideas to τ/BFCL does not establish a new training algorithm.

The residual value correction has older mathematical ancestry in [Hunt et al.](https://proceedings.mlr.press/v97/hunt19a.html) and [Adamczyk et al.](https://arxiv.org/abs/2212.01174). The semantic/evidence direction must distinguish [OmniOPD](https://arxiv.org/abs/2606.01476), [CROP](https://arxiv.org/abs/2608.13387) and [IGSD](https://arxiv.org/abs/2609.32694). [OPRD](https://arxiv.org/abs/2609.08798) is close to verifier-supported checkpoint-shift hybrids; [TCOD](https://arxiv.org/abs/2604.24005) is relevant to backward curricula.

The narrower plausible contribution is **identifying when model-pair capability composition fails because semantically compatible decisions have incompatible internal realizations, and developing a tractable correction that preserves a prescribed action marginal and improves multi-turn tasks beyond single-pair, mixture and outcome-only controls**. The equations alone are not a novelty proof. The empirical mechanism, estimator, and controlled advantage would need to carry the contribution.

## Reproducing the completed probes

```
python3 research/multiturn_algorithm/hierarchical_target_probe.py
python3 research/multiturn_algorithm/token_tree_probe.py
python3 research/multiturn_algorithm/continuation_probe.py
python3 research/multiturn_algorithm/information_state_probe.py
/private/tmp/agentic-research-20261008-venv/bin/python research/multiturn_algorithm/cache_structure_probe.py
```

The first four use Python's standard library. The cache probe requires NumPy, PyArrow and the downloaded files; hashes and Modal provenance are recorded above. Detailed derivations and negative cases live in the companion notes, rather than being hidden behind a single favorable toy result.
