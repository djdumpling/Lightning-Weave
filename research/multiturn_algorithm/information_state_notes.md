# Information sets, evidence-sensitive deltas, and a screened alternative portfolio

These proposals concern Lightning-Weave multi-turn accuracy and the mentor's agentic Direct OPD collection. They do not change, combine with, or make claims about the independent sparse-attention project. Nothing here requires the mentor's latest results. The CPU results are constructed mathematical witnesses, not new LLM/benchmark results.

## The most consequential extra failure mode

An action can be correct for a teacher with hindsight and impossible for a student with the present public history. If two possible worlds share the student's public prefix but require different final writes, copying each world's correct write teaches a mixture of guesses. The learnable common strategy is to acquire the distinguishing information and then respond to it. Merely filtering tool errors or regenerating a response at a WRITE prefix need not generate that strategy.

There is a second wrinkle: scoring each candidate by *the current student's* eventual success can punish useful information acquisition if the current student does not yet know how to use the answer. This means the proposed executed-branch diagnostic should measure both one-action intervention and a short assisted suffix. A low current-student value is not proof an action is intrinsically bad; it can identify an unlearned prerequisite/suffix pair.

The included [probe](information_state_probe.py) enumerates two hidden worlds and confirms:

| Constructed intervention | Exact result |
| --- | ---: |
| Imitate a clairvoyant teacher's final write from identical public history | 50% success |
| READ, then student uses the observed answer correctly with probability 0.8 | 80% conditional success |
| Outcome tilt from behavior `(0.45, 0.45, 0.10)` and action values `(0.5, 0.5, 0.8)`, temperature 0.1 | READ mass 69.06%; success 70.72%, from behavior 53% |
| Same tilt when observation-use skill is only 0.3 | READ mass falls to 1.48% |
| Same tilt after that suffix skill reaches 0.9 | READ mass becomes 85.85% |

This demonstrates a possible mechanism, not its incidence in either repository. Improving the suffix before optimizing the earlier decision is a sensible fix, but backward temporal curricula already have close precedent in [TCOD](https://arxiv.org/abs/2604.24005). Likewise, information-gathering imitation has a long history, including [Choudhury et al.](https://arxiv.org/abs/1705.07834). The research opportunity would have to be more specific than discovering either principle.

## A pair-specific proposal: transfer acquired evidence sensitivity

Suppose an action-log-odds capability delta contains two effects: a generic shift toward a preferred action/tool/format, and a genuine response to new evidence. The former can overwhelm the latter, even though the latter is exactly the capability needed after tool feedback.

For donor pair `j`, define the action delta at a public history `h,o`:

```text
Delta_j(h,o,a) = log P_post,j(a | h,o) - log P_pre,j(a | h,o)
```

Here `P(a|h,o)` must denote a probability on the same executable action space; it is not obtained simply by summing arbitrary token log-probabilities. Action marginal estimation and future-visible reasoning are treated separately in the hierarchy notes.

For two valid observations `o1,o2` and two candidate actions `a1,a2`, the diagnostic contrast is:

```text
I_j = [Delta_j(h,o1,a1) - Delta_j(h,o1,a2)]
    - [Delta_j(h,o2,a1) - Delta_j(h,o2,a2)]
```

This contrast measures how post-training changed the *relative* response to this factual difference. It cancels a context-invariant action bias. It is a diagnostic of local sensitivity, not automatically a causal estimate of general competence or a reward.

A candidate transfer signal on a small, validated set of matched observation worlds is:

```text
Delta_innovation,j(h,o,a)
  = Delta_j(h,o,a) - E_o'[Delta_j(h,o',a)]

q_A(a | h,o) proportional to
  b_A(a | h,o) * exp(sum_j w_j Delta_innovation,j(h,o,a) / alpha)
```

The expectation uses a specified distribution over *valid* alternative worlds consistent with the earlier public history. Initially this should be a controlled two-world experiment. It is not appropriate to invent a posterior for natural τ tasks and call it ground truth. Existing token Direct OPD is the control; this is a new, explicitly versioned action-target experiment.

The synthetic evidence witness uses action log-odds deltas `3 + 2` and `3 - 2` in two equally likely contexts where the correct action reverses. The common bias makes the raw target choose action 1 too often: 63.11% mean correctness. Centering leaves `+2,-2` and yields 88.08%. The same script supplies a counterexample: with a legitimate 90:10 world prior and 60%-accurate observations, discarding the prior lowers Bayesian MAP accuracy from 90% to 60%. Therefore **do not blindly remove all world-invariant delta components**. Use the contrast to diagnose, and require matched held-out outcomes to justify removal.

A stronger deployment candidate, if the simple residual works, retains the calibrated common component and separately controls the acquired observation response. A weaker result in which sensitivity improves but task outcome does not would falsify its practical benefit.

Nearest neighbors matter. [CROP](https://arxiv.org/abs/2608.13387) already uses semantic counterfactual versus harmless-paraphrase sensitivity to select OPD positions. The mentor's observation-increment study already examines visible-minus-hidden observation deltas. A narrower candidate contribution here is **signed pre/post action sensitivity measured on valid executable alternative worlds**, coupled to information acquisition and held-out branch success. This is not a broad novelty claim for counterfactual OPD, observation conditioning, or token selection. It is also independent of an attention mask: ordinary full-context models can implement the experiment.

## The smallest informative experimental design

Create 12–16 development task templates with controllable facts, separate from official held-out evaluation. Suitable facts include whether permission has been granted, whether a requested item is eligible, which of two reservations the user means, and whether a required field is still missing. Each template has two consistent worlds. Before the distinguishing READ/ASK, the public histories are identical. After the observation, the correct action changes. Include matched templates where new information should *not* change the action.

At the pre-information state, compare WRITE/STOP, READ/ASK, and the best teacher-proposed alternative. Execute the same candidate list in both worlds with two student-continuation seeds initially. At the post-information state, compare how student, teacher, raw donor delta, and evidence-sensitive delta rank the feasible actions. Separately score harmless paraphrases and harmless JSON formatting variations as negative controls. Freeze the student during diagnosis.

Use terminal success and policy compliance as outcomes. Also report acquisition frequency, correct use of observed facts, premature completion, action ranking, and the difference between one-action and short-suffix assistance. Candidate count, suffix budget, scorer calls, and training exposure must be matched. Two suffix samples estimate a noisy value; they are discovery evidence, not a reliable ranking of close action choices. Spend additional samples on genuinely uncertain comparisons and keep a separate held-out decision panel.

Valid twins require all of the following:

- Same initial public prefix, policy, tool schema, and task family; changed hidden facts do not contradict any earlier receipt.
- Simulator state, database, task goal, permitted actions, and verifier agree with the changed world. Altering a tool-output string while keeping contradictory hidden state or labels is invalid.
- No hindsight reservation ID, permission, or hidden task fact is inserted into the student's earlier prompt. Distill the query that obtains it.
- Regenerate every observation after a changed action. Never reuse a cached suffix from the old world.
- Separate development templates from confirmation templates and vary entities, surface wording, and tool schemas. Otherwise the student can memorize the paired answer pattern.
- In τ, clone/replay the user simulator state as well as database state. A changed public assistant utterance can alter later user behavior. In Lightning-Weave, retained private reasoning can alter later assistant behavior too; tool-call equality alone is not full state equality.

A one-action gain but no additional suffix gain supports prevention/selection work. A large assisted-suffix gain supports suffix teaching or a temporal curriculum. No useful teacher or retry branch means better targets are not yet available; investigate teacher competence, information availability, and branch validity before redesigning the loss.

## Alternative portfolio and kill criteria

The following are deliberately separate alternatives. They should not all be bolted onto a single method.

| Candidate | Concrete change or diagnostic | Evidence that promotes it | Evidence that demotes it | Relation to prior work |
| --- | --- | --- | --- | --- |
| Action marginalization before delta composition | Estimate each pre/post model's distribution over executable actions, then take ratios and combine, rather than combine literal rationale probabilities first | Donors agree on useful action but raw composition loses mass due to incompatible reasoning; semantic target restores ranking | Action rankings are stable across reasoning variants; marginal estimates have unusably low effective sample size | Semantic OPD exists, e.g. [OmniOPD](https://arxiv.org/abs/2606.01476); exact delta-composition versus marginalization mechanism is narrower |
| Hierarchical outcome target with conditional donor signal | Set verified action mass separately; use composed signal inside an action's genuinely equivalent response class | Increasing donor strength changes reasoning but cannot reverse verified action choice; training beats outcome-only control | Dense conditional signal adds no benefit to matched repair/action training | Generic outcome-calibrated branches are already [SPOT](https://arxiv.org/abs/2608.04419); preserve the narrower algebraic guarantee |
| Information acquisition plus suffix learning | Teach observation use at downstream states, then revisit READ/ASK decisions | Teacher-assisted suffix much better than one-action intervention; branch Q for READ rises after suffix microtraining | Student already uses evidence well; all errors are earlier irreversible writes | Closely related to TCOD and intervention methods; practical fix rather than stand-alone novelty |
| Evidence-sensitive capability innovation | Separate post-training's response to valid factual differences from common action/style preference | Counterfactual action flips become correct without harming invariant-control tasks or useful priors | Success falls on prior-dependent tasks; improvements match arbitrary paraphrase controls | CROP and existing observation-increment studies narrow novelty |
| Fresh-state refresh / reachability curriculum | After a short update, collect new student states and compare with more samples at old prefixes | Cached states become atypical; fresh collection fixes new failure types at equal cost | Same failures and decision distribution persist; refreshed data no better | Online imitation/state aggregation and [ReOPD](https://arxiv.org/abs/2607.04763) cover the general idea |
| Stage-specific donor attribution | Separate Base→Instruct from Instruct→Agentic; measure executed action ranking by READ/ASK/WRITE/STOP phase | Agentic increment helps feedback/writes while interface increment dominates text; phase effects stable on held-out states | Same endpoint dominates everywhere; phase routing unstable | Token-level multi-teacher routing exists in [MOPD-Router](https://arxiv.org/abs/2609.30837) |
| Future compatibility correction | Learn a small continuation correction to locally composed action preference | Single-step donors useful but combined policies reach successor states with systematic conflict | Failures already explained by support, semantics, or outcome noise | Entropic divergence correction is classical: [Hunt et al. 2019](https://proceedings.mlr.press/v97/hunt19a.html); no new theorem claim |
| Retained-reasoning intervention | At identical public state, compare original versus regenerated private working context, then fresh continuation | Frozen private mistaken plan predicts failure despite valid public evidence; replacing it changes continuation | Private reasoning is absent or replacement has no effect | Requires a separately labeled protocol; not an attention experiment and not permission to silently alter established evaluation |
| Precision on STOP and irreversible actions | Include STOP as explicit candidate and branch before/after suspect write; audit publicly satisfied obligations | Premature completion or irreversible actions dominate avoidable failures | Branch teacher cannot improve or errors occur in retrieval/interpretation | An important task-specific control, unlikely novel by itself |
| Common-support and token-exposure correction | Audit successful action's first divergent token in Top16; equalize turn-level effective gradient exposure | Useful actions absent from support or long reasoning dominates trainable terms | Successful alternatives already supported and balanced exposure does not help | Needed baseline/engineering control, not headline contribution |

## Two easily missed identifiability problems

The mentor's Base→Agentic and Instruct→Agentic pairs share an endpoint. Their weighted sum is

```text
w1*(log Agentic - log Base) + w2*(log Agentic - log Instruct)
= (w1+w2)*log Agentic - w1*log Base - w2*log Instruct.
```

They are not two independent experts. A gain can be amplification or denominator choice, not complementary capability composition. The exact stage separation is `(Base→Agentic) - (Base→Instruct) = Instruct→Agentic`. Score all three nodes on the *same candidates* before interpreting a mixture as combination of skills. True composition evidence needs distinct executed decision strengths and a matched best-single-pair baseline.

Second, a teacher/base delta is not automatically a task advantage. It can encode changed interface conventions, stylistic preferences, likelihood calibration, or rare-denominator amplification. The minimal attribution experiment compares teacher-only action ranking, a true pre/post ratio, a matched irrelevant/swap-denominator control, and the instruction-only versus agentic-stage ratios. A post-only log-probability reward with OTHER=0 in the current token objective is not a clean teacher-imitation baseline; use a correctly normalized teacher target on the same support/action space.

## Reproduce

```bash
python3 research/multiturn_algorithm/information_state_probe.py \
  --output research/multiturn_algorithm/information_state_probe.json
```

All nine mathematical assertions passed. No GPU generation, model scoring, production trainer edits, or benchmark accuracy claim is implied by this file.
