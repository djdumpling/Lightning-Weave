# Decision-level delta composition: derivation, limits, and decisive tests

These are proposed research modifications, not claims of demonstrated task gains.
The companion `hierarchical_target_probe.py` is a Python-standard-library finite
probability-space probe. It changes no production training or inference code.
The sparse-attention project is outside this proposal.

## The specific failure mode

Let a response be `y=(z,a)`, with reasoning `z` and a complete externally consumed
action `a`. A tool action includes its exact tool name, all arguments, ordered
call bundle, and any public text delivered alongside it. It is not merely a
`<tool_call>` token or an action type such as WRITE.

For a donor pair, the *joint response* log-ratio decomposes as

```
d_j(z,a|h) = d_j^A(a|h) + d_j^Z(z|h,a)
d_j^A = log P_post,j(a|h) - log P_pre,j(a|h)
d_j^Z = log P_post,j(z|h,a) - log P_pre,j(z|h,a)
```

Consider the idealized sequence-level composition

```
v(z,a|h) proportional to b(z,a|h) exp(sum_j w_j d_j(z,a|h) / alpha).
```

Its action marginal is

```
v_A(a|h) proportional to
  b_A(a|h) exp(sum_j w_j d_j^A(a|h) / alpha) C(h,a)

C(h,a) = E_{z ~ b(z|h,a)} exp(sum_j w_j d_j^Z(z|h,a) / alpha).
```

`C(h,a)` is a reasoning partition function. Donors can agree on a useful tool
action yet prefer different ways of thinking about it. Their response-level
product can then suppress that action. Marginalizing responses into actions and
composing donor ratios do not commute. This is a mathematical possibility, not
yet a diagnosis of the real training runs.

The existing cached Top16 loss constructs *local token targets*. Multiplying
those local distributions also introduces prefix-dependent normalizers. It is
not generally identical to the idealized sequence target above. The finite-space
witness demonstrates a mechanism to measure, not an exact prediction of the
current checkpoint's output distribution.

### A matched-base witness

Two donors assign probabilities over three responses:

```
y1 = (rationale 1, good action)
y2 = (rationale 2, same good action)
y3 = (rationale 3, bad action)
P1 = (.950, .001, .049)
P2 = (.001, .950, .049)
b  = (1/3, 1/3, 1/3)
```

Each donor chooses the good action with probability `.951`. Under equal donor
weights `.5,.5` and `alpha=2`, response composition gives the good action
probability `.613335`, below the base's `.666667`. Marginalizing each donor first
and then composing with the **same induced action base** `(2/3,1/3)` gives
`.861693`. At weights `1,1`, alpha 1, the corresponding probabilities are
`.441758` versus `.994718`.

Using a uniform action base for the second calculation changes the experiment:
it is not the marginal of the uniform response base. The probe preserves the
base exactly in both paths.

## An exact correction on represented support

Let `v(y)>0` be any proposed distribution over represented responses, and let
`g(y)=a` partition them into groups. Write `V(a)=sum_{g(y)=a}v(y)`.
Given an independently chosen normalized action target `Q_A(a)`, the unique
minimum-change distribution is

```
q*(y) = v(y) Q_A(g(y)) / V(g(y))
      = Q_A(a) v(y | a).
```

It minimizes `KL(q || v)` subject to `q_A=Q_A`. The proof follows from

```
KL(q || v) = KL(Q_A || V) + sum_a Q_A(a) KL(q(.|a) || v(.|a)).
```

Its log-energy correction is `log Q_A(a) - log V(a)`. The **minus log V** term is
necessary. Merely multiplying each response by a desired action weight leaves
its previous group partition mass in the result. In the probe, a donor assigns
READ mass `.035325`; multiplying by desired READ/STOP weights `.8/.2` only raises
READ to `.127762`. The exact projection produces `.8`, while preserving all
within-group odds. The numerical KL Pythagorean identity residual is below
`5e-16`.

This generalizes earlier decision-preservation ideas in two substantive ways:
the group is the executed action and resulting future state, rather than a
structural token fork, and the target action marginal may deliberately improve
rather than preserve the original policy. This mathematical generalization is
not a novelty claim; constrained KL projections are established mathematics.

### Choosing the action target

There are three scientifically distinct versions. Do not conflate them:

1. **Marginalize then compose:** obtain each donor's action marginal and compose
   its action-level post/pre ratio. This directly tests the compositional
   mechanism. Exact full-model marginals are generally intractable; a finite
   candidate calculation is only a conditional approximation.
2. **Outcome target:** let `Q_A(a) proportional to b_A(a) exp(Q_student(h,a)/tau)`,
   using fresh, state-restored continuations with the current student. This
   separates what decision works from which donor reasoning to learn. Student
   continuation is essential: an action a teacher can finish may strand the
   student. A toy branch ordering reverses from teacher `.95 versus .85` to
   student `.15 versus .75`.
3. **Sparse outcome correction of a donor prior:** retain ordinary composed
   DOPD on broadly sampled fresh student states, but project audited decisions
   where branch evidence establishes a harmful or beneficial action change.
   Retain the donor action prior where outcome evidence is weak, with explicit
   uncertainty and trust-region controls. This is the practical low-budget
   version; its constraints must be specified before looking at held-out data.

If the outcome target is wholly independent of donors and actions are true
future-state equivalence classes, donors' conditional reasoning has **no direct
causal effect on return at those same states** in an expressive exact model.
Any further advantage is through generalization or optimization. Accordingly,
outcome-only candidate training versus outcome-plus-pair-conditional training is
a necessary ablation. The projection theorem does not prove that pairs or
composition improve accuracy.

## The history-policy caveat is critical

Same public tool action is not automatically the same future policy state.
The mentor's latest collector reconstructs the public history and removes old
private reasoning. In that regime, two private rationales followed by the same
complete public action can induce the same future prompt and environment state.

Lightning-Weave's historical interleaved-thinking BFCL path preserves reasoning
within the current user turn when the next tool result is processed. In that
regime, the next student prompt can retain `z`. Equal action marginals therefore
do **not** imply equal return. The probe constructs a case where both policies
execute exactly the same tool action but shifting retained-rationale mass changes
success from `.9` to `.1`.

Before sharing a continuation outcome across responses, verify equality of:

- the environment database, pending simulator state, and relevant RNG state;
- the entire externally consumed response and ordered tool-call bundle;
- the exact rendered input to the next assistant, including retained reasoning.

If the last item differs, execute/score each rationale-specific continuation or
include retained memory in the group definition. It is legitimate to run a
public-history-reset diagnostic, but that is an explicit new history-policy arm
and needs matching train/evaluation settings. Do not silently replace the
historical interleaved-thinking protocol.

A useful small experiment is to take two natural responses with identical calls
but different thoughts, fix the tool outcome, and compare student continuation
with retained thoughts versus the same explicit reset rule. This measures whether
reasoning is an irrelevant style variable or a real memory variable before
choosing the abstraction.

## Candidate construction and sampling weights

A finite candidate-set target and a Monte Carlo estimate of a full distribution
are different objects.

**Enumerated finite set:** for a set of unique response strings `C`, normalize
exact frozen-student sequence probabilities once within C, compose the donor
energies there, and call the result a conditional candidate-set target. Appending
more teacher candidates changes C; do not call that an unbiased estimate of the
full action distribution. Duplicate strings should not receive accidental extra
mass merely because two teachers proposed them.

**Monte Carlo proposal:** if candidate draws come from proposal `mu`, expectations
under an unnormalized target `b(y) exp(E(y))` require importance weights

```
w(y) = b(y) / mu(y) * exp(E(y)).
```

For pure student proposals `mu=b`, the weights are `exp(E)`, not `b*exp(E)`.
Multiplying by b again double-counts the sampling distribution and effectively
squares its preference. Repeated draws represent multiplicity in the estimator;
if deduplicated, preserve summed weights. For mixed teacher/student proposals,
record the actual mixture and evaluate its likelihood, or avoid claiming an
importance-corrected full-distribution estimate. Report effective sample size.

Never estimate an action marginal by the likelihood of one rationale or the
mean of log-ratios. Even for one pair,

```
log[P_post(a)/P_pre(a)]
  = log E_{z~P_pre(z|a)} exp(d(z,a)).
```

It is a log expectation under the appropriate conditional base distribution.
Same-tokenizer matched donor pairs are the clean initial experiment. Cross-model
string likelihoods and token projection introduce additional approximation.

## Implementation boundary with the current Top16 contract

A token mask or a per-response multiplier on the existing local Top16 loss does
not implement the action projection. A repair token outside the fixed support
cannot receive an individual preferred target from the OTHER bucket.

For a pilot, use a separately versioned candidate/sequence objective: compute
student log likelihood for each candidate, aggregate it by the verified group,
and fit the group target and conditional donor target. A candidate-normalized
loss needs an anchor against uncontrolled probability outside C. Alternatively,
weighted response NLL has a clear joint target but changes the optimization
relative to local logit regression. Keep the original DOPD baseline unchanged.

For an exact autoregressive distribution `v`, projection is a Doob transform:

```
H(prefix) = E_v[Q_A(g(Y))/V(g(Y)) | prefix]
q*(token | prefix) = v(token | prefix) H(prefix+token)/H(prefix).
```

A finite response trie permits exact backward computation of these factors on
its support. Full-vocabulary or OTHER factors are not known from an ordinary
sealed Top16 cache; treating them as known would be an implementation error.

## Information acquisition needs public-history conditioning

A further multi-turn pitfall occurs when action alternatives are compared in
only one privileged hidden world. Suppose two equally likely worlds have the
same public conversation. GUESS-0 succeeds only in world 0; GUESS-1 only in world
1; READ reveals the world and then succeeds with probability `.9`.

Selecting the best action separately in each world yields contradictory guess
labels. A policy without hidden-world access succeeds `.5` after imitating those
labels. The shared-public-history values are `.5,.5,.9`, so READ is the correct
policy decision. Ordinary on-policy RL with a privileged critic is not inherently
invalid: averaging over latent worlds can be correct. The issue here is
best-action selection at one hidden state and then treating that label as a
reliable public-history decision target.

A targeted training variation can generate 2–4 valid training worlds sharing the
same prefix, vary only unobserved slots or intent, execute common candidate
actions in every world, and average student-continuation success. Keep the action
target shared until observations distinguish the worlds. Preserve all facts
already revealed in the conversation; arbitrary database perturbations are not
valid counterfactual tasks. This can test retrieval and clarification with less
hindsight leakage than a single-world repair label.

This is a candidate research direction, not automatically novel. Information
gathering and privileged imitation are longstanding topics, and recent IGSD
already verifies executed retrieval gains. The narrower question is whether
public-history-matched latent worlds fix action-level delta supervision on
transactional multi-turn agents.

## Smallest decisive experimental sequence

1. Collect new development states; mentor run artifacts are not required. Include
   after-tool decisions, clarification/READ, policy confirmation, WRITE, and STOP.
2. Generate several student and donor responses. Canonicalize exact action bundles
   and record the next rendered prompt to establish which groups are valid.
3. First measure oracle candidate headroom and then within-group versus between-
   group donor-score variance. Test response-first versus action-first composition
   with a common base and matched strength. Do not hide a base-prior change.
4. Execute distinct valid branches with student continuations; for retained
   reasoning, evaluate its separate future effect. A teacher continuation is a
   diagnostic for the handoff gap, not the main deployment value.
5. Train matched pilots: existing DOPD; refreshed-state DOPD; outcome-only candidate
   loss; outcome-corrected single-pair target; outcome-corrected composed target.
   Compare at matched unique states, branch budget, and optimizer exposure.
6. Only if missing-information cases remain a clear bottleneck, compare one-world
   and shared-public-history world targets on controlled training templates.

Reject this direction if action grouping is too sparse, candidate support has no
successful alternatives, marginalize-first scoring does not improve branch
ranking, outcome-only training matches all pair variants, or any apparent gain
vanishes under matched data/compute. Do not launch a large factorial study first.

## Closest sources and positioning

- [Composing Entropic Policies using Divergence Correction](https://arxiv.org/abs/1812.02216)
  establishes that sequential policy composition can require future divergence
  correction. A generic claim that composing policies needs correction is not new.
- [PivotOPD](https://arxiv.org/abs/2609.40285) already teaches prevention/recovery
  from pivotal errors. Branching and teacher repair are not sufficient novelty.
- [VG-OPD](https://arxiv.org/abs/2609.15404) routes expert supervision using verified
  counterfactual gain. Generic verifier-gated multi-teacher weighting is occupied.
- [STRIDE](https://arxiv.org/abs/2609.14636) investigates tau2 prefix stopping and
  restart. Prefix reuse/focusing alone is not a differentiating algorithm.
- [IGSD](https://arxiv.org/abs/2609.32694) executes paired query proposals to verify
  information gain before distillation. Verified retrieval credit alone is occupied.
- [Self-Retrospection Distillation](https://arxiv.org/abs/2610.08077) learns
  pre-interaction foresight from post-hoc experience, including reward-uniform
  groups. Hindsight-to-foresight supervision alone is not a differentiator.

The potential contribution is more specific: identify and correct the failure
of response-level donor composition under executable-state abstraction, account
for whether reasoning persists as memory, and demonstrate that corrected pair
transfer adds value beyond matched outcome-only supervision. The mathematics
supplies a testable mechanism; only the experiments can establish its importance.

## Exact token-local witness and the normalizer distinction

The probe also contains a **separate exact autoregressive witness** so the
sequence-level example is not mistaken for a demonstration about another loss.
At the first token, choose rationale R1/R2/R0. The frozen behavior and both pre
anchors assign `(1/3,1/3,1/3)`. The two post anchors assign
`(.95,.001,.049)` and `(.001,.95,.049)`. At the next token, choose GOOD/BAD.
Every model, including behavior and both pre/post anchors, has identical GOOD
probabilities `(.99,.99,.01)` conditional on R1/R2/R0. Consequently every leaf
log-ratio is exactly zero, and the local leaf target stays unchanged.

| Target | GOOD probability |
| --- | ---: |
| Frozen behavior | .663333 |
| Either donor | .941980 |
| Token-local composition, weights 1/1, alpha 1 | .442923 |
| Token-local composition, weights 1/1, alpha 2 | .555996 |
| Token-local composition, weights .5/.5, alpha 2 | .611068 |
| Action-first composition, weights .5/.5, alpha 2 | .849756 |

This tree uses at most three children, all represented; there is no omitted
repair-support explanation. The two donors favor different good rationale modes,
so composing them makes their shared bad rationale disproportionately likely.
All listed configurations reduce the good-action probability below the behavior.
The tree is terminal after the action, so no retained-memory assumption is hidden.

In a general autoregressive model, the local target's path probability is

```
q_local(y|h) = b(y|h) exp(sum_t d(h,y_<t,y_t)/alpha)
                        / product_t Z(h,y_<t).
```

A globally normalized response tilt instead has a single sequence normalizer.
The product of local normalizers is path dependent and cannot be dropped. The
simple response-space partition formula earlier is exact for its stated
sequence-level construction; it is not an identity for the production local
loss. The projection `q*=v Q_A/V` is exact for *any* normalized v, including the
actual local policy, but obtaining its action marginal still requires integration.

For a real probe, score each sampled path under the actual local target using its
frozen behavior likelihood, its per-prefix Top16 shifts, and every per-prefix
normalizer. A sampled token outside Top16 has shift zero; its individual frozen
behavior probability is still required. This provides a valid path likelihood
for importance estimation. Using only the sum of post-minus-pre token scores
would test the sequence proxy instead of the actual target.

## Can the action marginal be estimated economically?

This is the main feasibility risk. A pretrained base anchor may almost never
emit a valid tool call, and the important rationale modes can be disjoint across
anchors. Six response samples at a state do not establish the action marginal.

### Action-conditioned importance sampling

Fix a public prefix h and a complete candidate action a. Use a **known** proposal
`rho(r|h,a)` that is prompted with a to generate reasoning r. Then append a and
score the complete `(r,a)` under each original model on the **unhinted original
prefix h**. Under the support and termination assumptions,

```
P_M(a|h) = sum_r P_M(r,a|h)
         = E_{r~rho(.|h,a)} [P_M(r,a|h) / rho(r|h,a)].
```

The action hint belongs only to the proposal. If the numerator is scored with
that hint, it estimates a different privileged model and invalidates the claim.
The proposal need not be the actual posterior `P_M(r|h,a)`; closeness to that
posterior governs variance, however. The small absolute probability of a under a
base model does not by itself invalidate this identity, but poor posterior
coverage can make it unusable.

With K proposal components and fixed counts `n_k`, use the balance mixture

```
N = sum_k n_k
mu(r|h,a) = sum_k (n_k/N) rho_k(r|h,a)
w_M,i = P_M(r_i,a|h) / mu(r_i|h,a)
P_hat_M(a|h) = (1/N) sum_i w_M,i
Delta_hat_A(a) = logsumexp_i(log w_post,i)
                - logsumexp_i(log w_pre,i).
```

Score **every sampled rationale under every proposal component** to evaluate
mu, not merely under the component that generated it. Deterministic stratified
sampling has this unbiased normalizing-constant estimator because its component
counts match the mixture coefficients. The log and ratio estimates have finite-
sample bias even when the separate normalizing-constant estimates are unbiased.
Use float64 log-space reductions to avoid underflow.

A useful initial mixture uses student, post-anchor, and matched pre-anchor
rationale proposals, each conditioned on the same candidate action. Start with
the Qwen3-8B instruct-to-AgenticQwen pair, rather than a base model with an
unfamiliar chat interface. The broad base-to-post pair can be a second diagnostic.

### Density, boundary, and support requirements

- Record the actual proposal decoding density. Raw logits at temperature 1 do
  not give the probability of samples generated with another temperature or
  top-p truncation. A clean pilot uses temperature 1 and top-p 1, with exact
  token log probabilities and a defined reasoning termination token.
- Include the probability of the reasoning-ending token. A deterministic
  appended action is outside the proposal density but inside the numerator.
- A token cap restricts the rationale domain. Report a bounded-rationale
  marginal, or account for the censored tail; do not present it as a full
  marginal. Rejection sampling and post-hoc filtering also change the proposal.
- If proposal rejection introduces an unknown normalizer that is common to pre
  and post for the same action, that constant can cancel in their **ratio**.
  It does not restore omitted support or identify absolute action mass.
- The action set is also restricted. Reliable ratios for two candidate tool calls
  do not identify probabilities of every possible call, clarification, or STOP.
- Same-tokenizer pairs avoid additional string-probability ambiguities. A
  normalized finite candidate energy across tokenizers is useful experimentally
  but is not automatically the model's exact text distribution.

### Uncertainty and stopping criteria

Compute per-anchor `ESS=(sum w)^2/sum w^2`, normalized maximum weight, and a
paired uncertainty interval for the post/pre log ratio. With fixed-count mixture
sampling, bootstrap within proposal components so the design is preserved. Use
common sampled rationales for both anchors: their estimator errors are correlated
and independent-error formulas waste that pairing.

An approximate paired standard error can also use the sample variability of
`w_post/P_hat_post - w_pre/P_hat_pre`, divided by sqrt(N); for fixed-stratum designs,
combine the corresponding within-stratum variances. Bootstrap uncertainty does
not measure undiscovered rationale modes. Even high empirical ESS is insufficient
if every proposal misses the same important mode.

Pre-register practical feasibility checks, for example:

- min(pre ESS, post ESS) at least 16 out of 64, and no normalized weight above .2;
- action-rank decisions agree between nested 16- and 64-sample estimates except
  for states explicitly labeled unresolved;
- paired log-odds uncertainty is smaller than the action preference being claimed;
- independently designed proposal mixtures and held-out rationale batches produce
  compatible rankings; a leave-one-proposal-out sensitivity check is descriptive;
- the actual local-target path estimate and the sequence proxy are reported
  separately.

These are engineering acceptance thresholds, not a theorem guaranteeing accuracy.
Do not adapt them after seeing which settings support the hypothesis.

### A concrete diagnostic budget before any new trainer

1. Choose 12 fresh development decision states with two materially different,
   policy-valid candidate actions each. Include READ/clarify versus WRITE/STOP
   and after-tool follow-up. Preserve exact environment and rendered prompt state.
2. Sample 16 action-conditioned rationales per action: **384 rationale samples**.
   Score the original unhinted response likelihood under the frozen student and
   both anchors, plus proposal densities. Also score the actual local DOPD path
   probability rather than using only the joint-ratio proxy.
3. Extend unresolved or apparently interesting groups to 64 rationales, at most
   **1,536 total samples** for these 12 states. Keep the first 16 nested in 64;
   reserve an independent batch/proposal for sensitivity rather than counting
   nested samples as independent replications.
4. If estimates are stable and the corrected action ordering is more consistent
   with executed student-continuation outcomes, repeat on 12 new states. A full
   24-state panel at 64 rationales/action is at most **3,072 rationale samples**.
   Record generated/scored token count: a rationale cap of L implies up to N*L
   proposal tokens plus separate scorer work, not N cheap scalar operations.
5. Kill the **true-marginal estimator route** if 64 samples still have severe
   weight concentration, different proposals reverse rankings, or the actionable
   rank margin is smaller than uncertainty. Do not train a large model on those
   noisy pseudo-labels. A finite candidate-set target may still be useful, but
   describe it as such and compare against outcome-only candidate training.

A negative result here is informative: it says exact marginalization is too
expensive or the proposed nuisance explanation lacks support, before a trainer is
built. It does not establish that every possible action abstraction is useless.

## Why not simply average the donors?

That is an essential baseline. Choosing one donor for a whole response, or an
arithmetic mixture of complete-response donor distributions, avoids the simple
rationale-product conflict. Tokenwise averaging is a different baseline because
it may switch donor identity within a reasoning trace. If a cheap mixture fixes
the real failure equally well, action marginalization has not earned its cost.

The companion probe therefore adds a complementary-constraints example. Actions
`11,10,01,00` represent satisfying two requirements. Donor 1 probabilities are
`(.45,.45,.05,.05)` and donor 2 probabilities are `(.45,.05,.45,.05)`. The correct
11 action has probability `.45` under either donor and their mixture, but `.81`
under action-level product composition at weights 1/1, alpha 1. Give the two
donors incompatible rationale modes specifically for action 11, and exact
token-local composition instead assigns 11 probability `.144437`.

The interesting hypothesis is therefore **combine complementary decision
constraints without requiring agreement on the rationale that establishes
them**. A real contribution needs states exhibiting that complementarity and
must beat the best single donor and a whole-response donor mixture. The synthetic
example establishes possibility only.

Finally, local normalization can also *remove* an apparent global-product
failure. If the action token comes first and conflicting, behaviorally irrelevant
style tokens come afterward, later local normalizers cannot retroactively change
the action already selected. One exact example gives GOOD probability `.997238`
under token-local composition but `.126169` under the global response product.
Thus there is no universal ordering between the two constructions; the actual
autoregressive decision placement matters. This is another reason to test the
production local target directly instead of substituting a sequence-likelihood
proxy.

### Canonical tool calls are another marginalization

For the importance identity, `a` must be a precisely defined output event.
If canonicalization groups different JSON key orders, whitespace, or equivalent
public utterances, the semantic action probability is actually

```
P_M(a|h) = sum_{u : canonicalize(u)=a} sum_r P_M(r,u|h).
```

Appending and scoring one canonical JSON string integrates over r but not over
u. It estimates the probability of that exact serialization. A full semantic
estimate needs a known joint proposal over rationale and serialization, or an
explicitly restricted serialization protocol. The economical pilot can use exact
rendered action strings and separately report canonical-action agreement; it
must not quietly identify these two probabilities.

### Prefer native action counts when the event is common

Before implementing the importance estimator, a simpler first diagnostic is
64 ordinary generations from each matched instruct/pre and agentic/post model at
12 common development states: **1,536 responses total**. Count complete
canonicalized action events directly, with uncertainty and unseen-event bounds.
This uses no hinted rationale proposals and can expose disagreeing rationale
modes behind agreement on READ/STOP/tool choice. It is especially appropriate
for the instruct-to-agentic pair; an unadapted base may rarely generate a valid
protocol response.

Exact argument-bearing actions can remain too sparse to estimate, and collapsing
them into a tool name no longer measures the original action event. Label those
cases unresolved rather than assigning a large ratio from zero counts. Also
specify the generation distribution: temperature-1 unrestricted sampling targets
the original model distribution, whereas temperature/top-p changes define a
different action distribution. Use action-conditioned importance sampling only
where the event is too rare for native counting and the estimator's support and
variance checks succeed. For long reasoning traces, the IS pilot is a feasibility
kill-test rather than a promise of a cheap training signal.
