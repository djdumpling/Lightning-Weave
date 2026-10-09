# Continuation mismatch in checkpoint-delta transfer

This note concerns Lightning-Weave / multi-turn Direct-OPD only. The sparse-attention project is independent.

**Priority and scope:** this is an alternative sequential objective to investigate after branch diagnosis, not a demonstrated implementation bug in the current token-local loss. Normalizer rewards can promote loops; matched task reward/KL evidence is required.

The files in this directory are **exact synthetic finite-MDP calculations**, not LLM training, BFCL results, or tau results. They establish that a structural failure and a principled surrogate correction exist. They do not establish that this explains the real benchmark plateau or that the correction improves benchmark accuracy.

Run: python3 research/multiturn_algorithm/continuation_probe.py

Outputs: continuation_probe_results.json and continuation_probe_sweep.csv. Python standard library only.

## Finding

A donor's locally preferred action can be appropriate for the donor's continuation abilities and inappropriate for a stronger recipient. A post/pre checkpoint ratio can retain this continuation dependence even when the donor is exactly KL-optimal, all distributions are known, and support is complete.

Synthetic two-decision example:

- Choose Safe for reward 0.2 or Challenge for eventual success reward 1 / failure reward 0.
- The donor reference succeeds at Challenge with probability 0.01.
- The recipient reference succeeds with probability 0.99.
- Both initially choose Challenge with probability 0.5.
- The donor is trained to the exact KL-regularized optimum, with coefficient 1.

| Policy | Challenge probability | Expected task reward |
| --- | ---: | ---: |
| Recipient reference | 0.500000 | 0.595000 |
| Local checkpoint-delta transplant | 0.454386 | 0.561827 |
| Bellman-corrected transplant | 0.688616 | 0.748344 |

Both transplanted policies have the same improved Challenge completion probability, 0.996298. They differ at the earlier decision. The local method inherits the weak donor's reluctance to attempt the challenge, despite the recipient being able to complete it. It lowers both task reward and the intended regularized ratio objective in this example.

The probe sweeps 1,620 settings: 9 donor competence values, 9 recipient competence values, 5 safe rewards, and 4 KL coefficients. Local transplantation lowers task reward versus the recipient reference in 249 settings, including 174 of 720 settings where the recipient is stronger. Bellman correction increases task reward versus the local method in 1,105 settings, decreases it in 470, and ties in 45. These grid frequencies are design-dependent synthetic counts, not estimated prevalence in real tasks.

The 470 adverse cases matter: a correct optimizer of a KL-regularized objective need not maximize unregularized task reward. The local method can accidentally achieve higher task reward by taking more KL. Compare methods at matched realized KL or sweep their reward/KL frontiers before attributing an accuracy gain to better credit assignment.

## Derivation and candidate modification

Let b(a|h) be a fixed recipient reference, D(h,a) a fixed weighted sum of donor post/pre log ratios, and alpha > 0. A history includes all information the agent actually observes; for a finite-horizon problem it also includes the remaining budget.

The current local target has the form

q0(a|h) = b(a|h) exp(D(h,a)/alpha) / Z0(h).

The corresponding sequential objective is

J_D(pi) = E_pi sum_t [D(h_t,a_t) - alpha log(pi(a_t|h_t)/b(a_t|h_t))].

The exact causal soft Bellman optimum satisfies

F*(h) = alpha log sum_a b(a|h) exp((D(h,a) + E[F*(h')|h,a])/alpha),

q*(a|h) proportional to b(a|h) exp((D(h,a) + E[F*(h')|h,a])/alpha).

Discounting can be included consistently by multiplying successor values by gamma. The probe uses finite episodes and gamma = 1.

The local target omits the successor residual value. A lightweight policy-iteration modification is:

1. Collect refreshed recipient trajectories.
2. Evaluate a scalar residual value F^pi for the existing checkpoint-ratio objective.
3. Form a corrected target using D(h,a) + E[F^pi(h')|h,a].
4. Distill that target using the existing student-reference and composed-delta machinery.
5. Refresh states and repeat.

The useful implementation identity is

E_a~pi [D - alpha log(pi/b)]
= alpha log Z0(h) - alpha KL(pi(.|h) || q0(.|h)).

Therefore the critic can learn from existing local normalizers and KL terms, without asking a sparse verifier to supply the training reward. When pi = q0, the immediate regularized reward is simply alpha log Z0(h), and the missing correction is the expected future sum of these normalizers. Calling this a normalizer-return critic describes what it computes; it is ordinary soft policy evaluation, not a new Bellman theorem.

With exact evaluation, complete support, and exact policy improvement, this improves J_D. A learned critic, incomplete action candidates, stale states, and an approximate distillation step remove that guarantee.

For one donor, b = donor-pre, and alpha = 1, q0 = donor-post and Z0(h) = 1 everywhere. The correction is exactly zero. The probe checks this identity in 45 cases. If an implementation yields a nonzero correction in this setting, first investigate support approximation, token normalization, masking, or inconsistent references.

## Relation to a real donor reward

If donor-post is an exact causal KL-optimal policy relative to donor-pre, then, in donor temperature units,

D(h,a) = reward(h,a) + E[V_donor(h')|h,a] - V_donor(h).

The ratio is a donor-value-shaped reward. Its expected trajectory sum recovers the donor reward up to a starting-state constant when the terminal value is zero. Using it as the recipient's immediate advantage omits the recipient continuation correction. Full reward-to-go or appropriate Bellman evaluation handles that dependence.

This interpretation requires the pair to correspond to the assumed reward, KL reference, dynamics, information set, and terminal convention. An arbitrary SFT/RL checkpoint pair, a pair trained with a different reference, or an imperfect post-trained donor need not satisfy it. For arbitrary pairs the proposed critic still optimizes the specified ratio objective, but that may be unrelated to task success.

The probe includes an explicitly misaligned downstream objective: external reward is the complement of the donor reward. Correction raises the ratio objective from 0.443764 to 0.560885 while reducing external reward from 0.438173 to 0.251656. This is an intentional counterexample to an accuracy guarantee. A reliable surrogate cannot be inferred from a donor's aggregate leaderboard score.

## Stochastic-environment trap

A global exponential tilt over full trajectories can also tilt exogenous outcomes. That is not generally a policy the agent can execute.

In the probe, an action gives a fixed 0.5 chance of reward 1; another gives certain reward 0.6. Causal soft Bellman uses the lottery's expected value 0.5 and assigns it probability 0.475021. Incorrect full-path exponential tilting instead assigns it probability 0.505028 and implicitly changes its win probability to 0.731059, although the environment fixes that probability at 0.5.

Across model-controlled token choices, log-sum-exp is appropriate. Across stochastic tool/user/environment outcomes, take the expectation of the next value before optimizing the action. Do not replace it with log-mean-exp over lucky continuations. Full histories rather than inaccessible simulator state must define deployed policy inputs.

## Most informative real-data diagnostic

No latest mentor trajectories are required. Generate fresh development states from a frozen available recipient, preserving exact simulator state and user/environment randomness where possible.

For each state:

- Compare 2-4 executable alternatives, including the recipient's original action and a donor proposal.
- Continue each alternative with the same frozen recipient; on a smaller subset also continue with donor-post.
- Record local donor likelihood, post/pre delta, composed delta, normalizer/KL returns, realized task outcome, and support mass.
- Compare rankings within each state, then aggregate with task-clustered uncertainty.

The action-by-continuation-policy interaction is especially informative. If a donor-preferred action is good under donor continuation but bad under recipient continuation, or vice versa, there is evidence for a transfer-credit mismatch. If local delta already ranks recipient-successful actions well, investigate state/action coverage instead.

A cheap pilot estimates short normalizer returns first. It is only a diagnostic: short lookahead can miss information-gathering value and must not be treated as a success oracle. Use complete continuations on a stratified subset to detect this bias.

Do not train the critic unless the correction provides held-out information beyond the local delta and teacher likelihood. Falsification conditions:

1. Future normalizers do not vary meaningfully across consequential alternatives.
2. The correction tracks response length/style but not recipient task outcomes.
3. Local ranking errors disappear after merely changing candidate coverage or refreshing states.
4. The correction cannot beat a local target at matched realized KL and matched branch budget.
5. Ratio alignment degrades after policy updates; early calibration is insufficient.
6. Reward-based repair/SFT or ordinary outcome-based RL explains all gains with matched data.

At whole-response level, candidate sampling requires proposal correction. For candidates sampled from mu, weights targeting b(a) exp(score/alpha) are proportional to b(a)/mu(a) exp(score/alpha). If mu = b, b cancels. Softmax(log b + score) over b-sampled candidates double-counts the proposal. Whole-response likelihoods also include latent reasoning and variable-length text; masking thought tokens is an approximation, not exact semantic-action marginalization.

## Novelty assessment and primary sources

**The Bellman correction itself is not novel.** It is classical regularized policy iteration. A defensible contribution would be the specific diagnosis of recipient-reference/continuation mismatch in checkpoint-delta agentic transfer, a scalable estimator that works with existing OPD interfaces, and evidence that it improves long-horizon composition beyond matched alternatives.

- [Direct-OPD](https://arxiv.org/html/2607.05394v1) introduces post/pre policy-shift transfer and a local top-k surrogate. Our proposed analysis concerns the gap between that local update and its stated sequential reward objective.
- [Hunt et al., ICML 2019](https://proceedings.mlr.press/v97/hunt19a/hunt19a.pdf) already correct composition using future policy divergence.
- [Adamczyk et al., AAAI 2023](https://arxiv.org/abs/2212.01174) gives corrective value relations for changed rewards, dynamics, prior policies, and task composition. This is the closest mathematical ancestry and prevents claiming a new general transfer theorem.
- [Revisiting OPD](https://arxiv.org/html/2603.25562v1) already analyzes the bias/variance tradeoff between token-local and future-coupled estimators. Merely adding return-to-go is not a contribution.
- [FutureBridge-OPD](https://arxiv.org/html/2608.01953v2) executes a teacher intervention and tests paired student continuations using future teacher-preference density. The proposed normalizer correction has a different target, but lookahead and executed repair are already established.
- [LOLS](https://proceedings.mlr.press/v37/changb15.pdf) already uses learner/reference continuation policies for cost-sensitive search. A continuation-swap diagnostic is informative, not novel by itself.
- [OPRD](https://arxiv.org/abs/2609.08798) uses checkpoint shifts to amplify verifier-supported gradients while preserving the verifier objective's stationary points. It is an essential comparison if ratio calibration leads to a hybrid with outcome rewards.
- [TVKD](https://arxiv.org/abs/2509.16965) uses teacher values as potential-based shaping in preference distillation. Generic teacher-value shaping is also occupied.

Searches for Direct-OPD plus Bellman/continuation correction did not reveal an exact LLM transplant implementation in this review. That is limited evidence, not a certification of novelty. Before a paper claim, inspect these works' full algorithms and code and search new submissions again.


## Objective choice, not a demonstrated implementation bug

The production method's token-local normalized target is a legitimate choice of supervision. This proposal changes the trajectory-level incentives; it is not a claim that the implemented local loss computes its own intended objective incorrectly.

For an assistant response y, let B(y) be the autoregressive recipient reference. Three targets must be separated:

1. Global response tilt: Q_response(y) is proportional to B(y) exp(sum_i D_i(y_i)/alpha).
2. Token-local generation: Q_local(y) = B(y) exp(sum_i D_i(y_i)/alpha) / product_i Z0(prefix_i).
3. Semantic-action marginal: first sum each donor/reference probability over all valid traces realizing an action, then compose those action-level ratios with the induced recipient action prior.

The first two differ by a product of prefix normalizers. Repeated local imitation learns the second target; optimizing the unnormalized D-return chooses the first target in a deterministic token tree. Across stochastic environment transitions, causal Bellman planning is required instead of global path tilting.

Equivalently, the normalized local target can be viewed as using reward D - alpha log Z0(h). Restoring log Z0 therefore adds incentives for future state visitation. It is a substantive algorithmic hypothesis, not a neutral correction that must help.

Any strictly positive donor-post policy can be represented as optimal for reward log(post/pre) relative to donor-pre, with zero soft value. This constructed reward does not identify the actual post-training objective or establish alignment with task success.

The added high-normalizer-loop probe makes the danger explicit. Donor-pre(stop, continue)=(0.9,0.1), donor-post=(0.5,0.5), and recipient base=(0.1,0.9) produce log Z0=1.516347 in a live state. At an eight-turn time limit, the local policy stops successfully with probability 0.093497; the corrected D-objective policy stops with probability 0.015625 and consumes 7.995536 turns on average. At a sixteen-turn limit its stopping probability remains about 0.015625 while its expected length rises to 15.995536. Task success here explicitly means stopping before timeout; it is not the donor's constructed ratio objective.

Consequently, this proposal should follow branch diagnosis, not precede it. Reject it if future normalizer gains primarily predict length, looping, or stylistic changes, or if gains disappear at matched realized KL. Do not promote better surrogate value to an accuracy claim.

## Causally valid integration at tool boundaries

Only assistant choices receive checkpoint-ratio rewards. User and tool output tokens are observations, not actions whose model likelihoods should be rewarded.

For assistant response j containing tokens i, a sampled macro-step reward is

R_j = sum_i [D_ji - alpha log(pi_ji / b_ji)].

A critic at the next agent-observation boundary evaluates

F(h_j) = E_pi,env [R_j + F(h_(j+1))].

The current response is sampled from the agent, and the next history includes the actual tool/user response. All environment randomness remains distributed according to the environment. Use ordinary expectation of F(h_(j+1)) conditional on a proposed complete response; never exponentiate and reweight stochastic tool/user outcomes.

For policy evaluation under pi, the immediate token reward can be Rao-Blackwellized to alpha log Z0 - alpha KL(pi||q0), evaluated at each visited assistant-token prefix. Summing these state-wise expectations along fresh pi trajectories gives an unbiased value target under the ideal support assumptions. When scoring a fixed candidate response, however, use its actual summed D, not a sum of state-wise expected rewards: the candidate has already fixed its token choices.

A macro response target would be proportional to B(response|h) exp([D_response + E_env F(next_history)]/alpha). Implementing this exact target autoregressively requires its within-response continuation factors too. Merely adding a terminal critic bonus to an otherwise token-local loss is a useful macro-only approximation, not an exact implementation of the full Bellman optimum.

Declare whether discounting applies per token or per environment step; do not mix them. For per-turn discounting, use discount 1 inside the response and gamma at the environment boundary. Include the remaining budget in the value state and treat timeout separately from successful termination.

The production Top16+OTHER representation needs additional care. An OTHER bucket aliases tokens with different next states, and its coarse KL omits conditional distinctions inside the bucket. The full-support identities above do not automatically become exact for that representation. First validate the correction on expanded support / fully scored small vocabularies, measure missing mass, and state the resulting approximation.

## Additional token-tree check

The companion token_tree_probe.py separates global response products from current-style token-local normalization. It provides both a failure and a mitigation example, preventing the response-level result from being presented as a production diagnosis.

- Before-action rationale conflict: each donor favors the Good action with probability 0.941980, but they favor different rationale modes. The base's Good probability is 0.663333. Exact token-local composition produces 0.442923, while composing true action marginals gives 0.992581. Averaging the two donor weights still leaves token-local probability at 0.555996.
- After-action irrelevant style conflict: each donor favors Good with probability 0.95. Global response composition gives 0.126169, while token-local normalization and action-marginal composition both give 0.997238. Local normalization protects the action because the stylistic conflict happens after it.

These finite trees use at most three candidate tokens per prefix, so top-16 truncation is not responsible. All allowed actions have positive support. They demonstrate that token-local normalization can mitigate some global-product failures but cannot prevent every semantic-composition failure.

The Good/Bad action terminates these examples. For real multi-turn traces, grouping distinct rationales is justified only if they lead to equivalent future policy states, or if continuation differences are explicitly modeled. A retained rationale can influence later decisions even when the immediate tool call is identical.

