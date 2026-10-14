# Reconciled theory: what must change beyond local delta fitting

Phase 2 complete, 2026-10-14. This synthesis reconciles two independent analyses, [A](THEORY_DRAFTS/A.md) and [B](THEORY_DRAFTS/B.md), and an independent [mathematical audit](THEORY_DRAFTS/MATH_AUDIT.md). Neither theorist saw the other's draft. The audit toy was rerun by the root agent. **Recorded measurements** are in LEDGER; **code contracts** in ASSUMPTIONS and audit05; **deductions** below have explicit premises; **causal accounts remain hypotheses**. No model training or benchmark evaluation occurred.

The evidence does not identify one dominant cause. A useful next algorithm must supply or exploit a specific identifiable distinction that current DOPD misses, then preserve it through finite fitting and deployment. A cleaner surrogate, larger target, more complicated estimator, or residual benchmark headroom is insufficient by itself. The fixed-anchor ceiling remains a serious competing explanation.

## 1. The actual multi-turn POMDP

Let x contain hidden database state, the user's goal and private conversational state, pending transactions, and harness status. Let H contain realized external history, emitted reasoning, the partial response, and protocol/budget state. The augmented process s=(x,H) is Markov when all relevant simulator state is included. The actor receives h=Render(H), not x. If rendering drops relevant information, equal h need not imply equal future control states; a valid belief state must average over hidden x and omitted history conditional on available information.

Reasoning tokens are actions of the computational policy: they change the future input even when they do not change the database. Parsed tool calls, messages, requests for user input, and completion signals trigger different transitions. Ending an assistant response, yielding to a tool, completing a real-user turn, completing a task, and timing out are not interchangeable STOP events. Terminal reward is the benchmark's episode-success verifier under its actual protocol.

For current Qwen3.5, reasoning persists **since the latest real user message**, potentially through several tool exchanges. Equal tool JSON with different retained reasoning is not the same next actor state. Historical2507 replay dropped prior thought and is a different process. The current T=0 user removes deliberate positive-temperature user sampling; student continuation randomness is policy randomness. Hidden-state uncertainty can still make observations random conditional on h, even if a fully specified simulator world is deterministic. Conversely, branch-label disagreement does not prove exogenous transition noise. These distinctions are required before applying deterministic-transition, action-quotient, or counterfactual-replay theorems.

Write rho_theta for the deployed decoding policy, including temperature, truncation and presence penalties. It can differ from the raw full-vocabulary model pi_theta and be nonsmooth at truncation boundaries. Changing renderer, user model, task version, decoding or budgets changes the estimand J(theta)=E[R], not merely implementation detail.

## 2. The frozen measure, target and fitted loss

There are three distributions: executed episode occupancy under the collector; neutral completions at selected states; and full-normalized cached behavior probabilities b0 at the resulting token prefixes. Current collection uses T1/top-p .95/top-k20/presence1.5; four cached responses use neutral T1/full sampling. Hash selection, context limits, masking, censoring and repetitions determine the fixed empirical token measure mu. Sharing initial model weights does not make these measures equal.

At a cached prefix h, let S be behavior Top16, O its complement, b the aggregate distribution on S union {O}, and m=sum_{a in S}b_a. All model probabilities are normalized over the configured full vocabulary before gathering. Define

    D_a = sum_j w_j [log p_post,j(a|h) - log p_pre,j(a|h)], a in S;
    D_O = 0;
    q_a = b_a exp(D_a/alpha) / Z_h;
    L(theta) = E_mu [ alpha/(2m) sum_a b_a (log p_theta,a - log q_a)^2 ].

Synthetic composed pre/post fields encode D and are not themselves normalized probability models. O=0 is a reward convention, not an estimated mean tail advantage. The reduction is a global masked token mean, not equal turn or episode weighting. Token counts are exposure counts, not measured gradient shares.

With positive probabilities and inactive floors, q uniquely maximizes E_u[D]−alpha KL(u||b) at this prefix. The fitted loss is instead **behavior-weighted squared log error**, neither KL direction. In independent unconstrained bucket logits, q is the unique finite stationary distribution and unique zero-loss distribution: stationarity implies b_i r_i=p_i sum_j b_j r_j; a nonzero common sum would make every p_i greater than every corresponding q_i, or every p_i smaller, contradicting normalization. This does not imply unique neural stationarity: shared-parameter projected residuals can cancel.

Zero loss identifies represented-token probabilities and aggregate O mass, not its conditional allocation. The floor on O and current-log clamps can make extreme targets infeasible. A full-token lift preserving b0's conditional tail is a useful mathematical convention, but production does not enforce it. Neural fitting also provides no guarantees at uncached prefixes.

## 3. Exact gradient, curvature and the missing task-improvement bridge

For independent bucket logits u, p=softmax(u), r=log p−log q, R=sum b r, F(p)=diag(p)−pp^T, and J_log=I−1p^T,

    grad_u L_h = (alpha/m) [b elementwise r - p R];
    Hess_u L_h = (alpha/m) [J_log^T diag(b) J_log - R F(p)].

At a matching clean initialization p=b, let kappa=KL(b||q). Then

    -grad_theta L_h = (1/m) sum_a b_a D_a grad_theta log p_a;
    Hess_u L_h = (alpha/m)(1-kappa)F(b).

The first identity is exact only with compatible vocabularies, inactive floors and actual initial agreement with the cache. Backend differences make it approximate in a run. It identifies a local surrogate gradient, not a task advantage. Holding D fixed, the initial gradient is independent of alpha; later movement is not. The exact curvature uses reverse KL(b||q), unlike the reported forward target KL(q||b). It is not universally the positive Gauss–Newton/Fisher approximation. Negative curvature away from a stationary point does not establish an optimization pathology or diagnose L8.

Full-token logits add an OTHER log-sum-exp curvature term; see the audit for the exact formula. Tail probability redistributions at fixed aggregate mass are an exact invariance, yet straight tail-logit directions can change that mass at second order. Thus absence of tail-conditional supervision does not mean tail probabilities or every full-logit Hessian direction stay fixed.

Shared parameters induce cross-history effects. A local update at h moves log probabilities at h' through inner products of their parameter Jacobians. Desired shifts can generalize, cancel or conflict. For a preconditioned step theta' = theta−eta M grad L,

    J(theta')−J(theta) = −eta grad J^T M grad L + O(eta^2).

Neither reducing L nor matching mean target KL supplies the sign. Exact target invariance is not sampled-policy invariance, as L2 illustrates. Nonzero recorded minibatch losses and late plateaus are descriptive, not a matched convergence or realizability test (LEDGER S11).

On a common finite-horizon augmented process, the performance-difference identity is

    J(rho')−J(rho) = sum_t E_{s~d_t^rho'} E_{a~rho'(.|h(s))} A_t^rho(s,a).

It requires recipient continuation, the new policy's occupancy, actual terminal reward and unchanged environment/protocol. DOPD supplies none of those advantages or occupancies directly. In a hypothetical smooth full-policy tilt rho_epsilon proportional b exp(epsilon D), with raw and deployed policy identical and targets defined on all reached histories,

    dJ/d epsilon at 0 = sum_t E_{s~d_t^b} Cov_{a~b}[Q_t^b(s,a),D(h(s),a)].

This covariance identity is not the finite neural update and is not an AUC identity.

## 4. What a ratio and a chance AUC identify

If post truly solves a sequential KL-regularized control problem with pre as reference, shared dynamics/information state and appropriate boundary conditions, beta log(post/pre) can represent r+E[V(next)]−V. It includes donor continuation; it is not automatically recipient advantage or bare task reward. Arbitrary SFT/RL checkpoint pairs need not satisfy these premises. A ratio can always define a constructed surrogate, which does not identify the benchmark reward. Multiple checkpoints do not constitute independent identifying environments.

A hypothetical token-local q lift has log(Q(y)/B(y))=sum_t[D_t/alpha−log Z_t]. This differs from a globally normalized sequence tilt. Optimizing the sequential surrogate sum(D−alpha log(pi/b)) can favor future log Z and loops while hurting verifier success; existing prior toys already demonstrate this. Global conversation reweighting can also tilt realized environment outcomes that the actor cannot choose. Simple Bellman correction, action marginalization, privilege or normalization alone does not solve reward identification.

The exact toy in [toys/theory_math_audit](toys/theory_math_audit/check.py) establishes a narrower fact about P1. Let b be uniform over two failing actions and one successful OTHER action, alpha=1. Separately normalized pre=(1/5,3/10,1/2), post=(1/20,9/20,1/2) give ratios (1/4,3/2,1), q=(1/11,6/11,4/11). Within-state success-ranking AUC is exactly1/2, yet the initial success derivative is log(8/3)/18>0. Same-AUC normalized controls have zero or negative derivative. An exact K16 embedding is documented. AUC discards score spacings; shared conditional features provide additional possible discrepancies, but are not needed for this logical example.

This **does not show useful hidden signal in the real panel**, explain the measured gain, provide a new algorithm, or reject the ceiling. It prevents the false implication that AUC=.5 proves every learning direction useless. L6 still rules out treating available scalar scores and one-continuation outcomes as a reliable per-state oracle. Its selected mixed-state panel and enriched retest are not population-wide signal/noise measurements.

Bucket KL omits conditional-tail drift. Exact full-token trajectory KL under shared kernels is E_{P_pi} sum_t KL(pi(.|H_t)||b(.|H_t)); environment factors cancel even when transitions are stochastic. This uses actual induced occupancy and an episode sum. Frozen-cache target bucket KL, trained-policy KL and trajectory KL are three different quantities.

## 5. Six competing dominant-bottleneck hypotheses

**C — Budgeted transferable-information ceiling.** Current safe DOPD already extracts most useful complementary signal in these fixed pairs. More movement imports donor weaknesses; noisy outcomes or reprocessing add too little independent information. This allows the observed gain and is not a claim of Bayes optimality. A toy or residual error does not reject it. A matched, reproducible extra gain within budget does.

**R — Useful conditional signal mixed with task-misaligned shifts.** D contains enough remaining useful structure, but local q mixes it with format, calibration, style or incompatible behavior. Exact fitting would retain harmful target choices. Delta-minus-reference can preserve recipient strengths without a stronger generating teacher. This requires reproducible task-aligned conditional distinctions beyond chance scalar rankings; their existence is unproved. A simple “remove format/chat polish” version is weakened by useful legacy format shifts and L9.

**F — Finite shared-parameter realization.** The attainable target is materially better on consequential contexts, but its useful structure is lost or coupled to harm by representation, exposure, optimizer or off-cache generalization. The diagnostic distinction is a better feasible target versus the realized student, not just nonzero loss. Pure insufficient updates is weakened by the appropriately scoped weak-student null. Target-side projections alone do not repair F.

**T — Recipient continuation and sequential prerequisites.** Local changes alter retained reasoning, future obligations, information use, appropriate stopping or suffix competence despite little immediate next-call effect. Harm persists with adequate token support and exact local fitting. It predicts delayed divergence and continuation dependence. It does not establish that the donor ratio supplies the correct missing continuation value.

**I — Decision-relevant information-state mistakes.** Residual errors chiefly involve absent, forgotten or misused evidence, confirmation or hidden-goal distinctions. The actor must choose based on what h warrants; hindsight truth is not available evidence. It predicts dependence on decision sufficiency and faithful memory, but hidden state or sampled label variability alone is not evidence for this cause. It loses support if most errors occur with explicit correctly used evidence and intact memory.

**S — Consequential context or candidate coverage.** Current targets and fit are adequate where represented, but missing/censored prefixes, recovery states or individually unrepresented tail choices dominate residual errors. High average Top16 mass constrains broad tail stories but does not prove rare-event adequacy. P7 confounds coverage, censoring and token exposure. More same-distribution data and generic state priority need not repair the specific gap.

R separates target quality from F's target realization; T separates sequential utility from both. I and S identify additional potential structure but have weaker direct incidence evidence. C remains live against all five. These are competing **dominant** accounts, not mutually exclusive facts; no posterior probabilities are invented from overlapping observations.

## 6. Conservative evidence scoring

E means the stated mechanism directionally explains the observation **under its unverified premise**; N means compatible/neutral; T means tension with a specified strong version. An E is not confirmation. P and L rows overlap and cannot be counted as independent evidence. The two theorists differed in several generous E assignments; this table adopts the more conservative score when a mechanism needs an additional unmeasured premise. In particular, no hypothesis earns P8 simply by permitting either length sign.

| Evidence | C | R | F | T | I | S |
|---|---:|---:|---:|---:|---:|---:|
| P1 weak scalar rank, real transfer | N | E | N | N | N | N |
| P2 delta beats post, weak rescue | E | E | N | N | N | N |
| P3 own shift null, external useful | E | E | N | N | N | N |
| P4 stronger tilt collapses | E | E | E | N | N | N |
| P5 one-continuation broadcast fails | E | E | N | N | E | N |
| P6 multi-turn harm, weak next-call effect | N | N | N | E | E | E |
| P7 uncapped better, priority near uniform | N | N | N | E | N | E |
| P8 opposite length changes | N | N | N | N | N | N |
| L1 efficiency-arm tradeoff | E | E | N | E | N | N |
| L2 empirical invariance not realized | N | N | E | N | N | E |
| L3 one-step probes weak | N | N | N | E | N | N |
| L4 own-family reapplication null | E | E | N | N | N | N |
| L5 strengthened outcome term null | E | E | N | N | E | N |
| L6 ranking/retest limits | N | E | N | N | N | N |
| L7 weak teacher rescue | E | N | N | N | N | N |
| L8 overdose and scoped longer-fit null | E | E | T* | N | N | N |
| L9 parent reference gives no gain | E | T* | N | N | N | N |
| L10 state refresh trades domains | N | N | E | N | N | E |
| L11 focused states / existing mechanisms | N | N | N | N | N | N |
| L12 MCMC versus pools, unverified report | N | N | N | N | N | N |

R's P1/P5 explanations assume nuisance or misassigned noisy supervision obscures useful conditional shifts; this must be identified, not asserted. I's outcome explanation requires information-feasible comparisons, not an assumed stochastic user. T/S's P7 explanation assumes omitted continuations/contexts matter; exposure is a competitor. F's L8 tension is only with pure underoptimization; R's L9 tension is only with simple denominator purification. L11's unrun code is not a failed experiment. P8 rejects universal length mediation but gives no controlled sign prediction. Corrected W1 and current protocol details remain binding throughout.

## 7. Falsifiers and quantitative burden

| Contrast | Result that distinguishes causes |
|---|---|
| Target versus realized supported odds, with valid recipient continuation | Helpful q but harmful fit favors F; harmful q faithfully realized favors R/T. Cache loss alone cannot decide. |
| Matched current action, full retained history and later behavior | Delayed compatible-format failure favors T; immediate control corruption favors R; equality of JSON is insufficient matching. |
| Decision-sufficient versus genuinely confusable histories | Evidence/confirmation dependence favors I; failure despite sufficient correctly used evidence weakens it. Privileged success alone does not count. |
| Prefix/candidate coverage separately from exposure and censoring | Benefit specifically from previously missing consequential distinctions favors S; repeated exposure gains favor F. |
| Reproducible accessible conditional delta signal beyond current recipe | Weakens C's information component; only learnable matched-budget gains reject its full budgeted form. |

These are separating thought experiments, not a diagnostic-track proposal or authorized model runs. A future candidate should derive its target from a stated distinction and supply a decisive intervention, rather than merely request more logging.

Current mean success is .805799, residual failure .194201. If f is the fraction of current failures repaired and g the fraction of current successes broken, another2pp requires

    .194201 f - .805799 g >= .02.

Thus f>=10.30% at g=0 and f>=14.45% at g=1%. It is about55 additional successes per2736-rollout sweep. Restricting benefit to73 inconsistently solved tasks needs +3.123pp within that group; only20 hardest needs +11.4pp. Recovering every observed regression on41 base-perfect tasks yields only .256 aggregate pp. All114 baseline tasks have at least one observed success; no never-solved class is present in this artifact. This is reliability headroom, not proof that every prerequisite is possessed or every error preventable. Baseline stratification has regression-to-mean bias.

Three-model task-bootstrap intervals omit training-population uncertainty. A small positive point estimate is not validation. A finalist must identify enough preventable error mass, preserve success, use faithful history and equal information/compute controls, and state the assumptions under which its signal can be learned in at most about8 students. Retrodiction of five puzzles cannot be manufactured by changing the hypothesis after each row. No single root cause or proposal is selected at the end of this phase.


### Statistical qualification added after independent audit
The historical ~±1.6pp is a **paired difference** interval, not the per-model task interval (roughly ±5pp). Three observed seed-run scores cannot separate training variation from rollout noise. Four controls plus four candidates maximizes replication for one balanced contrast, but does not guarantee decisive +2pp power; a +1pp effect is generally harder. A future result can be unresolved rather than a forced success/failure, and superiority to zero is distinct from proving a lower bound above2pp. See [uncertainty audit](GROUND_TRUTH/uncertainty_design.md). Candidate effect estimates must remain hypotheses.
