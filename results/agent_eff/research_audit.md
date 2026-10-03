# Agent efficiency research audit October 2026

The replicated DECS accuracy cost is real in the completed BFCL outputs. I found no demonstrated sign, alpha, token alignment, or probability normalization error that explains it. The strongest current explanation is a mismatch between the quantity being transferred—local shortening preferences at frozen response prefixes—and the quantity being evaluated—successful tool execution over an interaction. This is a research diagnosis, not a proven causal explanation.

A useful new lead is that the earliest scored failure moves into the **first user turn**, where a task can already require several tool steps. Across all 800 multi-turn entries and the six paired core runs, first-turn failures contribute +1.50 points to the net +2.375-point failure increase. The previous “protect turn starts” treatment explicitly leaves initial user turns unprotected. Of that +1.50-point bin increase, +1.083 points come from net newly lost tasks and +0.417 from already failed tasks moving to an earlier failure. Initial-turn newly lost tasks therefore account for 45.6% of the total net loss, rather than 63%. This finding motivates replaying initial planning and recovery decisions before another training sweep; it does not yet establish their cause.

The audit also identified a real retry-log attribution bug and several artifact provenance weaknesses. They need repair, but the independently checked final accuracy and generated-token totals survive them.

Audit date: October 1, 2026. Audited checkout: `739eeffdbc1d2c99ea438debd0f1f1a5e3d4cf1b`, including the existing local concurrency changes in `configs/bfcl_eval/config.py` and `modal_eval.py`. Those changes were preserved. No training, evaluation, deployment, or paid GPU experiment was launched. This report and its CPU analysis artifacts are new; existing result reports and experimental code were not rewritten.

## Evidence and scope

The review covered both result reports, the experiment registry and synthesis notes, LoopTool preprocessing and prompt selection, frozen rollout collection, donor token audits and cross-tokenizer projection, score precomputation and numerical checks, synthetic and legacy target composition, sealed-cache validation, the executed Direct-OPD loss and Megatron routing/reduction, training and export provenance, BFCL serving and request rewriting, upstream pinned driver and checker, result aggregation/bootstrap/cost attribution, failure classification, and the existing Monte Carlo and GRPO/second-benchmark scaffolding. Unrelated math/code tasks and generic Slime features were not exhaustively audited.

The exact pinned NeMo-Skills BFCL driver was fetched and compared with the retained local copy; they were byte-identical. The exact pinned gorilla checker was read, including its per-turn state and response checks. I inspected locally retained raw runs rather than relying only on Markdown summaries.

The independent CPU recomputation checked **130 old and new runs**, including all **66 new runs**. Each contains 4,441 final entries across 17 categories. It found no duplicate final IDs, no cross-run entry-ID discrepancies, no discrepancy between raw weighted correctness and official aggregate/category score headers, and no discrepancy between per-step token sums and final generated-token totals. This establishes internal agreement among the retained artifacts; it does not replace a fresh execution of the grader or a GPU parity test.

Validation used existing CPU regression tests: 201 passed and one distributed-socket test skipped across the training/scoring/composition/configuration suites; 49 passed across the harness, evaluation metrics, failure-classification, and tau configuration suites. The CPU environment lacks Megatron, so native GPU forward/backward, context-parallel integration, checkpoint-conversion parity, and full distributed training were not rerun. Some retained training logs only record launches; final training curves could not be recovered.

The accompanying `research_audit_recompute.py` uses independently written BFCL weights and reads final outputs directly; `research_audit_usage.py` reproduces the invalid usage joins. `research_audit_evidence.json` stores independent aggregates, cluster sensitivity, failure-time transitions, and the positive base comparison. `research_audit_usage_evidence.json` records the observed usage inconsistencies. `research_audit_target_diagnostics.json` stores the inspected 800-row target diagnostic sample. Raw generations and model weights were not duplicated into the repository.

## What the replication establishes

The core comparison uses two independently trained checkpoints per treatment and three decoding seeds, paired within training and decoding seed. It is six evaluation pairs, not six independent training runs.

| Core comparison | New interleaved protocol | Old protocol |
| --- | --- | --- |
| DECS minus accuracy-only overall accuracy | −0.818 points | −1.080 points |
| DECS minus accuracy-only multi-turn accuracy | −2.375 points | −2.896 points |
| DECS generated tokens | −14.794% | −16.342% |
| DECS multi-turn accuracy by training seed | −2.583 / −2.167 points | −1.875 / −3.917 points |

The difference in multi-turn cost is +0.521 points. The original entry bootstrap interval includes zero and substantial improvement. Thus the careful conclusion is: **the harness defect was not the sole cause, and the cost persists after its correction**. The experiment does not establish that the harness correction had exactly no interaction with efficiency training.

The interleaved-thinking correction is well supported by the implementation, native-template parity test, prompt-growth diagnostic, unchanged thinking-off control, and complete final outputs. The correction reduces repeated reasoning, with about 20% fewer generated tokens for the trained thinking students and about 26% fewer on multi-turn. This is a useful efficiency gain from context handling. It is distinct from evidence that the learned donor composition improves the multi-turn accuracy–cost frontier.

Accuracy-only remains a successful capability transfer: the reported original OPD student gains +3.05 overall over base under the new harness, predominantly through multi-turn performance. Failure of the efficiency extension does not mean OPD itself failed.

There is also a narrower positive result for the composed student that already holds. Across the two DECS checkpoints and three matched decoding seeds, DECS mid versus Qwen3-4B base gains **+1.682 overall points [ +1.020, +2.336 ]**, gains **+4.833 multi-turn points [ +3.103, +6.542 ]**, and saves **9.188% of generated tokens [ 7.153, 11.236 ]**. These task-bootstrap intervals condition on the evaluated models. This is accuracy and token dominance over the starting model; it is not evidence that composition improves upon accuracy-only distillation at matched cost.

### Statistical qualifications

The bootstrap in `evaluation/bfcl_pooled.py:118` shares each category's resampled entries across runs, correctly preserving paired treatment comparisons and common-reference dependence. It freezes checkpoints and decoding replicates. Its confidence intervals therefore quantify task-sampling uncertainty conditional on these models; they do not estimate variation across newly trained checkpoints. Two training seeds cannot support a strong population-level claim about all training runs.

There is an additional task-family dependence. The four multi-turn categories reuse the underlying 200 task families with changed missing-function, missing-parameter, and long-context conditions. In inspected base comparisons, initial configurations match in 193–194 of 200 cases and first-user messages in 124–195 of 200, depending on the variant. Independent resampling within each category discards that cross-category dependence.

A sensitivity analysis resampling the common task suffix jointly across the four categories, with 2,000 draws and seed 0, gives:

| Quantity | Task-family clustered 95% interval |
| --- | --- |
| New overall DECS accuracy change | [−1.340, −0.260] points |
| New multi-turn DECS accuracy change | [−3.854, −0.875] points |
| New generated-token change | [−15.912, −13.681]% |
| Change in multi-turn cost between protocols | [−1.292, +2.334] points |

The substantive DECS cost survives this wider interval. Future inference should keep variants of one underlying task together and should separately report training-seed variation. The suffix grouping is a supported sensitivity model, not a claim that every variant has identical difficulty.

Averaging the three reference decoding seeds for single-seed exploratory arms is a reasonable noise-reduction choice when applied consistently to both protocols. Those comparisons have a different estimand from fully matched seed pairs and should be labeled accordingly. The fully paired six-run core is the strongest reference.

The reported through-origin accuracy-per-token fit is descriptive. Shared baselines, correlated task variants, noisy realized costs, an origin constraint, heterogeneous donors, and many selected arms prevent interpreting it as a fundamental Pareto frontier. Similar point slopes are not an equivalence test. Allowing an intercept instead of forcing the fit through zero changes the slopes from 0.05896/0.04751 to 0.04175/0.04106 points per percent saved; the corresponding intercepts are −0.2746/−0.1198 points. This sensitivity reinforces the descriptive interpretation. Likewise, the largest of 30 noisy residuals needs selection-aware validation; its nominal interval is not a confirmatory interval for the selected winner. L1-Max is unconfirmed, rather than mathematically disproven.

Several summary phrases should be narrowed in a future revision of the reports. “Single-turn is free on every arm” means little or no measured average loss in this matrix; it does not establish non-inferiority for every arm and slice. “Every efficiency arm loses 1–6 multi-turn points” is literally contradicted by the new L1-Max point estimate of +1.00. “Training length is not the lever” means the tested schedules did not improve aggregate BFCL; it does not demonstrate that rare decision targets were fitted. These qualifications do not erase the replicated core cost.

Early `results.md` tables use leaderboard-weighted token means, while later tables use empirical total tokens divided by 4,441 entries. For the old accuracy checkpoint these are approximately 3,146 and 1,966 tokens per entry. They are two different summaries, not a change in the checkpoint. Both need explicit labels. The replicated total-token comparison uses raw summed generated tokens and is unaffected by this naming problem.

## Where the failures occur

The earliest official failure decomposition across all 800 multi-turn entries is:

| Earliest scored failure bin | Net increase in failure rate for DECS |
| --- | --- |
| First user turn | +1.500 points |
| Second user turn | +0.438 points |
| Third user turn | +0.188 points |
| Fourth or later user turn | +0.396 points |
| Context overflow | −0.146 points |
| Total | +2.375 points |

The first user turn accounts for 63.2% of this net failure-bin increase. The original common non-overflowing 746-entry subset gives 58.3%; restoring equal weighting of the four categories on that subset gives 56.9%. The early concentration is robust to those choices. The full 800-entry analysis is preferable for the primary comparison because excluding overflows conditions on a treatment-affected outcome.

The checker loops through user turns, appends execution results, and returns at the first failed state/response check. Thus the inferred turn is a genuine earliest **scored** failure. It does not identify the first wrong token, the first wrong tool step, or a causal pivot. Shifting a task that both models fail from a later failure to an earlier failure changes the bins without changing task accuracy. Across 4,800 paired entry evaluations, 110 accuracy-only successes become DECS first-turn failures, while 58 accuracy-only first-turn failures become DECS successes: net +1.083 points. Among tasks both models fail, 117 move from a later/overflow failure to the first turn and 97 move the other way: net +0.417 points. Thus 45.6% of the net newly lost task count is descriptively in the initial turn, while the 63.2% bin fraction also includes earlier failure of existing unsuccessful tasks. A causal claim needs intervention and continuation from the same environment state.

The multi-turn category changes, pooled over the six core pairs, are:

| Category | New DECS accuracy change | Old change |
| --- | --- | --- |
| Base multi-turn execution | −5.750 points | −2.583 points |
| Missing function | −1.917 points | −3.250 points |
| Missing parameter | +0.833 points | −3.083 points |
| Long context | −2.667 points | −2.667 points |

These exploratory category estimates suggest the new dominant deficit is ordinary initial execution and planning, rather than solely missing-tool detection or rebuilding history after later user messages.

In a repeated rename-task example, both models initially receive a file-not-found error. Accuracy-only changes to the workspace directory and retries the move. DECS stops and asks the user to check the location. That difference recurs in four of six pairs and appears in related task variants. In another repeated example, DECS creates the requested backup but also creates an unintended directory in the wrong location, which fails the state check. These examples are consistent with deficient verification/recovery or planning. They are selected examples, not an estimate of their causal prevalence, and neither is an unfinished-thinking artifact.

### What the previous protection treatments tested

`data_curation/turn_positions.py:54` labels a user message with earlier assistant history `turn_start`; a first user message is `first_turn`. Line 72 protects only `turn_start`. `configs/agent_eff/config.py:450` registers that exact rule.

The selected cache contains 869 `single_turn:first_turn`, 1,934 `multi_turn:turn_start`, and 397 `multi_turn:after_tool` prompts. There is no `multi_turn:first_turn` stratum: conversation kind describes the supplied history, not the number of tool steps or user turns the next task will require.

Consequently, the turn-start treatments do **not** rule out damage to initial planning. The earlier `protect-first` treatment protects a reflection fork, which is also different from preserving the initial complete interaction turn. A new treatment protecting a validated initial action/plan would answer a different question. However, the current cache does not execute the generated action or identify its future horizon, so replay is a better first test than another cache gate.

## Confirmed bugs and reproducibility gaps

| Finding | Evidence | Consequence for current results |
| --- | --- | --- |
| Retry usage can be assigned to the wrong retained trajectory | `evaluation/bfcl_efficiency.py:243` uses unordered token-count counters and greedily chooses the earliest matching request | Observed across 16 of the 66 new runs, including the restarted core run. Prompt/reasoning/length-stop diagnostics can be wrong; final accuracy and generated-token totals are independently intact |
| Checkpoint identity omits weight bytes | `configs/bfcl_eval/modal_eval.py:762` hashes config files and safetensor filenames/sizes | A same-size changed checkpoint can collide and reuse results. Reproduced on temporary files; no evidence that a checkpoint was substituted in these runs |
| Legacy composer accepts misaligned rows and unequal stream lengths | `data_curation/build_direct_opd_composed_target.py:108` and `:173`; weight/source zip at `:61` | Proven latent bug. Current agent-eff arms use the safer synthetic pipeline, so this does not show current score corruption |
| Synthetic target reuse incompletely locks donor evidence | `configs/agent_eff/modal_pipeline.py:504`; `build_synthetic_shift_target.py:428` | Existing targets may be reused after donor score/evidence changes at the same paths. No demonstrated stale target in the core comparison |
| Full finite-score validation is incomplete | `data_curation/verify_donor_scores.py:133` restricts mapped finite/evidence reporting to the first 2,000 rows | Alignment is checked broadly, but this is not a comprehensive numerical validation of every synthetic shift |

The usage bug is more than a hypothetical ambiguity. Across all 66 new runs, 16 have 111 assigned user turns with impossible chronological request histories even though every multi-turn entry is labeled reconciled. It also occurs in the thinking-off control, so indistinguishable task prefixes and coincident token counts can cause ambiguity without a restart. In the restarted core run alone, eight user turns have impossible assigned histories. For one retained turn the assigned requests have message counts `[16,13]`; another has `[3,1,5]`. A genuine within-turn history grows as tool steps are appended. Counter reconciliation alone cannot determine which attempt was retained when different attempts produce the same token counts.

A synthetic reproduction makes the problem explicit: a discarded request generates five tokens with ten prompt tokens; the retained attempt generates five then six tokens with twenty and thirty prompt tokens. The current join can select the discarded five-token request and retained six-token request, report complete reconciliation, and attribute forty prompt tokens instead of fifty.

Future logs need entry ID, episode-attempt ID, turn/step index, request ID, request/response digest, and a final-output link to the retained request IDs. Until then, usage-dependent statistics should exclude or mark ambiguous attempts rather than infer their lineage from lengths. This includes some prompt-cost, reasoning-cost, cap-hit, and request-coverage diagnostics. It does not require throwing away valid final scores or final completion-token totals.

The checkpoint identity should use actual weight hashes or an immutable content-addressed export manifest, and it should be versioned so old result manifests are not silently treated as new identities. Target revisions should include the base/scored-shard checksums, tokenizer and projection identities, token-evidence audit identities, score dtype/backend and scorer version. Training reuse should compare that complete identity. The legacy composer should reject mismatched sample IDs/candidates/prefixes, unequal streams, and unequal numbers of weights and sources.

## Harness issues that remain relevant

The native Qwen3 template retains earlier reasoning between tool calls in the current user turn and drops reasoning from earlier user turns. The proxy's rewrite matches this template in the existing parity test. Its reported 99.6–99.9% coverage is nevertheless a **prompt-growth proxy**: prompt growth greater than the previous reasoning length does not directly prove token-for-token inclusion, because tool output and other content also grow the prompt. Code inspection and template parity are stronger evidence of the fix than this percentage alone.

The 32,768-token requested completion budget inside a 65,536-token window rejects prompts above roughly 32,768 tokens, even when a normal next reply would fit. This is an acknowledged protocol choice in `configs/bfcl_eval/config.py:143`. Overflows are fairly stable across treatments and their net change slightly favors DECS, so they do not explain the current accuracy cost. They still censor difficult tasks and confound the interpretation of a 4k completion-cap baseline: a lower cap both limits generation and permits longer prompts. An adaptive completion budget should be tested in its own protocol tree, with the original protocol retained for comparability.

The raw six-pair outputs contain 28 failed DECS multi-turn entries with an unclosed visible `<think>` block, compared with 27 for accuracy-only. Even the generous counterfactual of repairing every affected failed DECS entry to success could improve its mean multi-turn score by at most `28/(6*800)*100 = 0.583` points. Some such entries already have other errors, and several malformed final replies are in successful entries. This behavior therefore cannot explain the full −2.375-point cost, but “it has no effect on scores” is stronger than the evidence supports. It should be tied to missing-action versus already-complete turns and isolated by replay.

The pinned checker primarily grades tool execution state and requires the ground-truth execution responses to be contained in the accumulated model responses. It does not generally grade the usefulness of the final prose reply. This explains how a malformed final reply can coexist with success. It also means that an alternate strategy that reaches the goal while skipping a reference query may still fail. The observed frontier is a frontier for this BFCL protocol and its trace requirements, not automatically for all real agent tasks. Keep official scoring primary and add goal/state-only diagnostics where their semantics are valid.

A fixed sampling seed does not guarantee identical online outputs under different batching or hardware. The thinking-off control's small mismatch is consistent with that limitation, rather than evidence that the reasoning rewrite changes its requests. vLLM's official reproducibility guidance explicitly requires additional deterministic/batch-invariant settings and still restricts reproducibility to matching hardware and versions. Scheduling and H100/H200 fallback should therefore be recorded as possible nuisance variability, even if excluded from the task protocol hash. [vLLM reproducibility documentation](https://docs.vllm.ai/en/v0.22.0/usage/reproducibility/)

## What the training loss actually optimizes

The executed path in `configs/agent_eff/modal_pipeline.py:614`, through `configs/lightning_weave/train.py:126`, selects `slime.rollout.offline_direct_opd.megatron_loss` with the `tilted_target` mode and per-token reduction. In `slime/rollout/offline_direct_opd.py:898`, the bucket target and training loss are:

\[
q_i(s)=\frac{b_i(s)\exp[\delta_i(s)/\alpha]}{\sum_j b_j(s)\exp[\delta_j(s)/\alpha]},
\qquad
L_s=\frac{\alpha}{2m_s}\sum_i b_i(s)[\log p_i(s)-\log q_i(s)]^2,
\]

where the named candidates are the frozen student's Top-16, the remaining vocabulary is one `other` bucket, `delta_other=0`, and `m_s` is cached candidate mass. Alpha is 2.0 in the reviewed arms.

This is **squared log-probability fitting**, not direct minimization of forward KL or reverse KL. The forward/reverse KL and TV fields are diagnostics. Its initialization gradient matches the official Top-K immediate-reward surrogate and it has the intended zero-loss bucket target. Those properties are useful and pass the reviewed CPU tests. They do not make its later optimization equivalent to a sequence OPD gradient, a KL loss, or an interaction-return objective.

The probability normalization uses the full action vocabulary, then collapses the tail. Token offsets align each response action with the preceding logit. Masked-token reduction, CP=1 slicing, sealed shard checks, candidate bounds, and prompt-prefix identity checks were coherent in the reviewed path. No evidence supports blaming an accidental reversed shift, incorrect alpha division, padded vocabulary normalization, or one-token scoring offset.

Three approximations matter after this implementation check:

1. The state distribution is frozen. Each cached row is one assistant continuation from a source LoopTool history. Its generated calls are not executed; no subsequent student-produced observation enters that row. Histories include some source mistakes/error observations, but not the current trained policy's full failure-state distribution.
2. The target uses donor log ratios, with no task success, future call cost, future token cost, or recovery value. A sharp donor preference is not a measured agent advantage.
3. Shared transformer parameters must fit many local targets. The squared-log loss uses behavior weights, whereas the local curvature of KL fitting uses target weights. Agreement at initialization and at an unconstrained fixed point does not establish equivalent finite-capacity training.

Near the target, relative weighting between squared-log and KL geometry is approximately `b_i/q_i`. A candidate suppressed 100-fold by the target remains about 100-fold more important to the squared-log metric than to the analogous KL metric. Conversely, a newly valuable low-behavior-probability candidate may receive weak fitting weight. This is a reason to measure fitting error before attributing everything to donor quality.

## Target diagnostics and hypotheses rejected by evidence

I inspected a locally retained real 800-row shard, totaling 348,574 response tokens. It is a diagnostic sample, not a representative whole-cache estimate. DECS used its exact calibrated coefficient; E1 checks used approximately reported coefficients.

The cached candidate distribution is highly concentrated: mean Top-16 mass is about 0.999871, and Top-1 exceeds 0.99 at 74.4% of the inspected positions. The support-normalization multiplier is close to one at almost all positions. Only five of 800 responses hit the response cap, and four responses are distinct for 199 of 200 prompt groups. Broad truncation, duplicate generation, and widespread `1/support_mass` amplification are therefore weak explanations in this sample.

A mathematically real concern is that setting the tail reward to zero makes a statewise offset applied only to cached candidates act as a candidate-versus-tail preference. With `b=(.4,.4,.2)` and alpha 2, changing named rewards from `(0,0)` to `(2,2)` changes the target to roughly `(.4579,.4579,.0842)`. Centering and then releveling restores the original offsets; it does not remove that effect.

However, the actual DECS added-target KL decomposes almost entirely into rearranging named candidate choices: only **0.033%** of its added KL in this shard changes candidate-versus-tail mass. E1 sample estimates are also under 0.1%. Thus this offset issue is worth a sensitivity control, but it is not supported as the dominant explanation for the current DECS tradeoff.

The anonymous tail still hides individual actions. For full distributions,

\[
D_{KL}(p\Vert q)=D_{KL}(p_{K+1}\Vert q_{K+1})+
p_{other}D_{KL}(p(\cdot\mid other)\Vert q(\cdot\mid other)).
\]

A numerical counterexample gave essentially zero bucket loss while the full forward KL was 0.3586 after changing the tail's internal allocation. Tail mass preservation does not teach a rare recovery mode. This is particularly relevant to a complete tool action, whose probability depends on several successive token choices and visited prefixes.

The report already finds that explicit `</think>` is absent from the cached candidates at 99.70% of thinking positions; where it is present at the actual end, it is almost deterministic. Most added efficiency KL lies in ordinary reasoning and reflection choices rather than direct stopping decisions. This explains why more stop-token gates may accomplish little: the transfer changes the reasoning path that leads to stopping. Expanding support can test a limitation, but does not by itself supply a safe stop decision.

A small average KL is not a small change at every decision. Existing diagnostics put about 41–46% of added KL in 1% of positions, with reported per-position maxima of 8–9 nats. In the inspected DECS shard, reflection-fork mean added KL is about 0.836 despite global mean 0.0145. Uniform calibration therefore does not control risk at consequential states.

The precision audit compares FP32 arithmetic using the same BF16-rounded weights; it does not test original full-precision checkpoint rounding. Reported final-target disagreement is approximately 4% of the added KL for DECS, 11% for E1-Code, and 21% for E1-Math. Arithmetic sensitivity is a credible secondary concern for subtle E1 conclusions, but cannot simply explain every donor/gate/strength following a similar empirical tradeoff.

Final target fit by state type, decision horizon, and current-policy occupancy remains unknown. More updates failing to improve BFCL is insufficient to distinguish a poor target from inability to fit useful rare targets. This is one of the cheapest important gaps to close by teacher-forced checkpoint scoring.

## Diagnosis ranked by current evidence

| Explanation | Assessment | What would resolve it |
| --- | --- | --- |
| Compression removes useful initial execution, checking, or recovery behavior | Leading behavioral hypothesis. Repeated error-recovery examples, initial-turn losses, and reasoning-path KL support it; causality is untested | Shared-prefix turn/action/recovery replay |
| Local foreign-domain ratios do not encode agent success and remaining cost | Certain structural limitation; likely relevant to the transfer failure | Outcome-calibrated branch targets and a matched agent anchor |
| Frozen source/base occupancy misses states visited by the trained policy | Plausible and unmeasured after adaptation | Score/compare common prefixes, then a budget-controlled cache refresh |
| Student fails to fit consequential targets or fits them with poor squared-log geometry | Open. Aggregate schedules and pooled loss do not establish rare-state fit | Teacher-forced state-specific fit and same-target loss comparison |
| Sparse candidates omit valuable corrective modes | Proven information limitation, with unknown contribution here | Verified action coverage and controlled support expansion |
| Cross-tokenizer and BF16 score uncertainty distort fine donor signals | Plausible secondary concern, especially E1; protected action states did not restore accuracy | Precision/support audits on outcome-changing states |
| Candidate-versus-tail state offsets dominate the DECS loss | Weak explanation in the inspected shard: 0.033% of added KL | Tail-baseline sensitivity as a control |
| Missing within-turn reasoning, overflow, duplicate samples, or a sign/indexing error explain the whole cost | Poorly supported by corrected outputs, numerical checks, and tests | Retain protocol/parity checks; avoid another broad rerun absent new evidence |

This ranking separates facts about the objective from hypotheses about why a particular task fails. Several mechanisms can coexist. The present evidence is especially compatible with transferring a shortening style without transferring a task-aware allocation of computation.

## First principles explanation

In a single-step decision with a shared reference and a valid reward signal, an exponential tilt is a principled policy-improvement target. Interactive decisions additionally determine later observations, useful checking opportunities, failure states, and recovery cost. A policy can imitate a shorter local continuation while reducing the chance of completing the task.

If the local target is `q(a|h) proportional to b(a|h) exp(r(h,a)/beta)`, then along a generated trajectory:

\[
\log\frac{q(\tau)}{b(\tau)}=
\sum_t r(h_t,a_t)/\beta-\sum_t\log Z(h_t).
\]

The normalizers depend on the visited histories. A globally tilted complete-trajectory objective instead has a single trajectory partition. In deterministic or replay-fixed environments, conditioning that target on a current action introduces a future partition/value term. More generally, with fixed stochastic environment dynamics, KL-regularized policy improvement uses a soft Bellman value:

\[
q^*(a|h)\propto b(a|h)
\exp\{[r(h,a)+\mathbb E V(h')]/\beta\}.
\]

The missing quantity is continuation value. It cannot generally be recovered by increasing the coefficient of a token-local shortening ratio. Donor ratios can already contain policy-specific future advantages under the donor's original training objective, but those future policies and values change when capabilities are combined or transferred to tool tasks.

This is the same structural issue addressed by entropic policy composition with a future divergence correction. That theory applies under stronger shared-dynamics/reference/optimality assumptions than arbitrary LLM anchors satisfy; it is a useful precedent, not a proof of the present cause. [Composing Entropic Policies using Divergence Correction](https://proceedings.mlr.press/v97/hunt19a/hunt19a.pdf)

Small local errors can also produce visible task losses without one catastrophic category. For illustration, reducing each of 20 independent decision success probabilities from .970 to .968 reduces their joint success by roughly 2.2 points. Real agent decisions are dependent and have unequal importance; the example only shows that diffuse small errors are compatible with a multi-turn task-level cost.

Finally, reasoning and actions are coupled. A turn factors as `pi(z,a|h)=pi(z|h) pi(a|h,z)`. Changing the reasoning `z` changes the context used to generate the action `a`. Leaving tool-token loss untouched, zeroing the donor term at a boundary, or protecting a training prompt does not constrain the action marginal after the reasoning changes. The protocol-protection experiment successfully removing extra calls without restoring accuracy is consistent with this distinction.

## Literature findings

The sources below were reviewed as primary research, with methods/limitations rather than only abstracts. Recent agent papers help identify mechanisms; they do not establish those mechanisms in these runs.

| Source | Useful connection to this audit |
| --- | --- |
| [Lightning Weave](https://arxiv.org/html/2609.14708) | Its guarantee concerns fixed cached states and initialization of a zero-discount token surrogate. It explicitly separates those guarantees from later state distributions and lists agentic/tool-use extension as future work. The present failure does not contradict that scoped result. |
| [Direct-OPD](https://arxiv.org/html/2607.05394) | Immediate post/pre token ratios can transfer useful shifts, but increasing the transferred shift does not guarantee validation improvement; unreliable long prefixes matter. |
| [S²D-OPD](https://arxiv.org/html/2609.29142) | A post/pre ratio can remain unchanged as both donors' absolute mass on the student's candidates tends to zero. Donor-pair divergence then disappears. This motivates support and divergence audits, rather than equating ratio magnitude with reliable capability evidence. |
| [Revisiting OPD](https://arxiv.org/html/2603.25562) | Distinguishes token-local surrogates from sequence reverse-KL gradients and discusses prefix and special-token mismatch. |
| [TurnOPD](https://arxiv.org/html/2607.05804) | Finds uneven supervision across turns and late-turn loss that can lose outcome discrimination. Measure loss by turn before blindly balancing or upweighting it. |
| [Guided-OPD](https://arxiv.org/html/2606.15912) | Intervenes at complete-turn boundaries and anneals teacher intervention, providing access to useful states rarely visited by the student. |
| [Multi-Turn OPD with Prefix Replay](https://arxiv.org/abs/2607.04763) | Treats prefix design as a balance between student occupancy and teacher reliability. More student-like histories can be less reliable teacher-query states. |
| [STRIDE](https://arxiv.org/abs/2609.14636) | Uses teacher endorsement to stop unreliable rollouts and restart from useful prefixes. Its tau evidence links loss of endorsement to the first error, motivating error-conditioned audits. |
| [SPOT](https://arxiv.org/html/2608.04419) | Uses verifier-scored student continuations to calibrate teacher-proposed branches. This is close to the missing downstream-value experiment; its principal evidence is mathematical reasoning. |
| [ATOD](https://arxiv.org/html/2606.27814) | Combines teacher guidance with outcome RL to address imitation ceilings and task misalignment. |
| [Entropy-Aware OPD](https://arxiv.org/html/2603.07079) | Selective teacher-supported forward supervision can expand coverage, but entropy alone does not determine recovery value. |
| [CRISP](https://arxiv.org/html/2603.05433) | Same-family concision-conditioned teachers are a useful compression baseline. Its iterative forward-KL issues should not be generalized to verified recovery demonstrations. |
| [Does OPD Really Distill](https://arxiv.org/html/2608.31046) | Teacher-free low-probability token suppression can rival sampled-token OPD in its setting. This motivates sharpening/concision placebos; it does not invalidate the analytic K+1 target used here. |
| [Prune-OPD](https://arxiv.org/html/2605.07804) | Truncating incompatible suffixes can improve training efficiency, but can remove recoverable states and does not supply an outcome signal. |
| [Position Bias and IW-OPD](https://arxiv.org/html/2606.22600) | Position-dependent discrepancy motivates drift diagnostics; practical stabilized weights are not exact occupancy correction. |
| [Interpolated Policy Distillation](https://arxiv.org/html/2609.37170) | Explicit teacher/student mixtures offer another coverage tradeoff. Token interpolation inside a tool action needs separate validation from complete-turn intervention. |
| [Canonical-Context OPD](https://arxiv.org/abs/2605.30251) | Self-generated earlier assumptions can distort later evidence use. Its gradually revealed evidence tasks differ from executable tool-state recovery. |
| [Structured Agent Distillation](https://arxiv.org/html/2505.13820) | Separate reasoning/action supervision is a useful baseline; disjoint token masks do not generally imply orthogonal parameter gradients in a shared transformer. |
| [The Danger of Overthinking](https://arxiv.org/html/2502.08235) | Agent inefficiency includes analysis paralysis, bad actions, and premature disengagement. Better interaction decisions may matter more than shorter prose. |
| [AgentDiet](https://arxiv.org/html/2509.23586) | Context compression is an independent efficiency route when useful generated deliberation needs preservation. |

Older principles remain relevant. DAgger makes visited-state coverage central; AggreVaTe supplies cost-to-go supervision; policy-distillation analyses distinguish objectives with superficially similar updates; generalized policy improvement evaluates proposals under the target reward. These support a task-value and occupancy approach rather than assuming arbitrary checkpoint differences recover portable rewards. [DAgger](https://arxiv.org/abs/1011.0686), [AggreVaTe](https://arxiv.org/abs/1406.5979), [Distilling Policy Distillation](https://proceedings.mlr.press/v89/czarnecki19a.html), [Successor Features](https://proceedings.neurips.cc/paper/2017/file/350db081a661525235354dd3e19b8c05-Paper.pdf)

### Critical reading of PivotOPD

The supplied paper provides useful counterfactual diagnostics and explicit generation of recovery continuations from post-mistake states. That mechanism is relevant to the observed file-error example, but pivotal mistakes have not yet been causally identified in these BFCL runs. [PivotOPD](https://arxiv.org/html/2609.40285)

Its implemented recovery PPO signal is teacher-sampled and weighted by clipped log probability ratios. For categorical actions, the update at the old policy has the form `g_a=q_a*phi_a−p_a*E_q(phi)`, while exact forward-KL descent is `q_a−p_a`. Equality of a frozen surrogate's scalar value with forward KL at initialization does not imply equal gradients. Use its recovery-state sampling idea; if the intended loss is forward supervision, implement ordinary cross-entropy or actual forward KL and call the objective accurately. Its multi-candidate, within-one-turn detection metric also does not establish pivot-label precision.

## Recommended research sequence

The next work should distinguish target quality, support, target fitting, and occupancy. A larger donor sweep cannot do that. The following stages are proposed experiments, not completed experiments.

### Recover trustworthy diagnostics

Repair future log lineage and content identities. Recompute or exclude ambiguous usage-dependent observations. Retain all final entries in the primary accuracy/generated-token comparison. Label empirical versus leaderboard-weighted costs. Use task-family clustered confidence intervals and report per-training-seed outcomes separately.

Teacher-force a fixed sample of initial prompts, post-tool prefixes, reflection choices, action boundaries, and failed-task prefixes through accuracy-only and DECS checkpoints. Measure target KL/TV, candidate mass, full/sampled reference divergence, action-sequence coverage, and fit against the intended target by state class. Reuse existing checkpoints; no retraining is needed. Compare the cached base-policy states with states visited by the trained models.

Compute donor pre/post mass on the common student support and donor-pair coarse JSD alongside ratio magnitude. Examine whether strong pushes coincide with little absolute donor evidence or with candidate actions the student never executes. This tests a different question from the already tried reflection/protocol/difficulty gates. Checkpoint precision and tokenizer projection sensitivity should be concentrated on the decisions that change outcomes, not only random rows.

### Run counterfactual replay on initial execution and recovery

Use a disjoint diagnostic task split, selected before reading intervention outcomes. Sample initial-turn losses, later losses, both-model successes, and both-model failures; do not select only memorable examples. Clone or replay identical environment states with isolated identities so branches cannot share mutable execution state.

At each selected state, compare:

| Intervention | What it distinguishes |
| --- | --- |
| Accuracy-only complete initial turn, then DECS continuation | Whether an initial planning/action replacement restores performance |
| DECS reasoning with a validated action substituted | Whether action choice is the immediate deficit |
| Concise reasoning conditioned on the same validated action, then unhinted student continuation | Whether safe compression exists at that state |
| Keep the mistake, then provide one or two verified recovery turns | Whether missed recovery coverage is a useful training target |
| Accuracy-only and DECS on the same post-tool prefix | Within-turn policy difference without different earlier occupancy |

Use multiple continuation seeds, the official grader, complete cost-to-go, and a goal/state-only diagnostic where valid. Keep failed and truncated branches in unconditional cost accounting. Report first-turn success, whole-task success, recovery probability, repeated calls, and remaining tokens/calls separately. A swapped turn should use the correct current observations; replaying a gold action against an incompatible state is invalid. Keep later user instructions out of the student prompt and verify that supervised actions are feasible from the current evidence.

This study can reveal that concise policies stop checking, that action candidates are missing, that both donors are unreliable, or that student fitting is poor. Each outcome has a different remedy. It also prevents attributing generic earlier failure-bin movement to initial underthinking without causal evidence.

The current `data_curation/mc_advantage_probes.py:184` completes a single forced-token assistant response and scores call structure against a LoopTool target. It is useful for local compression validation but does not execute the call and measure future episode success/cost. It must be extended for this question. The existing BFCL GRPO adapter is also one user-turn, base-category scaffolding with a one-tool-per-step prompt and truncated observations; it is not already a matched full-conversation efficiency anchor.

### Change the target using measured continuation value

For complete candidate turns `y` at history `h`, estimate success and remaining operational cost under a common continuation policy. A proposed target is:

\[
q_{turn}(y|h)\propto\pi_{acc}(y|h)
\exp\{[\eta A_{success}(h,y)-\lambda A_{cost}(h,y)]/\beta(h)\}.
\]

Success advantage must reflect eventual task success; cost advantage must include future generated tokens, prompt processing, tool calls, retries, and recovery. A constant immediate minus-one reward at an ordinary token position cannot distinguish actions by itself. Donors can propose concise continuations or provide a prior, while verified continuation outcomes calibrate or veto their effect.

Operationally, minimize expected cost subject to a success constraint relative to accuracy-only, with a trust region. Allocate more computation to decisions where it prevents costly failure and remove waste where validated branches preserve success. Choose the accuracy tolerance before evaluation; require the constraint on the multi-turn slice as well as overall score, since a one-point overall tolerance can hide a three-point multi-turn loss when single-turn is unchanged.

Preserve critical alternatives explicitly: use a bounded union of base/current-student/teacher candidate support and validated termination/protocol/action candidates, retaining the residual bucket. At rare recovery states, train verified teacher-sampled complete turns without privileged hints using actual cross-entropy/forward supervision. The existing squared-log loss can remain the control for ordinary supported states. A support expansion is a diagnostic treatment, not a promise that arbitrary early stopping is safe.

For action-preserving compression, generate a concise turn conditioned on a verified action, remove that hint from the student's input, and validate the produced action and subsequent outcome. This directly supervises what compression must preserve. Masking reasoning or action tokens alone does not do so.

After target quality is established, compare the present squared-log residual with actual forward/reverse KL on the same states and target. Report per-state fit and realized behavior. If a refresh is tested, compare one new on-policy cache refresh with equal extra updates on the original cache; the prior 400-update result is an appropriate control for additional optimization budget, not for occupancy.

### Test a matched agent anchor and independent benchmark

A matched in-house pair is the cleanest test of the project's transfer premise. Start from the same accuracy checkpoint, with identical task split, tools, initialization, optimizer, updates, and rollout/context protocol; vary only the explicit cost objective or success constraint. Evaluate whether this agent-trained direction beats the empirical BFCL tradeoff and whether synthesized math/code directions agree with its continuation-value effect.

If the matched agent anchor beats the tradeoff while external ratios do not, transfer/signal mismatch is implicated. If both fail while verified safe concise branches exist, fitting/support is implicated. If replay finds little safe compression in the lost states, preserve their useful reasoning and move efficiency to redundant calls, context, or easier tasks. If a single cache refresh repairs losses at matched savings, occupancy is implicated.

Keep variants of each underlying task family in the same training, diagnostic, validation, or test split. Recheck contamination against the exact evaluator data pin before generating new training rollouts, and keep all BFCL-derived training tasks separate from reported evaluation tasks. Mechanical prompt/schema decontamination alone does not eliminate repeated development-set selection.

Use BFCL as development evidence and an independent held-out benchmark such as tau2 for confirmation. Before using the existing tau path, verify its actual within-turn reasoning transport, add the agent-eff checkpoints, and preserve benchmark/user-simulator versions and identity. It does not automatically inherit the BFCL proxy fix. Do not compare scores across materially changed task versions. No second benchmark was run in this audit.

## Criteria for a primary result

A credible next result should demonstrate a reproducible gain at matched realized operational cost or a reproducible cost reduction under a predeclared success non-inferiority constraint. Use held-out task families, multiple independent training seeds, decoding replicates, and an independent benchmark. Keep the number of confirmatory arms small. Separate training-seed variability from task-bootstrap uncertainty and account for selecting a winner.

A useful initial matrix is accuracy-only, reproduced DECS, one outcome-calibrated/action-preserving target, and a matched in-house efficiency anchor. Include support or refresh variants only when the replay/fit diagnostics predict their benefit. Match actual cost, rather than only target KL, and retain an inference-time cap/concise control.

Operational metrics should include raw total generated tokens, uncached prompt processing, wall-clock/accelerator cost under comparable serving, calls and failed/repeated calls, success under a fixed total budget, and unconditional cost per attempt. Successful-only cost is secondary because cheap failures can make it look favorable. A generic sharpening/concision placebo matched in cost and entropy helps test whether the donor conveys decisions beyond broad policy reshaping.

There are already defensible findings: accuracy-anchor transfer improves agent performance, interleaved context reuse avoids substantial repeated computation, and the tested cross-domain efficiency shifts produce persistent BFCL multi-turn costs while leaving single-turn averages largely intact. The evidence does not yet support a universal impossibility claim or a novel multi-turn frontier improvement. The most promising route to that improvement is **validated initial execution and recovery coverage with outcome-calibrated complete-turn targets**, while retaining the useful sparse cache machinery where its assumptions hold.

## Reproducing the CPU checks

Use a Python environment with NumPy for the independent aggregate audit. The raw directory has `runs/` and `runs_old/` children; its original local location is recorded in `research_audit_evidence.json`. Run `python results/agent_eff/research_audit_recompute.py --raw-root RAW_DIRECTORY --output OUTPUT_JSON --cluster-multiturn`. Remove `--cluster-multiturn` to reproduce the original category-stratified intervals; `--draws 0` checks all point totals without bootstrapping.

The usage check imports the current repository analysis code so it can reproduce its assignment behavior. Run `python results/agent_eff/research_audit_usage.py --repo REPOSITORY_DIRECTORY --new-root RAW_DIRECTORY/runs --output OUTPUT_JSON`. Its impossible-history check detects some attribution failures; a clean history does not establish an unambiguous join. All commands read existing outputs and run on CPUs.
