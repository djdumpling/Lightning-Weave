# Multi turn agent distillation research review

**Recommendation as of October 8, 2026:** make the next accuracy experiment about **which decisions improve the eventual task outcome**, then use Direct OPD to learn those decisions. Your mentor’s new collection pipeline and your Lightning-Weave machinery fit together well for this. The missing component is executing alternative actions and testing their continuations. Another efficiency donor, token gate, or unverified failure-prefix resampling sweep is a lower priority.

There are two independent research projects: multi-turn accuracy through better agentic training, and sparse-attention distillation. Keep their experiments and claims separate. The updated [algorithm investigation](../multiturn_algorithm/README.md) develops the multi-turn direction without an attention component. Generic multi-turn OPD, teacher intervention, and self-distillation on τ/BFCL already have close precedents; the potential contribution is the combination of **verified action improvement, information available at the decision, and transferable pre/post model deltas**.

## Evidence and scope

| Project | Inspected revision | Evidence available |
| --- | --- | --- |
| self-distillation | `5847c58a265ceb4a8ff350bef8c5d92c92f018ac` | Latest mentor push, confirmed against remote `main`; collection, scoring, packing, attention, study protocols and tests. Agentic run outputs are excluded from Git and absent in this checkout. |
| Lightning-Weave | `1c725e9` on `agent-eff-synthesis` | Retained BFCL results, current target composition and loss, Modal evaluation code; earlier experiment code preserved at `6f73aa2`. |
| agentic-eval | `d1bc6a1` | τ evaluator and BFCL scoring/protocol tests; cluster bring-up instructions. No new SWE/Terminal-Bench oracle run. |

Small existing summaries and 100 native τ2 conversation records were read from `alex-dev-2`, volume `lightning-weave-tau-eval`. No new GPU generation or training was launched. New CPU probes, downloaded summaries, and source hashes are in this directory. This is an audit of the relevant training/evaluation paths, not an exhaustive correctness proof of all three repositories.

The mentor’s reported +2–3 percentage points overall and up to +5 multi-turn remain meeting reports until the corresponding run receipts are available. Do not substitute hardcoded historical scores or natural episode terminations for those receipts. Here, `base` in your evaluation files means the original **post-trained Qwen3-4B student**, not the pre-training anchor in the teacher/base ratio.

## What the latest mentor pipeline actually trains

The [agentic guide](/Users/alexwa/self-distillation/docs/AGENTIC_TRAINING.md) and [focused collector](/Users/alexwa/self-distillation/self_distillation/studies/qwen3_tau2_focused_turns_20261007/collection.py) establish this sequence:

1. Collect 1,024 AReaL τ2 interactions from clean Qwen3-4B. Rebuild each request from public history, omitting old private reasoning. Allow `min(32768, 40960 - prompt_length)` response tokens, with no cumulative 2,048-token episode cap.
2. Select declared WRITE turns and the initiating turn of the first observable assistant tool error. Exact call IDs locate the pre-action prefix. After four oversized prefixes are excluded, the protocol describes 1,893 states from 914 episodes.
3. Sample 12,800 fresh responses at those prefixes, six or seven per state. **These regenerated responses are not executed.** There is no resulting observation, continuation reward, successful-repair label, or branch preference.
4. Score fixed responses with reference/pre/post models; merge checks exact row, prefix, support, behavior and paired eligibility identity. Train a clean Qwen3-4B with alpha 2, learning rate `1e-6`, 200 updates, and 64 responses per global batch.
5. Evaluate on **τ2-bench Airline**, 50 tasks × four repeats, using mean single-trial success (Avg@4 / pass^1). This is neither τ-bench v1 nor probability of at least one success in four attempts.

The three anchor pairs are Qwen3.5-9B-Base→Qwen3.5-9B, Qwen3-8B-Base→AgenticQwen-8B, and Qwen3-8B→AgenticQwen-8B. This latest study uses dense scoring and external anchors; the repository also contains separate weakened-anchor self-distillation and sparse-attention studies.

This is a useful expansion of **response coverage at selected states**. It does not refresh the states reached by an updated student or establish recovery competence. A tool error can follow a valid action; a wrong write can execute successfully; a missing clarification can cause an error several turns later. Selecting all WRITE turns partly broadens coverage, but is not a causal error locator. Natural user stop is also not task success: the recorded 979 natural stops out of 1,024 source episodes cannot be interpreted as 95.6% accuracy.

The changed source budget, task coverage, selection, repeats, and logical sample unit confound any comparison with earlier episode training. Isolate state selection with the same source trajectories and training budget before crediting a focusing algorithm.

## What your existing results already establish

The retained [interleaved-thinking results](../../results/agent_eff/results_2.md) show that keeping within-turn reasoning fixed a real harness mismatch and reduced generation substantially. Nevertheless, DECS mid still costs **2.37 multi-turn points** versus accuracy-only (95% paired interval −3.67 to −1.12) while saving 14.8% total generated tokens, across two training seeds and three decoding seeds. The corresponding single-turn difference is −0.04 points. That is a real accuracy/efficiency tradeoff in this setting, not simply the old reasoning-history bug.

The [projection study](../../results/agent_eff/results_3.md) found premature first-turn completion to be an important correlate of lost accuracy. Target projection suppressed one requested decision-shift direction in training, but did not resolve the BFCL accuracy problem. Protection gates, donor changes, and strength changes repeatedly stayed near the same tradeoff. Do not spend another round rediscovering that line.

Coverage is more promising: only **397/3,200 training prompts (12.4%)** were after-tool states, while approximately 63% of BFCL steps were after tool results in the retained analysis. The new τ2 audit below has about 40% of assistant messages immediately after a tool message. These are different benchmark distributions, but both motivate deliberate coverage of feedback-conditioned decisions. They do not prove that weighting those states alone will improve accuracy.

## Completed probes using existing data and CPU models

### Matched τ2 Airline trial

[Analysis code](probes/analyze_tau.py) reads native records, rejects duplicate/mismatched task-trial identities, verifies equal recorded seeds, and bootstraps paired task outcomes. [Results and 100 source hashes](artifacts/tau_airline_matched_trial0.json) identify the exact input files.

| Historical trial 0 | Original student | LoopTool OPD student |
| --- | ---: | ---: |
| Successes | 11/50, 22% | 19/50, 38% |
| Failed tasks | 39 | 31 |
| Failures with an explicit assistant tool error | 8 | 3 |
| Failures without an explicit assistant tool error | 31 | 28 |
| Successes despite an explicit tool error | 1 | 3 |
| Assistant messages immediately after tool feedback | 209/526, 39.7% | 242/600, 40.3% |
| Generation tokens | 421,892 | 430,676 |
| Length-truncated turns | 0 | 0 |

The paired delta is **+16 points, bootstrap interval [+2, +30]**: 12 gains, four losses, seven both correct, 27 both wrong. This is exploratory evidence from one historical trial, not confirmation of the latest mentor method or training-seed robustness. OPD has only one saved τ2 trial, whereas the baseline summary averages five; the matched comparison avoids conflating their repeat sets. All 39 baseline failures naturally reached `user_stop`, directly illustrating why termination is not correctness.

The main algorithmic lesson is coverage: the explicit-error signal is absent from most failures. This does **not** measure coverage of your mentor’s full union with WRITE states. It supports adding information acquisition, clarification, confirmation, and final completion decisions to a diagnostic panel.

One concrete audit example is task 2: both models transfer to a human, while the grader expects user/reservation retrieval and a certificate action. It is a candidate missing-information/early-transfer case, not proof that every grader-required action is semantically necessary. Full policy and task adjudication must precede using it as a training label.

The saved top-level `comparison.json` currently contains only the base model although both per-model summaries exist. The comparison writer replaces that file with the models selected in its latest invocation. Rebuild comparisons from per-model records; the top-level file is not an inventory of all completed results.

### User simulator sensitivity

Saved Thinking-2507 results show why the simulator belongs in the scientific protocol:

| User profile | Repeats | τ2 Retail | τ2 Airline | τ2 Telecom |
| --- | ---: | ---: | ---: | ---: |
| GPT-4.1 through Prime | 1 | 52.63 | 54.00 | 28.95 |
| Qwen3-30B-A3B-Thinking-2507 | 5 | 39.30 | 43.20 | 27.19 |
| Qwen3-235B-A22B-Instruct-2507 | 5 | 53.51 | 50.40 | 27.89 |

These are descriptive profile comparisons, not a paired causal model-only ablation: simulator settings and repeat counts differ. The saved summaries are in `artifacts/tau_thinking_*_summary.json`. There are no `user_error` terminations in these summaries. A weaker/otherwise different user can materially change task difficulty; “baseline below the paper” should first trigger protocol reconstruction, not a training conclusion.

### Direct OPD objective and support

For a token position, let `b` be cached frozen behavior probabilities, `S` the fixed Top16 support, and `O` the aggregate remaining vocabulary. The actual [tilted loss](../../slime/rollout/offline_direct_opd.py) is:

```text
r_i = sum_j w_j (log p_post,j,i - log p_pre,j,i),    r_O = 0
q_i = b_i exp(r_i / alpha) / sum_u b_u exp(r_u / alpha)
m   = sum_{i in S} b_i
L_t = alpha / (2m) * sum_{i in S union O} b_i (log p_theta,i - log q_i)^2
```

Normalization happens over the full model vocabulary before gathering candidates. Training averages **trainable token terms**, not equally weighted episodes or responses. Longer reasoning therefore contributes more terms; support mass `m` also affects their scale. The clean-reference field exists in the schema but is not an additional reference KL in tilted-target mode. The behavior distribution supplies the anchor. The two repositories’ core loss functions match.

[The numerical probe](probes/direct_opd_geometry.py) found initialization-gradient error of `2.93e-8` against the analytic expression, and gradient magnitude `3.64e-8` at `p=q`: no sign or fixed-point bug was found. It also demonstrates three substantive limitations:

- Swapping two OTHER-token probabilities from 0.01/0.29 to 0.29/0.01 leaves the loss exactly unchanged. The objective cannot individually favor a repair token outside cached support. This does not preclude indirect generalization, but motivates measuring support at the first action divergence and supplying successful repair text if needed.
- A post-only log-probability reward with OTHER reward zero favors OTHER over all nonpositively rewarded candidates. In the toy case OTHER mass rises from 0.30 to 0.50. This existing composition mode is **not a clean teacher-only KD baseline**.
- Doubling every mixture weight and alpha preserves the target but doubles loss/gradient scale. Report weights, alpha, and learning rate together; “same target” is not identical optimization.

Keep the established Top16/OTHER baseline intact. If support is the bottleneck, add a separate supervised repair term or explicitly version a new support experiment; do not silently change the frozen data contract.

### Attention mechanism and geometry

[CPU probe](probes/probe_attention.py), [output](artifacts/attention_probe.json): changing an old token with no direct edge to the final query has zero final-logit effect in a one-layer random Qwen3, but maximum change 0.25050 in four layers. Retained hidden states relay information. This is expected causal computation, not future leakage; it disproves equating sparse direct edges with erased history. These random-model numbers say nothing about task accuracy.

At synthetic context length 4,096, sink4/window512 permits 23.61% of dense causal edges. Task/observation retention with initial prefix256 permits 46.48%; prefix2048 permits **90.62%**. Chunk union gathering processes 1.49×, 1.21×, and 1.09× the allowed pairs respectively. A sparse label or edge count is not evidence of realized speedup.

## Bugs fixed and remaining validity risks

| Finding | Action and scope |
| --- | --- |
| Lightning-Weave composition silently accepted missing weights, mismatched candidate/prefix/behavior alignment, broadcastable shapes, and unequal streams | Restored strict score, manifest and row alignment checks; reject truncated streams before publishing. Added regressions. This is a confirmed latent bug, **not evidence that historical inputs were corrupted**. |
| Uncapped τ2 trajectories set cumulative budget to `None`, but shared packing compared an integer with it | Fixed live packing to accept explicit no-cap while retaining per-turn limits and numeric-budget validation. Added actual collection-to-packing regression beyond 2,048 generated tokens. |
| Reference scoring still passed that `None` to a bounded legacy validator | Detached one-turn scoring view now uses its recorded turn allowance. Parent receipts remain uncapped. Latest focused-response experiment used a numeric cap and is unaffected by these two bugs. |
| Scoring test’s FP32 normalization oracle varied on this ARM CPU | Changed the test oracle to FP64; production normalization and tolerances unchanged. |
| Evaluation documentation inferred Qwen’s repeat count from percentage granularity and called a wide interval “reproduced” | Corrected both copies. Repeated trials can produce the same fractions; numerical agreement does not establish a reproduced protocol. |

Relevant fixes are [composition](../../data_curation/build_direct_opd_composed_target.py), [packing](/Users/alexwa/self-distillation/self_distillation/training/turn_context_packing.py), and [scoring adapter](/Users/alexwa/self-distillation/self_distillation/turn_context_scoring.py). No historical snapshots or result files were edited.

Additional issues to resolve before interpreting small new gains:

- The mentor’s historical baseline used an effective transport timeout of 600 seconds versus 3,600 seconds for the new student. The study discloses this. Refresh the baseline under the corrected transport and inspect retries/missingness.
- Latest source generation allows 40,960 total tokens, whereas focused response scoring/training allows 32,768. Training and evaluation also use different user models. These may be deliberate, but should be separate variables in the next experiment.
- Cross-tokenizer pairs can train on different eligible token subsets. Report eligibility by role, tool-call argument, and decision type; compare on common eligibility for mechanism analyses.
- No executable task-level τ2 overlap audit was located. Compare normalized tasks, initial databases, goals, templates and tool schemas against evaluation, not only dataset names or exact transcript strings. Mere absence of distinctive tool names is not a proof of no contamination.
- BFCL still has intentional hard context-overflow failures under the historical protocol. Preserve the published comparison, but run any adaptive-budget repair as a separately labeled protocol, with the same models on both sides. Similar overflow counts do not imply identical affected tasks.
- Empty-user errors are intentionally zero-scored in the historical τ evaluator; distinguish that convention from model-caused failures. Generic exception fallthrough can also label a programming error as an agent zero, so inspect tracebacks and validate harness oracles before assigning a model explanation.
- The shared CPU environment is not the pinned production stack. Native-template, frozen-runtime, distributed, and GPU parity checks remain separate requirements for a new launch.

## Literature that changes the research positioning

Read these first; each changes a concrete design choice.

| Work | Relevant result and consequence |
| --- | --- |
| [PivotOPD, September 30](https://arxiv.org/html/2609.40285v1) | Prevent suspected pivotal errors and teach recovery using action-hinted self-teacher responses. ALFWorld, WebShop, Search-QA, SWE-Bench; no τ/BFCL. Its implemented PPO recovery update is explicitly not the gradient of its stated forward-KL loss. Reproduce the implemented objective, not just the headline equation. |
| [HERO, June 10](https://arxiv.org/html/2606.11559v1) | τ Retail training and Airline OOD, plus WebShop. Next observations and local hindsight reflections condition a privileged self-teacher. For 4B, GRPO→HERO improves Retail 33.3→34.7 and Airline 18.0→19.5 using mean@4. Large full-demo privilege can hurt; local actionable feedback matters. |
| [CrEST, August](https://arxiv.org/html/2608.13179v1) | Verifier-derived turn advantages determine update sign; privileged self-teacher scores modulate magnitude. BFCL V3/WildToolBench. BFCL uses 100 training IDs and 400 disjoint evaluation examples, **not the full official leaderboard**. An immediate turn verifier is easier here than in transactional τ. |
| [Guided-OPD, June 14](https://arxiv.org/html/2606.15912v1) | Whole-turn teacher interventions decay to zero; student turns receive reverse KL and teacher turns forward KL. ALFWorld, ScienceWorld, WebShop. Teacher takeover is an established baseline. [Code](https://github.com/Zzzz-166/Guided-OPD). |
| [D-CORE, ICML 2026](https://arxiv.org/html/2602.02160v1) | BFCLv3 and τ, with public [code/data links](https://proceedings.mlr.press/v306/xu26bn.html). Decomposed/composed reasoning self-distillation for SFT then diversity-aware GRPO. This is not online logit self-OPD, but disproves the broad “no self-distillation on these benchmarks” claim. |
| [OPD², July 16](https://arxiv.org/html/2607.15161v1) | Post-minus-base token delta, student-distribution centering, and agreement gate; math/science/code. Closest pair-based inspiration, but a different loss from repository Direct OPD. Ratios can capture style and interface changes as well as capability. |
| [PIVOT, September 28](https://arxiv.org/html/2609.35303v1) | A different paper from PivotOPD: counterfactual rollback for VLM agents. Rollback without hints often explains recovery gains. Include equal-budget same-student retry before claiming teacher guidance helped. |

Other close work: [ATOD](https://arxiv.org/html/2606.27814v1) already combines turn-aware distillation with GRPO; [Dual OPD](https://arxiv.org/html/2606.30626v1) studies privileged teacher/student gap routing and is not your **Direct** OPD; [Latent OPSD](https://arxiv.org/html/2608.13040v1) includes BFCL-v3 multi-turn; [G-OPD/ExOPD](https://arxiv.org/abs/2602.12125) uses flexible references and teacher pre-RL correction. These narrow any novelty claim around generic turn weighting or delta composition.

[Behavior Leverage Imbalance](https://arxiv.org/html/2607.07050v1) motivates logging structural action/stop tokens and unnecessary calls, but its BFCL loop diagnostic is not official leaderboard scoring. [ReOPD](https://arxiv.org/html/2607.04763v1) replays teacher histories without tool execution, clarifying the difference between on-policy responses and on-policy states. [DivOPD](https://arxiv.org/html/2609.34838v1) addresses asynchronous update diversity; useful only if the queue bottleneck exists here. [Lightning OPD 2.0](https://arxiv.org/html/2607.28449v1) offers style-disagreement residualization, but another residualization sweep is lower priority than measuring whether the delta ranks useful actions.

## The next accuracy experiments

### First establish whether repair is possible and teachers help

Build a **40-state discovery panel from training/development environments**, stratified across missing information, wrong arguments, confirmation/policy, silent bad writes, post-error recovery, repeated actions, and premature completion. The historical benchmark analysis above informs categories only; do not train on those held-out evaluation episodes. Include READ, clarification, and stop states, not only WRITE/error states.

At each state compare four branches, with two simulator/continuation seeds initially:

| Branch | Question answered |
| --- | --- |
| Fresh student retry | How much is recoverable through another sample alone? |
| One teacher-selected action, then student | Does the teacher supply a useful decision? |
| Teacher action plus up to two recovery turns, then student | Does success require a recovery skill absent from the student? |
| Continue after the original error and intervene there | Is this state still repairable, or must the earlier error be prevented? |

This is at most 320 suffixes for the first 40-state panel, with the fourth branch used only when valid. Within each matched decision state, use the identical public prefix and cloned **database plus user-simulator state**, the same suffix turn/token budget, and freshly executed feedback. Analyze pre-action prevention separately from post-error recovery: branch 4 starts from a different state, so its score cannot be treated as a pure intervention effect against the pre-action branches. Replaying later observations after changing an earlier write fabricates a trajectory. Equal initial random seeds reduce variance but do not keep user utterances identical once histories diverge.

Use existing remote resources only after verifying snapshot/replay equivalence. The present focused-response collector does not implement this branch experiment. Runtime restoration must reproduce the original prefix’s database and pending user state before any accuracy conclusion. Preserve explicit natural, censored and infrastructure outcomes.

Measure terminal success and policy correctness; teacher gain over student retry; successful alternative-action support in student Top16; delta ranking of better versus worse branches; and error/recovery categories. Limit privileged hints to information the student could have obtained then. A hidden reservation ID learned in the future should produce a retrieval or clarification target, not an earlier guessed ID.

If teacher branches do not beat same-student retry, stop elaborating the distillation loss. Diagnose teacher competence, branch validity, or irrecoverable states. If teacher one-step intervention helps but teacher-assisted continuation adds little, prioritize prevention. If only assisted continuation helps, prioritize learning recovery trajectories.

### Then isolate data improvement from objective improvement

Use one validated same-tokenizer anchor pair first, with clean-student initialization and one fixed held-out evaluation protocol. Compare:

1. Existing DOPD on states sampled without failure focusing.
2. Same DOPD on decision-balanced states from the same source pool, equalizing effective trainable-token exposure and optimizer budget.
3. Arm 2 plus supervised successful repair continuations from executed branches.

Include a repair-only SFT baseline if arm 3 improves, to establish what the delta contributes. Use equal-turn/episode weighting as an explicit ablation; current token-mean training implicitly overweights long reasoning, but changing that simultaneously with selection obscures the explanation. Normalize fractional weights with a denominator that actually sums those weights—do not blindly repurpose a loss mask whose reducer counts positions.

A conservative proposed objective is:

```text
L = L_DirectOPD(fresh student decision states)
    + lambda * NLL(verified successful, information-feasible repair responses)
```

This separates local prevention from teaching low-support recovery behavior and reuses most of the current trainer. It is not a novelty claim by itself. Match optimizer exposure so adding repair examples does not simply add more training. Begin with one small pilot, then two independent training seeds for confirmation; choose a practically meaningful improvement before looking at results.

Only after verified repair helps should you add branch-value advantages, preference learning, or a CrEST-style verifier-sign modulation. Adding the same scalar return to every vocabulary reward cancels under a fully normalized tilt; adding it only to Top16 changes OTHER mass instead. Outcome integration must affect action/sequence preference or explicit sample weighting, not masquerade as token credit.

### Refresh states and combine experts only when evidence warrants it

Frozen original-student prefixes become stale as the student improves. A later experiment can recollect from the updated student after a small training block, reseal exact behavior probabilities and support, and compare against repeated use of the old cache at matched collection/training cost. This changes visitation, which generating more responses at fixed prefixes cannot do.

Use the mentor’s three pairs initially as competing experts at identical states. Choose based on executed branch ranking, not individual anchor likelihood or global benchmark strength. Your composition machinery can then combine complementary deltas, with weights calibrated on development states and frozen before evaluation. For cross-tokenizer experts, first measure action-token eligibility and common-support ranking. Do not interpret “three pairs” as three independent discoveries when their post model is shared.

A plausible contribution is **branch-verified delta distillation for information acquisition, safe writes and recovery in conversational agents**. The differentiating evidence would be autonomous held-out task success beyond teacher retry/repair-SFT baselines, with valid mutable-state branches and a demonstrated advantage from the model pair. Confirm novelty again after the exact algorithm is fixed.

## StreamingLLM and the attention project

[StreamingLLM](https://arxiv.org/abs/2309.17453) stabilizes generation with initial sink tokens plus recent KV context; its [official FAQ](https://github.com/mit-han-lab/streaming-llm) makes clear that evicted middle history is unavailable. It is not a persistent task-memory mechanism.

The mentor’s [attention implementation](/Users/alexwa/self-distillation/self_distillation/attention/README.md) masks frozen scorer attention while preserving token IDs/positions. Students still train and run with dense attention. There is no KV eviction or StreamingLLM position relocation; Qwen3.5 recurrent Gated DeltaNet layers remain unchanged. A verified BFCL gain would support the value of a changed **training signal**, not establish a faster deployed model or a clean intervention removing historical information.

For the accuracy track, ask whether sparse scoring improves **correct-action ranking in the post-minus-pre delta**:

```text
r_sparse - r_dense
= (log p_post_sparse - log p_post_dense)
  - (log p_pre_sparse - log p_pre_dense)
```

A benefit can arise from changing the denominator or confidence rather than improving the post model. Candidate-common shifts alter Top16-versus-OTHER mass because OTHER reward stays zero. Analyze relative candidate ranking and aggregate support mass separately, then validate ranking against executed branches.

The repository already contains pre/post factorial, feedback-memory, observation-increment, phase-balance and dense-history/sparse-read studies. Retrieve those results before proposing the same grid again. The [observation-increment integration](/Users/alexwa/self-distillation/self_distillation/studies/agentic_observation_increment_20261004/REWARD_INTEGRATION.md) already sketches dense delta plus a gated visible-minus-hidden observation increment; it is marked CPU-only/NOT_GPU_READY. Its interpretation as task advantage remains a hypothesis.

For a separate future inference experiment, compare full public history; initial system/tools plus three recent complete interactions; and that window plus a structured persistent-fact store. Retain user constraints, entity IDs, confirmations, action receipts, unresolved subgoals, and policy. Keep call/return groups valid. Define “turn” explicitly: user turn, assistant generation and tool round are different units. Literal `[:-3]` retains everything except the last three; `[-3:]` selects the last three. Neither token slicing nor raw message slicing safely preserves structured exchanges by itself.

To distinguish attention sinks from semantic memory, vary 0/1/4 sink tokens at fixed semantic context, then vary old constraints at fixed sinks. Exempt those candidate sink positions from semantic-prefix retention, or use a separate sink/window-only diagnostic: if the full initial prefix is already globally retained, this sink ablation changes no visibility. Use full-history task success, write correctness and latency/KV memory, not attention entropy alone. [When Attention Sink Emerges](https://arxiv.org/abs/2410.10781) and [DuoAttention](https://arxiv.org/abs/2410.10819) provide mechanism and retrieval-head baselines. [The Complexity Trap](https://arxiv.org/abs/2508.21433) studies observation masking versus summaries in SWE agents; [Masking Stale Observations](https://arxiv.org/abs/2606.00408) shows dependence on retrieval/model capacity. Separate fixed-turn from fixed-token budgets so extra interaction opportunity is not mislabeled better memory.

## Benchmark choice and confirmation

Epoch’s [September 10 review](https://epoch.ai/benchmarks/berkeley-function-calling-leaderboard/review) labels **BFCL v4** flawed, finding grading-affecting defects in 24/50 stratified sampled items. The sample includes newer memory/web categories. **That 48% figure is not a BFCL v3 defect estimate.** Some multi-turn examples justify inspecting overlapping task families, but membership and defect presence must be checked against your exact pin. No τ review appears in the [review inventory](https://epoch.ai/data/benchmark-reviews-documentation/included-benchmarks); absence is not endorsement.

Keep pinned BFCL v3 for continuity, reporting multi-turn categories separately and manually adjudicating a blinded sample of model disagreements. Preserve official scores and separately report adjudicated sensitivity; do not silently edit the benchmark. τ2 is useful because it tests changing external state and permissions, but Airline alone is small. Use disjoint development states for discovery and held-out Airline plus another τ2 domain or another stateful environment for confirmation.

At 50 tasks × four repeats, +3 points is six additional successful conversations. Use task-cluster paired intervals, per-training-seed results, fixed simulator/settings and explicit retry accounting. A paired task bootstrap conditional on two trained models does not cover all training randomness. Avg@4, pass^1, pass^4 (all four succeed), and pass@4 (at least one succeeds) answer different questions.

The highest-value missing inputs are the mentor’s frozen protocols, model/checkpoint identities, per-task repeated outcomes, and trajectories for the latest focused and attention studies. Those decide which already-run mechanisms should be retained. The completed code audit and probes already support prioritizing action coverage and verified repair over further efficiency tuning.

## Validation and reproduction

Focused checks completed after fixes:

- Lightning-Weave: **135 passed, one distributed test skipped** across target composition, loss, BFCL diagnostics and τ configuration.
- self-distillation: **26 passed, 18 subtests passed** for complete rollout, portable turn packing and turn scoring; four frozen-run-dependent cases excluded.
- Attention: **36 passed, two subtests passed** across the five documented mask/context files.
- agentic-eval: **32 passed** for BFCL scoring/context and τ configuration.

Dependencies were isolated at `/private/tmp/agentic-research-20261008-venv` (Python 3.12.13, torch 2.14.1, Transformers 5.19.0); production package locks were not changed. CPU success is not native GPU parity.

From Lightning-Weave:

```bash
PYTHONPATH=. /private/tmp/agentic-research-20261008-venv/bin/python \
  research/followup/probes/direct_opd_geometry.py
PYTHONPATH=/Users/alexwa/self-distillation /private/tmp/agentic-research-20261008-venv/bin/python \
  research/followup/probes/probe_attention.py /private/tmp/attention_probe.json
/private/tmp/agentic-research-20261008-venv/bin/python \
  research/followup/probes/analyze_tau.py \
  --base /private/tmp/agentic-tau-audit/base_records \
  --opd /private/tmp/agentic-tau-audit/opd_records \
  --output /private/tmp/tau_airline_matched_trial0.json
```

To restore raw records, first create two destination directories, then use `modal volume get` on `tau-b712b7c60edc-full/{base,opd}/tau2_airline/trial0` in `lightning-weave-tau-eval`, with `MODAL_ENVIRONMENT=alex-dev-2`. The analysis records every source SHA-256. Summaries are preserved here as downloaded; historical Modal outputs remain untouched.
