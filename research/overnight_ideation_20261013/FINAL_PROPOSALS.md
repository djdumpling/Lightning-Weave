1. No proposal met the requested standard for a defensible, novel extra ≥2-point gain after two fresh red-team rounds.
2. The process covered48 original candidates across eight mechanism families, three tournament generations and74 blind comparisons.
3. Four additional candidates from a final change of lens also failed screening; their useful ingredients and rejection reasons are preserved.
4. The literature inventory contains186 verified primary works, including41 full deep reads; other reads are explicitly narrower.
5. The strongest remaining target idea isolates a contextual interaction in the composed pair-relative policy, retaining recipient lower-order odds.
6. Its favorable toy gains3.9842 points over ordinary DOPD, but an equal-information rule target ties it exactly.
7. The second idea estimates a bounded, moment-preserving response contrast without ranking individual successful branches.
8. Its intervention theorem is valid, but native decoding and repeated interventions can reverse the intended gain.
9. The third deepened idea, a relation-informed optimizer, was eliminated after unfavorable compute-matched CPU results.
10. No training, benchmark evaluation, GPU job, paid model call, model download, commit or existing-repository edit was performed.

# Overnight research: best leads and unresolved failures

**Final assessment: the research process is complete; the scientific stopping bar was not met.** This file records the best two research leads in the requested format; they are not being passed off as successful finalists. The overnight window elapsed during a service interruption, and the user explicitly requested daytime continuation. No unattended progress during that interruption is claimed.

Evidence notation: **M** is an existing measurement checked in source; **D** a derivation; **T** a small authored CPU calculation; **H** an unverified mechanism or subjective forecast; **P** a proposed experiment, never executed here. A toy percentage-point gain is not a benchmark forecast. All paths below are relative to this output directory.

The strongest ceiling explanation remains viable: these anchor pairs may contain little additional useful competence beyond what current DOPD already extracts at a tolerable distance. More movement can import donor weaknesses. A likelihood ratio identifies a probability change, not task utility, and a locally valid target can still be poorly realized by the shared neural model and deployed decoder. Neither the ceiling nor a particular alternative cause has been established.

Three corrections materially change the research case. W1's weak-student LoopTool positive used Base4B→2507, so own-family transfer is not universally null. Current Qwen3.5 reasoning persists across tool exchanges until the next real user message. The approximately±1.6-point interval is for the paired trained-minus-base difference conditional on the observed models, not an absolute-model interval or a guarantee of future test power. See [verified ledger](LEDGER.md) and [uncertainty audit](GROUND_TRUTH/uncertainty_design.md).

**Headroom (A):** current mean80.5799% leaves19.4201 points of error; another2 points requires repairing10.30% of those errors without regressions. Every one of114 baseline tasks had at least one success in24 trials. That supports a reliability question, not proof of an identified never-solved capability. Repairing15% of failures while breaking.5% of successes yields2.5101 points; those repair rates are a hypothetical scenario, not a measurement. Preserving the base-perfect group alone offers only.256 observed points.

## Lead 1 — Contextual interaction transfer

### 1. Name and pitches

**Two-sentence pitch.** On valid four-cell public-history comparisons, transfer only the interaction component of the pair-relative composed policy odds. Keep the recipient's lower-order odds and fit the resulting exact target with the existing offline DOPD loss.

**For the mentor.** A weak donor's useful increment might be a relation between a retained user obligation and a newly observed fact, while its main effects import shortcuts the strong student already handles better. A four-cell target decomposition can separate those components in a way a scalar strength cannot. The mathematical separation is real, and a toy reproduces chance within-state ranking together with positive transfer. The missing contribution is a concrete acquisition family in which the pair supplies useful information that the same public rule does not already provide; without that, this is not a strong new algorithm recommendation.

### 2. Root cause, evidence and competing explanations

**H:** useful contextual interactions coexist with harmful lower-order increments. This conditionally explains pair-over-post transfer, recipient-dependent own/external transfer, amplified-target harm and delayed multi-turn failures. None of the existing measurements identifies that decomposition. Data coverage, finite fitting, native decoding and the limited-complementarity ceiling remain alternatives.

The mechanism targets the *ideal target*, not an optimizer problem. Its exact distinction from scalar attenuation survives. Its claim to isolate donor-acquired competence does not survive unchanged: recipient-dependent path normalizers can create an interaction even with a context-independent donor shift.

### 3. Objective and derivation

For x=(x1,x2)∈{−1,+1}², use actual actor prefixes h_x, including retained thought. Declare two positive, disjoint complete suffix events y0,y1 with the same semantic correspondence across cells. Let B be the frozen recipient response law and Q_D the product of its normalized token-local DOPD laws, with the documented Top16+OTHER lift.

    ell_b,x = log[B(y1|h_x)/B(y0|h_x)]
    d_x = log[Q_D(y1|h_x)/Q_D(y0|h_x)] − ell_b,x
    chi_x = x1*x2
    k = (1/4) sum_x chi_x*d_x
    ell_star,x = ell_b,x + k*chi_x

Let m_D,x=Q_D({y0,y1}|h_x). Set Q_star(y1)=m_D,x·sigmoid(ell_star,x), Q_star(y0)=m_D,x−Q_star(y1), leaving every outside complete response unchanged. For each affected trie prefix u, add descendant leaf mass changes to Q_D(u), then take child/parent cylinder-mass ratios. This gives a normalized causal actor law. It preserves lower-order **log-odds coefficients**, candidate mass and outside raw law, not deployed probability marginals.

Fit Q_star with the unchanged behavior-weighted squared-log loss. With T the count of unmasked cached token positions and M_t the frozen Top16 mass, its objective is `L(theta) = (1/T) sum_t [alpha/(2 M_t)] sum_{a in Top16+OTHER} b_t(a) [log pi_theta(a|h_t) − log q_star(a|h_t)]²`, using full normalization and the existing numerical floors. For represented atomic choices, D'_a=D_a+alpha·log(q_star,a/q_D,a), with OTHER unchanged, realizes the target. Longer tries need individually represented modified tokens and the stated tail gauge. Disabling replacement recovers ordinary DOPD; **k=0 is not the disabled limit**, because it can remove useful ordinary main effects.

For one reached opportunity, with a_x the continuation-value difference and rho_x its episode reach probability,

    Delta J = sum_x rho_x*m_D,x*a_x*
              [sigmoid(ell_b,x+k*chi_x) − sigmoid(ell_b,x+d_x)].

Writing d_x=l_x+k*chi_x, a_x*l_x≤0 is a sufficient cellwise nonharm condition versus ordinary DOPD. Against same-mass base odds, a_x*k*chi_x≥0 suffices. Valid factorials alone imply neither sign. Multiple opportunities cannot be added without accounting for overlapping episodes and changed occupancy.

**Revised interpretation:** d_x equals raw alpha-scaled donor path contrast minus the path-normalizer contrast. Accordingly k belongs to the composed recipient target. The independent normalization-only example yields k=.161189832 even though donor contextual dependence and raw donor path-odds contrast are zero. See [revised technical packet](REDTEAM/ROUND2/M84_REVISED.md).

### 4. Algorithm and estimator

```text
Declare training-only task/template split and public factorial constructor.
Generate four coherent native histories and two prespecified semantic alternatives.
Preserve served tokens; assign each actual root to at most one quad.
Score frozen recipient, pre and post on every required full-context suffix prefix.
Construct normalized Q_D, including local normalizers and declared tail treatment.
Compute k exactly from the four frozen odds increments; no return ranking.
Replace the two-event law, reconstruct trie conditionals, encode offline targets.
Fit with existing DOPD alpha=2, LR=1e-6, 200 updates and matched token exposure.
```

No teacher runs during training. Score precision is not uncertainty about task value. Reject conflicting/ancestor-overlapping roots, missing required support, incompatible native histories and unexplained semantic correspondence. These are validity conditions, not outcome-tuned gates.

### 5. Genealogy

The positive parents are pair-relative composition (W1/W3/W4; EFT/Proxy/LW), relational distillation and public factorial decomposition. The negative parents are scalar amplification collapse (L8), target invariance without realized invariance (L2), unreliable response ranking (L6) and occupied action/phase gates. The original E05 target branch was consolidated during screening and reopened against a competing finite-fit explanation; G3_B/M84 is the resulting target-side candidate. No novelty is claimed for Fourier coefficients, counterfactual prompts or ratio arithmetic alone.

### 6. Ten nearest works and exact differences

The following table records narrow distinctions, not a proof that the combination is publishably novel. Primary identifiers/titles were verified; [the literature inventory](LIT/ALL_inventory.json) records actual reading depth.

| Verified primary work | Established mechanism | Remaining difference for this lead |
|---|---|---|
| [Lightning Weave: Improving the Accuracy-Efficiency Frontier of Reasoning Models through Capability Composition — 2609.14708](https://arxiv.org/abs/2609.14708) | Frozen checkpoint differences define token-local tilted targets, fitted offline. | Decomposes the transferred odds change across public contexts and deletes its lower-order components on declared native events. The local tilt itself is inherited. |
| [An Emulator for Fine-Tuning Large Language Models using Small Language Models — 2310.12962](https://arxiv.org/abs/2310.12962) | Composes a tuned/base likelihood ratio with another recipient reference. | Adds a particular functional projection before transplanting the increment. Reference replacement and ratio arithmetic are already occupied. |
| [Tuning Language Models by Proxy — 2401.08565](https://arxiv.org/abs/2401.08565) | Uses small-model tuned-minus-untuned logits to guide a larger model. | Offline interaction-only target instead of the whole local proxy difference. A weaker donor helping a stronger recipient is not new. |
| [Rethinking Relative Policy Differences Transfer of On-Policy Distillation — 2610.13205](https://arxiv.org/abs/2610.13205) | Separates expert, source reference and recipient reference; studies source and recipient compatibility. | Proposes a specific cross-context component as the useful difference, rather than a source/reference sweep. Its utility is still an assumption. |
| [Relational Knowledge Distillation — 1904.05068](https://arxiv.org/abs/1904.05068) | General n-tuple relation matching; distance/angle implementations. | Changes the ideal recipient target, rather than merely adding a relation-matching penalty to ordinary targets. Generic relational transfer is occupied. |
| [CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation — 2608.13387](https://arxiv.org/abs/2608.13387) | Uses teacher sensitivity to counterfactual versus paraphrased inputs to select token supervision. | Signed four-cell checkpoint increment changes target odds; CROP selects where to apply the original teacher update. Public counterfactual construction alone is not novelty. |
| [The Shapley Taylor Interaction Index — 1902.05622](https://arxiv.org/abs/1902.05622) | Axiomatizes attribution of feature interactions. | The four-cell coefficient is elementary existing decomposition; novelty could only be useful transfer of that coefficient relative to a separate recipient. |
| [A Unified Approach to Interpreting Knowledge Distillation for Large Language Models via Interactions — 2607.08776](https://arxiv.org/abs/2607.08776) | CIP penalizes complex masked-word interactions during LLM distillation. | Preserves the recipient's lower-order odds and adds a signed donor-pair public-condition interaction, rather than penalizing student word-mask interactions. Selective interaction-aware KD is already occupied. |
| [Selecting Feature Interactions for Generalized Additive Models by Distilling Foundation Models — 2604.13332](https://arxiv.org/abs/2604.13332) | TabDistill extracts important teacher interactions to choose terms of a supervised GAM. | Transfers the value of a checkpoint-pair increment to an existing policy, not only a selected interaction index into a redesigned predictor. |
| [Fast Generic Interaction Detection for Model Interpretability and Compression — OpenReview fQTlgI2qZqE](https://openreview.net/forum?id=fQTlgI2qZqE) | Detects interactions and uses them to build a compact ParaACE student. | No new architecture or interaction-detection claim; the narrow object is recipient-relative checkpoint-increment transplantation. |

### 7. Retrodictions and ledger consistency

“Conditional E” predicts the direction under the named extra premise; it is not empirical confirmation.

| Puzzle | Status | Directional premise and falsifier |
|---|---|---|
| P1 | Conditional E | A main effect reverses within-state ranks across cells while useful interaction survives; existing toy AUC=.5 with positive ordinary transfer. Fails if actual useful cross-cell structure is absent. |
| P2 | Conditional E | Pair subtraction cancels inherited liabilities while retaining useful contextual change; fails if matched post-only has the same useful component and no extra liability. |
| P3 | Conditional E | Own-family useful relation is saturated in the strong recipient while an external relation is not; weak-recipient W1 remains possible. Fails if signed increments/margins are comparable. |
| P4 | Conditional E, partial | Amplified lower-order liability crosses a bad decision/decoder boundary after useful effects saturate. Does not explain the separate400-update result. |
| P5 | Neutral | Avoiding noisy outcome labels does not explain the old broadcast estimator's null. |
| P6 | Conditional E | Retained obligation and later observation jointly determine value; a shortcut fails later despite small immediate probe effects. AND requires all-corner accounting. |
| P7 | Neutral | No separation of cap, exposure, truncation and coverage effects. |
| P8 | Neutral | No directional explanation of opposite length changes. |

Five conditional accounts are possible, as both algebra and the first review establish. Closest negatives: L1/L8 require the trained attenuation control; L2 requires native realized behavior; L4 preserves recipient-specific own-shift scope; L5/L6 prohibit silently adding a ranking oracle; L7 prevents assuming a stronger rescue teacher; L10 requires a same-data control; L11 denies novelty for phase/observation mechanisms. Full P/L table: [deep derivation](DEEPEN/G3_B.md).

### 8. What differs from failed approaches—and what still fails

Unlike a coefficient sweep, projection removes lower-order terms while keeping nonzero interaction; no single scalar can generally do both. Unlike decision-preserving reasoning projection, it intentionally changes a declared decision relation and acknowledges native realization risk. It introduces neither Bellman return, teacher rescue nor a per-state winner label.

However, in the demonstrated labeled-parity regime, the correct-rule target ell_b+k·chi is identical at the same conditional KL. Identical targets/rows/optimization cannot gain a distinct finite-fitting effect merely from being named “pair transfer.” The pair might supply useful cross-quad calibration if only a global rule budget is matched, but that is a new, unearned calibration hypothesis, not proof of total pair redundancy or of pair necessity. The unlabeled regime instead needs a substantive reason for useful value alignment. That bridge remains unresolved.

### 9. Completed toy and negative controls

| Authored fixture | Ordinary DOPD | Interaction target | Matched scalar | Matched rule |
|---|---:|---:|---:|---:|
| Favorable compatibility |80.2197%|84.2039%|81.0447%|84.2039%|
| Pure useful interaction |84.3360%|84.3360%|84.3360%|84.3360%|
| AND with weak negative corner |88.3974%|80.0971%|85.4338%|85.4338%|
| Useful main effect only |84.3360%|80.1085%|80.1085%|80.1085%|
| Wrong interaction sign |69.9025%|74.2573%|77.4524%|84.6989%|

Favorable base79.9163%; the AND base81.0968%, so projection can harm even relative to base. Zero donor signal produces no difference from ordinary. The favorable within-state ratio AUC is exactly.5. Local normalizers matter: omitting them incurs a .3836-nat path-odds error in that fixture. Miniature native decoding changes the probabilities; raw conservation is not a deployment invariant.

Root read and independently reran [toy source](toys/g3_b/toy.py), its assertions and [reviewer checks](REDTEAM/ROUND1/M84_independent_checks.py). These establish mathematical possibilities and failures. They do not measure real event incidence, neural fit or extra retail gain.

### 10. Repo implementation and compute

| Existing hook | Required future change |
|---|---|
| `data_curation/fresh_states.py` | New valid factorial constructor and disjoint ownership manifest; preserve captured-token identity. |
| `configs/qwen35_transfer/modal_pipeline.py` | Preserve actual/served token equality and latest-real-user reasoning retention. |
| `data_curation/precompute_direct_opd_scores.py:score_qwen35_sequence` | Full-context recipient/pre/post suffix scoring; no assumed streamed-GDN cache shortcut. |
| `data_curation/shift_geometry.py` and `build_synthetic_shift_target.py:encode` | Grouped normalized-odds projection, trie reconstruction and provenance-preserving target encoding. |
| `slime/rollout/offline_direct_opd.py:offline_direct_opd_tilted_target_terms` | Existing loss unchanged when required tokens are represented; new support must be identical across controls. |

256quads×4histories×2suffixes×3scorers=6,144 full-context passes. At the historical median6,940 prompt tokens, that is42.64M aggregate prompt-token **proxy**, plus≤.394M suffix-scoring tokens at64-token cap. New native histories add≤2.10M generated tokens under a2,048 cap. Eight students mean1,600 updates, exports and21,888 proposed final evaluation episodes. No measured throughput establishes that this fits two nights. Detailed hooks/floor/tail qualifications: [implementation audit](DEEPEN/ROOT_IMPLEMENTATION_AUDIT.md), [full B report](DEEPEN/G3_B.md).

### 11. Designed falsification experiment and readiness gap

Exactly eight students: current recipe2; same factorial data with ordinary target1; projected target2; trained KL-matched attenuation2; same-specification rule1. All use contemporary user/renderer/sampler, matched exposure/support, training-only construction and untouched114 evaluation tasks. Select attenuation by outcome-free KL bisection, never by reward tuning.

**P:** the main hypothesis predicts ≥2pp over current DOPD, positive effects in both projected seeds, superiority to attenuation and an advantage not explained by factorial data or public rules. Its necessary conditional mechanism is enough unique affected-episode mass and beneficial odds movement to clear2pp after regressions. A general headroom identity is not evidence that these quantities hold.

Stop before students if valid native events lack enough mass, every useful target is already rule-encoded without a distinct calibration mechanism, or the utility case incorrectly treats AND as parity. After a legitimate launch, a rule/attenuation tie deletes the claimed contribution; data-only gain attributes it to augmentation; better target fit without native reward gain rejects the realization hypothesis. An unresolved interval is inconclusive, not a positive result.

**Final design clarification:** use114pinned tasks×24trials for each of eight models,21,888 episodes total, with the same contemporary user and paired task/trial protocol. The primary practical criterion is a pooled point improvement≥2pp, both projected seeds positive and a paired task interval excluding0; that is not proof that the true gain exceeds2pp. Report seed spread separately. Under the merely illustrative historical task-difference SD8.715pp, a normal approximation gives about69% power for+2pp and23% for+1pp before additional seed uncertainty. Those are conditional design illustrations, not new-method power estimates; do not expand the budget post hoc after an unresolved result.

**Readiness failure:** one-seed data/rule controls and two-seed main contrasts cannot guarantee a decisive mechanistic comparison at the audited noise floor. Showing a positive effect differs from showing that its lower bound exceeds2pp. The experiment is a specified falsification pilot; this run has not established the requested decisive two-night design for this lead.

### 12. Risks, premortem and forecasts

Most likely +.3pp result: coherent quads have little native probability; the retained interaction is not useful; direct rules contain the benefit; or shared fitting/decoding changes other behavior. The exact normalization and rule-equality objections survive a larger sample.

The Phase6 author's **subjective**, uncalibrated forecasts were P(extra≥1pp)=15%, P(extra≥2pp)=5%; for an attributed gain over both attenuation and rule, about5%/1%. They refer to true average effects, not crossing a noisy estimate threshold. After both reviews, root uses approximate planning forecasts **P(extra≥1pp)=10%, P(extra≥2pp)=3%** for this fixed design. These are subjective judgments, not externally calibrated probabilities: the conceptual utility/calibration gap, low-probability native events and weak control replication lower confidence. No forecast is inferred from the toy. A defensible reopening must provide a concrete validity-without-labels acquisition family plus a substantive utility premise, or a nontrivial calibration contribution with its appropriate controls.

## Lead 2 — Moment-preserving boundary circulation

### 1. Name and pitches

**Two-sentence pitch.** From an iid native response pool, change the probability of handing control back to the user while preserving the pool's mean donor-pair score and response length. Estimate the signed return effect with one uniformly executed candidate per pool, then distill the bounded response law offline.

**For the mentor.** Rather than ranking noisy successful branches, this buys one low-dimensional, known-probability intervention across many tasks. Both directions average back to the current native policy, and a centered return estimator identifies their contrast exactly. That is a clean estimator, but its pair moment must be useful rather than decorative; the current raw-MLE fitting step can reverse the gain under the deployed sampler, and a one-boundary theorem does not govern a policy changed everywhere. It also explains only two requested puzzles.

### 2. Root cause and alternatives

**H:** a stable global handoff direction exists after removing donor-score and length variation, while broadcast outcome labels spread weak signal across irrelevant response differences. A premature handoff can remove later recovery opportunities. The premise is motivated by L5/P6 but is not an error taxonomy of the current strong student.

Competing explanations are generic stopping, irrelevant or harmful moment constraints, changed native sampling from uniform MLE, ordinary second-phase data fitting, and the pair-complementarity ceiling. C does not diagnose an observational confounding flaw in L5: those responses were actually executed.

### 3. Law, objective, estimator and limits

At the first assistant generation after a tool result, with current-DOPD parent native policy mu, draw K=8 iid complete responses, retaining duplicates, malformed/capped cases and no-arrival episodes. Let e indicate user handoff without a tool call, g the true normalized complete-response log p_post/p_pre, and l response length.

    F = columns(1,g,l)
    s = (I − projection_span(F))*e
    delta_i = s_i / [2K*max_j |s_j|]    (zero for a zero residual)
    w_plus,i = 1/K + delta_i
    w_minus,i = 1/K − delta_i

Each weight lies in[.5/K,1.5/K]. Define nu_plus/minus as the expectation of these weighted empirical measures over fresh pools. Then `(nu_plus+nu_minus)/2=mu`; both preserve the population g and l moments, their density ratios to mu lie in[.5,1.5], and TV(nu,mu)≤.25. These are response-law identities at one history, not trajectory identities.

Choose I uniformly from the pool, execute Y_I with frozen parent continuation, observe R, and form

    X = K*delta_I*(R − .5),       |X|≤.25.

Its expectation is the one-intervention gain of nu_plus over mu. Freeze a global sign lambda on discovery tasks before validation. Under `V_h(y)=c_h+kappa*e(y)+eta_h*g(y)+zeta_h*l(y)+epsilon_h(y)`, moment cancellation isolates the event component plus a residual term; stable kappa and a small adverse residual are utility assumptions.

The proposed fitting objective is full-sequence weighted raw NLL, `sum_h b_h sum_i w_lambda,hi[-log p_theta(Y_hi|h)]`, with task-balanced b_h. Under realizability it matches the raw target nu. **The deployed law is D_protocol(p_theta), not p_theta.** Uniform weights recover the parent native sampling law as a raw-MLE target, which is not generally the parent's original raw model. Skipping phase2 recovers the original DOPD parent; setting delta=0 does not guarantee it after training. See [full derivation](DEEPEN/G3_C.md) and [revised packet](REDTEAM/ROUND2/M37_REVISED.md).

### 4. Algorithm and what is precomputed

```text
Train two contemporary current-DOPD parents; freeze models and native protocol.
Verify training-manifest unique IDs and train/evaluation disjointness.
Hash-split training tasks into discovery and validation; preserve equal task weights.
For each prespecified unit, roll in to first post-tool assistant boundary or termination.
At arrival, draw8 iid native responses; execute one uniformly with parent continuation.
Score all actual tokens with both frozen donors; construct e,g,length and delta.
Select one sign on discovery; validate against own-sign deletion controls.
If value, necessity and cost gates pass, fit uniform/constrained/generic descendants.
Evaluate native descendants against parents and controls on untouched evaluation tasks.
```

All donor scoring precedes student fitting. No runtime selector is included. Numerically tiny projection residuals are zeroed by a prespecified rank tolerance; outcome-tuning it would change the method. Native renderer/sampling parity is required: historical replay drops reasoning and omits some normal sampling arguments.

### 5. Genealogy

Parents are bounded finite resampling, moment-constrained distribution matching, controlled local continuation experiments and offline distillation. The negative parents are noisy broadcast outcomes (L5), weak ranking (L6), rejected teacher rescue (L7), generic stop/length controls (L1) and target-to-realization failure (L2). D06/C01 evolved toward a population law rather than a fixed sampled trie. The pair is used as a preserved feature, not an identified reward. That is a materially weaker Lightning Weave identity and must earn its necessity.

### 6. Ten nearest works

| Primary work | Occupied contribution | Remaining distinction / weakness |

|---|---|---|
| [Reinterpreting Importance-Weighted Autoencoders, arXiv1704.02916](https://arxiv.org/abs/1704.02916) — targeted method | Finite resampling induces a population distribution. | C uses bounded symmetric perturbations and native return contrasts; the law identity itself is old. |
| [RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment, arXiv2304.06767](https://arxiv.org/abs/2304.06767) —§3.2–3.3 | Sample/select responses and distill the induced policy. | C has no per-response reward ranking; constrained direction must beat generic sample-and-fit. |
| [A Distributional Approach to Controlled Text Generation, arXiv2012.11635](https://arxiv.org/abs/2012.11635) — abstract | Distributional constraints and autoregressive target matching. | Pair-referenced moment choice and executed structural contrast are provisional, not generic constrained distillation. |
| [Policy-Conditioned Counterfactual Credit for Verifiable Reinforcement Learning of Long-Horizon Language Agents, arXiv2606.05263](https://arxiv.org/abs/2606.05263) — full shared method audit | Controlled agent interventions with frozen-reference continuations and causal credit. | C estimates one bounded known-probability moment-preserving contrast; no novelty for “causal branch credit.” |
| [IAPO: Influence-Aware Policy Optimization for Credit Assignment in Multi-Turn Service Agents, arXiv2608.24588](https://arxiv.org/abs/2608.24588) — full available method audit | Routes realized outcome credit through annotated dependency relations. | C uses randomized population contrasts, no annotator edges; pair specificity remains unproved. |
| [ESTAR: Early-Stopping Token-Aware Reasoning For Efficient Inference, arXiv2602.10004](https://arxiv.org/abs/2602.10004) — targeted method | Stop interventions/labels and stopping-aware training. | C concerns native user handoff and terminal task return, while retaining score/length moments. Generic stopping is occupied. |
| [TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning, arXiv2603.12529](https://arxiv.org/abs/2603.12529) — abstract | Learned reasoning exit decisions. | Different public event and preserved moments; distinction needs deletion results, not terminology. |
| [CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation, arXiv2608.13387](https://arxiv.org/abs/2608.13387) — deep shared read | Nuisance-controlled counterfactual relevance and selective OPD. | C executes response distributions and estimates return, rather than sensitivity-based token selection. |
| [Reinforcement and Imitation Learning via Interactive No-Regret Learning, arXiv1406.5979](https://arxiv.org/abs/1406.5979) — targeted theory | Learner roll-in, action exploration and continuation cost supervision. | C's continuation is its own trained parent and its contrast is restricted; no novelty for local interventions plus imitation. |
| [Double/Debiased Machine Learning for Treatment and Causal Parameters, arXiv1608.00060](https://arxiv.org/abs/1608.00060) — abstract | Orthogonal scores/cross-fitting for low-dimensional causal parameters. | C does not fit nuisance regressions or obtain a DML robustness theorem. Its projection changes the policy rather than only debiasing an estimator. |

### 7. Retrodiction table

| Puzzle | Status | Reason |
|---|---|---|
| P1 | Neutral | Estimable global contrast is compatible with weak ranking; it does not explain existing delta gains. |
| P2 | Neutral | Preserving pair score does not explain pair-over-post superiority. |
| P3 | Neutral | No derived recipient-specific own/external-shift account. |
| P4 | Neutral | Bounded resampling weights do not predict amplified DOPD or duration effects. |
| P5 | Conditional E, partial | A stable low-dimensional return direction amid irrelevant variation can favor pooling over diffuse broadcast; this premise is not measured. |
| P6 | Conditional E | Wrong handoff removes later opportunities, creating delayed terminal cost. |
| P7 | Neutral | First-post-tool intervention explains neither cap nor uniform-selection comparison. |
| P8 | Neutral | Preserving length does not explain the observed opposite length changes. |

This fails the five-puzzle requirement. L5 is not repaired by merely naming randomization; L6 forbids per-state winner claims; L2 remains active under native fitting; L10 requires contemporary task/protocol controls; L11 means error-step/phase selection is not novel. Full L1–L12 accounting is preserved in [the deep report](DEEPEN/G3_C.md).

### 8. Distinctions from failed attempts

It uses a prespecified signed distribution contrast, not a winner score, coefficient sweep, teacher replacement or suffix MCMC. Both one-moment deletions and generic stopping receive independent discovery-selected signs and the same uniform-index return observations. This prevents an intentionally wrong-sign control from manufacturing necessity.

Projection and constrained stopping are established. Affine changes of g do not alter its span; a second-review audit verified495 exact pool/transform cases for the stronger invariance to g→a*g+b*l+c, a≠0, including pre/post reversal. Such invariance is compatible with nuisance control, but excludes an interpretation as directly transferring the pair's directional improvement. Moment preservation alone is not competence preservation.

### 9. Completed toy and decisive counterexamples

The exact K8 enumeration covers4^8=65,536 ordered pools via165 count vectors, with an independent ordered K4 check. At baseline41.6667%, full constraints gain4.1567pp, versus3.1918pp for either deletion and3.0273pp for generic stopping, each using its best sign. Constant utility gives exact zero effect. A utility depending only on the preserved moments also gives zero full effect; utility=e favors generic stopping.

Affinely raising the same witness to80.5799% yields only1.2470pp. That is **not** a global bound. Root's exact LP over all bounded reward means at that baseline, for the fixed toy law and both selected signs, finds maximum minimum deletion advantage.46572pp; a full gain2.11433pp is possible only at the maximum with a zero necessity gap. This constrains this fixture's discrimination, not all possible real pools.

Two distinct failures remain. Applying the same favorable local intervention twice changes+.1247pp once to−.2358pp twice in an exact episode example. Separately, presence penalty1.5 can turn intended raw gain+.8729pp into **−2.4860pp relative to decoded uniform fitting**, even with perfect raw targets and all four alternatives retained by top_p.95/top_k20. Neither sampling noise nor poor fitting causes those reversals.

Root independently reran [exact pool toy](toys/g3_c/exact_pool.py), [native counterexample](REDTEAM/ROUND1/M37_NATIVE_COUNTEREXAMPLE.py) and [rational reward envelope](toys/phase7_root_c/reward_envelope.py). No model evidence is implied.

### 10. Implementation and all-in cost

A new full-sequence weighted-NLL path is needed: offline DOPD currently pins a squared-log token-mean loss in `slime/utils/arguments.py` and `slime/rollout/offline_direct_opd.py`. `data_curation/branch_outcomes.py` and the branch path in `configs/tau_bench_eval/harness.py` need native renderer/sampling parity; ordinary evaluation passes `agent_llm_args(protocol)`, while the branch constructor does not. True-token donor scores must use full normalization, not sampled OTHER-bucket scores.

Maximum acquisition:1,536 initial episode units,12,288 candidate responses and1,536 executed continuations, **total across two parents**. Caps:2M roll-in generated tokens,6M candidate tokens,3.072M continuation tokens;300M native prefill;200M/2M user prompt/output. Two serial9B scorers each get≤6M response and90M prompt tokens; the historical-median prompt proxy is85.26M per scorer. These are proposed envelopes, not measured bills.

Two200-update parents plus six200-update descendants exhaust eight students. At batch64 the eight runs imply102,400 row exposures; the historical-median prefix proxy is710.45M tokens before attention/backward factors. Final evaluation adds21,888 episodes. Current training memory is already near the recorded GPU capacity, and no throughput supports a promised two-night completion. A future protocol-correct timing/memory smoke plus20% reserve is mandatory to claim feasibility; none ran here.

### 11. Designed experiment and failure thresholds

Two parents, then one uniform, one constrained and one generic-stop descendant from each: eight students. Each matched triplet shares parent, cache/order, optimizer reset convention and exposure. Only discovery training pools enter fitting; validation pools are held out. Conditional on verifying563 unique training IDs, use281 discovery/282 validation tasks;384 units per parent per split, with repeat counts and equal parent/task weights. The114 evaluation tasks remain untouched until final model comparisons. The earlier written-plan evaluation split error was corrected before any model work.

**P:** the gain scenario requires an arrival-weighted local intervention around+4pp, with≤1.5pp total loss from task transport, native realization and other changed decisions. The latter is an unverified assumption, not a theorem. Native/whole-policy comparisons must test it directly. A final constrained point estimate≥2pp, positive effects for both parent seeds, and superiority to uniform/generic are predeclared goals; wide intervals remain inconclusive.

Corrected validation effective independent-unit n=688.5195; a conservative95% single-contrast Hoeffding half-width is2.5879pp. Using282 task clusters gives4.0437pp. Paired necessity contrasts have a wider range and need their own variance/multiplicity treatment. In the strong toy, necessity gaps.2895/.3388pp are smaller than optimistic paired half-widths.9084/2.0271pp before clustering. More repeats do not add independent tasks.

Kill the mechanism if attained TV cannot support the effect, no global sign validates, either deletion/generic matches it, missing-return bounds erase the effect, or native/episode outcomes fail despite a positive local contrast. A new native-NLL objective might repair one decoder mismatch, but it would be a revised algorithm requiring its own support/implementation contract; it does not supply the missing five-puzzle theory. This is not yet a decisive two-night experiment meeting the user's bar.

### 12. Risks, premortem and forecasts

Most likely +.3pp outcome: legitimate handoffs need opposite signs, preserved moments are irrelevant, pools are homogeneous, the constrained direction is too weak to distinguish from generic stopping, or native and downstream changes consume the local gain. These are substantive failure modes, not just lack of a model run tonight.

The Phase6 author's subjective forecasts were P(extra≥1pp)=12%, P(extra≥2pp)=4%, with broad judgment ranges5–25%/1–10%. They are not empirical posteriors. After both reviews, root uses approximate planning forecasts **P(extra≥1pp)=5%, P(extra≥2pp)=1%** for the unchanged raw-MLE design. These are subjective judgments, not externally calibrated probabilities, reflecting the explicit native mismatch, weak necessity separation and missing explanatory scope. Reopening needs a meaningful pair-constraint necessity case at the available precision, a native realization algorithm, an explicit episode transport hypothesis and at least three additional genuine directional puzzle accounts.

## Final selection and what would justify reopening

Both candidates received two substantive fresh blind reviews, with a root revision between rounds. Failed launches during the earlier service interruption were not counted. All four completed reviews declined finalist promotion. The separate [selection-standard audit](REDTEAM/SELECTION_STANDARD_AUDIT.md) agrees that neither clears the full bar while defending B as credible conditional research. Missing measured prevalence, a universal neural guarantee, or distribution-free significance is **not** an automatic kill. A harmful toy limits a claimed implication; it does not establish universal failure.

| Requirement | Contextual interaction transfer | Boundary circulation |
|---|---|---|
| Ten verified nearest works with differences | Complete | Complete |
| Five directional conditional puzzles | Five: P1/P2/P3/P4/P6 | Two: P5/P6; unmet |
| Completed relevant toy and controls | Complete; rule tie and AND harm retained | Complete; native and repeated-intervention reversals retained |
| Two fresh review rounds | Complete; unresolved central contribution | Complete; unresolved explanatory/target mismatch |
| Defensible extra2pp case and decisive budget | Conditional scale witness; substantive pair role unresolved | No supported native/episode and necessity chain at stated budget |
| Qualified finalist | **No** | **No** |

B is the better starting point for a future conceptual rescue. It needs one concrete validity-without-labels acquisition family and a domain-grounded reason for useful pair interaction, or an explicitly different hypothesis about heterogeneous pair-derived teaching calibration with a capacity-limited witness and genuine calibration controls. Matching each quad's movement preserves its calibration, so the current rule equality deletes a directional-information claim in the shown regime; it does not prove that all possible pair calibration is redundant. No such rescue was established here.

C's estimator can be reused as a clean local experiment, but the current trained-policy proposal needs a native-law fitting objective, three additional substantive puzzle explanations and a credible separation/variance case. Simply adding diagnostics does not change the loss's probability object. Neither lead is recommended as a ready eight-student launch from this report. The limited-complementarity ceiling remains a live explanation, not a proved impossibility.

## Eliminated optimizer and late alternatives

The third deepened candidate retained the original DOPD target and loss but added a semantic interaction Jacobian to an Adam-containing numerical subspace. The corrected high-support CPU suite lost to ordinary Adam in8/9 mean settings at matched proxy compute; largest mean separation from generic Krylov was only.0583pp. Even free corrections in the representative setting gained only.0142pp. The small-radius derivation explains why weak modes remain suppressed; implementation also needs costly finite secants/snapshots. See [full report](DEEPEN/G3_R.md) and [root audit](DEEPEN/ROOT_TOY_AUDIT.md). This is a negative mechanism/cost result, not proof that all structured optimizers fail.

The final four-candidate reopening found no survivor: signed-measure continuation derivatives are established and consolidate into C01; progressive checkpoint transfer and Lie-bracket order are occupied; feedback prediction plus hindsight self-distillation is closely occupied by [SELF](https://arxiv.org/abs/2610.11384); a whole-episode mixture cannot exceed its best constituent without a new selector/component. [Full algebra, priors and kill reasons](EVOLUTION/LATE_SIGNAL_CANDIDATES.md). A separate six-paper pass tightened abstraction, implementation-law and dynamic-regime assumptions without creating a new signal.

## Process record and appendices

| Artifact | Contents |
|---|---|
| [LEDGER](LEDGER.md), [PUZZLES](PUZZLES.md), [ASSUMPTIONS](ASSUMPTIONS.md) | Corrected facts, protocol scopes and assumptions. |
| [THEORY](THEORY.md) | Competing causes and multi-turn formalization. |
| [LIT_MAP](LIT_MAP.md), [canonical inventory](LIT/ALL_inventory.json), [literature cards](LIT/) | Verified primary works, read-depth limits and mechanism boundaries. |
| [Original generation validation](GENERATION/VALIDATION.json), [candidates](CANDIDATES/), [screening](SCREENING/INTEGRATED.md) |48original cards, kills, mutations and four late attempts. |
| [TOURNAMENT](TOURNAMENT.md) | Three generations,74blind comparisons, preserved inputs and rankings. |
| [Round1](REDTEAM/ROUND1/), [revision response](REDTEAM/REVISION_RESPONSE.md), [Round2](REDTEAM/ROUND2/) | Fresh adversarial reviews, corrections and exact counterexamples. |
| [CPU toys](toys/), [root toy audit](DEEPEN/ROOT_TOY_AUDIT.md) | Reproducible positive/null/harmful mathematical cases. |
| [DECISIONS](DECISIONS.md), [STATE](STATE.md) | Every major selection/rejection and final stopping-rule status. |

The [final process audit](FINAL_AUDIT.json) distinguishes completing the requested research process from meeting the scientific finalist bar. The user explicitly permits an honest best-candidates report when the night ends before that bar is reached; that fallback must not be labeled a successful discovery.
