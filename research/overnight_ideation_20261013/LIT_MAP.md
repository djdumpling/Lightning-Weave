# Literature mechanism map

**Phase 1 complete.** Six scouts produced 87 cards with 30 deep reads; root added five deep cards. There are 92 unique primary-verified records, including 91 arXiv records and one official-proceedings-only record (SMILe, requested arXiv/OpenReview identifier unverified). All 16 supplied arXiv pointers matched titles; the Lightning Weave author attribution and several method aliases required correction. This map separates useful mechanisms from the assumptions that make them work. All numerical results are source reports, not experiments performed tonight. Canonical metadata and read depth live in each scout inventory. A verified title is not a verified theorem or replication.

## What kind of information is being transferred?

| Family | Representative primary sources / cards | Enabling assumption | What it does not give us for free |
|---|---|---|---|
| Recipient-prefix distillation | GKD2306.13649, MiniLLM2306.08543, Semi-OPD2610.11291; T1 | Teacher feedback is useful on recipient prefixes; finite student can realize enough of it | A stronger generation teacher, success ranking, or a benefit from refreshing the cache |
| Capability differences | EFT2310.12962, Proxy Tuning2401.08565, relative transfer2610.13205, LW2609.14708; T2/ROOT | Pair change is useful and compatible at recipient states | Identification of environmental reward, safe extrapolation, sequence optimality |
| Contextual self-teaching / subtraction | OPSD2601.18734, Purified OPSD2607.02234, CoDA2608.08764; T1 | Hints, references or consensus contain additional useful information | Correctness from confidence; new information from a renamed self-ratio |
| Sparse branch calibration | SPOT2608.04419; T1 | Local interventions and verifier continuations have informative contrasts | Reliable local credit from the existing one-sample noisy outcomes |
| Future bridges and recovery | FutureBridge2608.01953, PivotOPD2609.40285; T1 | Replay is faithful, corrections are executable, donor or privileged hint is useful | Teacher rescue under our renderer; future preference as a task-success guarantee |
| Delta as update geometry | OPRD2609.08798, ICSD2608.14945; T2/T3 | An informative recipient advantage/gradient already exists | Signal creation; a population parameter-space guarantee from token-logit algebra |
| Outcome-trained ratio / prefix value | Free PRM2412.01981, PRIME2502.01456, Q#[2502.20548](https://arxiv.org/abs/2502.20548); T2 | Labels ground the learned surrogate; value model generalizes; exact reduction has transition assumptions | A calibrated value function from arbitrary frozen checkpoints |
| Hierarchical critics and privilege | ArCHer2402.19446, SWEET-RL2503.15478; T3 | Cross-state conditional means or privileged reference information are predictable | Identifiable local causal values merely from trajectory preference sums |
| Repeated-state credit | GiGPO2505.10978; T3 | Valid common information state and enough return variation after trajectory collection | Equivalence of database state and full agent history, or free rollouts |
| Diversity and conditionality | RAGEN2504.20073, RAGEN2 [2604.06268](https://arxiv.org/abs/2604.06268); T3 | Return variation reflects controllable differences; cross-context dependence tracks useful reasoning | Separation of policy signal from user noise; correctness from input dependence |
| Executable synthetic environments | EnvFactory2605.18703, AReaL-SEA2601.22607, LoopTool2511.09148, AgentForge2511.05951; T3 | Checked tool semantics, valid user/world simulation, sufficient data diversity | A drop-in precomputable signal already stored in this cache |
| Broad prerequisite coverage | Coverage Not Targeting2609.02417; ROOT | Terminal feedback can train distributed prerequisites; compare matched concentration and scale | Proof that every credit method must be uniform, or large gains on the full strong-student protocol |
| Counterfactual variance control | Event-keyed CRN2603.11084, Counterfactual Shapley2607.16999; ROOT | Replayable simulator, explicit coupling, positive covariance / valid counterfactual estimand | Causal identity from a shared seed, or unbiased local labels without additional execution |
| Recipient-compatible trace sampling | Karan2610.02140; ROOT | Useful semantic equivalence class, valid proposals, adequate mixing | Cheap causal suffix replay through tools/users; improvement over a matched sample pool |

## The ratio identification ladder

A pair always identifies a change in normalized probabilities. An exact one-step KL optimum further identifies reward differences up to a state constant and scale. A sequential optimum instead entangles reward with continuation value. A ratio fitted to outcomes obtains its task semantics from those labels, not from the algebraic density identity. A donor direction used to transform gradients obtains improvement semantics from the independently supplied gradient. These are different objects with different assumptions.

Exact sequence normalization repairs the target for its **chosen surrogate**. It cannot certify that the surrogate is episode success. Likewise, adding an outcome term is informative only to the extent its estimator captures a controllable contrast. A mathematically elegant transformation can preserve all the original information limitations.

## Three implementation distinctions that theories must retain

1. Current DOPD fits behavior-weighted **squared log residuals** over a full-normalized Top16+OTHER distribution, using a global token mean. It does not optimize the conceptual reverse-KL objective directly. Tabular target equivalence is weaker than neural training equivalence.
2. Current Qwen3.5 retains reasoning until the next real user message, including intervening tool exchanges. Identical visible tool calls need not yield identical later agent inputs. State abstractions and replay must account for that.
3. Agent action sampling and stochastic environment transitions are distinct. The current retail user is temperature zero; neither that fact nor unstable sampled continuations settles whether transitions on the actor's information state satisfy a deterministic theorem. See A19.

## What the negative ledger demands of a new mechanism

L1/L4/L8/L9 eliminate strength, donor, reference and simple gate changes as a headline. L2 eliminates inference from target invariance to sampled invariance. L5/L6 force an explicit signal-identification or noise argument for outcome grounding. L7 forces separation of generation competence from privileged evaluation or structured knowledge. L10 makes state refresh an existing ingredient with transfer tradeoffs. L11 reserves observation increments, phase balancing, error selection and memory controls. L12 requires a same-budget plain sample-pool control for any suffix chain.

The positive evidence still matters: a weak donor's *difference* can help a strong recipient; every current baseline task has at least one observed success; opposite token effects can both improve accuracy. These facts support several competing explanations. They do not uniquely establish reward transfer, belief repair, finite-fit interference, temporal credit, or a reachable extra two points.

## Proof and evaluation cautions

Output-coordinate stationarity does not automatically survive shared neural Jacobians and sample averaging. A stopped-target score estimator does not imply that its policy-dependent literal objective has the same optimum; the ROOT Shapley card records a one-step exact counterexample to that conflation. A deterministic response tree is not automatically an interactive POMDP. A small-subset single-update study is a source of mechanisms and controls, not a directly transferable effect estimate.

## Partial observability and decision value

T5 distinguishes latent-state decoding from belief recovery. A privileged expert's chosen action need not be executable from the public history, even with unlimited imitation data. The relevant decision object conditions on the recipient's history and, when needed, averages value over its belief. Belief averaging, privileged critics, and this impossibility distinction are established theory ([2412.00985](https://arxiv.org/abs/2412.00985), [2510.03207](https://arxiv.org/abs/2510.03207)), not a new DOPD idea.

Information gain is not decision value: an observation may reveal an irrelevant fact, duplicate what native continuation would discover, or satisfy a required confirmation rule regardless of uncertainty. REVOIR2609.37588 and CLEAT2610.12655 occupy explicit ask/act value comparisons. IGSD2609.32694 occupies executed, paired query interventions with controlled evidence scoring; CROP2608.13387 occupies nuisance-controlled prompt sensitivity. Both require extra information or scoring and neither supplies a ready retail action-value oracle.

These mechanisms become relevant only if residual errors actually involve missing or misused information, a fact not established by aggregate benchmark headroom. Strong-recipient saturation could instead arise from limited donor novelty, finite fitting, or target corruption. The theory phase must keep these alternatives alive.

## Imitation and regularized control: the conditions that matter

DAgger1011.0686 and cost-to-go imitation require useful feedback and recipient recovery; interaction alone is not a guarantee. Modern horizon analyses ([2407.15007](https://arxiv.org/abs/2407.15007),[2606.30445](https://arxiv.org/abs/2606.30445)) separate reward scale, realizability and probability-query access. DOPD already has the latter at sampled recipient prefixes. Expert demonstrations, probability queries, and occupancy refresh must not be conflated.

Hunt1812.02216 and Adamczyk2212.01174 derive continuation/divergence corrections under identified component rewards, values and transition assumptions. They do not turn arbitrary signed checkpoint differences into those quantities. IRL identifiability results ([2106.03498](https://arxiv.org/abs/2106.03498),[2209.10974](https://arxiv.org/abs/2209.10974),[2411.15951](https://arxiv.org/abs/2411.15951)) make that missing bridge explicit. RL's Razor2509.04259 uses idealized projection geometry; the actual squared-log neural fit does not inherit it by name. Cached token-mean KL also differs from policy-occupancy-weighted trajectory KL.

## Frontier boundaries through October 13

OPD²2607.15161 and CompassOPD2609.10154 further occupy centered/anchored checkpoint differences. Lightning OPD2.0 [2607.28449](https://arxiv.org/abs/2607.28449) occupies cross-fitted recurring-style residualization. Their selected or abstract-level cards limit the precision of novelty claims; any close finalist must trigger a full mechanism comparison.

ASCENT2610.05303 adds verified retrospective context to a self-teacher; ActFirst2609.36608 changes collection through reference-conditioned inverse dynamics; STRIDE2609.14636 changes stopping/restarts using teacher confidence. Each buys a different signal or distribution change, with exact replay, reference quality and protocol costs. None establishes an extra two points over the current strong DOPD baseline. Skills-not-Knowledge2610.09639 separates prefix source from loss direction in controlled transfer, but its operational definitions do not prove a general no-new-knowledge theorem.

Top-K2607.07050 studies teacher-restricted normalized JSD, whose omitted-coordinate zero-gradient property differs from our behavior-support OTHER objective. Its temporal migration and required-call regressions make broad support restoration an occupied and risky remedy. Average support mass alone is still insufficient to dismiss rare consequential coordinates.

## Search coverage and verification record

All six requested themes meet the >=10-paper and 3–5-deep-read bar. Inventories and source hashes: LIT/ALL_inventory.json and LIT/VALIDATION.json. Prompt pointer resolutions: LIT/PROMPT_CITATIONS_VERIFIED.json. T1–T6 indices link every card and disclose exact read depth; no abstract-only item is presented as a theorem audit.

Visible ICLR2027 searches encountered OpenReview browser challenges and API403. ASCENT's arXiv paper is verified; its discovered OpenReview/venue pointer remains unverified. No primary-verified citing paper for Lightning Weave was found in the bounded frontier scan; this is not an absence claim. SMILe is verified at official PMLR, with no requested arXiv/OpenReview identifier found. The independent three-paper claim audit and exact Shapley arithmetic check are retained in LIT/ and toys/.

## What enters Phase 2

Keep the ceiling null alongside at least four genuinely competing causes. Distinguish measured ledger evidence, conditional literature mechanisms, and new hypotheses. A useful theory must specify the information state, whose continuation matters, what the actual loss learns, how that changes occupancy, and which facts would falsify the account. No candidate has been generated or favored in this phase.

## Adjacent-field supplement during blind generation
Six primary-abstract cards added, bringing the map to98 unique records,35 deep reads. [Successor features](https://arxiv.org/abs/1606.05312) require transferable dynamics/value features; [reward machines](https://arxiv.org/abs/2010.03950), [coupled agendas](https://arxiv.org/abs/2510.27329), and [uncertain event interpretation](https://arxiv.org/abs/2406.00120) occupy structured temporal-task mechanisms. [Proximal RL](https://arxiv.org/abs/2110.15332) requires identifying proxy/bridge assumptions; it is not a name for removing arbitrary branch noise. [Offline bisimulation pitfalls](https://arxiv.org/abs/2310.17139) warn that missing transitions can undermine state equivalence. All six have primary metadata verification; proximal RL, offline bisimulation, and coupled agendas additionally received targeted method/assumption reads. None is counted as a complete deep read or proof audit; see ROOT_ADJ cards.

Seven additional cross-field primary-abstract cards from generator D bring the inventory to105 unique records (104 arXiv, one PMLR),35 deep reads. [Control barriers](https://arxiv.org/abs/1609.06408) require a valid dynamics/safe-set model; [double/debiased learning](https://arxiv.org/abs/1608.00060) requires orthogonality and nuisance-rate conditions; [information-directed sampling](https://arxiv.org/abs/1403.5556) requires a meaningful posterior and regret/information tradeoff. [Refined minimax regret](https://arxiv.org/abs/2402.12284) and [aleatoric environment design](https://arxiv.org/abs/2207.05219) distinguish learnable gaps from environmental difficulty/noise. [Residual robot control](https://arxiv.org/abs/1812.03201) assumes a useful base controller and learnable residual; [counterexample-driven probabilistic program synthesis](https://arxiv.org/abs/1904.12371) assumes formal program sketches and checkable specifications. These sources establish occupied ingredients, not automatic agentic novelty or ready-made supervision.

## Phase4 primary-source additions and sharper boundaries

The inventory now covers140 unique primary records (139 arXiv, one PMLR),142 cards with duplicate independent reviews retained, and39 full/deep reads. Two earlier abstract-only delta papers and two new causal-credit papers have separate full-method supplements. Read depth remains explicit; a full read does not imply a verified theorem or replication.

[Agent-C](https://arxiv.org/abs/2512.23738) occupies temporal-rule/SMT-constrained language-agent action generation, including retail/airline; [MemTX](https://arxiv.org/abs/2607.23929) occupies transactional belief commitment, provenance and cascading memory repair. [PaW](https://arxiv.org/abs/2606.02388) occupies shared-policy next-observation auxiliary training. These are stronger novelty boundaries than generic classical citations alone. Merely rescoring the resulting wrapper or labels with a pair is insufficient as a new core.

[CVT-RL](https://arxiv.org/abs/2606.05263) occupies frozen-continuation intervention credit and outcome-model/importance-weight correction. Its displayed terminal-ratio estimator is consistent with a correct unclipped ratio, but its stated generic outcome-model robustness does not follow when the model conditions only on pre-continuation variables. A relevance predictor is also not automatically the actual step-inclusion propensity. [IAPO](https://arxiv.org/abs/2608.24588) occupies typed support-use graph routing of trajectory advantage; conservation of scalar token mass does not imply gradient conservation or causal credit. Its cited appendices are missing from the retrieved v2 paper, and its BFCL-MT gains are small relative to reported seed spread. See `LIT/SUPP_causal_credit_mechanisms.md` for exact equations, controls, protocol and cost limits. Neither licenses an extra-two-point prediction for strong DOPD.

Other occupied ingredients include [frozen-unembedding contextual edits](https://arxiv.org/abs/2610.01170), [entity-binding/provenance defenses](https://arxiv.org/abs/2606.30531), [set invariance](https://arxiv.org/abs/2505.15433), [offline harness control](https://arxiv.org/abs/2607.05458), [agentic evolution-strategy training](https://arxiv.org/abs/2608.17310), [symbolic semantic loss](https://arxiv.org/abs/1711.11157), [orthogonal gradient descent](https://arxiv.org/abs/1910.07104), and [geometry-adaptive meta-preconditioning](https://arxiv.org/abs/2304.01552). Exact differences and source read depths are in SCREEN_ABC/DEF cards and screening reports. No absence-of-search-hit argument establishes novelty.

## Phase5 adjacent novelty boundary
Five more verified primary records bring the inventory to145 unique records/147 cards/39 full deep reads. Targeted reads of Multi-Agent Imitation Learning with Copulas (2107.04750) and RouteOPD (2609.08337) sharpen the occupied space: separate marginal/dependence modeling and marginal substitution, and integrable pairwise odds fitting, respectively. RKD (1904.05068), MAGDi (2402.01620), and bi-causal transport (2605.17271) delimit relational supervision and nonanticipative coupling. These additions are not finalist endorsements and do not change sealed G1 judge inputs. Cards record exact read depth.

## G2 relational full-method audit
Inventory now149 unique primary records,151 paper cards,41 full deep reads. SUPP_relational_mechanisms.md upgrades copula imitation and RouteOPD, adds discrete copula reconstruction (1901.08741) and invariant-teacher/equivariant-student distillation (2012.09398), and checks EFT/RKD methods. F06 equals a familiar multiplicative ratio followed by IPF; C03 changes interaction-residual weighting; F01 preserves the base odds discrepancy. Exact constructions were not located, but routine-combination risk is high. Correctness requirements do not themselves earn novelty credit.

## Interaction-specific follow-up after G3 sealing
CIP2607.08776 already makes interaction-aware LLM distillation an explicit training method: it penalizes complex masked-word interactions while ordinary KD transfers salient teacher behavior. TabDistill2604.13332 and ParaACE (OpenReview fQTlgI2qZqE) transfer interaction structure into an interpretable student function class. Therefore neither interaction extraction, selective structure transfer nor crossed differences can be the novelty claim. The G3 distinction is narrower: a signed checkpoint-pair policy-odds component retaining recipient lower-order terms, or a semantic numerical basis for fitting the unchanged DOPD objective. These works provide adjacent positive mechanisms and strong controls, not empirical support for our multi-turn headroom. See G3_INTERACTION cards and search log. They were not injected into already sealed blind judging.

## Late adjacent mechanism audit

Six additional primary records bring the canonical inventory to180 (178arXiv, one PMLR, one OpenReview);41 full deep reads remain unchanged. Targeted reads are not full reads. [The synthesis](EVOLUTION/LATE_MECHANISM_INSIGHTS.md) investigates lossy causal abstractions, module interfaces and partial ordering of adaptive regimes. Root reverified metadata, checked current revisions for2402.14942 and2509.25123, and reran all three finite counterexamples.

No candidate is promoted. An abstract action requires an explicit law over its concrete implementations; endpoint behavior does not identify internal module conventions; marginal fixed-action effects need not identify an adaptive regime. These are precise limits, not empirical diagnoses or evidence of a new transferable signal. Full randomized conditional records can resolve the toy adaptive ambiguity. The appendix records that distinction.

## Final new-signal reopening

Six more primary records bring the inventory to186unique works (184arXiv, onePMLR, oneOpenReview), with41full deep reads unchanged. [Four additional algebraic attempts](EVOLUTION/LATE_SIGNAL_CANDIDATES.md) tested signed policy derivatives, donor checkpoint paths, feedback modeling and complete-episode mixtures. None survived. Weak-derivative rollouts are occupied by2004.04843; progressive checkpoints by2410.05464/2601.10114; transfer-order commutators by2606.24993. SELF2610.11384 directly joins feedback prediction with hindsight self-distillation in agents, including retail, under a materially different protocol. An episode mixture cannot exceed its best component without a new selection or component-learning signal.

The signed-derivative estimator is a useful C01-family implementation residue, not a novel standalone proposal. No external paper's benchmark gain is used as a forecast for this strong-student recipe. Root independently reverified all six primary metadata records and inspected the closest SELF/weak-derivative/order method pages.
