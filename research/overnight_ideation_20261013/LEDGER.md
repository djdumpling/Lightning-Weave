# Verified evidence ledger

Phase0 consolidation, 2026-10-13 America/New_York. **M** means reported measurement checked in source, **A** means CPU arithmetic on existing artifacts, **C** means code contract, **H** means hypothesis; none of the benchmark results was rerun tonight. Source hashes: `GROUND_TRUTH/source_inventory.json`. LW HEAD `5c521a288eff744c464536896d797661203760b2`; mentor HEAD `5847c58a265ceb4a8ff350bef8c5d92c92f018ac`. Detailed source audit reports live in `GROUND_TRUTH/`.

## Corrections to the request that change the research

1. **W1 pair identity:** weak-student LoopTool BFCL +3.05 overall/+9.25 multi-turn used Qwen3-4B-Base→Qwen3-4B-Thinking-2507, not 8B-Base→AgentForge. The latter belongs to mentor-reported separate EnvFactory results. `configs/looptool_opd/config.py:21–27,58–60`, `docs/agent_efficiency_synthesis.md:3–4`, `GROUND_TRUTH/01_efficiency.md`.
2. **Latest 2507 precision update:** final working arm is +2.0 [+0.2,+4.0] pinned and +2.6 [+0.7,+4.5] corrected at avg@24, superseding historical +2.1/+2.8 estimates with trained models at avg@12. `results/agent_eff/results_7.md:173–198`.
3. **Delta-vs-post advantage:** +1.8 [+0.2,+3.5] pinned, +2.6 [+0.7,+4.4] corrected; the request's +2.6 must carry its task version. `results/agent_eff/results_6.md:185,258`.
4. **Historical vs current user:** results4–7 use Qwen3-235B user; current Qwen3.5 results use DeepSeek-V4-Flash-0731. Comparisons across them are not model-only effects. `results_4.md:19`, `results_9.md:40–48`.
5. **No observed never-solved baseline task:** all114 Qwen3.5 task rates exceed0/24. Reliability, not empirically identified zero-support capability, is the available headroom in this artifact. `GROUND_TRUTH/headroom.md`.
6. **No universal current-step erasure at tool boundaries:** Qwen3.5 retains reasoning after the latest real user message, potentially over several tool calls. Prior2507 collection discarded earlier reasoning. `results_9.md:50–71`, `results_5.md:31`.
7. **Existing observation-delta and phase mechanisms:** mentor already implemented observation double increments, phase balancing, dense/sparse pre×post, and feedback memory controls. Several are NOT_GPU_READY; existence is verified, downstream failure is not. `GROUND_TRUTH/03_mentor_latest.md`.
8. **Nulls have scope:** AUC panel is382 initially mixed states;71.5% agreement is an enriched retest; teacher retry is reexecution of the same sampled turn. Format failures were excluded in teacher rescue. `results_6.md:94–164`.
9. **Cap comparison is not pure coverage:** cap alters selected states, censoring, and5.6M vs12.3M loss tokens. Uniform-vs-priority CI still admits ~1pp. `results_7.md:29–49,96–105,194–206`.
10. **L12 remains unverified locally:** no retained Markdown source located for suffix-MCMC result. Carry as user-reported negative example, not audited measurement.

## Positive evidence

| ID | Verified observation | Interpretation limit / provenance |
|---|---|---|
| W1 | **M:** weak Qwen3-4B LoopTool transfer +3.05 overall,+9.25 BFCL multi-turn under corrected interleaved history. | Pair is Base4B→2507, not AgentForge. See audit01 for exact intervals/config provenance. |
| W2 | **M:** 2507 own uncapped retail states, all3 seeds positive; final+2.0 pinned,+2.6 corrected. Hard unchanged tasks gain~5.4 final/~6 historical; easy unchanged flat. About10–12% shorter. Stay-success rises from53→56% historically, with transfer rate~10% flat. | Stay-success conditions on behavior and is descriptive; difficulty split used disjoint baseline trials in historical analysis. `results_6.md:177–217,266–281`; `results_7.md:184–198`. |
| W3 | **M:** pair delta beats same post's matched geometric-mixture distillation at mean target KL~.069. All distillation seed scores below all delta seed scores in recorded comparison. | Pinned advantage1.8, corrected2.6; not pure full-vocabulary KD. `results_6.md:73–75,168,258,293`. |
| W4 | **M:** Qwen3.5 base78.8012→seed-mean80.5799, +1.7788 [+0.2437,+3.3748] pp. Tokens+13.70/+18.53/+16.54%. Full-cache KL .0214294. | CI conditions on three models;2 individual seed intervals cross0. No capped Qwen3.5 comparator. `results_10.md:9–29,200–215`; saved comparison/audit JSON. |
| W5 | **M:** 2507 cap+0.9 vs uncapped+2.0 pinned; uniform+2.1. Uniform−priority+0.0 [−1.2,+1.3]. | “Coverage matters” plausible but not isolated from token exposure/censoring; cap not worse on BFCL. `results_7.md:140–149,184–206`. |
| W6 | **M:** efficiency shifts cut single-turn token usage with little resolved accuracy effect; replicated harm concentrates in multi-turn. | Magnitude and donor-specific uncertainty vary; no universal physical frontier. `results.md`; `results_2.md`; audit01. |

## Negative evidence and what it actually rules out

| ID | Audited outcome | Ruled-out claim / remaining scope |
|---|---|---|
| L1 | **M:** roughly30 trained efficiency arms yield a strong descriptive accuracy/token relationship; no convincing replicated escape. | Reject another donor/strength/gate sweep as a headline. Does not prove universal bound; regression depends on intercept/model, and individual noisy arms depart. |
| L2 | **M:** empirical decision-marginal projection exactly preserved cached groups and suppressed one requested shift direction; sampled transfer partial and BFCL benefit inconclusive. | Target invariance does not guarantee neural generalization. Not a theorem against all structured objectives. `results_3.md:189–216,239–269`. |
| L3 | **M:** one-step Wait/So and DECS shortening probes find little next-call effect while multi-turn costs persist. | Next-call matching cannot diagnose all trajectory effects. `results.md` sections8,11,12; audit01. |
| L4 | **M:** own Base4B→2507 shift applied to2507: BFCL+0.04 [−.50,.59], retail−.3 [−2.7,2.2]. | This shift/recipient/recipe null; does not invalidate weak-student own-family transfer W1. `results_5.md:104,114`. |
| L5 | **M/C:** broadcast A=log(p_k/pbar), p_k=(R_k+.5)/2, one continuation/response.739/3200 mixed states.×13.3 outcome strength adds BFCL−.37 [−.75,.02], retail−1.1 [−3.2,.9] vs delta. | No demonstrated benefit; outcome-only causal credit not established. Coef1 reused same seed/order, not independent replication. `results_5.md:41,67,121–137`; `results_6.md:13`. |
| L6 | **M/C:** no reliable whole-response AUC in382 mixed states/1253pairs; best local action-span exception fails replication.71.5% agreement on2008 enriched reexecutions. | Do not use per-state success ranking without new identification/noise solution. Does not exclude across-state learning or other representations. `results_6.md:23–27,94–140`; audit05 for estimator semantics. |
| L7 | **M:** all-fail teacher next-turn rescue .031/.033 vs student own-turn reexecution .037; all-success retention AgentForge .896 vs student .976. | Teacher/sample strata differ;34/600 AgentForge malformed excluded; student-formatted prompts. No strong accessible rescue demonstrated. `results_6.md:144–164`. |
| L8 | **M:** coef×2 KL .0697→.2109; BFCL overall−13.22, parallel95→6.3 across seeds. Retail pooled−5.8 with seed9012−22 and48% empty turns. Weak-student400vs200 updates unresolved/null. | Reject unprincipled stronger tilt or longer training. Not proof of optimal target or perfect fitting. `results_7.md:96–105,128,147–168`; `results_2.md:174,245`. |
| L9 | **M:** matched parent→AgentForge roughly same as base→AgentForge (corrected−.5 [−2.3,+1.3]); parent variant+5% tokens. | Shared-post reference contrast is nonlinear at target/task level. It does not isolate “chat polish” by benchmark-score subtraction. `results_6.md:177–181,258`; prior revision:26–32. |
| L10 | **M:** fresh weak-student states vs LoopTool, tau all+2.9 [0,5.7], retail+3.4 [−.3,6.9], BFCL multi-turn−1.40 [−2.77,−.13]. Lower unnecessary escalation explains descriptive retail pattern. | Retail escalation policy improvement is real under policy; stronger grading artifact is airline no-write tasks. Data/strength/exposure change together. `results_4.md:118,134–172,208–229`. |
| L11 | **M/C:** mentor cap, error regeneration, WRITE/first-observed-tool-error selection, sparse attention implemented/reported; fresh6–7 responses are unexecuted. Extra phase/observation/memory mechanisms already present. | Distinguish reported marginal/null outcomes from source-only NOT_GPU_READY mechanisms; do not declare unrun code a failed experiment. Audit03. |
| L12 | **User report, unverified:** suffix-MCMC loses to plain sample pools for math compression. | Use as an idea-screening warning, not a verified quantified result; no relevant local retained Markdown located. |

## Structural facts and unresolved claims

- **S1:** Hard-task concentration is strongest in2507 unchanged-task analysis. Current Qwen3.5 baseline-defined20hard tasks contribute+.402 aggregate pp;53medium contribute+1.633;41perfect contribute−.256. These groups are noisy and descriptive.
- **S2:** Token-mean reduction is code-verified; historical reasoning counts2,891/3,201overall and7,590/8,386multi-turn are about90% (`results_0.md:75–80`). This measures token exposure, **not actual loss or gradient share**; residuals and Jacobians differ. No universal90% for current caches is established.
- **S3:** Legacy Base4B has untrained format-token denominators and large positive special-token shifts; zeroing them hurt. It does not establish identical14–20nat contamination for every external pair. Audit01 and `results_0.md`.
- **S4:** 78.34% singleton visible-response groups after tools is from historical8-response efficiency cache, not current4-response accuracy cache. JSON canonicalization barely changes it. `research/multiturn_algorithm/README.md:25–42`.
- **S5:** Premature first-turn completion and more rejected bookkeeping calls correlate with efficiency harm. Exact16.7→18.3 pair needs its source; audited report supports baseline16.3–17.4% and DECS+1.6 at14.8% savings. Selected wrong-path/cp/mv examples are not a full error taxonomy.
- **S6:** Rendering is model/protocol-specific. Current Qwen3.5 reasoning persists after latest real user; BFCL interleaved thinking does likewise within user turn; historical2507 collection omits old thought.
- **S7:** Anchors are not demonstrably stronger than strong recipients on the measured tau profiles. Leaderboard inferiority does not preclude a useful complementary shift; rescue weakness does constrain naive teacher imitation.
- **S8:** Historical800-row audit Top16 mass .999871, top1>.99 at74.4%of positions, and only.033%of DECS incremental KL changes candidate-vs-tail mass (`research_audit.md:161–180`). Current Qwen3.5 smoke mean Top16 support mass .9998593 is high. It neither proves full-cache support sufficiency nor supports a broad “missing most token mass” story. Full corrected cache has5,539,958tokens.
- **S9:** Qwen3.5 frozen53→78 jump is joint history/sampling repair; **not training**. Protocol integrity dominates some apparent model gaps. `results_8.md:98–111`; `results_9.md:12–23`.

## Headroom and power constraints

From existing CSV: baseline78.8012%, current trained mean80.5799%, remaining error19.4201pp. Additional2pp means removing10.30% of remaining error, about55 extra successes per2736 rollouts. If all gain is on73inconsistent tasks, required within-group gain3.123pp; if on20hardest,11.4pp. Pure preservation of base-perfect tasks has only.256pp observed headroom. These are arithmetic requirements, not evidence of reachability. See reproducible `GROUND_TRUTH/headroom_analysis.py`.

Task bootstrap intervals condition on observed models; training seed spread is a separate uncertainty source. With only114tasks, avg@24 and~1.6–2pp task intervals, a decisive future comparison should use paired task/trial protocols and multiple seeds, pre-register a practically meaningful effect and avoid treating a nonsignificant +.3 as validation. Proposed eight-student budgets must include contemporary matched controls; historical users/renderers are not interchangeable.

## Prior-ideation quarantine

No novelty credit for plain soft Bellman/normalizer correction, action-marginalize-before-compose, hierarchical outcome targets, backward suffix curriculum, matched hidden worlds, evidence-sensitive odds, fixed-anchor visitation refresh, phase balancing, observation double increments, or teacher/error-step intervention alone. Full map in audit04. The existing Bellman sweep improved the ratio surrogate while reducing task reward in470/1620 constructed settings; loops and exogenous-outcome tilting are known counterexamples. New use must identify and repair the precise old failure, not rename it.

- **S10:** Mean target KL hides concentration:41–46%of added KL lies in1%of positions, with maxima8–9nats (`results.md:518–519`). This is an existing diagnostic, not a license to propose another clipping/gating knob.

## Final implementation qualifications

The code audit (`GROUND_TRUTH/05_code.md`) confirms full configured-vocabulary normalization, fixed cached behavior and global masked token-mean squared-log fitting. It adds three consequential limits: synthetic pre/post arrays are reward encodings, not normalized models; the outcome bonus broadcasts onto the sampled bucket (whole OTHER for a tail token); historical branch replay drops reasoning and cannot be assumed valid for current Qwen3.5. AUC bootstraps prompt states, not parent trajectories, and the student-logprob control uses coarse tail-bucket probability. These narrow existing probe interpretations without inventing a positive signal.

- **S11:** Recorded Qwen3.5 first/final training minibatch losses are .072788/.027311, .068245/.020757 and .076056/.018898 across the three seeds (`results/agent_eff/results_10.md:210–213`). These are different minibatches, not a matched fixed-cache convergence audit. They do not establish perfect target fitting, a positive irreducible floor, or that longer training would help. The fixed-batch four-step smoke reduced .061834→.036424 (`:154–168`), an optimizer sanity check only.

  CPU block summaries of all 200 recorded minibatches confirm early decline followed by slower/flat last-half loss: last four 25-update means are .02644/.02522/.02580/.02653 (1234), .02193/.02099/.02084/.02040 (5678), .02189/.02088/.02109/.02048 (9012). This is descriptive optimizer behavior, not a convergence theorem; the schedule and changing batches remain confounded. Source hashes and full block means: `GROUND_TRUTH/training_loss_summary.json`.

- **S12 / noise-floor correction (A):** The current Qwen3.5 approximately ±1.6pp interval is for the **paired trained-minus-base difference**, conditional on the three trained models, not for an individual model's absolute score. CPU task bootstrap absolute-score intervals are roughly ±5pp (base [73.83,83.52], trained-mean [75.55,85.27]); paired difference [+0.195,+3.399] independently reproduces the saved estimate's scale. Across-task SD of paired mean differences is8.715pp (SE.816). The observed three seed-run scores have SD.428pp, while conditional independent-rollout SE per trained model is.54–.58pp; training-only variance is not identified. New-method paired covariance is unknown. Four-vs-four +2pp power is conditional on task/seed variance and can be poor; +1pp is harder. A two-sided exact sign-flip test over four seed pairs cannot reach5% (minimum p=.125). Detecting a positive effect differs from showing its lower bound exceeds2pp. See `GROUND_TRUTH/uncertainty_design.md` and reproducible `toys/uncertainty_design.py/json`. No benchmark was rerun.
