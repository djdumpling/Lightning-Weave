# What eight future trained students can establish

**A +2 percentage-point gain could be detectable with eight students, but the existing artifacts cannot establish its power. A +1 pp gain would usually be difficult to resolve.** The defensible upper-budget comparison is four contemporary control students and four new-method students, with a prespecified comparison. Neither four seeds nor 114 tasks guarantees a decisive result. This document concerns statistical design only; it proposes no algorithm and executes no training or evaluation.

All calculations use the existing `outputs/qwen35_transfer_20261012/task_rates.csv`, `comparison.json`, `summary_bootstrap_crosscheck.json`, and protocol metadata. These are 114 retail tasks, 24 recorded trials per task/model, one frozen base and three trained students. CPU reproduction is in `toys/uncertainty_design.py`; numerical outputs and the source SHA-256 are in `toys/uncertainty_design.json`. All effects below are percentage points, not relative percentages.

**First separate the claims.** A fixed-suite claim asks for expected success on these 114 tasks under fresh rollouts. A task-population claim generalizes beyond this task set, requiring an exchangeable-task assumption that benchmark selection itself does not establish. A recipe claim also generalizes over future training runs. A confidence interval conditional on three particular trained models does not automatically support the recipe claim. Trials improve estimates within a task; tasks measure heterogeneous task effects; independently trained students measure training variation. These three replication units are not interchangeable.

The observed contemporary recipe scores 80.2632, 80.4094, and 81.0673; its mean is 80.5799, versus frozen-base 78.8012. The reported +1.7788 interval [+0.2437,+3.3748] resamples paired tasks and conditions on the observed trained models. It answers a different comparison from the intended *additional* +1/+2 beyond this recipe. A future +2 target would be about 82.58 if that control mean held, but the actual contemporary control must be measured.

**Task uncertainty and the value of pairing.** The between-task SD of each model's observed rate is about 26.7 pp, whereas the SD of the paired current-minus-base task differences is much smaller:

| Observed quantity | Across-task SD | SE across 114 tasks |
|---|---:|---:|
| Frozen-base task rates | 26.789 | 2.509 |
| Trained three-seed mean task rates | 26.652 | 2.496 |
| Seed 1234 minus base | 10.191 | 0.954 |
| Seed 5678 minus base | 9.066 | 0.849 |
| Seed 9012 minus base | 10.850 | 1.016 |
| Three-seed mean minus base | 8.715 | 0.816 |

A separate 20,000-draw Python task bootstrap gives a base-score interval [73.83,83.52], a trained-mean score interval [75.55,85.27], and a paired-difference interval [+0.195,+3.399]. It uses a different RNG and seed from the saved 10,000-draw NumPy calculation; these are independent numerical checks, not exact reproductions of its endpoints. The absolute-score intervals are roughly ±5 pp; the paired-difference interval is roughly ±1.6 pp because task difficulties largely cancel. Overlap of absolute-score intervals is consequently not a valid test of the paired effect.

The observed current/base task-rate correlation is 0.947 for the trained mean. **This does not identify the future new/current effect covariance.** A method that changes different tasks can have much larger paired variance despite similar average accuracy. Pair tasks in every comparison, but do not promise the historical paired interval will recur.

**Rollout uncertainty.** For independent Bernoulli outcomes within each task, the estimated conditional variance of a model's fixed-suite mean is

`V_roll = sum_j[p_hat_j (1 − p_hat_j)/(m − 1)] / T²`,

with rates expressed as probabilities, `m=24`, `T=114`. The factor `m−1` supplies the finite-sample unbiased estimate of `Var(p_hat_j)`. This gives SE 0.605 pp for base and 0.575, 0.576, and 0.542 pp for the trained students. It is a conditional calculation under independent trials, not proof that the simulator/reexecution process obeys that assumption. Zero estimated variance for an observed 24/24 task does not establish true probability one.

If all model evaluations were independent, the three-student mean minus the **one shared base evaluation** would have rollout SE 0.687 pp. Averaging three differences does not divide that shared baseline noise by three. Four current versus four new students, each avg@24, would have rollout SE about 0.399 pp if both recipes have the current average rollout variance. With more trials it scales approximately as `sqrt(24/m)`. Positive paired rollout covariance could reduce this; negative covariance increases it. Matching a seed label or task/trial ID is not evidence of a particular outcome covariance, especially after trajectories diverge. The CSV has marginal task rates, not the paired binary outcomes needed to estimate trial-level covariance.

Observed task bootstrap variation already contains finite-rollout noise. Do not add 0.687² to the observed 0.816²: that double counts it. Subtracting these variances suggests residual task SE about 0.441 pp, but this is only an illustrative variance-component subtraction under the above assumptions, not a robust estimate of future method heterogeneity. Treating all 2,736 conversations as independent identical binomials would also erase the task-population uncertainty and ignore unequal task probabilities.

**Training-seed uncertainty.** The three trained aggregate scores have sample SD **0.428 pp**, range 0.804 pp. That spread combines training differences and evaluation noise. It is smaller than the conditional per-model rollout SE, so these three scores cannot separately identify a positive training-only variance; a negative variance subtraction is not evidence of zero training variance. Even under iid normal *seed-run* scores, a 95% interval for their population SD is about **0.223–2.692 pp**. This interval is itself assumption-dependent, and three runs cannot diagnose tail risk. Resampling three seeds does not manufacture information about unobserved failures.

Matched training seeds/data order can help if they induce positive new/current covariance. For population training SDs `sigma_new`, `sigma_control`, and correlation `rho`, the paired-training variance is `sigma_new² + sigma_control² − 2 rho sigma_new sigma_control`. None of those new-method quantities is observed. The historical 0.428 pp is not a warranted plug-in for this paired training SD.

**Transparent conditional power for four versus four.** The following is a sensitivity analysis, not an empirical power estimate. Assume independent, exchangeable tasks; four independent seed pairs; independent rollout noise between arms; comparable per-model rollout variance; an additive task/seed decomposition; and no additional seed-by-task interaction. Let `sigma_T` be SD of the true paired task effects and `sigma_S` the SD of paired training effects. With variance components treated as known,

`SE² = sigma_T²/114 + sigma_S²/4 + 0.399²`.

An extra interaction component would need its own term. All table powers use a two-sided 5% normal test. They are optimistic about variance estimation with only four pairs. The exact normal power calculation is `Phi(delta/SE − 1.96) + Phi(−delta/SE − 1.96)`; 80% minimum detectable effect (MDE) is approximately `2.802 SE`.

| Assumed task-effect SD | Assumed paired-training SD | Total SE | Power: +1 pp | Power: +2 pp | 80% MDE |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.25 | 0.562 | 43% | 95% | 1.57 |
| 4 | 0.50 | 0.602 | 38% | 91% | 1.69 |
| 4 | 1.00 | 0.741 | 27% | 77% | 2.08 |
| 8 | 0.25 | 0.858 | 21% | 64% | 2.40 |
| 8 | 0.50 | 0.885 | 20% | 62% | 2.48 |
| 8 | 1.00 | 0.985 | 17% | 53% | 2.76 |
| 12 | 0.25 | 1.199 | 13% | 39% | 3.36 |
| 12 | 0.50 | 1.219 | 13% | 38% | 3.41 |
| 12 | 1.00 | 1.293 | 12% | 34% | 3.62 |

These assumptions span useful scenarios; they are not confidence bounds inferred from three old seeds. In particular, the 8.715 observed task SD already includes rollout noise and cannot be inserted as a noise-free `sigma_T` without qualification.

For sample-size intuition, under this same optimistic normal calculation with paired-training SD 0.5 and avg@24, approximately 527/1,266/2,499 exchangeable tasks would support 80% power for +1 at task-effect SD 4/8/12. For +2, those counts fall to 77/184/363. These are hypothetical independent task samples, not permission to reuse trials or closely related tasks as extra task units. With paired-training SD 1 and four pairs, even infinitely many tasks/trials leave SE 0.5; that training floor alone misses 80% normal power for +1. More trials cannot eliminate task or training uncertainty.

**The four-seed limitation persists on a fixed suite.** Suppose instead the claim is explicitly limited to these 114 tasks. Under iid normal aggregate paired seed-run differences, including their evaluation noise, the paired t test has only three degrees of freedom and critical value 3.182. Exact noncentral-t power, numerically integrated in the script, is:

| Assumed aggregate paired-run SD | Power: +1 pp | Power: +2 pp | 80% MDE |
|---:|---:|---:|---:|
| 0.5 | 75% | >99% | 1.06 |
| 0.8 | 41% | 90% | 1.70 |
| 1.0 | 29% | 75% | 2.13 |
| 1.5 | 16% | 45% | 3.19 |
| 2.0 | 11% | 29% | 4.26 |

Here the SD includes rollout noise; do not add another rollout term. Independent avg@24 evaluations alone imply about 0.798 pp paired-run SD at the observed variance, so the 0.5 row requires favorable covariance, greater evaluation replication, or lower outcome variance. This fixed-suite table excludes uncertainty from generalizing to other tasks and cannot replace the task-population table. Normality of four differences is also untestable in practice. A conventional two-sided exact sign-flip test over four prespecified pairs has only 16 assignments and minimum attainable p=2/16=0.125; a distribution-free 5% paired sign-flip rejection is unavailable at that sample size.

**Decision implications within eight students.** Four controls plus four candidate students uses the budget for the strongest balanced single contrast under similar arm variances. Allocating students to multiple candidates or ablations reduces replication for the decisive comparison. Preserve the contemporary renderer, user, task version, generation settings, and evaluation schedule; the historical frozen base is not a trained-current-recipe control. Preserve task/trial identities and record joint outcomes for assessing covariance, without assuming it in advance.

Specify the primary estimand, practical effect threshold, one primary comparison, and uncertainty procedure before seeing outcomes. Report individual runs, paired task effects, rollout assumptions, and recipe uncertainty separately. Crossed task/seed resampling or a hierarchical analysis can reflect both sampling axes, but four seed pairs still provide weak information about the seed distribution. Screening candidates on the same benchmark and reporting only the winner introduces selection bias; a large candidate funnel does not justify an unadjusted confirmatory claim.

Finally, detecting superiority to zero is easier than demonstrating that the gain is **at least** +1 or +2. A 95% interval whose lower endpoint exceeds +2 requires the estimate to exceed +2 by its uncertainty margin; a true effect exactly +2 will rarely satisfy that stronger criterion. Conversely, a nonsignificant estimate is not equivalence to the current recipe. The useful outcome is a calibrated interval relative to the practical threshold, including an honest “unresolved” region, rather than a forced positive/negative verdict.

**Allocation supplement (root CPU calculation).** Under the same hypothetical fixed-suite iid-normal paired-run model, using three primary seed pairs rather than four reduces +2pp t-test power from89.9% to61.9% at paired-run SD.8pp, and from75.5% to47.1% at SD1pp. A3+3+2 ablation allocation therefore trades substantial primary replication for mechanism identification. These are conditional scenarios, not measured future-method power; raw rows are in `toys/uncertainty_allocation.json`. The integration is the same `nct_power` function in `toys/uncertainty_design.py` with pairs=3 versus4.
