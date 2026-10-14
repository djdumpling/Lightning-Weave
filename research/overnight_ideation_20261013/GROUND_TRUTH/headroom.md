# Existing-artifact headroom (descriptive)

Source: `outputs/qwen35_transfer_20261012/task_rates.csv` (114 paired tasks, base and three trained seed rates, each avg@24). Reproducible CPU calculation: `headroom_analysis.py`; original calculation output `headroom.json`.

Measured arithmetic: base 78.8012%, seed mean 80.5799%, gain 1.77875 pp. There are **zero** tasks with base rate 0/24, **73** with rates strictly between 0 and 1, and **41** with 24/24. Thus this artifact cannot partition a never-solved set versus a sometimes-solved set as suggested in the request: every task is observed solvable at least once by base. It does not prove nonzero probability of every possible required subskill.

| Baseline-defined group | Tasks | Base mean | Trained mean | Contribution to aggregate gain | Remaining aggregate error |
|---|---:|---:|---:|---:|---:|
| Sometimes solved | 73 | 66.895% | 70.072% | +2.035 pp | 19.164 pp |
| Always observed solved | 41 | 100.000% | 99.289% | −0.256 pp | 0.256 pp |
| Hard, base ≤50% | 20 | 29.792% | 32.083% | +0.402 pp | 11.915 pp |
| Medium, 50%<base<100% | 53 | 80.896% | 84.408% | +1.633 pp | 7.249 pp |

The claim that gains concentrate on hard tasks is established in historical 2507 comparisons; this Qwen3.5 split gives most aggregate gain on the medium group. No causal extrapolation across students is justified. Baseline-defined groups have selection and regression-to-mean bias.

An additional +2 pp above 80.58 is 54.72 expected extra successes per 2,736 rollouts, or 10.30% of remaining errors. If confined to the 73 inconsistent tasks, it requires +3.123 pp average within them; if confined to the 20 hardest, +11.4 pp within them. These are arithmetic requirements, **not forecasts**. Recovering all loss on base-perfect tasks supplies only 0.256 pp, so pure preservation cannot plausibly account for +2 pp by itself. A task-wise best observed training-seed oracle reaches 84.686%, but is selection-biased, uses test labels, and is not deployable or evidence for any proposed training mechanism.

Use official held-out labels only for this retrospective headroom calculation; do not choose future training states, algorithms, or hyperparameters by task ID. New decisive experiments must preregister on development tasks disjoint from the 114 evaluation tasks.
