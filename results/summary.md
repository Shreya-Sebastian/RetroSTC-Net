# Results summary

## E1: behavioural-tagging window (5 seeds; consolidated share of task A)

| event offset | STC | no tags | PRPs per synapse |
|---|---|---|---|
| none | 0.00 | 0.00 | 0.00 |
| -300 | 0.13 | 0.13 | 0.00 |
| -150 | 0.58 | 0.60 | 0.00 |
| +0 | 0.91 | 0.92 | 0.91 |
| +150 | 0.71 | 0.79 | 0.71 |
| +300 | 0.42 | 0.68 | 0.42 |
| +600 | 0.03 | 0.51 | 0.03 |

Size-triggered capture without any event: 0.62.

## E2: cross-task rescue (5 seeds)

| gap | weight-bound tags: share A / acc A | change-bound tags: share A / acc A |
|---|---|---|
| 0 | 0.92 / 0.875 | 0.91 / 0.872 |
| 75 | 0.85 / 0.872 | 0.82 / 0.866 |
| 150 | 0.78 / 0.870 | 0.71 / 0.852 |
| 300 | 0.66 / 0.862 | 0.43 / 0.779 |
| 450 | 0.53 / 0.854 | 0.16 / 0.622 |
| 600 | 0.37 / 0.862 | 0.03 / 0.556 |
| 900 | 0.24 / 0.852 | 0.00 / 0.534 |

B not salient: share A 0.00, acc A 0.593, acc B 0.516. Task B accuracy with salient onset: 0.972.

## E3: faithfulness (deletion test, 5 seeds)

| ranking | reset 1%: acc A / acc B | reset 5%: acc A / acc B |
|---|---|---|
| log_A | 0.843 / 0.968 | 0.797 / 0.963 |
| log_B | 0.940 / 0.828 | 0.951 / 0.691 |
| fisher_A | 0.927 / 0.965 | 0.902 / 0.950 |
| fisher_B | 0.911 / 0.961 | 0.946 / 0.929 |
| random | 0.873 / 0.972 | 0.876 / 0.971 |
| none | 0.872 / 0.971 | |

Memory-source attribution at the end of task B (before the delay):
- task A: full weights 0.846, long-term part only 0.873
- task B: full weights 0.977, long-term part only 0.971

Provenance log, seed 0:

- event 'B onset' consolidated 63% of source 'B' (total change 274.47); strongest hidden units [135, 132, 200, 211, 181]
- event 'B onset' consolidated 85% of source 'A' (total change 223.65); strongest hidden units [131, 52, 200, 214, 233]

## E4: Split-MNIST (5 tasks, 5 seeds, average final accuracy)

| method | task-incremental | domain-incremental |
|---|---|---|
| Plain SGD | 0.980 ± 0.003 | 0.616 ± 0.010 |
| EWC | 0.987 ± 0.002 | 0.637 ± 0.011 |
| SI | 0.987 ± 0.001 | 0.657 ± 0.016 |
| STC, event at task onset | 0.980 ± 0.002 | 0.619 ± 0.012 |
| STC, surprise-gated | 0.980 ± 0.002 | 0.621 ± 0.015 |
| STC log as importance | 0.980 ± 0.002 | 0.636 ± 0.015 |

Selected hyperparameters (tuning seed 100): ewc/domain: {'lam': 30.0}; plain/domain: {}; si/domain: {'lam': 0.1}; stc_anchor/domain: {'lam': 1.0}; stc_oracle/domain: {'stc': {'tau_E': 1000.0}}; stc_surprise/domain: {'stc': {'tau_E': 1000.0}}; ewc/task: {'lam': 100.0}; plain/task: {}; si/task: {'lam': 0.1}; stc_anchor/task: {'lam': 1.0}; stc_oracle/task: {'stc': {'tau_E': 3000.0}}; stc_surprise/task: {'stc': {'tau_E': 3000.0}}
