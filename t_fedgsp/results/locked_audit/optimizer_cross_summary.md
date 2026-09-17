# Reviewer-requested optimizer-crossed validation control

- Scope: post-lock, validation only; the locked test was not evaluated.
- All conditions start from the same seed-specific 110-round Graph-GRU checkpoint.
- Each condition receives 30 additional rounds with all 79 hospitals and one local epoch.

| Condition | Mean Micro-AUPRC | SD | Trainable | Branch bytes |
|---|---:|---:|---:|---:|
| full_graph_adamw_fedavg | 0.041662 | 0.002215 | 18,365 | 348,200,400 |
| residual_sgd_fedadam | 0.043599 | 0.001763 | 6,521 | 123,638,160 |
| full_graph_sgd_fedadam | 0.047258 | 0.001747 | 18,365 | 348,200,400 |
| residual_adamw_fedavg | 0.041652 | 0.002218 | 6,521 | 123,638,160 |

The full-graph SGD/FedAdam minus residual SGD/FedAdam deltas are +0.003678, +0.004077, +0.003223 (mean +0.003659).
Residual AdamW/FedAvg selects round 0 in every seed.

## Interpretation

The locked-test comparison remains a valid comparison of preregistered recipes, but it does not isolate the residual structure. The paper must present T-FedGSP as a communication--utility operating point and disclose optimizer sensitivity.
