# Locked eICU test evaluation

This is the single evaluation of the previously sealed test partition.

| Model | Micro-AUPRC | Macro-AUPRC | Micro-AUROC | Brier (cal.) | ECE (cal.) |
|---|---:|---:|---:|---:|---:|
| static_mlp | 0.033021 | 0.029333 | 0.735446 | 0.014183 | 0.019033 |
| temporal_gru | 0.040853 | 0.035563 | 0.762351 | 0.015374 | 0.021074 |
| graph_gru | 0.043277 | 0.042024 | 0.780619 | 0.015255 | 0.020541 |
| fulljoint | 0.045562 | 0.041677 | 0.789628 | 0.023843 | 0.040582 |

Full-joint minus graph mean seed delta: +0.001530.
Patient-cluster bootstrap 95% CI: [+0.000656, +0.004191].
