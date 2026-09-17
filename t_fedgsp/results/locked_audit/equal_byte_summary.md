# Reviewer-requested equal-byte controls

All results are validation-only and use seeds 1--3. The locked test was not reopened.

| Condition | Mean Micro-AUPRC | SD | Trainable parameters | Final-stage bytes |
|---|---:|---:|---:|---:|
| Time--graph hybrid adapter (30 rounds) | 0.044421 | 0.001311 | 5,173 | 98,080,080 |
| Generic hybrid adapter (30 rounds) | 0.044474 | 0.001587 | 5,167 | 97,966,320 |
| Full Graph-GRU (8 affordable rounds) | 0.044477 | 0.001844 | 18,365 | 92,853,440 |

Hybrid minus generic per seed: -0.000389, -0.000036, +0.000265 (mean -0.000054).

Hybrid minus equal-byte full model per seed: -0.000672, +0.000293, +0.000209 (mean -0.000057).

**Decision.** The seed-0 screening result did not replicate. The time--graph hybrid is retained only as a negative diagnostic control; it is not promoted as a superior method.
