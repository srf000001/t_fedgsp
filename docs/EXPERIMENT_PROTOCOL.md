# Locked experimental protocol v1

## Status and purpose

This protocol is locked before reading any new test-set prediction. Pilot tuning may use train and validation data only. Deviations require a dated `note/` record and a new protocol version; the original outcome remains reported.

## Dataset and task

- Full eICU-CRD v2.0 only; demo/synthetic data are ineligible for main results.
- Locked 107,306 stays, 84,696 patients, and 79 hospital clients.
- Patient-level 75,108/10,825/21,373 train/validation/test stay split.
- Inputs: structured events from 0–24 h; targets: 50 diagnoses first observed after 24 h.
- Main temporal resolution: 2 h (12 bins), aggregated exactly from the audited one-hour cache.
- Early prediction: 6 h and 12 h use prefixes of the same representation without retraining unless explicitly labeled as window-specific.

## Confirmatory methods

1. Static-MLP: time-summed concept signal, capacity-matched hidden head.
2. Temporal-only GRU/causal filter: (K_g=0), same projection and classifier budget.
3. Graph-only filter: (K_t=0), same graph basis.
4. Graph-GRU: capacity-matched temporal graph baseline using the same (B_{ptkd}) basis.
5. Separable T-FedGSP: rank (R_f=1).
6. T-FedGSP: mask normalization, (K_t=2), (K_g=2), (R_f=4).
7. Centralized counterparts for methods 1, 4, and 6 as non-private optimization references.

FedAvg uses all 79 clients per round, sample-size weighting, one local epoch, and 20 confirmatory rounds. A 10-round seed-0 pilot may be used to reject broken configurations. FedProx is added if FedAvg validation curves show client-drift instability.

## Model selection

- Pilot seed: 0; confirmatory seeds: 1, 2, 3.
- Primary validation criterion: Micro-AUPRC.
- Allowed pilot grid: bin width {1 h, 2 h, 4 h}; projection dimension {16, 32, 64}; (K_t,K_g\in\{1,2\}); rank {2,4}; mask exponent \(\gamma\in\{0,0.5,1\}\); learning rate {1e-3, 3e-3}; dropout {0,0.2}.
- Successive optimization must follow the ladder below; test metrics cannot select settings.

## Outcomes

Primary outcome: test Micro-AUPRC.

Secondary outcomes: Micro-AUROC, Macro-AUPRC, Macro-AUROC where defined, P@5, R@5, Brier score, expected calibration error, per-hospital Micro-AUPRC distribution, worst-20%-hospital mean, parameter count, wall-clock time, peak RSS, and measured per-round communication bytes.

## Perturbation protocol

The locked test tensor is perturbed at inference only:

- independent event deletion: 10%, 30%, 50%;
- Gaussian timestamp jitter before rebinning: sigma 30, 60, 120 minutes, clipped to [0,24 h];
- single-modality absence for each of diagnosis, medication, lab, and treatment.

Each perturbation uses three fixed seeds shared across methods. “Robustness” is limited to these perturbations. Report clean performance and area under each degradation curve.

## Statistical analysis

- Report mean ± sample SD across seeds 1–3 for every confirmatory metric.
- For the locked primary comparison, compute a paired patient-cluster bootstrap (2,000 resamples) of the pooled three-seed prediction average and report the 95% percentile CI for the Micro-AUPRC difference.
- Compare per-hospital paired Micro-AUPRC by two-sided Wilcoxon signed-rank with rank-biserial effect size; report median paired difference and bootstrap CI.
- Holm-correct only the family of secondary confirmatory comparisons; exploratory ablations are labeled exploratory rather than promoted through p-values.
- Do not infer equivalence from non-significance. For clean-performance non-inferiority in a robustness-first fallback, use a predeclared margin of -0.0005 Micro-AUPRC.

## Paper-evidence success gate

The clean-utility claim passes only if T-FedGSP has positive mean delta over the strongest matched federated baseline across all three seeds, a paired cluster-bootstrap 95% CI above zero, and an absolute Micro-AUPRC gain of at least 0.0010.

The robustness fallback passes only if clean performance meets the -0.0005 non-inferiority margin, deletion/jitter robustness AUC improves by at least 5% relative, and the worst-20%-hospital mean does not degrade by more than 0.0005.

If neither gate passes, the current contribution is not paper-ready and optimization continues; results may not be rewritten as success.

## Pre-registered optimization ladder

1. Verify labels, train-only preprocessing, numerical scale, and baseline convergence.
2. Select mask exponent, graph/temporal order, rank, and projection dimension using validation only.
3. Tune class-balanced loss clipping/logit adjustment and learning rate.
4. Test graph top-k {8,12,20} and bin width {1 h,2 h,4 h} while preserving train-only construction.
5. If client drift is diagnosed, add FedProx and client-balanced sampling; do not change the test set.
6. If clean gains fail but robustness is promising, activate Option C consistency regularization as a separately labeled protocol version.

## Federated protocol v2 addendum (2026-07-18)

The original 20-round, locally weighted AdamW setting was executed before this addendum and failed the baseline-convergence gate (D64 backbone validation Micro-AUPRC 0.016290 after 20 rounds). The run and its logs are retained under `t_fedgsp/results/federated_seed0_d64/`.

Validation-only optimizer probes showed that locally absent rare labels and randomly initialized output biases consumed most early communication. The following method-neutral changes are therefore locked before any test evaluation:

- positive-class weights use counts aggregated from the training split and are identical across clients;
- the linear prediction bias is initialized from the same aggregate weighted label prior;
- all 79 clients still participate and FedAvg remains sample-size weighted;
- each client still performs one local epoch per round;
- local AdamW learning rate is 0.03 for the backbone/continued-backbone branches and 0.003 for residual-only adapters;
- the shared backbone receives 80 rounds, followed by 30 equal-budget branch rounds;
- graph-GRU continuation, rank-one residual, and rank-four T-FedGSP all start from the same best validation-selected federated backbone state;
- communication includes actual download and upload bytes for every participating client.

This addendum changes optimization and convergence budget only. It does not alter the cohort, split, labels, temporal window, graph, projection, primary metric, perturbation protocol, statistical tests, or success thresholds. Test data remain sealed until the v2 configuration and confirmatory seeds are complete.

## Final validation freeze v3 (2026-07-18; before any test prediction)

This addendum supersedes the v1 candidate-method list and the v2 adapter parameterization, while retaining every earlier outcome as development evidence. All selections below were made from train/validation runs only. At this freeze, no test prediction or test metric has been computed.

### Frozen models and optimization

- Main proposed model: a direct signed-L1-normalized 2-by-3 causal time–graph coefficient surface with (K_t=1), (K_g=2), a zero-initialized logit residual head, and 6,521 trainable adapter parameters. The transferred Graph-GRU backbone is frozen.
- Capacity-matched structural control: the rank-one factorized adapter. Its six filter scalars equal the direct surface's six filter scalars. The rank-four factorized adapter is exploratory; validation shows no rank advantage, so the paper will not claim that nonseparability is the source of the utility gain.
- Static MLP and temporal GRU baselines train for 110 common rounds plus a 30-round validation-selected continuation, using all 79 hospitals, one local epoch, local AdamW, and FedAvg.
- The Graph-GRU trains for 80 rounds plus 30 continuation rounds using local AdamW/FedAvg. Its best validation checkpoint is the identical strong initialization for all final branches.
- The matched continued-Graph-GRU baseline receives a further 30 AdamW/FedAvg rounds. The proposed and factorized adapters receive 30 residual-only rounds with local SGD and server FedAdam. Optimizers were selected on validation per parameterization; all branches share the cohort, representation, initialization checkpoint, all-client participation, local-epoch count, round budget, seed, and selection metric.
- Confirmatory model seeds are exactly 1, 2, and 3. Seed 0 remains development-only.

### Frozen reporting and statistics

- Primary ranking metric: raw-probability Micro-AUPRC. Secondary ranking metrics are Macro-AUPRC, Micro/Macro-AUROC, precision@5, and recall@5.
- Scalar temperature is fitted independently for each seed/model using validation predictions only. Calibrated Brier score and 15-bin ECE are secondary calibration outcomes; temperature scaling cannot alter ranking metrics.
- The main hypothesis is the direct-surface adapter versus the strongest matched continued Graph-GRU. Report per-seed mean and sample SD, the pooled seed-averaged prediction result, and a 2,000-resample paired patient-cluster bootstrap 95% percentile CI for the pooled Micro-AUPRC difference.
- Per-hospital Micro-AUPRC uses the pooled seed-averaged predictions and a two-sided Wilcoxon signed-rank test with paired median, positive-hospital fraction, and rank-biserial effect size. No cross-hospital fairness claim is permitted unless these results support it.
- Communication is computed from actual transmitted trainable-parameter bytes for all 79 clients. End-to-end totals include the common 110-round strong-backbone training and the final branch.

### Single locked-test execution

- Checkpoints and validation-prediction files for all four confirmatory models and all three seeds must exist before execution. Their SHA-256 hashes and validation-fitted temperatures are written to a checkpoint lock before inference.
- The test event cache may be constructed beforehand only to reproduce the already locked test features and raw minute offsets. It must contain no patient/stay identifiers and must reconstruct the locked concept and modality matrices exactly.
- The test partition is evaluated exactly once by `t_fedgsp/scripts/evaluate_locked_test.py`. The script refuses execution if a completed summary already exists and writes no patient-level prediction file.
- The same single run executes clean evaluation and the predeclared inference-only robustness suite: shared event deletion at 10%, 30%, and 50%; raw timestamp jitter at 30, 60, and 120 minutes; and absence of each of four modalities. Three fixed perturbation seeds are shared across methods.
- After this execution, no architecture, hyperparameter, checkpoint, loss, split, outcome definition, statistical test, or success threshold may change. Any weak or null test result must be reported rather than optimized against.
