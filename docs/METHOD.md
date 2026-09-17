# T-FedGSP method specification (final validation-frozen direct-surface route)

## 1. Task and federated signals

For ICU stay (p) at hospital (c(p)), structured events in the first 24 hours form

\[
X_p\in\mathbb{R}_{+}^{T\times V},\qquad
M_p\in\mathbb{R}_{+}^{T\times Q},
\]

where the main configuration uses (T=12) two-hour bins, (V=1{,}740) train-derived diagnosis/medication/abnormal-lab/treatment concepts, and (Q=4) modality-count channels. The target (y_p\in\{0,1\}^{50}) contains diagnoses first recorded after 24 hours. Patient-level train/validation/test partitions and hospital clients are fixed before training.

The normalized graph shift

\[
S=D^{-1/2}(A+I)D^{-1/2}
\]

is constructed only from within-bin concept co-occurrence among training stays. No raw event row or patient-specific graph is exchanged during federated optimization.

## 2. Observation-aware projected graph basis

Let (n_{pt}=\sum_q M_{ptq}). Event counts are stabilized and normalized by observation density:

\[
\bar X_{ptv}=
\frac{\log(1+X_{ptv})}
{\left(1+\log(1+n_{pt})\right)^{\gamma}},
\qquad \gamma=1.
\]

The model also receives (\log(1+M_p)), preventing an all-zero concept vector from conflating an unobserved interval with an observed interval containing no vocabulary-matched concept.

A fixed seeded Achlioptas projection (R\in\mathbb{R}^{V\times d}) makes full-eICU training tractable. For (k=0,\ldots,K_g),

\[
P_k=S^kR,\qquad
B_{ptkd}=\bar X_{pt:}P_k.
\]

The selected configuration uses (d=64) and (K_g=2). Projection matrices, graph powers, vocabularies, and scale statistics are shared and fixed; every matched method receives the same basis.

## 3. Matched graph-GRU backbone

The backbone learns graph-order weights

\[
u_{pt}=\sum_{k=0}^{2}\operatorname{softmax}(\omega)_k B_{ptk:}.
\]

Each (u_{pt}) is concatenated with the four log-modality counts and encoded by a GRU with hidden size 42. Masked temporal mean and final-observed-state pooling are concatenated and mapped to 50 base logits (z_p^{\mathrm{base}}). This exact network is the strongest matched graph-GRU baseline and the shared initialization for T-FedGSP.

## 4. Capacity-matched direct causal time–graph residual

The validation-selected adapter learns a direct coefficient surface

\[
\widetilde\Theta=\tanh(\Phi),\qquad
\Theta_{\ell k}=\frac{\widetilde\Theta_{\ell k}}
{\sum_{i,j}|\widetilde\Theta_{ij}|+\epsilon},
\qquad
\Phi\in\mathbb R^{(K_t+1)\times(K_g+1)}.
\]

Its causal joint state is

\[
J_{ptd}=\sum_{\ell=0}^{K_t}\sum_{k=0}^{K_g}
\Theta_{\ell k}\,\mathbf 1[t\ge\ell]B_{p,t-\ell,k,d}.
\]

The selected orders are (K_t=1) and (K_g=2), so the surface contains six scalars and can have matrix rank two. A separable rank-one control has exactly the same filter-scalar count: two temporal scalars, three graph scalars, and one mixture scalar. This makes the structural comparison capacity matched, although validation results show the two parameterizations are statistically indistinguishable and no superiority claim is assigned to nonseparability.

After feature-wise layer normalization, a learned channel scale, and GELU, masked mean and last-state pooling yield a (2d)-dimensional joint-evidence vector. A linear residual head produces (z_p^{\mathrm{joint}}), and the final logits are

\[
z_p=z_p^{\mathrm{base}}+\sigma(g)z_p^{\mathrm{joint}}.
\]

The coefficient surface is initialized from a smooth rank-one outer product, while the residual head is initialized exactly to zero. Consequently, before adapter training, T-FedGSP is bitwise prediction-equivalent to the transferred graph-GRU. During adapter training the backbone is frozen and only 6,521 residual parameters are optimized.

## 5. Strong-backbone federated optimization

Hospitals first train the shared graph-GRU backbone. A method-neutral weighted label-prior bias is initialized from training-only positive/negative counts aggregated across hospitals:

\[
b_j=\log\frac{w_j(\pi_j+\epsilon)}{1-\pi_j+\epsilon},
\]

where (w_j\le20) is the clipped class weight. This avoids spending early communication rounds learning rare-label intercepts and is identical for every method.

At federated round (r), every one of the 79 hospitals performs one local epoch and the server applies sample-size-weighted FedAvg:

\[
\theta^{r+1}=\sum_{c=1}^{79}
\frac{N_c}{\sum_jN_j}\theta_c^{r+1}.
\]

The final protocol first trains the graph-GRU for 80 rounds and then continues it for 30 rounds, using local AdamW and server FedAvg. The best validation checkpoint after these common 110 rounds is frozen as the shared strong backbone. From that exact checkpoint, two additional 30-round branches are evaluated: (i) continued full Graph-GRU training with its validation-selected AdamW/FedAvg optimizer, and (ii) residual-only adapter training with local SGD and server FedAdam. The latter communicates only the 6,521 trainable adapter parameters. Rank-one, rank-four factorized, and direct-surface adapters use the same frozen checkpoint, local SGD, server FedAdam, and branch budget.

The optimizer family is therefore selected on validation for each trainable parameterization rather than asserted to be identical. Fairness is enforced through the same cohort, representation, seed, strong starting checkpoint, all-client participation, local-epoch count, additional-round budget, selection metric, and one-time test protocol. Reported end-to-end communication includes the common 110-round backbone cost plus the branch cost.

Ordinary parameter federation is not described as a cryptographic or differential-privacy guarantee. Communication is measured from actual transmitted parameter bytes for all participating clients.

## 6. Complexity and nested controls

The one-time sparse basis cost is

\[
O\!\left(K_g|E_S|d+\operatorname{nnz}(X)(K_g+1)d\right).
\]

Per-stay direct joint filtering costs

\[
O\!\left(T(K_t+1)(K_g+1)d\right).
\]

The direct surface can represent rank-one or rank-two time–graph responses. A factorized rank-one adapter is the capacity-matched separable control, and a rank-four factorized adapter is retained as an exploratory over-parameterized control. Zero residual logits recover the graph-GRU exactly. These nested controls permit structural attribution without changing the input basis or data split.
