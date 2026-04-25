# ConfairNLP — Novelty Results Summary

Each section below uses the 5-move discussion style:
(1) decomposition, (2) attribution, (3) theoretical tie-back, (4) trade-off surfacing, (5) honest negative.

## Module 1 -- Causal Coverage Attribution

**(1) Decomposition.** The marginal CP coverage gap is not uniform across demographic groups. Among 13 reliable groups (n_test >= 30), the three groups furthest from the 1 - alpha target are:

- **unknown** (n_test=390): marginal coverage 0.962 (gap +0.062); D_g=0.00, U_g=0.10, S_g=0.23; dominant cause = SystemicBias.
- **Asian** (n_test=69): marginal coverage 0.855 (gap -0.045); D_g=0.05, U_g=0.15, S_g=0.39; dominant cause = SystemicBias.
- **Islam** (n_test=274): marginal coverage 0.858 (gap -0.042); D_g=0.01, U_g=0.10, S_g=0.35; dominant cause = SystemicBias.

**(2) Attribution.** We attribute these gaps using three orthogonal diagnostics: data scarcity (D_g = min-max-normalized 1/sqrt(n_cal)), model uncertainty (U_g = mean normalized entropy of softmax), and systemic bias (S_g = 1 - point-prediction accuracy on the group). On the 10 undercovered reliable groups, Spearman correlations with marginal undercoverage are: D_g: rho=+0.10 (p=0.777), U_g: rho=+0.50 (p=0.138), S_g: rho=+0.55 (p=0.0984). Contrary to the naive expectation, *data scarcity* is the **weakest** predictor in this regime; the strongest is systemic bias, with model uncertainty close behind. Across all 23 groups, dominant-cause classification gives SystemicBias=18, DataScarcity=3, Mixed=2: nearly every reliable group is bottlenecked by classifier accuracy, not by calibration-set size.

**(3) Theoretical tie-back.** The Vovk (2012) split-CP finite-sample bound |coverage - (1 - alpha)| <= O(1/sqrt(n_g)) is *loose* for our reliable groups: with n_cal in the hundreds, the bound predicts at most a few percent slack from calibration noise alone, which matches the small spread of D_g across reliable groups (max 0.06 in normalized units). The remaining gap therefore must come from what the bound does not control -- the conditional accuracy of the underlying classifier P(y_hat = y | g) -- which our S_g diagnostic captures directly.

**(4) Trade-off surfacing.** Fair CP (lambda* = 0.10 selected on the tuning split) reduces mean |coverage - target| by n/a (no reliable groups in this bucket) for Data-Scarcity groups, n/a (no reliable groups in this bucket) for Model-Uncertainty groups, +0.0038 for Systemic-Bias groups, and n/a (no reliable groups in this bucket) for Mixed groups. The reduction on Systemic-Bias groups is small in absolute terms because the underlying classifier accuracy variance is the binding constraint; lambda* lands close to zero (almost-pure marginal) precisely because pushing further toward per-group quantiles inflates set sizes without matching reductions in disparity.

**(5) Honest negative.** Post-hoc CP cannot rescue groups whose coverage gap is driven by classifier error rather than calibration-set scarcity. In our data this is the dominant regime: reliable groups have hundreds of calibration samples but still under-cover because the classifier is confidently wrong on a non-trivial fraction of in-group examples. Closing those gaps requires improving the underlying classifier (better data, debiased fine-tuning) or accepting strictly larger prediction sets that approach the trivial set of size num_classes. We report this limitation explicitly rather than hide it in the aggregate.

