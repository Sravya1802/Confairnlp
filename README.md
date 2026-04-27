# ConfairNLP: Conformal Prediction for Equitable Hate Speech Detection

CS 517: Socially Responsible AI - Course Project, UIC

## Overview

ConfairNLP studies whether conformal prediction coverage is distributed evenly
across demographic target groups in hate speech detection. The default pipeline
fine-tunes BERT and HateBERT on HateXplain, calibrates conformal prediction
sets, and compares:

- Marginal split conformal prediction
- Group-conditional conformal prediction
- Fairness-regularized conformal prediction with lambda interpolation

The final test split is reserved for reporting. Fair CP lambda selection is
performed on a separate tuning split to avoid test-set tuning.

On top of the baseline, three novelty modules diagnose *why* per-group
disparities exist (Causal Coverage Attribution), test whether the apparent
disparity reduction is statistically significant (Bootstrap CIs and paired
permutation tests over Set-Size Disparity), and audit whether any CP method is
robust to demographic-token shortcut learning (Counterfactual SGT-Swap stress
test). The narrative writeup is in
[`results/NOVELTY_SUMMARY.md`](results/NOVELTY_SUMMARY.md); baseline numbers
are in [`results/BASELINE_README.md`](results/BASELINE_README.md).

## Headline findings (HateBERT, alpha = 0.10, lambda* = 0.10)

- **Where the gaps come from.** Among reliable groups (n_test >= 30),
  classifier accuracy variance (`S_g`) is the strongest predictor of marginal
  undercoverage (Spearman rho = +0.55, p = 0.10). Calibration scarcity
  (`D_g`) is the *weakest* (rho = +0.10, p = 0.78). 18 of 23 groups are
  classified SystemicBias-dominated. Contrary to the naive expectation, the
  bottleneck is the underlying classifier, not the per-group calibration set
  size.
- **What Fair CP buys.** Reliable-group coverage disparity drops from
  Marginal 0.0615 to Fair 0.0449. Bootstrap 95% CIs on the three methods
  overlap, but a paired permutation test on iteration-level differences shows
  the gain is significant at p = 0.0005 - pairing controls for resample-level
  noise that dominates unpaired CIs.
- **What Fair CP does not buy.** Counterfactual SGT-token swaps reveal that
  all three CP methods are essentially indistinguishable on stability under
  input perturbation. The 51% set-flip rate on Homosexual->Heterosexual is an
  order of magnitude larger than the next-worst pair. Coverage-fairness and
  counterfactual-fairness are different criteria; post-hoc CP only addresses
  the former.

The unifying claim: **post-hoc conformal prediction is a calibration tool, not
a debiasing tool.**

## Project structure

```text
confairnlp/
|-- README.md
|-- requirements.txt
|-- run_all.py                       Single entry point for the full pipeline
|-- conformal/                       Three CP methods (softmax + APS scores)
|   |-- marginal_cp.py
|   |-- group_conditional_cp.py
|   |-- fair_cp.py
|-- data/
|   |-- download_data.py             HateXplain (default), ToxiGen, Davidson
|-- models/
|   |-- train_classifier.py          BERT and HateBERT fine-tuning
|-- evaluation/
|   |-- coverage_analysis.py         Per-group coverage tables, Wilson CIs, plots
|   |-- ablation.py                  Model x score-function x alpha grid
|   |-- _novelty_setup.py            Shared helper: cached softmax + CP results
|   |-- causal_attribution.py        Module 1: D/U/S diagnostics, dominant cause
|   |-- set_size_fairness.py         Module 2: set-size disparity + bootstrap CIs
|   |-- counterfactual.py            Module 3: SGT-swap stress test
|-- scripts/
|   |-- make_poster_figures.py       4 publication-quality novelty figures
|   |-- regen_baseline_plots.py      3 baseline plots at 300 DPI from cache
|-- tests/
|   |-- test_conformal.py            Unit tests for quantile + fallback logic
|-- figures/                         Poster figures (PDF + PNG, 300 DPI)
|-- results/                         Generated CSVs, PDFs, narrative writeup
    |-- NOVELTY_SUMMARY.md           Full 5-move discussion of the 3 modules
    |-- BASELINE_README.md           Baseline snapshot headline numbers
    |-- baseline_snapshot/           Frozen baseline outputs for reference
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full baseline pipeline (data download + train BERT and HateBERT +
all three CP methods + plots + ablation):

```bash
python run_all.py --batch-size 8
```

Useful options:

```bash
python run_all.py --help
python run_all.py --skip-training            # require existing saved models
python run_all.py --device cuda --epochs 5 --batch-size 8
python run_all.py --alpha 0.10 --lambda-steps 11 --skip-ablation
```

`--skip-training` means "require saved models and do not train." If a saved
model is missing, the command fails with a clear error. Use the default
behavior to train missing models, or `--force-retrain` to ignore saved models
and retrain.

## Reproducing the novelty experiments

After `run_all.py` has produced trained models in `models/trained/` and
splits in `data/hatexplain_splits.pkl`, run each novelty module from the
repo root:

```bash
python -m evaluation.causal_attribution           # Module 1
python -m evaluation.set_size_fairness            # Module 2 (B=500 bootstrap)
python -m evaluation.counterfactual               # Module 3 (CF GPU inference)

# Optional: also write target-group-threshold-policy rows for Module 3
python -m evaluation.counterfactual --threshold-policy both
```

Each module reuses cached softmax probabilities and CP results stored in
`results/_novelty_cache.pkl` (created on first run; auto-invalidated by
mtime/size signature when splits or model directory change), so subsequent
modules do not re-run BERT inference.

## Poster figures

Generate seven publication-quality figures (300 DPI, PDF + PNG, ColorBrewer
Set2 palette, serif fonts):

```bash
# 4 novelty figures from results/ CSVs (no model inference, ~10 seconds)
python scripts/make_poster_figures.py

# 3 baseline figures regenerated from the novelty cache at 300 DPI
python scripts/regen_baseline_plots.py
```

Output goes to `figures/` (novelty) and `results/` plus
`results/baseline_snapshot/` (baseline). All seven figures are at 300 DPI;
the rcParams pin `pdf.fonttype: 42` so embedded text is searchable TrueType,
not Type 3 outlines.

## Data protocol

The default experiment uses HateXplain. It creates deterministic splits:

- 60% train
- 20% calibration
- 10% tuning (used only for Fair CP lambda selection)
- 10% final test (reserved for reporting)

Splits are stratified by label plus primary target group when possible, with
rare group-label strata collapsed so very small groups do not break the
split. The pipeline saves `data/hatexplain_split_distributions.csv` so group
and label distribution drift can be inspected.

ToxiGen and Davidson preprocessing helpers remain in `data/download_data.py`,
but they are not part of the default evaluation pipeline because their label
spaces and group metadata differ from HateXplain.

## Methods

### Marginal Conformal Prediction

Standard split conformal prediction using either softmax or APS nonconformity
scores. It targets marginal coverage:

```text
P(Y in C(X)) >= 1 - alpha
```

### Group-Conditional Conformal Prediction

Computes separate thresholds by demographic target group. Groups with fewer
than 30 calibration examples use the marginal threshold as a fallback.

### Fairness-Regularized Conformal Prediction

Interpolates between marginal and group-conditional thresholds:

```text
q_fair_g = lambda * q_g + (1 - lambda) * q_marginal
```

Lambda is selected on the tuning split using an objective that first avoids
undercoverage, then minimizes reliable-group coverage disparity, then average
set size.

## Novelty modules

### Module 1 - Causal Coverage Attribution

For each demographic group, computes three orthogonal diagnostic scores and
classifies the dominant failure mode:

- `D_g` (data scarcity): min-max-normalized `1 / sqrt(n_cal)`
- `U_g` (model uncertainty): mean normalized softmax entropy
- `S_g` (systemic bias): `1 - argmax accuracy on the group`

Reports per-group dominant cause (DataScarcity / ModelUncertainty /
SystemicBias / Mixed) and Spearman correlations of each diagnostic with
marginal undercoverage on reliable groups. Validates the framework: in our
data, `S_g` (not `D_g`) drives gaps.

Outputs: `results/attribution_scores.csv`, `results/failure_taxonomy.csv`,
`results/attribution_validation.txt`.

### Module 2 - Set-Size Disparity and Bootstrap CIs

Three contributions on top of per-group coverage:

1. Per-group mean and 95th-percentile set sizes, plus Gini coefficient of
   mean-set-size across reliable groups (Marginal 0.026, GC 0.053, Fair
   0.031).
2. Size-stratified coverage: bins test samples by `|C(x)|` and recomputes
   coverage in each bin. Singletons cover 0.84, pairs 0.91, triples 1.00 -
   the marginal 1-alpha target averages over heterogeneous size
   sub-populations.
3. Bootstrap 95% CIs on reliable-group coverage disparity (B = 500
   iterations resampling cal+test, lambda* held fixed). Paired permutation
   test on iteration-level differences confirms the Fair-CP gain is
   statistically significant despite overlapping unpaired CIs.

Outputs: `results/set_size_disparity.csv`,
`results/size_stratified_coverage.csv`, `results/disparity_bootstrap.csv`.

### Module 3 - Counterfactual SGT-Swap Stress Test

Constructs identity-preserving token swaps (e.g. African <-> Caucasian,
Islam <-> Christian, Homosexual <-> Heterosexual) on test posts that contain
at least one source-group lexicon token, and re-runs HateBERT on the swapped
text. Measures coverage stability, set-flip rate, and label-flip rate per
swap pair per CP method.

Supports three threshold policies via `--threshold-policy`:
`fixed_source` (isolates text sensitivity, default), `target_group` (proper
protected-attribute-intervention semantics), `both`.

Outputs: `results/counterfactual_stability.csv`,
`results/counterfactual_comparison.csv`,
`results/counterfactual_swap_stats.csv`,
`results/counterfactual_lexicon.json`. Raw per-post softmax tables go to
`results/counterfactual_posts.csv` (gitignored).

## Metrics

The primary disparity metric is computed on reliable groups only, where a
reliable group has at least 30 final-test examples by default. The output
table still reports all groups, flags small groups, and includes Wilson 95%
confidence intervals for coverage estimates.

## Generated outputs

Baseline pipeline (`results/`, snapshot in `results/baseline_snapshot/`):

- `per_group_coverage.csv` -- per-group coverage with Wilson CIs
- `coverage_bar_chart.pdf` -- per-group coverage across the three CP methods
- `lambda_tradeoff.pdf` -- Fair-CP Pareto frontier
- `multi_alpha_disparity.pdf` -- disparity across alpha in {0.05, 0.10,
  0.15, 0.20}
- `ablation_summary.csv` -- model x score-function x alpha grid
- `full_results_summary.txt` -- text summary

Novelty modules (`results/`):

- `attribution_scores.csv`, `failure_taxonomy.csv`,
  `attribution_validation.txt` (Module 1)
- `set_size_disparity.csv`, `size_stratified_coverage.csv`,
  `disparity_bootstrap.csv` (Module 2)
- `counterfactual_stability.csv`, `counterfactual_comparison.csv`,
  `counterfactual_swap_stats.csv`, `counterfactual_lexicon.json` (Module 3)
- `NOVELTY_SUMMARY.md` -- full 5-move discussion writeup

Poster figures (`figures/`, 300 DPI, PDF + PNG):

- `fig1_causal_attribution.{pdf,png}`
- `fig2_bootstrap_ci.{pdf,png}`
- `fig3_size_stratified.{pdf,png}`
- `fig4_counterfactual.{pdf,png}`

## Reproducibility

Random seeds are fixed to 42 for NumPy, PyTorch, and CUDA. Exact
reproducibility also depends on hardware, CUDA/cuDNN behavior, and installed
package versions. The full baseline pipeline takes ~57 minutes on a
GTX 1660 Ti at batch size 8. Novelty modules add ~5 min total once the
softmax cache is warm.

## Tests

```bash
python -m unittest discover -s tests
```

Covers quantile threshold computation, score-function validation, small-group
fallback flag, and the lambda-selection objective.
