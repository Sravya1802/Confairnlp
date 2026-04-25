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

## Project Structure

```text
confairnlp/
|-- requirements.txt
|-- README.md
|-- run_all.py
|-- data/
|   |-- download_data.py
|-- models/
|   |-- train_classifier.py
|-- conformal/
|   |-- marginal_cp.py
|   |-- group_conditional_cp.py
|   |-- fair_cp.py
|-- evaluation/
|   |-- coverage_analysis.py
|   |-- ablation.py
|-- results/
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run_all.py
```

Useful options:

```bash
python run_all.py --help
python run_all.py --skip-training
python run_all.py --device cuda --epochs 5 --batch-size 16
python run_all.py --alpha 0.10 --lambda-steps 11 --skip-ablation
```

`--skip-training` now means "require saved models and do not train." If a saved
model is missing, the command fails with a clear error. Use the default behavior
to train missing models, or `--force-retrain` to ignore saved models and retrain.

## Data Protocol

The default experiment uses HateXplain. It creates deterministic splits:

- 60% train
- 20% calibration
- 10% tuning
- 10% final test

Splits are stratified by label plus primary target group when possible, with
rare group-label strata collapsed so that very small groups do not break the
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

## Metrics

The primary disparity metric is computed on reliable groups only, where a
reliable group has at least 30 final-test examples by default. The output table
still reports all groups, flags small groups, and includes Wilson confidence
intervals for coverage estimates.

Generated outputs include:

- `results/per_group_coverage.csv`
- `results/coverage_bar_chart.pdf`
- `results/lambda_tradeoff.pdf`
- `results/multi_alpha_disparity.pdf`
- `results/ablation_summary.csv`
- `results/full_results_summary.txt`

## Reproducibility

Random seeds are fixed to 42 for NumPy and PyTorch. Exact reproducibility also
depends on hardware, CUDA/cuDNN behavior, and installed package versions.

## Tests

Run the lightweight conformal unit tests with:

```bash
python -m unittest discover -s tests
```
