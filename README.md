# ConfairNLP: Conformal Prediction for Equitable Hate Speech Detection

CS 517: Socially Responsible AI — Course Project, UIC

## Overview

ConfairNLP investigates whether conformal prediction coverage guarantees hold equally across demographic groups in hate speech detection. We fine-tune BERT and HateBERT classifiers, apply split conformal prediction, measure per-group coverage disparities, and propose fairness-regularized fixes.

## Project Structure

```
confairnlp/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── run_all.py                # Single script to run the full pipeline
├── data/
│   └── download_data.py      # Downloads and preprocesses HateXplain, ToxiGen, Davidson
├── models/
│   └── train_classifier.py   # Fine-tunes BERT and HateBERT on HateXplain
├── conformal/
│   ├── marginal_cp.py        # Standard split conformal prediction (+ APS)
│   ├── group_conditional_cp.py  # Per-group conformal thresholds
│   └── fair_cp.py            # Fairness-regularized CP with lambda interpolation
├── evaluation/
│   ├── coverage_analysis.py  # Per-group coverage metrics, tables, and plots
│   └── ablation.py           # Ablation over alphas, scores, models
└── results/                  # Generated plots and tables (after running)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_all.py
```

### Command-line options

```bash
python run_all.py --help

# Key flags:
#   --skip-training    Load pre-trained models instead of training
#   --alpha 0.10       Set coverage level (default: 0.10)
#   --device cuda      Use GPU (auto-detects if not specified)
#   --output-dir results/  Directory for output files
#   --skip-ablation    Skip ablation studies to save time
```

## Methods

### Marginal Conformal Prediction
Standard split CP using either softmax or APS (Adaptive Prediction Sets) nonconformity scores. Provides marginal coverage guarantee: P(Y in C(X)) >= 1 - alpha.

### Group-Conditional Conformal Prediction
Computes separate conformal thresholds per demographic group. Targets per-group coverage: P(Y in C(X) | G=g) >= 1 - alpha for all groups g. Falls back to marginal threshold for small groups (< 30 samples).

### Fairness-Regularized Conformal Prediction
Interpolates between marginal and group-conditional thresholds:

```
q_fair_g = lambda * q_g + (1 - lambda) * q_marginal
```

Lambda controls fairness-efficiency tradeoff:
- lambda=0: pure marginal (small sets, poor group fairness)
- lambda=1: pure group-conditional (fair coverage, larger sets)

## Datasets

- **HateXplain**: 3-class hate speech with demographic target annotations
- **ToxiGen**: Machine-generated toxic text with target group labels
- **Davidson**: Tweet-level hate/offensive/neither (no demographic labels)

## Output

After running, `results/` contains:
- `per_group_coverage.csv` — coverage rates per group per method
- `coverage_bar_chart.pdf` — grouped bar chart of per-group coverage
- `lambda_tradeoff.pdf` — Pareto frontier (disparity vs. set size)
- `multi_alpha_disparity.pdf` — disparity across alpha values
- `ablation_summary.csv` — ablation results table
- `full_results_summary.txt` — text summary of key findings

## Reproducibility

All random seeds are fixed to 42. Results are deterministic given the same hardware and library versions.
