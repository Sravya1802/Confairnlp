# Baseline snapshot (regenerated 2026-04-24)

Primary classifier: HateBERT fine-tuned on HateXplain. Splits 60/20/10/10 (train/calibration/tuning/test). Alpha = 0.10, target coverage 0.90. Lambda* selected on the dedicated tuning split (leak-free).

## Headline numbers (test split)

| Method | Coverage | Avg set size | Reliable-group disparity | All-group disparity |
|---|---|---|---|---|
| Marginal CP | 0.9059 | 1.85 | 0.0615 | 0.3000 |
| Group-Conditional CP | 0.9121 | 1.95 | 0.0459 | 0.3000 |
| Fair CP (lambda* = 0.10) | 0.9017 | 1.83 | **0.0449** | 0.3000 |

Fair CP shrinks reliable-group disparity from 0.062 to 0.045 (~27% relative reduction) while *also* slightly reducing average set size. Total pipeline runtime: 56.6 minutes on a GTX 1660 Ti at batch size 8.

## Methodology contrast vs the previous (leaky) run

The previous summary reported lambda* = 0.50, marginal disparity 0.114, group-cond disparity 0.100. After fixing the test-set leakage in lambda selection (lambda is now picked on the tuning split and then frozen for test evaluation) and reporting reliable-group disparity (n_test >= 30) as the primary metric, the headline disparity gap shrinks dramatically and the optimal lambda lands at 0.10, not 0.50. The methodological correction matters: the "leaky" pipeline was overfitting the fairness adjustment to the test set.

## Top 3 under-covered reliable groups under Marginal CP

(target = 0.90; 95% Wilson CI shown in parentheses)

| Group | Coverage | 95% CI | n_test |
|---|---|---|---|
| Islam | 0.8577 | (0.811, 0.894) | 274 |
| Asian | 0.8551 | (0.753, 0.919) | 69 |
| Refugee | 0.8758 | (0.814, 0.919) | 153 |

## Top 3 over-covered reliable groups under Marginal CP

| Group | Coverage | 95% CI | n_test |
|---|---|---|---|
| unknown | 0.9615 | (0.938, 0.977) | 390 |
| Hispanic | 0.9189 | (0.834, 0.962) | 74 |
| Jewish | 0.9057 | (0.863, 0.936) | 244 |

The "unknown" group (no demographic tag) over-covers strongly under Marginal CP, which is what makes the marginal *average* coverage hit target while specific tagged groups under-cover.

## Files

- `per_group_coverage.csv` -- full per-group coverage with Wilson CIs and reliable-group flags
- `coverage_bar_chart.pdf` -- per-group coverage across the three CP methods
- `lambda_tradeoff.pdf` -- Pareto frontier for Fair-CP lambda
- `multi_alpha_disparity.pdf` -- disparity across alpha in {0.05, 0.10, 0.15, 0.20}
- `ablation_summary.csv` -- model x score-function x alpha grid
- `full_results_summary.txt` -- text summary

## Reproducibility

```
python run_all.py --batch-size 8
```

Seed = 42. ~57 min on GTX 1660 Ti, 6 GB VRAM.
