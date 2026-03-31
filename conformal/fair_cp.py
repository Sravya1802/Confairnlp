"""
fair_cp.py — Fairness-Regularized Conformal Prediction.

Interpolates between marginal and group-conditional thresholds:

    q_fair_g = lambda * q_g + (1 - lambda) * q_marginal

where lambda in [0, 1] controls the fairness-efficiency tradeoff:
  - lambda=0 -> pure marginal CP (smallest sets, worst fairness)
  - lambda=1 -> pure group-conditional CP (best fairness, largest sets)

Sweeps lambda and computes coverage disparity vs. set size to produce
a Pareto frontier visualization.
"""

import numpy as np

from conformal.marginal_cp import (
    aps_nonconformity_scores,
    build_prediction_sets_aps,
    build_prediction_sets_softmax,
    compute_quantile_threshold,
    evaluate_prediction_sets,
    softmax_nonconformity_scores,
)
from conformal.group_conditional_cp import get_group_indices, MIN_GROUP_SIZE

SEED = 42
np.random.seed(SEED)


def compute_fair_thresholds(
    q_marginal: float,
    group_thresholds: dict,
    lam: float,
) -> dict:
    """Compute fairness-regularized thresholds for each group.

    q_fair_g = lam * q_g + (1 - lam) * q_marginal

    Args:
        q_marginal: Marginal (global) threshold
        group_thresholds: Dict mapping group -> group-specific threshold
        lam: Interpolation parameter in [0, 1]

    Returns:
        Dict mapping group -> fairness-regularized threshold
    """
    fair_thresholds = {}
    for group, q_g in group_thresholds.items():
        fair_thresholds[group] = lam * q_g + (1 - lam) * q_marginal
    return fair_thresholds


def run_fair_cp_single_lambda(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_groups: list,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_groups: list,
    alpha: float,
    lam: float,
    score_function: str = "softmax",
) -> dict:
    """Run fairness-regularized CP for a single lambda value.

    Args:
        cal_probs, cal_labels, cal_groups: Calibration data
        test_probs, test_labels, test_groups: Test data
        alpha: Significance level
        lam: Fairness interpolation parameter
        score_function: 'softmax' or 'aps'

    Returns:
        Dict with coverage, set sizes, per-group results, disparity.
    """
    # Identify all groups
    all_groups = set()
    for g in cal_groups:
        if isinstance(g, list):
            all_groups.update(g)
        else:
            all_groups.add(g)
    all_groups = sorted(all_groups)

    # Compute calibration scores
    if score_function == "softmax":
        cal_scores = softmax_nonconformity_scores(cal_probs, cal_labels)
    else:
        cal_scores = aps_nonconformity_scores(cal_probs, cal_labels)

    # Marginal threshold
    q_marginal = compute_quantile_threshold(cal_scores, alpha)

    # Per-group thresholds
    group_thresholds = {}
    group_cal_sizes = {}
    for group in all_groups:
        group_idx = get_group_indices(cal_groups, group)
        group_cal_sizes[group] = len(group_idx)

        if len(group_idx) < MIN_GROUP_SIZE:
            group_thresholds[group] = q_marginal
        else:
            if score_function == "softmax":
                group_scores = softmax_nonconformity_scores(
                    cal_probs[group_idx], cal_labels[group_idx]
                )
            else:
                group_scores = aps_nonconformity_scores(
                    cal_probs[group_idx], cal_labels[group_idx]
                )
            group_thresholds[group] = compute_quantile_threshold(group_scores, alpha)

    # Compute fair thresholds
    fair_thresholds = compute_fair_thresholds(q_marginal, group_thresholds, lam)

    # Build prediction sets
    prediction_sets = []
    for i in range(len(test_labels)):
        sample_groups = test_groups[i] if isinstance(test_groups[i], list) else [test_groups[i]]
        q_hat_i = max(
            fair_thresholds.get(g, q_marginal) for g in sample_groups
        )

        if score_function == "softmax":
            psets = build_prediction_sets_softmax(test_probs[i:i+1], q_hat_i)
        else:
            psets = build_prediction_sets_aps(test_probs[i:i+1], q_hat_i)
        prediction_sets.append(psets[0])

    # Evaluate overall
    overall = evaluate_prediction_sets(prediction_sets, test_labels)

    # Evaluate per group
    per_group_results = {}
    for group in all_groups:
        group_idx = get_group_indices(test_groups, group)
        if len(group_idx) == 0:
            continue
        group_psets = [prediction_sets[i] for i in group_idx]
        group_eval = evaluate_prediction_sets(group_psets, test_labels[group_idx])
        group_eval["n_test"] = len(group_idx)
        group_eval["q_fair"] = fair_thresholds[group]
        per_group_results[group] = group_eval

    # Coverage disparity
    coverages = [r["coverage"] for r in per_group_results.values()]
    coverage_disparity = max(abs(c - (1 - alpha)) for c in coverages) if coverages else 0.0

    return {
        "lambda": lam,
        "overall_coverage": overall["coverage"],
        "overall_avg_set_size": overall["avg_set_size"],
        "per_group": per_group_results,
        "fair_thresholds": fair_thresholds,
        "q_marginal": q_marginal,
        "coverage_disparity": coverage_disparity,
        "prediction_sets": prediction_sets,
        "alpha": alpha,
        "score_function": score_function,
    }


def run_fair_cp_sweep(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_groups: list,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_groups: list,
    alpha: float = 0.10,
    score_function: str = "softmax",
    lambda_steps: int = 11,
) -> list:
    """Sweep lambda from 0.0 to 1.0 and compute results for each.

    Args:
        cal_probs, cal_labels, cal_groups: Calibration data
        test_probs, test_labels, test_groups: Test data
        alpha: Significance level
        score_function: 'softmax' or 'aps'
        lambda_steps: Number of lambda values (default 11 for 0.0 to 1.0 in 0.1)

    Returns:
        List of result dicts, one per lambda value.
    """
    lambdas = np.linspace(0.0, 1.0, lambda_steps)
    all_results = []

    print(f"\n[Fair CP Sweep] alpha={alpha}, score={score_function}")
    print(f"{'lambda':>8s} {'coverage':>10s} {'avg_size':>10s} {'disparity':>12s}")
    print("-" * 45)

    for lam in lambdas:
        result = run_fair_cp_single_lambda(
            cal_probs, cal_labels, cal_groups,
            test_probs, test_labels, test_groups,
            alpha, lam, score_function,
        )
        all_results.append(result)

        print(f"{lam:8.2f} {result['overall_coverage']:10.4f} "
              f"{result['overall_avg_set_size']:10.4f} "
              f"{result['coverage_disparity']:12.4f}")

    # Find the best lambda (lowest disparity with acceptable coverage)
    best_idx = min(range(len(all_results)), key=lambda i: all_results[i]["coverage_disparity"])
    best = all_results[best_idx]
    print(f"\nBest lambda = {best['lambda']:.2f} "
          f"(disparity={best['coverage_disparity']:.4f}, "
          f"avg_size={best['overall_avg_set_size']:.4f})")

    return all_results
