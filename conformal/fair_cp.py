"""
Fairness-regularized conformal prediction.

Interpolates between marginal and group-conditional thresholds:

    q_fair_g = lambda * q_g + (1 - lambda) * q_marginal

Lambda should be selected on a tuning split, not on the final test split.
"""

import numpy as np

from conformal.group_conditional_cp import (
    MIN_GROUP_SIZE,
    MIN_RELIABLE_TEST_GROUP_SIZE,
    collect_unique_groups,
    compute_coverage_disparity,
    get_group_indices,
    sample_group_list,
)
from conformal.marginal_cp import (
    build_prediction_sets,
    compute_quantile_threshold,
    evaluate_prediction_sets,
    nonconformity_scores,
    validate_alpha,
    validate_probability_inputs,
    validate_score_function,
)

SEED = 42
np.random.seed(SEED)


def compute_fair_thresholds(
    q_marginal: float,
    group_thresholds: dict,
    lam: float,
) -> dict:
    """Compute q_fair_g = lam * q_g + (1 - lam) * q_marginal."""
    if not 0 <= lam <= 1:
        raise ValueError(f"lambda must be between 0 and 1, got {lam!r}")
    return {
        group: lam * q_g + (1 - lam) * q_marginal
        for group, q_g in group_thresholds.items()
    }


def _calibrate_group_thresholds(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_groups: list,
    eval_groups: list,
    alpha: float,
    score_function: str,
) -> tuple[float, dict, dict, set]:
    all_groups = collect_unique_groups(cal_groups, eval_groups)
    cal_scores = nonconformity_scores(cal_probs, cal_labels, score_function)
    q_marginal = compute_quantile_threshold(cal_scores, alpha)

    group_thresholds = {}
    group_cal_sizes = {}
    fallback_groups = set()
    for group in all_groups:
        group_idx = get_group_indices(cal_groups, group)
        group_cal_sizes[group] = len(group_idx)

        if len(group_idx) < MIN_GROUP_SIZE:
            group_thresholds[group] = q_marginal
            fallback_groups.add(group)
            continue

        group_scores = nonconformity_scores(
            cal_probs[group_idx],
            cal_labels[group_idx],
            score_function,
        )
        group_thresholds[group] = compute_quantile_threshold(group_scores, alpha)

    return q_marginal, group_thresholds, group_cal_sizes, fallback_groups


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
    min_test_group_size: int = MIN_RELIABLE_TEST_GROUP_SIZE,
) -> dict:
    """Run fairness-regularized CP for a single lambda value."""
    validate_alpha(alpha)
    validate_score_function(score_function)
    validate_probability_inputs(cal_probs, cal_labels)
    validate_probability_inputs(test_probs, test_labels)
    if cal_probs.shape[1] != test_probs.shape[1]:
        raise ValueError("calibration and test probabilities must have the same class count")
    if len(cal_groups) != len(cal_labels):
        raise ValueError("cal_groups length must match cal_labels length")
    if len(test_groups) != len(test_labels):
        raise ValueError("test_groups length must match test_labels length")

    q_marginal, group_thresholds, group_cal_sizes, fallback_groups = (
        _calibrate_group_thresholds(
            cal_probs,
            cal_labels,
            cal_groups,
            test_groups,
            alpha,
            score_function,
        )
    )
    fair_thresholds = compute_fair_thresholds(q_marginal, group_thresholds, lam)

    prediction_sets = []
    for i in range(len(test_labels)):
        q_hat_i = max(
            fair_thresholds.get(group, q_marginal)
            for group in sample_group_list(test_groups[i])
        )
        psets = build_prediction_sets(test_probs[i : i + 1], q_hat_i, score_function)
        prediction_sets.append(psets[0])

    overall = evaluate_prediction_sets(prediction_sets, test_labels)

    per_group_results = {}
    for group in collect_unique_groups(cal_groups, test_groups):
        group_idx = get_group_indices(test_groups, group)
        if len(group_idx) == 0:
            continue

        group_psets = [prediction_sets[i] for i in group_idx]
        group_eval = evaluate_prediction_sets(group_psets, test_labels[group_idx])
        group_eval["n_cal"] = group_cal_sizes.get(group, 0)
        group_eval["n_test"] = len(group_idx)
        group_eval["q_fair"] = fair_thresholds.get(group, q_marginal)
        group_eval["used_marginal_fallback"] = group in fallback_groups
        group_eval["reliable_group"] = len(group_idx) >= min_test_group_size
        per_group_results[group] = group_eval

    coverage_disparity_all = compute_coverage_disparity(per_group_results, alpha)
    coverage_disparity_reliable = compute_coverage_disparity(
        per_group_results,
        alpha,
        min_test_group_size=min_test_group_size,
    )
    coverage_disparity = coverage_disparity_reliable or coverage_disparity_all

    return {
        "lambda": float(lam),
        "overall_coverage": overall["coverage"],
        "overall_avg_set_size": overall["avg_set_size"],
        "per_group": per_group_results,
        "fair_thresholds": fair_thresholds,
        "q_marginal": q_marginal,
        "coverage_disparity": coverage_disparity,
        "coverage_disparity_reliable": coverage_disparity_reliable,
        "coverage_disparity_all": coverage_disparity_all,
        "prediction_sets": prediction_sets,
        "alpha": alpha,
        "score_function": score_function,
        "min_test_group_size": min_test_group_size,
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
    lambda_values: list[float] | None = None,
    min_test_group_size: int = MIN_RELIABLE_TEST_GROUP_SIZE,
    verbose: bool = True,
) -> list:
    """Evaluate Fair CP over a lambda grid."""
    if lambda_values is None:
        lambdas = np.linspace(0.0, 1.0, lambda_steps)
    else:
        lambdas = np.array(lambda_values, dtype=float)

    all_results = []
    if verbose:
        print(f"\n[Fair CP Sweep] alpha={alpha}, score={score_function}")
        print(
            f"{'lambda':>8s} {'coverage':>10s} {'avg_size':>10s} "
            f"{'disp_rel':>10s} {'disp_all':>10s}"
        )
        print("-" * 56)

    for lam in lambdas:
        result = run_fair_cp_single_lambda(
            cal_probs,
            cal_labels,
            cal_groups,
            test_probs,
            test_labels,
            test_groups,
            alpha,
            float(lam),
            score_function,
            min_test_group_size=min_test_group_size,
        )
        all_results.append(result)

        if verbose:
            print(
                f"{lam:8.2f} {result['overall_coverage']:10.4f} "
                f"{result['overall_avg_set_size']:10.4f} "
                f"{result['coverage_disparity_reliable']:10.4f} "
                f"{result['coverage_disparity_all']:10.4f}"
            )

    if verbose:
        best = select_lambda_by_tuning(all_results, alpha)
        print(
            f"\nLowest selection objective in this sweep: lambda={best['lambda']:.2f} "
            f"(coverage={best['overall_coverage']:.4f}, "
            f"reliable disparity={best['coverage_disparity_reliable']:.4f}, "
            f"avg_size={best['overall_avg_set_size']:.4f})"
        )

    return all_results


def select_lambda_by_tuning(sweep_results: list, alpha: float) -> dict:
    """Select lambda from tuning results without looking at test labels.

    The objective first avoids undercoverage, then minimizes reliable-group
    disparity, then average set size.
    """
    target_coverage = 1 - alpha

    def key(result):
        coverage_shortfall = max(0.0, target_coverage - result["overall_coverage"])
        return (
            coverage_shortfall,
            result["coverage_disparity"],
            result["overall_avg_set_size"],
            result["lambda"],
        )

    return min(sweep_results, key=key)


def result_for_lambda(sweep_results: list, selected_lambda: float) -> dict:
    """Return the sweep result matching selected_lambda, tolerant of float grids."""
    return min(sweep_results, key=lambda result: abs(result["lambda"] - selected_lambda))
