"""
group_conditional_cp.py — Group-conditional conformal prediction.

Computes separate conformal thresholds for each demographic group,
ensuring per-group coverage guarantees. Falls back to marginal
threshold for groups with fewer than 30 calibration samples.
"""

import warnings

import numpy as np

from conformal.marginal_cp import (
    aps_nonconformity_scores,
    build_prediction_sets_aps,
    build_prediction_sets_softmax,
    compute_quantile_threshold,
    evaluate_prediction_sets,
    softmax_nonconformity_scores,
)

SEED = 42
np.random.seed(SEED)

MIN_GROUP_SIZE = 30  # Minimum calibration samples per group


def get_group_indices(groups: list, target_group: str) -> np.ndarray:
    """Get indices of samples belonging to a specific group.

    Handles the case where each sample can belong to multiple groups
    (stored as lists).

    Args:
        groups: List of group labels. Each element can be a string or
                a list of strings (for multi-group membership).
        target_group: The group to filter for.

    Returns:
        Array of indices belonging to the target group.
    """
    indices = []
    for i, g in enumerate(groups):
        if isinstance(g, list):
            if target_group in g:
                indices.append(i)
        elif g == target_group:
            indices.append(i)
    return np.array(indices, dtype=int)


def run_group_conditional_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_groups: list,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_groups: list,
    alpha: float = 0.10,
    score_function: str = "softmax",
) -> dict:
    """Run group-conditional conformal prediction.

    Computes a separate threshold q_hat_g for each demographic group g
    using only that group's calibration data. At test time, uses the
    threshold matching the test sample's group.

    Args:
        cal_probs: Calibration softmax probabilities, shape (n_cal, num_classes)
        cal_labels: Calibration true labels, shape (n_cal,)
        cal_groups: Calibration group labels (list of str or list of lists)
        test_probs: Test softmax probabilities, shape (n_test, num_classes)
        test_labels: Test true labels, shape (n_test,)
        test_groups: Test group labels
        alpha: Significance level
        score_function: 'softmax' or 'aps'

    Returns:
        Dict with per-group thresholds, coverage rates, set sizes, etc.
    """
    # Identify all unique groups
    all_groups = set()
    for g in cal_groups:
        if isinstance(g, list):
            all_groups.update(g)
        else:
            all_groups.add(g)
    all_groups = sorted(all_groups)

    # Compute overall calibration scores for fallback
    if score_function == "softmax":
        cal_scores_all = softmax_nonconformity_scores(cal_probs, cal_labels)
    else:
        cal_scores_all = aps_nonconformity_scores(cal_probs, cal_labels)

    q_marginal = compute_quantile_threshold(cal_scores_all, alpha)

    # Compute per-group thresholds
    group_thresholds = {}
    group_cal_sizes = {}

    for group in all_groups:
        group_idx = get_group_indices(cal_groups, group)
        group_cal_sizes[group] = len(group_idx)

        if len(group_idx) < MIN_GROUP_SIZE:
            warnings.warn(
                f"Group '{group}' has only {len(group_idx)} calibration samples "
                f"(< {MIN_GROUP_SIZE}). Using marginal threshold as fallback."
            )
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

    # Build prediction sets per test sample using its group's threshold
    # For multi-group samples, use the most conservative (largest) threshold
    prediction_sets = []
    for i in range(len(test_labels)):
        sample_groups = test_groups[i] if isinstance(test_groups[i], list) else [test_groups[i]]
        # Use the maximum threshold among all groups the sample belongs to
        q_hat_i = max(
            group_thresholds.get(g, q_marginal) for g in sample_groups
        )

        if score_function == "softmax":
            psets = build_prediction_sets_softmax(
                test_probs[i : i + 1], q_hat_i
            )
        else:
            psets = build_prediction_sets_aps(
                test_probs[i : i + 1], q_hat_i
            )
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
        group_labels = test_labels[group_idx]
        group_eval = evaluate_prediction_sets(group_psets, group_labels)
        group_eval["n_cal"] = group_cal_sizes.get(group, 0)
        group_eval["n_test"] = len(group_idx)
        group_eval["q_hat"] = group_thresholds[group]
        per_group_results[group] = group_eval

    # Coverage disparity
    coverages = [r["coverage"] for r in per_group_results.values()]
    coverage_disparity = max(abs(c - (1 - alpha)) for c in coverages) if coverages else 0.0

    results = {
        "overall_coverage": overall["coverage"],
        "overall_avg_set_size": overall["avg_set_size"],
        "per_group": per_group_results,
        "group_thresholds": group_thresholds,
        "q_marginal": q_marginal,
        "coverage_disparity": coverage_disparity,
        "prediction_sets": prediction_sets,
        "alpha": alpha,
        "score_function": score_function,
    }

    print(f"\n[Group-Conditional CP] alpha={alpha}, score={score_function}")
    print(f"  Overall coverage = {overall['coverage']:.4f}")
    print(f"  Overall avg set size = {overall['avg_set_size']:.4f}")
    print(f"  Coverage disparity = {coverage_disparity:.4f}")
    print(f"  Per-group coverage:")
    for group, r in sorted(per_group_results.items()):
        print(f"    {group:20s}: coverage={r['coverage']:.4f}, "
              f"avg_size={r['avg_set_size']:.4f}, n_test={r['n_test']}, "
              f"n_cal={r['n_cal']}")

    return results
