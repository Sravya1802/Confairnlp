"""
Group-conditional conformal prediction.

Computes per-group conformal thresholds and falls back to the marginal
threshold for groups with too few calibration examples.
"""

import warnings

import numpy as np

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
MIN_GROUP_SIZE = 30
MIN_RELIABLE_TEST_GROUP_SIZE = 30
np.random.seed(SEED)


def sample_group_list(groups) -> list[str]:
    if isinstance(groups, str):
        values = [groups]
    elif isinstance(groups, (list, tuple, set, np.ndarray)):
        values = list(groups)
    elif groups is None:
        values = []
    else:
        values = [str(groups)]

    cleaned = []
    for group in values:
        if group is None:
            continue
        text = str(group).strip()
        if text and text.lower() not in {"none", "nan"}:
            cleaned.append(text)
    return cleaned or ["unknown"]


def collect_unique_groups(*group_lists: list) -> list[str]:
    all_groups = set()
    for group_list in group_lists:
        for groups in group_list:
            all_groups.update(sample_group_list(groups))
    return sorted(all_groups)


def get_group_indices(groups: list, target_group: str) -> np.ndarray:
    """Get indices of samples belonging to a specific group."""
    indices = []
    for i, sample_groups in enumerate(groups):
        if target_group in sample_group_list(sample_groups):
            indices.append(i)
    return np.array(indices, dtype=int)


def compute_coverage_disparity(
    per_group_results: dict,
    alpha: float,
    min_test_group_size: int | None = None,
) -> float:
    """Compute max absolute group coverage deviation from target coverage."""
    target_coverage = 1 - alpha
    coverages = []
    for result in per_group_results.values():
        if min_test_group_size is not None and result.get("n_test", 0) < min_test_group_size:
            continue
        if not np.isnan(result["coverage"]):
            coverages.append(result["coverage"])
    return max(abs(coverage - target_coverage) for coverage in coverages) if coverages else 0.0


def run_group_conditional_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_groups: list,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_groups: list,
    alpha: float = 0.10,
    score_function: str = "softmax",
    min_test_group_size: int = MIN_RELIABLE_TEST_GROUP_SIZE,
) -> dict:
    """Run group-conditional conformal prediction."""
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

    all_groups = collect_unique_groups(cal_groups, test_groups)
    cal_scores_all = nonconformity_scores(cal_probs, cal_labels, score_function)
    q_marginal = compute_quantile_threshold(cal_scores_all, alpha)

    group_thresholds = {}
    group_cal_sizes = {}
    fallback_groups = set()

    for group in all_groups:
        group_idx = get_group_indices(cal_groups, group)
        group_cal_sizes[group] = len(group_idx)

        if len(group_idx) < MIN_GROUP_SIZE:
            warnings.warn(
                f"Group '{group}' has only {len(group_idx)} calibration samples "
                f"(< {MIN_GROUP_SIZE}). Using marginal threshold as fallback."
            )
            group_thresholds[group] = q_marginal
            fallback_groups.add(group)
            continue

        group_scores = nonconformity_scores(
            cal_probs[group_idx],
            cal_labels[group_idx],
            score_function,
        )
        group_thresholds[group] = compute_quantile_threshold(group_scores, alpha)

    prediction_sets = []
    for i in range(len(test_labels)):
        q_hat_i = max(
            group_thresholds.get(group, q_marginal)
            for group in sample_group_list(test_groups[i])
        )
        psets = build_prediction_sets(test_probs[i : i + 1], q_hat_i, score_function)
        prediction_sets.append(psets[0])

    overall = evaluate_prediction_sets(prediction_sets, test_labels)

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
        group_eval["q_hat"] = group_thresholds.get(group, q_marginal)
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

    results = {
        "overall_coverage": overall["coverage"],
        "overall_avg_set_size": overall["avg_set_size"],
        "per_group": per_group_results,
        "group_thresholds": group_thresholds,
        "q_marginal": q_marginal,
        "coverage_disparity": coverage_disparity,
        "coverage_disparity_reliable": coverage_disparity_reliable,
        "coverage_disparity_all": coverage_disparity_all,
        "prediction_sets": prediction_sets,
        "alpha": alpha,
        "score_function": score_function,
        "min_test_group_size": min_test_group_size,
    }

    print(f"\n[Group-Conditional CP] alpha={alpha}, score={score_function}")
    print(f"  Overall coverage = {overall['coverage']:.4f}")
    print(f"  Overall avg set size = {overall['avg_set_size']:.4f}")
    print(f"  Coverage disparity (reliable groups) = {coverage_disparity_reliable:.4f}")
    print(f"  Coverage disparity (all groups) = {coverage_disparity_all:.4f}")
    print("  Per-group coverage:")
    for group, result in sorted(per_group_results.items()):
        marker = "" if result["reliable_group"] else " [small n]"
        print(
            f"    {group:20s}: coverage={result['coverage']:.4f}, "
            f"avg_size={result['avg_set_size']:.4f}, n_test={result['n_test']}, "
            f"n_cal={result['n_cal']}{marker}"
        )

    return results
