"""
Ablation studies across models, score functions, and alpha values.

Every Fair CP configuration selects lambda on the tuning split and reports
metrics on the final test split.
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from conformal.fair_cp import result_for_lambda, run_fair_cp_sweep, select_lambda_by_tuning
from conformal.group_conditional_cp import (
    MIN_RELIABLE_TEST_GROUP_SIZE,
    compute_coverage_disparity,
    run_group_conditional_cp,
)
from conformal.marginal_cp import run_marginal_cp
from evaluation.coverage_analysis import compute_per_group_marginal_coverage

SEED = 42
np.random.seed(SEED)

ALPHA_VALUES = [0.05, 0.10, 0.15, 0.20]
SCORE_FUNCTIONS = ["softmax", "aps"]


def run_single_ablation(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_groups: list,
    tuning_probs: np.ndarray,
    tuning_labels: np.ndarray,
    tuning_groups: list,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_groups: list,
    alpha: float,
    score_function: str,
    model_name: str,
    dataset_name: str,
    lambda_steps: int = 11,
    min_test_group_size: int = MIN_RELIABLE_TEST_GROUP_SIZE,
) -> dict:
    """Run one ablation configuration."""
    marginal = run_marginal_cp(
        cal_probs,
        cal_labels,
        test_probs,
        test_labels,
        alpha=alpha,
        score_function=score_function,
    )

    marginal_per_group = compute_per_group_marginal_coverage(
        marginal["prediction_sets"],
        test_labels,
        test_groups,
        min_test_group_size=min_test_group_size,
    )
    marginal_disparity = compute_coverage_disparity(
        marginal_per_group,
        alpha,
        min_test_group_size=min_test_group_size,
    )

    gc = run_group_conditional_cp(
        cal_probs,
        cal_labels,
        cal_groups,
        test_probs,
        test_labels,
        test_groups,
        alpha=alpha,
        score_function=score_function,
        min_test_group_size=min_test_group_size,
    )

    fair_tuning = run_fair_cp_sweep(
        cal_probs,
        cal_labels,
        cal_groups,
        tuning_probs,
        tuning_labels,
        tuning_groups,
        alpha=alpha,
        score_function=score_function,
        lambda_steps=lambda_steps,
        min_test_group_size=min_test_group_size,
        verbose=False,
    )
    selected_tuning = select_lambda_by_tuning(fair_tuning, alpha)

    fair_test = run_fair_cp_sweep(
        cal_probs,
        cal_labels,
        cal_groups,
        test_probs,
        test_labels,
        test_groups,
        alpha=alpha,
        score_function=score_function,
        lambda_steps=lambda_steps,
        min_test_group_size=min_test_group_size,
        verbose=False,
    )
    selected_fair = result_for_lambda(fair_test, selected_tuning["lambda"])

    return {
        "model": model_name,
        "dataset": dataset_name,
        "alpha": alpha,
        "score_function": score_function,
        "marginal_coverage": marginal["coverage"],
        "marginal_avg_set_size": marginal["avg_set_size"],
        "marginal_disparity": marginal_disparity,
        "gc_coverage": gc["overall_coverage"],
        "gc_avg_set_size": gc["overall_avg_set_size"],
        "gc_disparity": gc["coverage_disparity"],
        "gc_disparity_all": gc["coverage_disparity_all"],
        "fair_coverage": selected_fair["overall_coverage"],
        "fair_avg_set_size": selected_fair["overall_avg_set_size"],
        "fair_disparity": selected_fair["coverage_disparity"],
        "fair_disparity_all": selected_fair["coverage_disparity_all"],
        "selected_lambda": selected_fair["lambda"],
    }


def run_ablation_study(
    model_probs: dict,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Run ablations over supplied model probability dictionaries."""
    os.makedirs(output_dir, exist_ok=True)

    configs = []
    for (model_name, dataset_name), data in model_probs.items():
        for alpha in ALPHA_VALUES:
            for score_fn in SCORE_FUNCTIONS:
                configs.append((model_name, dataset_name, alpha, score_fn, data))

    print(f"\n[Ablation] Running {len(configs)} configurations...")
    all_results = []
    for model_name, dataset_name, alpha, score_fn, data in tqdm(configs, desc="Ablation"):
        result = run_single_ablation(
            cal_probs=data["cal_probs"],
            cal_labels=data["cal_labels"],
            cal_groups=data["cal_groups"],
            tuning_probs=data["tuning_probs"],
            tuning_labels=data["tuning_labels"],
            tuning_groups=data["tuning_groups"],
            test_probs=data["test_probs"],
            test_labels=data["test_labels"],
            test_groups=data["test_groups"],
            alpha=alpha,
            score_function=score_fn,
            model_name=model_name,
            dataset_name=dataset_name,
            lambda_steps=data.get("lambda_steps", 11),
            min_test_group_size=data.get(
                "min_test_group_size",
                MIN_RELIABLE_TEST_GROUP_SIZE,
            ),
        )
        all_results.append(result)

    df = pd.DataFrame(all_results)

    csv_path = os.path.join(output_dir, "ablation_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[Ablation] Results saved to {csv_path}")
    print("\n[Ablation Summary]")
    print(df.to_string(index=False))

    return df
