"""
ablation.py — Ablation studies across models, scores, alphas, and datasets.

Compares:
  - BERT vs HateBERT as base classifiers
  - Softmax score vs APS score
  - Alpha values: 0.05, 0.10, 0.15, 0.20
  - Datasets: HateXplain, ToxiGen

Outputs a summary table (CSV) with coverage disparity and avg set size
for each configuration.
"""

import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from conformal.marginal_cp import run_marginal_cp
from conformal.group_conditional_cp import run_group_conditional_cp
from conformal.fair_cp import run_fair_cp_sweep
from evaluation.coverage_analysis import compute_per_group_marginal_coverage

SEED = 42
np.random.seed(SEED)

ALPHA_VALUES = [0.05, 0.10, 0.15, 0.20]
SCORE_FUNCTIONS = ["softmax", "aps"]


def run_single_ablation(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_groups: list,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_groups: list,
    alpha: float,
    score_function: str,
    model_name: str,
    dataset_name: str,
) -> dict:
    """Run a single ablation configuration.

    Returns a dict with the configuration and key metrics.
    """
    # Marginal CP
    marginal = run_marginal_cp(
        cal_probs, cal_labels, test_probs, test_labels,
        alpha=alpha, score_function=score_function,
    )

    # Compute per-group marginal coverage for disparity
    marginal_per_group = compute_per_group_marginal_coverage(
        marginal["prediction_sets"], test_labels, test_groups,
    )
    marginal_disparity = 0.0
    if marginal_per_group:
        marginal_disparity = max(
            abs(v["coverage"] - (1 - alpha))
            for v in marginal_per_group.values()
        )

    # Group-conditional CP
    gc = run_group_conditional_cp(
        cal_probs, cal_labels, cal_groups,
        test_probs, test_labels, test_groups,
        alpha=alpha, score_function=score_function,
    )

    # Fair CP sweep
    fair_results = run_fair_cp_sweep(
        cal_probs, cal_labels, cal_groups,
        test_probs, test_labels, test_groups,
        alpha=alpha, score_function=score_function,
    )
    best_fair = min(fair_results, key=lambda r: r["coverage_disparity"])

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
        "fair_coverage": best_fair["overall_coverage"],
        "fair_avg_set_size": best_fair["overall_avg_set_size"],
        "fair_disparity": best_fair["coverage_disparity"],
        "best_lambda": best_fair["lambda"],
    }


def run_ablation_study(
    model_probs: dict,
    hatexplain_splits: dict,
    toxigen_data: dict = None,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Run full ablation study across all configurations.

    Args:
        model_probs: Dict mapping (model_name, dataset) -> {
            'cal_probs', 'cal_labels', 'cal_groups',
            'test_probs', 'test_labels', 'test_groups'
        }
        hatexplain_splits: HateXplain data splits
        toxigen_data: Optional ToxiGen data for cross-dataset ablation
        output_dir: Directory to save results

    Returns:
        DataFrame with ablation results
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    configs = []
    for (model_name, dataset_name), data in model_probs.items():
        for alpha in ALPHA_VALUES:
            for score_fn in SCORE_FUNCTIONS:
                configs.append((model_name, dataset_name, alpha, score_fn, data))

    print(f"\n[Ablation] Running {len(configs)} configurations...")
    for model_name, dataset_name, alpha, score_fn, data in tqdm(configs, desc="Ablation"):
        result = run_single_ablation(
            cal_probs=data["cal_probs"],
            cal_labels=data["cal_labels"],
            cal_groups=data["cal_groups"],
            test_probs=data["test_probs"],
            test_labels=data["test_labels"],
            test_groups=data["test_groups"],
            alpha=alpha,
            score_function=score_fn,
            model_name=model_name,
            dataset_name=dataset_name,
        )
        all_results.append(result)

    df = pd.DataFrame(all_results)

    # Save
    csv_path = os.path.join(output_dir, "ablation_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[Ablation] Results saved to {csv_path}")
    print("\n[Ablation Summary]")
    print(df.to_string(index=False))

    return df
