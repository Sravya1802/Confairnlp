"""
regen_baseline_plots.py -- Regenerate the three baseline PDFs in results/
(coverage_bar_chart, lambda_tradeoff, multi_alpha_disparity) at 300 DPI
from the existing _novelty_cache.pkl, without re-running the full pipeline.

Run from the repo root:
    python scripts/regen_baseline_plots.py
"""

from __future__ import annotations

import os
import pickle
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from conformal.fair_cp import result_for_lambda, run_fair_cp_sweep, select_lambda_by_tuning
from conformal.group_conditional_cp import compute_coverage_disparity, run_group_conditional_cp
from conformal.marginal_cp import run_marginal_cp
from evaluation.coverage_analysis import (
    compute_per_group_marginal_coverage,
    plot_multi_alpha_disparity,
    run_coverage_analysis,
)

CACHE_PATH = os.path.join(PROJECT_ROOT, "results", "_novelty_cache.pkl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, "results", "baseline_snapshot")


def _multi_alpha_results(setup: dict) -> dict:
    """Re-run the alpha sweep on the cached probabilities (numpy-only, fast)."""
    cal_probs = setup["cal_probs"]
    cal_labels = setup["cal_labels"]
    cal_groups = setup["cal_groups"]
    tuning_probs = setup["tuning_probs"]
    tuning_labels = setup["tuning_labels"]
    tuning_groups = setup["tuning_groups"]
    test_probs = setup["test_probs"]
    test_labels = setup["test_labels"]
    test_groups = setup["test_groups"]
    score_function = setup["score_function"]
    min_test_group_size = setup["min_test_group_size"]

    alpha_results = {}
    for alpha_val in [0.05, 0.10, 0.15, 0.20]:
        m = run_marginal_cp(
            cal_probs, cal_labels, test_probs, test_labels,
            alpha=alpha_val, score_function=score_function,
        )
        m_pg = compute_per_group_marginal_coverage(
            m["prediction_sets"], test_labels, test_groups,
            min_test_group_size=min_test_group_size,
        )
        m_disp = compute_coverage_disparity(
            m_pg, alpha_val, min_test_group_size=min_test_group_size,
        )
        gc = run_group_conditional_cp(
            cal_probs, cal_labels, cal_groups,
            test_probs, test_labels, test_groups,
            alpha=alpha_val, score_function=score_function,
            min_test_group_size=min_test_group_size,
        )
        fair_tuning = run_fair_cp_sweep(
            cal_probs, cal_labels, cal_groups,
            tuning_probs, tuning_labels, tuning_groups,
            alpha=alpha_val, score_function=score_function,
            min_test_group_size=min_test_group_size, verbose=False,
        )
        selected = select_lambda_by_tuning(fair_tuning, alpha_val)
        fair_test = run_fair_cp_sweep(
            cal_probs, cal_labels, cal_groups,
            test_probs, test_labels, test_groups,
            alpha=alpha_val, score_function=score_function,
            min_test_group_size=min_test_group_size, verbose=False,
        )
        selected_fair = result_for_lambda(fair_test, selected["lambda"])
        alpha_results[alpha_val] = {
            "marginal_disparity": m_disp,
            "group_conditional_disparity": gc["coverage_disparity"],
            "fair_cp_disparity": selected_fair["coverage_disparity"],
        }
    return alpha_results


def main() -> int:
    if not os.path.exists(CACHE_PATH):
        print(f"Missing cache: {CACHE_PATH}")
        print("Run any of the novelty modules first to populate it (or run_all.py).")
        return 1

    with open(CACHE_PATH, "rb") as f:
        setup = pickle.load(f)

    print("Regenerating coverage bar chart and lambda tradeoff at 300 DPI...")
    run_coverage_analysis(
        setup["marginal"],
        setup["group_conditional"],
        setup["fair_test_sweep"],
        setup["test_labels"],
        setup["test_groups"],
        setup["alpha"],
        OUTPUT_DIR,
        selected_lambda=setup["fair_lambda"],
        min_test_group_size=setup["min_test_group_size"],
    )

    print("\nRegenerating multi-alpha disparity plot...")
    alpha_results = _multi_alpha_results(setup)
    plot_multi_alpha_disparity(
        alpha_results,
        os.path.join(OUTPUT_DIR, "multi_alpha_disparity.pdf"),
    )

    print("\nCopying refreshed PDFs and PNGs into baseline_snapshot/...")
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    for name in [
        "coverage_bar_chart.pdf",
        "coverage_bar_chart.png",
        "lambda_tradeoff.pdf",
        "lambda_tradeoff.png",
        "multi_alpha_disparity.pdf",
        "multi_alpha_disparity.png",
    ]:
        src = os.path.join(OUTPUT_DIR, name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(SNAPSHOT_DIR, name)
        shutil.copy2(src, dst)
        print(f"  {dst}")

    print("\nDone. All baseline PDFs and PNGs refreshed at 400 DPI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
