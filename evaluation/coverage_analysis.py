"""
Per-group coverage analysis and visualization.

The primary disparity metric excludes groups with very small final-test counts
by default, while still reporting all-group disparity and marking small groups
in the output table.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conformal.fair_cp import result_for_lambda
from conformal.group_conditional_cp import (
    MIN_RELIABLE_TEST_GROUP_SIZE,
    compute_coverage_disparity,
    get_group_indices,
    sample_group_list,
)

SEED = 42
np.random.seed(SEED)


def _save_pdf_and_png(save_path: str, dpi: int = 300) -> None:
    """Save the active figure to ``save_path`` and a sibling PNG.

    GitHub markdown renders PNG but not PDF, so every plot in this module
    writes both formats so they can be embedded in README.md / NOVELTY_SUMMARY.md.
    """
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if save_path.lower().endswith(".pdf"):
        png_path = save_path[:-4] + ".png"
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson confidence interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    half_width = (
        z
        * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _group_ci(result: dict) -> tuple[float, float]:
    return wilson_interval(result.get("n_covered", 0), result.get("n_test", 0))


def build_coverage_table(
    marginal_per_group: dict,
    group_cond_results: dict,
    fair_cp_results: dict,
) -> pd.DataFrame:
    """Build a per-group coverage comparison table."""
    groups = sorted(
        set(marginal_per_group)
        | set(group_cond_results["per_group"])
        | set(fair_cp_results["per_group"])
    )

    rows = []
    for group in groups:
        marginal = marginal_per_group.get(group, {})
        gc = group_cond_results["per_group"].get(group, {})
        fair = fair_cp_results["per_group"].get(group, {})
        marginal_low, marginal_high = _group_ci(marginal)
        gc_low, gc_high = _group_ci(gc)
        fair_low, fair_high = _group_ci(fair)

        rows.append(
            {
                "Group": group,
                "Reliable_Group": bool(
                    gc.get("reliable_group")
                    or fair.get("reliable_group")
                    or marginal.get("reliable_group")
                ),
                "N_Test": int(
                    gc.get("n_test")
                    or fair.get("n_test")
                    or marginal.get("n_test", 0)
                ),
                "N_Calibration": int(gc.get("n_cal", fair.get("n_cal", 0))),
                "Marginal_Coverage": marginal.get("coverage", np.nan),
                "Marginal_CI_Low": marginal_low,
                "Marginal_CI_High": marginal_high,
                "Marginal_SetSize": marginal.get("avg_set_size", np.nan),
                "GroupCond_Coverage": gc.get("coverage", np.nan),
                "GroupCond_CI_Low": gc_low,
                "GroupCond_CI_High": gc_high,
                "GroupCond_SetSize": gc.get("avg_set_size", np.nan),
                "GroupCond_Fallback": bool(gc.get("used_marginal_fallback", False)),
                "FairCP_Coverage": fair.get("coverage", np.nan),
                "FairCP_CI_Low": fair_low,
                "FairCP_CI_High": fair_high,
                "FairCP_SetSize": fair.get("avg_set_size", np.nan),
                "FairCP_Fallback": bool(fair.get("used_marginal_fallback", False)),
            }
        )

    return pd.DataFrame(rows)


def compute_per_group_marginal_coverage(
    prediction_sets: list,
    test_labels: np.ndarray,
    test_groups: list,
    min_test_group_size: int = MIN_RELIABLE_TEST_GROUP_SIZE,
) -> dict:
    """Compute marginal CP coverage broken down by group."""
    all_groups = sorted({group for groups in test_groups for group in sample_group_list(groups)})

    per_group = {}
    for group in all_groups:
        idx = get_group_indices(test_groups, group)
        if len(idx) == 0:
            continue
        n_covered = sum(1 for i in idx if test_labels[i] in prediction_sets[i])
        sizes = [len(prediction_sets[i]) for i in idx]
        ci_low, ci_high = wilson_interval(n_covered, len(idx))
        per_group[group] = {
            "coverage": n_covered / len(idx),
            "avg_set_size": float(np.mean(sizes)),
            "n_test": len(idx),
            "n_covered": n_covered,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "reliable_group": len(idx) >= min_test_group_size,
        }

    return per_group


def plot_coverage_bar_chart(
    marginal_per_group: dict,
    group_cond_results: dict,
    fair_cp_results: dict,
    alpha: float,
    save_path: str,
):
    """Plot grouped bar chart of per-group coverage for all methods."""
    groups = sorted(group_cond_results["per_group"].keys())

    marginal_covs = [marginal_per_group.get(g, {}).get("coverage", np.nan) for g in groups]
    gc_covs = [group_cond_results["per_group"][g]["coverage"] for g in groups]
    fair_covs = [fair_cp_results["per_group"].get(g, {}).get("coverage", np.nan) for g in groups]

    x = np.arange(len(groups))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, marginal_covs, width, label="Marginal CP", color="#4C72B0")
    ax.bar(x, gc_covs, width, label="Group-Conditional CP", color="#DD8452")
    ax.bar(x + width, fair_covs, width, label="Fair CP", color="#55A868")

    ax.axhline(
        y=1 - alpha,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Target (1-alpha = {1-alpha:.2f})",
    )

    ax.set_xlabel("Demographic Group", fontsize=12)
    ax.set_ylabel("Coverage Rate", fontsize=12)
    ax.set_title(f"Per-Group Coverage Comparison (alpha = {alpha})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    _save_pdf_and_png(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Coverage bar chart saved to {save_path}")


def plot_lambda_tradeoff(
    sweep_results: list,
    alpha: float,
    save_path: str,
    selected_lambda: float | None = None,
):
    """Plot the Fair CP tradeoff: reliable disparity vs. average set size."""
    lambdas = [result["lambda"] for result in sweep_results]
    disparities = [result["coverage_disparity"] for result in sweep_results]
    set_sizes = [result["overall_avg_set_size"] for result in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        set_sizes,
        disparities,
        c=lambdas,
        cmap="viridis",
        s=100,
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )
    ax.plot(set_sizes, disparities, "--", color="gray", alpha=0.5, zorder=1)

    for i, lam in enumerate(lambdas):
        ax.annotate(f"lambda={lam:.1f}", (set_sizes[i], disparities[i]), xytext=(5, 5),
                    textcoords="offset points", fontsize=7)

    if selected_lambda is not None:
        selected = result_for_lambda(sweep_results, selected_lambda)
        ax.scatter(
            [selected["overall_avg_set_size"]],
            [selected["coverage_disparity"]],
            s=180,
            facecolors="none",
            edgecolors="red",
            linewidth=2,
            label=f"Selected lambda={selected['lambda']:.2f}",
            zorder=10,
        )
        ax.legend(fontsize=9)

    plt.colorbar(scatter, ax=ax, label="lambda")
    ax.set_xlabel("Average Prediction Set Size", fontsize=12)
    ax.set_ylabel("Coverage Disparity on Reliable Groups", fontsize=12)
    ax.set_title(f"Fairness-Efficiency Tradeoff (alpha = {alpha})", fontsize=14)
    plt.tight_layout()
    _save_pdf_and_png(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Lambda tradeoff curve saved to {save_path}")


def plot_multi_alpha_disparity(alpha_results: dict, save_path: str):
    """Plot coverage disparity across multiple alpha values."""
    alphas = sorted(alpha_results.keys())
    methods = [
        ("Marginal CP", "marginal_disparity"),
        ("Group-Conditional CP", "group_conditional_disparity"),
        ("Fair CP", "fair_cp_disparity"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, key in methods:
        disparities = [alpha_results[alpha].get(key, 0) for alpha in alphas]
        ax.plot(alphas, disparities, "o-", label=label, markersize=8)

    ax.set_xlabel("alpha (Significance Level)", fontsize=12)
    ax.set_ylabel("Reliable-Group Coverage Disparity", fontsize=12)
    ax.set_title("Coverage Disparity Across Alpha Values", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_pdf_and_png(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Multi-alpha disparity plot saved to {save_path}")


def run_coverage_analysis(
    marginal_results: dict,
    group_cond_results: dict,
    fair_sweep_results: list,
    test_labels: np.ndarray,
    test_groups: list,
    alpha: float,
    output_dir: str,
    selected_lambda: float | None = None,
    min_test_group_size: int = MIN_RELIABLE_TEST_GROUP_SIZE,
) -> dict:
    """Run coverage analysis and generate output files."""
    os.makedirs(output_dir, exist_ok=True)

    if selected_lambda is None:
        fair_result = min(fair_sweep_results, key=lambda result: result["coverage_disparity"])
        print(
            "\n[Coverage Analysis] No tuned lambda was provided; using lowest "
            "test-sweep disparity for this analysis only."
        )
    else:
        fair_result = result_for_lambda(fair_sweep_results, selected_lambda)
        print(
            f"\nSelected Fair CP lambda={fair_result['lambda']:.2f} "
            f"(chosen on tuning split)"
        )

    marginal_per_group = compute_per_group_marginal_coverage(
        marginal_results["prediction_sets"],
        test_labels,
        test_groups,
        min_test_group_size=min_test_group_size,
    )

    coverage_df = build_coverage_table(
        marginal_per_group,
        group_cond_results,
        fair_result,
    )

    csv_path = os.path.join(output_dir, "per_group_coverage.csv")
    coverage_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[Coverage Table] Saved to {csv_path}")
    print(coverage_df.to_string(index=False))

    plot_coverage_bar_chart(
        marginal_per_group,
        group_cond_results,
        fair_result,
        alpha,
        os.path.join(output_dir, "coverage_bar_chart.pdf"),
    )

    plot_lambda_tradeoff(
        fair_sweep_results,
        alpha,
        os.path.join(output_dir, "lambda_tradeoff.pdf"),
        selected_lambda=fair_result["lambda"],
    )

    marginal_disparity_all = compute_coverage_disparity(marginal_per_group, alpha)
    marginal_disparity_reliable = compute_coverage_disparity(
        marginal_per_group,
        alpha,
        min_test_group_size=min_test_group_size,
    )

    return {
        "coverage_table": coverage_df,
        "marginal_per_group": marginal_per_group,
        "marginal_disparity": marginal_disparity_reliable or marginal_disparity_all,
        "marginal_disparity_reliable": marginal_disparity_reliable,
        "marginal_disparity_all": marginal_disparity_all,
        "gc_disparity": group_cond_results["coverage_disparity"],
        "gc_disparity_reliable": group_cond_results["coverage_disparity_reliable"],
        "gc_disparity_all": group_cond_results["coverage_disparity_all"],
        "fair_disparity": fair_result["coverage_disparity"],
        "fair_disparity_reliable": fair_result["coverage_disparity_reliable"],
        "fair_disparity_all": fair_result["coverage_disparity_all"],
        "selected_lambda": fair_result["lambda"],
        "selected_fair_results": fair_result,
        "min_test_group_size": min_test_group_size,
    }
