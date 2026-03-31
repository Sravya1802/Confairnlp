"""
coverage_analysis.py — Per-group coverage analysis and visualization.

Generates:
  1. Per-group coverage table (CSV + printed)
  2. Coverage bar chart (PDF) comparing methods
  3. Lambda tradeoff curve (PDF) — Pareto frontier
  4. Multi-alpha analysis
"""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 42
np.random.seed(SEED)


def build_coverage_table(
    marginal_results: dict,
    group_cond_results: dict,
    fair_cp_results: dict,
    alpha: float,
) -> pd.DataFrame:
    """Build a per-group coverage comparison table.

    Args:
        marginal_results: Results from marginal CP (includes prediction_sets)
        group_cond_results: Results from group-conditional CP
        fair_cp_results: Results from fair CP at the best lambda
        alpha: Significance level

    Returns:
        DataFrame with columns: Group, Marginal_Coverage, Marginal_SetSize,
        GroupCond_Coverage, GroupCond_SetSize, FairCP_Coverage, FairCP_SetSize
    """
    groups = sorted(group_cond_results["per_group"].keys())

    rows = []
    for group in groups:
        gc = group_cond_results["per_group"][group]
        fc = fair_cp_results["per_group"].get(group, {})

        row = {
            "Group": group,
            "GroupCond_Coverage": gc["coverage"],
            "GroupCond_SetSize": gc["avg_set_size"],
            "GroupCond_N": gc["n_test"],
            "FairCP_Coverage": fc.get("coverage", np.nan),
            "FairCP_SetSize": fc.get("avg_set_size", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add marginal coverage (same for all groups in standard marginal CP)
    df["Marginal_Coverage"] = marginal_results["coverage"]
    df["Marginal_SetSize"] = marginal_results["avg_set_size"]

    # Reorder columns
    df = df[
        [
            "Group",
            "Marginal_Coverage",
            "Marginal_SetSize",
            "GroupCond_Coverage",
            "GroupCond_SetSize",
            "FairCP_Coverage",
            "FairCP_SetSize",
            "GroupCond_N",
        ]
    ]

    return df


def compute_per_group_marginal_coverage(
    prediction_sets: list,
    test_labels: np.ndarray,
    test_groups: list,
) -> dict:
    """Compute marginal CP coverage broken down by group.

    Args:
        prediction_sets: Prediction sets from marginal CP
        test_labels: True labels
        test_groups: Group labels for test samples

    Returns:
        Dict mapping group -> {coverage, avg_set_size, n_test}
    """
    from conformal.group_conditional_cp import get_group_indices

    all_groups = set()
    for g in test_groups:
        if isinstance(g, list):
            all_groups.update(g)
        else:
            all_groups.add(g)

    per_group = {}
    for group in sorted(all_groups):
        idx = get_group_indices(test_groups, group)
        if len(idx) == 0:
            continue
        covered = sum(1 for i in idx if test_labels[i] in prediction_sets[i])
        sizes = [len(prediction_sets[i]) for i in idx]
        per_group[group] = {
            "coverage": covered / len(idx),
            "avg_set_size": np.mean(sizes),
            "n_test": len(idx),
        }

    return per_group


def plot_coverage_bar_chart(
    marginal_per_group: dict,
    group_cond_results: dict,
    fair_cp_results: dict,
    alpha: float,
    save_path: str,
):
    """Plot grouped bar chart of per-group coverage for all 3 methods.

    Args:
        marginal_per_group: Per-group coverage from marginal CP
        group_cond_results: Group-conditional CP results
        fair_cp_results: Fair CP results (best lambda)
        alpha: Significance level
        save_path: Path to save the PDF plot
    """
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

    # Target coverage line
    ax.axhline(y=1 - alpha, color="red", linestyle="--", linewidth=1.5,
               label=f"Target (1-α = {1-alpha:.2f})")

    ax.set_xlabel("Demographic Group", fontsize=12)
    ax.set_ylabel("Coverage Rate", fontsize=12)
    ax.set_title(f"Per-Group Coverage Comparison (α = {alpha})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Coverage bar chart saved to {save_path}")


def plot_lambda_tradeoff(
    sweep_results: list,
    alpha: float,
    save_path: str,
):
    """Plot the Pareto frontier: coverage disparity vs. avg set size.

    Args:
        sweep_results: List of results from fair_cp_sweep
        alpha: Significance level
        save_path: Path to save the PDF plot
    """
    lambdas = [r["lambda"] for r in sweep_results]
    disparities = [r["coverage_disparity"] for r in sweep_results]
    set_sizes = [r["overall_avg_set_size"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(set_sizes, disparities, c=lambdas, cmap="viridis",
                         s=100, edgecolors="black", linewidth=0.5, zorder=5)
    ax.plot(set_sizes, disparities, "--", color="gray", alpha=0.5, zorder=1)

    # Annotate lambda values
    for i, lam in enumerate(lambdas):
        ax.annotate(f"λ={lam:.1f}", (set_sizes[i], disparities[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    plt.colorbar(scatter, ax=ax, label="λ")
    ax.set_xlabel("Average Prediction Set Size", fontsize=12)
    ax.set_ylabel("Coverage Disparity (max |cov_g - (1-α)|)", fontsize=12)
    ax.set_title(f"Fairness-Efficiency Tradeoff (α = {alpha})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Lambda tradeoff curve saved to {save_path}")


def plot_multi_alpha_disparity(
    alpha_results: dict,
    save_path: str,
):
    """Plot coverage disparity across multiple alpha values.

    Args:
        alpha_results: Dict mapping alpha -> {marginal_disp, gc_disp, fair_disp}
        save_path: Path to save the PDF plot
    """
    alphas = sorted(alpha_results.keys())
    methods = ["Marginal", "Group-Conditional", "Fair CP"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        key = method.lower().replace("-", "_").replace(" ", "_")
        disparities = [alpha_results[a].get(f"{key}_disparity", 0) for a in alphas]
        ax.plot(alphas, disparities, "o-", label=method, markersize=8)

    ax.set_xlabel("α (Significance Level)", fontsize=12)
    ax.set_ylabel("Coverage Disparity", fontsize=12)
    ax.set_title("Coverage Disparity Across α Values", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
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
) -> dict:
    """Run full coverage analysis and generate all outputs.

    Args:
        marginal_results: Results from marginal CP
        group_cond_results: Results from group-conditional CP
        fair_sweep_results: Results from fair CP lambda sweep
        test_labels: Test set true labels
        test_groups: Test set group labels
        alpha: Significance level
        output_dir: Directory to save results

    Returns:
        Dict with all analysis results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find best lambda result (lowest disparity)
    best_fair = min(fair_sweep_results, key=lambda r: r["coverage_disparity"])
    print(f"\nBest Fair CP: lambda={best_fair['lambda']:.2f}, "
          f"disparity={best_fair['coverage_disparity']:.4f}")

    # Compute per-group marginal coverage
    marginal_per_group = compute_per_group_marginal_coverage(
        marginal_results["prediction_sets"], test_labels, test_groups
    )

    # Build coverage table
    coverage_df = build_coverage_table(
        marginal_results, group_cond_results, best_fair, alpha
    )

    # Add per-group marginal coverage to the table
    for i, row in coverage_df.iterrows():
        group = row["Group"]
        if group in marginal_per_group:
            coverage_df.at[i, "Marginal_Coverage"] = marginal_per_group[group]["coverage"]
            coverage_df.at[i, "Marginal_SetSize"] = marginal_per_group[group]["avg_set_size"]

    # Save coverage table
    csv_path = os.path.join(output_dir, "per_group_coverage.csv")
    coverage_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n[Coverage Table] Saved to {csv_path}")
    print(coverage_df.to_string(index=False))

    # Plot coverage bar chart
    plot_coverage_bar_chart(
        marginal_per_group, group_cond_results, best_fair, alpha,
        os.path.join(output_dir, "coverage_bar_chart.pdf"),
    )

    # Plot lambda tradeoff
    plot_lambda_tradeoff(
        fair_sweep_results, alpha,
        os.path.join(output_dir, "lambda_tradeoff.pdf"),
    )

    # Compute disparities for summary
    marginal_disparity = max(
        abs(v["coverage"] - (1 - alpha))
        for v in marginal_per_group.values()
    ) if marginal_per_group else 0.0

    analysis_results = {
        "coverage_table": coverage_df,
        "marginal_per_group": marginal_per_group,
        "marginal_disparity": marginal_disparity,
        "gc_disparity": group_cond_results["coverage_disparity"],
        "fair_disparity": best_fair["coverage_disparity"],
        "best_lambda": best_fair["lambda"],
        "best_fair_results": best_fair,
    }

    return analysis_results
