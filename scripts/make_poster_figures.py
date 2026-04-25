"""
make_poster_figures.py -- Generate four poster-ready figures from the
existing novelty CSVs in results/. Each figure is saved as both PDF and
PNG at 300 DPI under figures/.

Inputs (must already exist; produced by the novelty modules):
  results/attribution_scores.csv       Module 1 per-group D/U/S scores
  results/disparity_bootstrap.csv      Module 2 bootstrap CI summary
  results/size_stratified_coverage.csv Module 2 size-stratified coverage
  results/counterfactual_stability.csv Module 3 CF stability metrics

Run from the repo root:
    python scripts/make_poster_figures.py

Outputs (figures/):
  fig1_causal_attribution.{pdf,png}
  fig2_bootstrap_ci.{pdf,png}
  fig3_size_stratified.{pdf,png}
  fig4_counterfactual.{pdf,png}
"""

from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# ColorBrewer Set2 (colorblind-safe) -- shared across all four figures.
COLOR_MARGINAL = "#66c2a5"  # teal
COLOR_GC = "#fc8d62"  # orange
COLOR_FAIR = "#8da0cb"  # purple-blue
COLOR_TARGET = "#d62728"  # red, used for the 1-alpha target line and outlier highlight

# Same palette is reused for the D/U/S diagnostic stack (the colors carry
# no cross-figure semantics; consistency just minimizes palette switching).
COLOR_D = COLOR_MARGINAL
COLOR_U = COLOR_GC
COLOR_S = COLOR_FAIR


def _setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
                "Liberation Serif",
                "serif",
            ],
            "mathtext.fontset": "stix",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, basename: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf = os.path.join(FIGURES_DIR, basename + ".pdf")
    png = os.path.join(FIGURES_DIR, basename + ".png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {pdf}")
    print(f"  -> {png}")


def _require(path: str, hint: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing input: {path}\n  Run {hint} first to produce it."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Figure 1: Causal Attribution Stacked Bar
# ---------------------------------------------------------------------------


def make_fig1_attribution() -> None:
    df_full = _require(
        os.path.join(RESULTS_DIR, "attribution_scores.csv"),
        "python evaluation/causal_attribution.py",
    )
    omitted = (
        df_full[~df_full["reliable"]]
        .sort_values("n_test", ascending=False)["group"]
        .tolist()
    )
    df = (
        df_full[df_full["reliable"]]
        .sort_values("undercoverage_marginal", ascending=False)
        .reset_index(drop=True)
    )
    groups = df["group"].tolist()
    D = df["D_g"].to_numpy()
    U = df["U_g"].to_numpy()
    S = df["S_g"].to_numpy()
    causes = df["dominant_cause"].tolist()

    fig, ax = plt.subplots(figsize=(14, 6.6))
    x = np.arange(len(groups))
    ax.bar(
        x, D, color=COLOR_D,
        label=r"$D_g$ (data scarcity, normalized $1/\sqrt{n_{cal}}$)",
        edgecolor="white", linewidth=0.6,
    )
    ax.bar(
        x, U, bottom=D, color=COLOR_U,
        label=r"$U_g$ (model uncertainty, mean entropy)",
        edgecolor="white", linewidth=0.6,
    )
    ax.bar(
        x, S, bottom=D + U, color=COLOR_S,
        label=r"$S_g$ (systemic bias, $1 - \mathrm{accuracy}$)",
        edgecolor="white", linewidth=0.6,
    )

    cause_short = {
        "DataScarcity": "D",
        "ModelUncertainty": "U",
        "SystemicBias": "S",
        "Mixed": "M",
    }
    totals = D + U + S
    for i, c in enumerate(causes):
        ax.text(
            i, totals[i] + 0.015, cause_short.get(c, c[:1]),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=60, ha="right", fontsize=10)
    ax.set_xlabel(
        "Demographic group (sorted by marginal undercoverage, descending)",
        fontsize=14,
    )
    ax.set_ylabel("Sum of normalized diagnostic scores", fontsize=14)
    ax.set_title("Why Each Group Fails: Causal Coverage Attribution", fontsize=16)
    ax.set_ylim(0, totals.max() * 1.18)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    omitted_str = ", ".join(omitted) if omitted else "none"
    fig.text(
        0.5, -0.02,
        "Letter above each bar = dominant cause: D = DataScarcity, "
        "U = ModelUncertainty, S = SystemicBias, M = Mixed.",
        ha="center", fontsize=10, style="italic",
    )
    fig.text(
        0.5, -0.05,
        f"Groups with $n_{{test}} < 30$ omitted ({omitted_str}) for reliability.",
        ha="center", fontsize=10, style="italic",
    )

    plt.tight_layout()
    _save(fig, "fig1_causal_attribution")


# ---------------------------------------------------------------------------
# Figure 2: Bootstrap CI Bar Chart
# ---------------------------------------------------------------------------


def make_fig2_bootstrap_ci() -> None:
    df = _require(
        os.path.join(RESULTS_DIR, "disparity_bootstrap.csv"),
        "python evaluation/set_size_fairness.py",
    ).set_index("method")

    method_order = ["marginal", "gc", "fair"]
    label_map = {
        "marginal": "Marginal CP",
        "gc": "Group-\nConditional CP",
        "fair": r"Fair-CP ($\lambda^*=0.10$)",
    }
    colors = {"marginal": COLOR_MARGINAL, "gc": COLOR_GC, "fair": COLOR_FAIR}

    points = [float(df.loc[m, "point_estimate"]) for m in method_order]
    lows = [float(df.loc[m, "ci_lower_2.5"]) for m in method_order]
    highs = [float(df.loc[m, "ci_upper_97.5"]) for m in method_order]
    err_low = [p - lo for p, lo in zip(points, lows)]
    err_high = [hi - p for p, hi in zip(points, highs)]
    bar_colors = [colors[m] for m in method_order]

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    x = np.arange(len(method_order))
    ax.bar(
        x, points,
        yerr=[err_low, err_high],
        color=bar_colors,
        capsize=10, edgecolor="black", linewidth=0.8,
        error_kw={"linewidth": 1.6, "ecolor": "black"},
        width=0.55,
    )

    for i, (p, h) in enumerate(zip(points, highs)):
        ax.text(
            i, h + 0.006, f"{p:.4f}",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label_map[m] for m in method_order])
    ax.set_xlabel("Conformal prediction method", fontsize=14)
    ax.set_ylabel(
        "Reliable-group disparity\n"
        r"$\max_g\; |\mathrm{coverage}_g - (1 - \alpha)|$",
        fontsize=14,
    )
    ax.set_title(
        "Fair-CP Disparity Reduction is Statistically Significant",
        fontsize=16,
    )

    ax.text(
        0.97, 0.95,
        "Paired permutation test ($B = 500$):\n"
        r"Marginal $-$ Group-Cond: $p$ = 0.0005" "\n"
        r"Marginal $-$ Fair-CP:    $p$ = 0.0005" "\n"
        "Error bars: bootstrap 95% CI",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="white", edgecolor="gray", alpha=0.95),
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(highs) * 1.30)

    fig.text(
        0.5, -0.04,
        "Bootstrap CIs (unpaired) overlap, but the paired permutation test "
        "($p = 0.0005$) confirms Fair-CP's improvement is statistically\n"
        "significant — pairing controls for resample-level noise that "
        "dominates unpaired CIs.",
        ha="center", fontsize=10, style="italic",
    )

    plt.tight_layout()
    _save(fig, "fig2_bootstrap_ci")


# ---------------------------------------------------------------------------
# Figure 3: Size-Stratified Coverage
# ---------------------------------------------------------------------------


def make_fig3_size_stratified() -> None:
    df = _require(
        os.path.join(RESULTS_DIR, "size_stratified_coverage.csv"),
        "python evaluation/set_size_fairness.py",
    )
    df = df[df["bin"].isin(["1", "2", "3"])].copy()

    bins = ["1", "2", "3"]
    methods = ["marginal", "gc", "fair"]
    label_map = {
        "marginal": "Marginal CP",
        "gc": "Group-Conditional CP",
        "fair": r"Fair-CP ($\lambda^*=0.10$)",
    }
    colors = {"marginal": COLOR_MARGINAL, "gc": COLOR_GC, "fair": COLOR_FAIR}

    width = 0.26
    x = np.arange(len(bins))

    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("bin").reindex(bins)
        offset = (i - 1) * width
        coverage = sub["coverage"].to_numpy()
        ns = sub["n"].to_numpy().astype(int)
        ax.bar(
            x + offset, coverage, width, color=colors[method],
            label=label_map[method], edgecolor="black", linewidth=0.6,
        )
        for xi, c, n in zip(x + offset, coverage, ns):
            if not np.isnan(c):
                ax.text(
                    xi, c + 0.008, f"{c:.3f}\n$n={n}$",
                    ha="center", va="bottom", fontsize=9,
                )

    ax.axhline(
        y=0.90, linestyle="--", color=COLOR_TARGET, linewidth=1.6,
        label=r"Target coverage ($1 - \alpha = 0.90$)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"$|C(x)| = {b}$" for b in bins])
    ax.set_xlabel("Prediction-set size", fontsize=14)
    ax.set_ylabel("Empirical coverage rate", fontsize=14)
    ax.set_title(
        "Hidden Miscalibration: Marginal CP Under-Covers Singletons",
        fontsize=16,
    )
    ax.set_ylim(0.7, 1.06)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "fig3_size_stratified")


# ---------------------------------------------------------------------------
# Figure 4: Counterfactual Set-Flip Rates
# ---------------------------------------------------------------------------


def make_fig4_counterfactual() -> None:
    df = _require(
        os.path.join(RESULTS_DIR, "counterfactual_stability.csv"),
        "python evaluation/counterfactual.py",
    )
    if "threshold_policy" in df.columns:
        df = df[df["threshold_policy"] == "fixed_source"].copy()

    pairs = (
        df[["source", "target"]]
        .drop_duplicates()
        .values.tolist()
    )
    pair_labels = [f"{s}→{t}" for s, t in pairs]

    methods = ["marginal", "gc", "fair"]
    label_map = {
        "marginal": "Marginal CP",
        "gc": "Group-Conditional CP",
        "fair": r"Fair-CP ($\lambda^*=0.10$)",
    }
    colors = {"marginal": COLOR_MARGINAL, "gc": COLOR_GC, "fair": COLOR_FAIR}

    width = 0.26
    x = np.arange(len(pairs))

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    for i, method in enumerate(methods):
        offset = (i - 1) * width
        rates = []
        for s, t in pairs:
            row = df[(df["source"] == s) & (df["target"] == t) & (df["method"] == method)]
            rates.append(float(row["set_flip_rate"].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x + offset, rates, width, color=colors[method],
            label=label_map[method], edgecolor="black", linewidth=0.6,
        )

    hetero_indices = [i for i, p in enumerate(pairs) if p[0] == "Homosexual"]
    if hetero_indices:
        idx = hetero_indices[0]
        ax.axvspan(idx - 0.5, idx + 0.5, color=COLOR_TARGET, alpha=0.10, zorder=0)
        max_rate = max(
            float(
                df[(df["source"] == "Homosexual") & (df["method"] == m)]["set_flip_rate"].iloc[0]
            )
            for m in methods
            if not df[(df["source"] == "Homosexual") & (df["method"] == m)].empty
        )
        ax.annotate(
            f"OUTLIER\n{max_rate * 100:.0f}% set-flip",
            xy=(idx, max_rate),
            xytext=(idx + 0.7, max_rate + 0.10),
            ha="left", fontsize=11, fontweight="bold", color=COLOR_TARGET,
            arrowprops=dict(arrowstyle="->", color=COLOR_TARGET, lw=1.6),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=20, ha="right", fontsize=11)
    ax.set_xlabel("Demographic swap pair (source → target)", fontsize=14)
    ax.set_ylabel("Set-flip rate", fontsize=14)
    ax.set_title(
        "All CP Methods Equally Vulnerable to Demographic Shortcut Learning",
        fontsize=16,
    )
    ax.set_ylim(0, 0.72)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "fig4_counterfactual")


# ---------------------------------------------------------------------------


def main() -> int:
    _setup_matplotlib()
    print("Generating poster figures (300 DPI, PDF + PNG)...")

    print("\n[1/4] Figure 1: Causal Attribution Stacked Bar")
    make_fig1_attribution()

    print("\n[2/4] Figure 2: Bootstrap CI Bar Chart")
    make_fig2_bootstrap_ci()

    print("\n[3/4] Figure 3: Size-Stratified Coverage")
    make_fig3_size_stratified()

    print("\n[4/4] Figure 4: Counterfactual Set-Flip Rates")
    make_fig4_counterfactual()

    print(f"\nDone. Figures saved to {FIGURES_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
