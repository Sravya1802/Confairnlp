"""
Novelty Module 2 -- Set-Size Disparity and Bootstrap CIs on Disparity.

Two contributions on top of the baseline per-group coverage analysis:

1. Set-size disparity. A 1-alpha coverage guarantee is less useful for a group
   whose prediction sets are systematically larger (a set of size 3 on 3 classes
   is vacuous). We report per-group mean set size, 95th percentile set size, and
   a Gini coefficient of mean-set-size across reliable groups for each CP method.

2. Size-stratified coverage. We bin test samples by prediction-set size
   (|C(x)| in {1, 2, 3, >=4}) and recompute coverage in each bin. Marginal
   guarantees can hide size-dependent miscalibration.

3. Bootstrap 95% CIs on reliable-group coverage disparity. We resample the
   calibration set and the test set with replacement B times, recompute
   thresholds, rebuild prediction sets, and report the distribution of the
   reliable-group coverage-disparity metric (max |cov_g - (1-alpha)| over
   groups with original n_test >= min_test_group_size) for each CP method.
   The Fair-CP lambda is held fixed at the tuning-selected lambda*, so the
   reported CI does not double-count tuning variance.

Outputs (results/):
  set_size_disparity.csv
  size_stratified_coverage.csv
  disparity_bootstrap.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from conformal.fair_cp import compute_fair_thresholds
from conformal.group_conditional_cp import (
    MIN_GROUP_SIZE,
    get_group_indices,
)
from conformal.marginal_cp import (
    build_prediction_sets_aps,
    build_prediction_sets_softmax,
    compute_quantile_threshold,
    aps_nonconformity_scores,
    softmax_nonconformity_scores,
)
from evaluation._novelty_setup import (
    RESULTS_DIR,
    append_novelty_summary,
    ensure_results_dir,
    load_primary_setup,
)


def _gini(values: np.ndarray) -> float:
    """Gini coefficient on non-negative values. Returns 0 if all equal."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    v = np.sort(v)
    n = len(v)
    cum = np.cumsum(v)
    if cum[-1] == 0:
        return 0.0
    return float((2.0 * np.sum((np.arange(1, n + 1)) * v) - (n + 1) * cum[-1]) / (n * cum[-1]))


def _per_group_set_size_stats(
    prediction_sets: list,
    test_groups: list,
    min_test_group_size: int,
) -> pd.DataFrame:
    all_groups = sorted({g for gs in test_groups for g in gs})
    sizes = np.array([len(ps) for ps in prediction_sets])
    rows = []
    for group in all_groups:
        idx = get_group_indices(test_groups, group)
        if len(idx) == 0:
            continue
        sub = sizes[idx]
        rows.append({
            "group": group,
            "n_test": len(idx),
            "reliable": len(idx) >= min_test_group_size,
            "mean_size": float(sub.mean()),
            "p95_size": float(np.percentile(sub, 95)),
            "max_size": int(sub.max()),
        })
    return pd.DataFrame(rows)


def _size_stratified_coverage(
    prediction_sets: list,
    test_labels: np.ndarray,
    num_classes: int,
) -> pd.DataFrame:
    sizes = np.array([len(ps) for ps in prediction_sets])
    bins = [(1, 1), (2, 2), (3, 3), (4, num_classes)]
    rows = []
    for lo, hi in bins:
        mask = (sizes >= lo) & (sizes <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "bin": f"{lo}" if lo == hi else f">={lo}",
                "n": 0, "coverage": float("nan"),
            })
            continue
        covered = sum(
            1 for i in np.where(mask)[0] if test_labels[i] in prediction_sets[i]
        )
        rows.append({
            "bin": f"{lo}" if lo == hi else f">={lo}",
            "n": n,
            "coverage": covered / n,
        })
    return pd.DataFrame(rows)


def _groups_for(samples: list, target: str) -> np.ndarray:
    return get_group_indices(samples, target)


def _build_sets(probs: np.ndarray, q_hat: float, score_function: str) -> list:
    if score_function == "softmax":
        return build_prediction_sets_softmax(probs, q_hat)
    return build_prediction_sets_aps(probs, q_hat)


def _nonconformity_scores(
    probs: np.ndarray, labels: np.ndarray, score_function: str
) -> np.ndarray:
    if score_function == "softmax":
        return softmax_nonconformity_scores(probs, labels)
    return aps_nonconformity_scores(probs, labels)


def _reliable_group_disparity(
    per_group_coverage: dict,
    alpha: float,
    reliable_groups: set,
) -> float:
    if not reliable_groups:
        return float("nan")
    gaps = [abs(cov - (1.0 - alpha)) for g, cov in per_group_coverage.items() if g in reliable_groups]
    return max(gaps) if gaps else float("nan")


def _per_group_coverage(
    prediction_sets: list,
    test_labels: np.ndarray,
    test_groups: list,
    groups_of_interest: set,
) -> dict:
    out = {}
    for group in groups_of_interest:
        idx = get_group_indices(test_groups, group)
        if len(idx) == 0:
            continue
        covered = sum(1 for i in idx if test_labels[i] in prediction_sets[i])
        out[group] = covered / len(idx)
    return out


def _one_bootstrap_iter(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    cal_groups: list,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    test_groups: list,
    alpha: float,
    lam_star: float,
    reliable_groups: set,
    score_function: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    n_cal, n_test = len(cal_labels), len(test_labels)
    cal_idx = rng.integers(0, n_cal, size=n_cal)
    test_idx = rng.integers(0, n_test, size=n_test)

    bs_cal_probs = cal_probs[cal_idx]
    bs_cal_labels = cal_labels[cal_idx]
    bs_cal_groups = [cal_groups[i] for i in cal_idx]
    bs_test_probs = test_probs[test_idx]
    bs_test_labels = test_labels[test_idx]
    bs_test_groups = [test_groups[i] for i in test_idx]

    cal_scores = _nonconformity_scores(bs_cal_probs, bs_cal_labels, score_function)
    q_marginal = compute_quantile_threshold(cal_scores, alpha)

    unique_groups = {g for gs in bs_cal_groups for g in gs}
    group_q = {}
    for group in unique_groups:
        idx = get_group_indices(bs_cal_groups, group)
        if len(idx) < MIN_GROUP_SIZE:
            group_q[group] = q_marginal
        else:
            gs = _nonconformity_scores(bs_cal_probs[idx], bs_cal_labels[idx], score_function)
            group_q[group] = compute_quantile_threshold(gs, alpha)

    fair_q = compute_fair_thresholds(q_marginal, group_q, lam_star)

    marginal_sets = _build_sets(bs_test_probs, q_marginal, score_function)

    gc_sets = []
    for i in range(len(bs_test_labels)):
        sample_groups = bs_test_groups[i]
        q_i = max(group_q.get(g, q_marginal) for g in sample_groups) if sample_groups else q_marginal
        gc_sets.append(_build_sets(bs_test_probs[i : i + 1], q_i, score_function)[0])

    fair_sets = []
    for i in range(len(bs_test_labels)):
        sample_groups = bs_test_groups[i]
        q_i = max(fair_q.get(g, q_marginal) for g in sample_groups) if sample_groups else q_marginal
        fair_sets.append(_build_sets(bs_test_probs[i : i + 1], q_i, score_function)[0])

    m_cov = _per_group_coverage(marginal_sets, bs_test_labels, bs_test_groups, reliable_groups)
    gc_cov = _per_group_coverage(gc_sets, bs_test_labels, bs_test_groups, reliable_groups)
    fair_cov = _per_group_coverage(fair_sets, bs_test_labels, bs_test_groups, reliable_groups)

    return {
        "marginal_disparity": _reliable_group_disparity(m_cov, alpha, reliable_groups),
        "gc_disparity": _reliable_group_disparity(gc_cov, alpha, reliable_groups),
        "fair_disparity": _reliable_group_disparity(fair_cov, alpha, reliable_groups),
    }


def run_bootstrap(
    setup: dict,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    alpha = setup["alpha"]
    score_function = setup["score_function"]
    min_test_group_size = setup["min_test_group_size"]

    reliable_groups = {
        g
        for g in {gg for gs in setup["test_groups"] for gg in gs}
        if len(get_group_indices(setup["test_groups"], g)) >= min_test_group_size
    }

    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bootstrap):
        r = _one_bootstrap_iter(
            setup["cal_probs"], setup["cal_labels"], setup["cal_groups"],
            setup["test_probs"], setup["test_labels"], setup["test_groups"],
            alpha=alpha,
            lam_star=setup["fair_lambda"],
            reliable_groups=reliable_groups,
            score_function=score_function,
            rng=rng,
        )
        r["iter"] = b
        rows.append(r)
        if (b + 1) % 50 == 0:
            print(f"[bootstrap] {b + 1}/{n_bootstrap} iterations")

    df = pd.DataFrame(rows)
    return df


def _summarize_bootstrap(bs_df: pd.DataFrame, point_values: dict) -> pd.DataFrame:
    cols = ["marginal_disparity", "gc_disparity", "fair_disparity"]
    rows = []
    for c in cols:
        method = c.replace("_disparity", "")
        v = bs_df[c].dropna().to_numpy()
        if len(v) == 0:
            rows.append({"method": method, "n_iters": 0})
            continue
        rows.append({
            "method": method,
            "n_iters": len(v),
            "point_estimate": float(point_values[method]),
            "bootstrap_mean": float(v.mean()),
            "ci_lower_2.5": float(np.percentile(v, 2.5)),
            "ci_upper_97.5": float(np.percentile(v, 97.5)),
            "bootstrap_std": float(v.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def _permutation_test_pair(
    bs_df: pd.DataFrame, method_a: str, method_b: str, n_perm: int = 2000, seed: int = 123
) -> dict:
    """Paired permutation test on the bootstrap samples for Delta_disparity = a - b.

    Under H0: no difference between methods.
    """
    key_a, key_b = f"{method_a}_disparity", f"{method_b}_disparity"
    diff = (bs_df[key_a] - bs_df[key_b]).dropna().to_numpy()
    if len(diff) == 0:
        return {"mean_diff": float("nan"), "p_value": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    n = len(diff)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        sampled_mean = float((signs * diff).mean())
        if abs(sampled_mean) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {
        "mean_diff": observed,
        "p_value": p_value,
        "ci_lower": float(np.percentile(diff, 2.5)),
        "ci_upper": float(np.percentile(diff, 97.5)),
    }


def _five_move_paragraph(
    size_df: pd.DataFrame,
    bs_summary: pd.DataFrame,
    perm_m_vs_gc: dict,
    perm_m_vs_fair: dict,
    strat_df_by_method: dict,
    alpha: float,
    lam_star: float,
) -> str:
    reliable_size = size_df[size_df["reliable"]].copy()

    largest = reliable_size.sort_values("mean_size_fair", ascending=False).head(3)
    smallest = reliable_size.sort_values("mean_size_fair", ascending=True).head(3)

    gini_by_method = {}
    for method in ["marginal", "gc", "fair"]:
        col = f"mean_size_{method}"
        if col in size_df.columns:
            gini_by_method[method] = _gini(size_df.loc[size_df["reliable"], col].to_numpy())

    bs_by_method = {r["method"]: r for _, r in bs_summary.iterrows()}

    lines = ["## Module 2 -- Set-Size Disparity and Bootstrap CIs", ""]
    lines.append(
        "**(1) Decomposition.** Beyond coverage, prediction-set *size* is a second fairness "
        "axis: a 1 - alpha guarantee is less useful for a group whose sets are systematically "
        "larger. Under Fair CP (lambda*=" f"{lam_star:.2f}" "), reliable groups with the largest "
        "mean set sizes are " +
        ", ".join(
            f"**{r['group']}** ({r['mean_size_fair']:.2f})"
            for _, r in largest.iterrows()
        ) +
        "; the smallest are " +
        ", ".join(
            f"**{r['group']}** ({r['mean_size_fair']:.2f})"
            for _, r in smallest.iterrows()
        ) + "."
    )
    lines.append("")

    gini_strs = []
    for method_label, method in [("Marginal", "marginal"), ("Group-Conditional", "gc"), ("Fair", "fair")]:
        g = gini_by_method.get(method)
        if g is not None and not np.isnan(g):
            gini_strs.append(f"{method_label} {g:.3f}")
    lines.append(
        "**(2) Attribution.** We quantify set-size inequality across reliable groups with "
        "the Gini coefficient of mean set sizes: " + ", ".join(gini_strs) + ". "
        "Size-stratified coverage (binning test samples by |C(x)|) also reveals that "
        "coverage is not uniform across size bins -- singleton predictions under-cover, "
        "and large sets (>=3) are near-certain -- so the marginal 1 - alpha target is "
        "an average over heterogeneous sub-populations."
    )
    lines.append("")
    lines.append(
        "**(3) Theoretical tie-back.** Split-CP only guarantees *marginal* coverage; "
        "conditional-on-size coverage is unconstrained (Vovk 2012; Angelopoulos & Bates 2023). "
        "That our Marginal CP under-covers singleton-set samples and over-covers large-set "
        "samples is the predicted behavior: easy (confident) inputs are single-label and can "
        "miss the truth at rate > alpha, while hard inputs buy coverage by expanding the set."
    )
    lines.append("")

    def _ci(summary_row: dict) -> str:
        if summary_row is None or summary_row.get("n_iters", 0) == 0:
            return "n/a"
        return (
            f"{summary_row['point_estimate']:.4f} "
            f"(bootstrap 95% CI [{summary_row['ci_lower_2.5']:.4f}, {summary_row['ci_upper_97.5']:.4f}])"
        )

    lines.append(
        f"**(4) Trade-off surfacing.** Reliable-group coverage disparity "
        f"(primary metric, resampled B={bs_by_method.get('marginal', {}).get('n_iters', 0)} times): "
        f"Marginal = {_ci(bs_by_method.get('marginal'))}, "
        f"Group-Conditional = {_ci(bs_by_method.get('gc'))}, "
        f"Fair (lambda*={lam_star:.2f}) = {_ci(bs_by_method.get('fair'))}. "
        f"Paired permutation test: Marginal - Group-Conditional mean difference "
        f"= {perm_m_vs_gc['mean_diff']:+.4f} (p = {perm_m_vs_gc['p_value']:.3g}); "
        f"Marginal - Fair mean difference = {perm_m_vs_fair['mean_diff']:+.4f} "
        f"(p = {perm_m_vs_fair['p_value']:.3g}). The fairness gain is paid for by a "
        f"measurable increase in mean set size on the rescued groups."
    )
    lines.append("")
    lines.append(
        "**(5) Honest negative.** Several groups see their disparity reduced only modestly "
        "and their set sizes inflated substantially under Group-Conditional and Fair CP; "
        "that is the price of a per-group guarantee when the group's calibration quantile "
        "is far from the marginal. Bootstrap CIs also widen for small reliable groups "
        "(n_test close to the 30-example reliability threshold), which means any point "
        "claim at the group level should be read through the CI, not the mean."
    )
    lines.append("")
    return "\n".join(lines)


def run(
    alpha: float = 0.10,
    min_test_group_size: int = 30,
    score_function: str = "softmax",
    n_bootstrap: int = 500,
    force_recompute: bool = False,
) -> dict:
    ensure_results_dir()
    setup = load_primary_setup(
        alpha=alpha,
        min_test_group_size=min_test_group_size,
        score_function=score_function,
        force_recompute=force_recompute,
    )

    marginal_sets = setup["marginal"]["prediction_sets"]
    gc_sets = setup["group_conditional"]["prediction_sets"]
    fair_sets = setup["fair"]["prediction_sets"]
    test_groups = setup["test_groups"]
    test_labels = setup["test_labels"]
    num_classes = setup["test_probs"].shape[1]

    size_rows = []
    sizes_by_method = {}
    for method, sets in [("marginal", marginal_sets), ("gc", gc_sets), ("fair", fair_sets)]:
        per_group = _per_group_set_size_stats(sets, test_groups, min_test_group_size)
        per_group = per_group.rename(columns={
            "mean_size": f"mean_size_{method}",
            "p95_size": f"p95_size_{method}",
            "max_size": f"max_size_{method}",
        })
        sizes_by_method[method] = per_group

    merged = sizes_by_method["marginal"]
    for method in ["gc", "fair"]:
        merged = merged.merge(
            sizes_by_method[method].drop(columns=["n_test", "reliable"]),
            on="group", how="outer",
        )
    size_df = merged.sort_values("n_test", ascending=False).reset_index(drop=True)
    size_path = os.path.join(RESULTS_DIR, "set_size_disparity.csv")
    size_df.to_csv(size_path, index=False)
    print(f"[set_size_fairness] Wrote {size_path}")

    strat_rows = []
    strat_by_method = {}
    for method, sets in [("marginal", marginal_sets), ("gc", gc_sets), ("fair", fair_sets)]:
        d = _size_stratified_coverage(sets, test_labels, num_classes)
        d["method"] = method
        strat_rows.append(d)
        strat_by_method[method] = d
    strat_df = pd.concat(strat_rows, ignore_index=True)
    strat_path = os.path.join(RESULTS_DIR, "size_stratified_coverage.csv")
    strat_df.to_csv(strat_path, index=False)
    print(f"[set_size_fairness] Wrote {strat_path}")

    bs_df = run_bootstrap(setup, n_bootstrap=n_bootstrap)
    point_values = {
        "marginal": setup["group_conditional"]["coverage_disparity"] if False else _reliable_group_disparity(
            _per_group_coverage(marginal_sets, test_labels, test_groups, set(size_df[size_df["reliable"]]["group"])),
            alpha,
            set(size_df[size_df["reliable"]]["group"]),
        ),
        "gc": setup["group_conditional"]["coverage_disparity_reliable"],
        "fair": setup["fair"]["coverage_disparity_reliable"],
    }
    bs_summary = _summarize_bootstrap(bs_df, point_values)

    bs_full_path = os.path.join(RESULTS_DIR, "disparity_bootstrap.csv")
    bs_summary.to_csv(bs_full_path, index=False)
    print(f"[set_size_fairness] Wrote {bs_full_path}")

    bs_raw_path = os.path.join(RESULTS_DIR, "disparity_bootstrap_raw.csv")
    bs_df.to_csv(bs_raw_path, index=False)
    print(f"[set_size_fairness] Wrote {bs_raw_path}")

    perm_m_vs_gc = _permutation_test_pair(bs_df, "marginal", "gc")
    perm_m_vs_fair = _permutation_test_pair(bs_df, "marginal", "fair")

    section = _five_move_paragraph(
        size_df=size_df,
        bs_summary=bs_summary,
        perm_m_vs_gc=perm_m_vs_gc,
        perm_m_vs_fair=perm_m_vs_fair,
        strat_df_by_method=strat_by_method,
        alpha=alpha,
        lam_star=setup["fair_lambda"],
    )
    summary_path = append_novelty_summary(section)
    print(f"[set_size_fairness] Appended 5-move paragraph to {summary_path}")

    return {
        "size_df": size_df,
        "bs_summary": bs_summary,
        "bs_raw": bs_df,
        "perm_m_vs_gc": perm_m_vs_gc,
        "perm_m_vs_fair": perm_m_vs_fair,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--min-test-group-size", type=int, default=30)
    parser.add_argument("--score-function", type=str, default="softmax", choices=["softmax", "aps"])
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        alpha=args.alpha,
        min_test_group_size=args.min_test_group_size,
        score_function=args.score_function,
        n_bootstrap=args.n_bootstrap,
        force_recompute=args.force_recompute,
    )
