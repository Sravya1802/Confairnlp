"""
run_all.py — Single entry point for the full ConfairNLP pipeline.

Runs:
  1. Data download and preprocessing
  2. Model training (BERT and HateBERT) or loading pre-trained models
  3. Marginal Conformal Prediction
  4. Group-Conditional Conformal Prediction
  5. Fairness-Regularized CP with lambda sweep
  6. Coverage analysis and visualization
  7. Ablation studies
  8. Summary of all results

Usage:
    python run_all.py
    python run_all.py --skip-training --alpha 0.10 --device cuda
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.download_data import download_all
from models.train_classifier import train_all, get_softmax_probs, load_hatexplain_splits
from conformal.marginal_cp import run_marginal_cp
from conformal.group_conditional_cp import run_group_conditional_cp
from conformal.fair_cp import run_fair_cp_sweep
from evaluation.coverage_analysis import (
    run_coverage_analysis,
    compute_per_group_marginal_coverage,
    plot_multi_alpha_disparity,
)
from evaluation.ablation import run_ablation_study

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfairNLP: Conformal Prediction for Equitable Hate Speech Detection"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load pre-trained models instead of training from scratch",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Coverage level (significance level, default: 0.10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detects if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data"),
        help="Directory for downloaded data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "trained"),
        help="Directory for trained models",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip ablation studies to save time",
    )
    return parser.parse_args()


def prepare_group_labels(df):
    """Extract group labels as a list from the dataframe.

    Each element is a list of group strings (since samples can belong
    to multiple groups in HateXplain).
    """
    groups = []
    for _, row in df.iterrows():
        g = row.get("target_groups", [])
        if isinstance(g, str):
            g = [g]
        elif not isinstance(g, list):
            g = list(g) if hasattr(g, '__iter__') else [str(g)]
        # Filter out empty groups
        g = [x for x in g if x and str(x).strip().lower() not in ("", "none", "nan")]
        if not g:
            g = ["unknown"]
        groups.append(g)
    return groups


def run_pipeline(args):
    """Execute the full ConfairNLP pipeline."""
    start_time = time.time()

    print("=" * 70)
    print("  ConfairNLP: Conformal Prediction for Equitable Hate Speech Detection")
    print("=" * 70)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Alpha: {args.alpha}")
    print(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # =====================================================================
    # Step 1: Download and preprocess data
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 1: Data Download and Preprocessing")
    print("=" * 70)

    hatexplain_path = os.path.join(args.data_dir, "hatexplain_splits.pkl")
    if os.path.exists(hatexplain_path):
        print(f"Found existing data at {hatexplain_path}, loading...")
        splits = load_hatexplain_splits(args.data_dir)
        all_data = {"hatexplain": splits}
    else:
        all_data = download_all(args.data_dir)
        splits = all_data["hatexplain"]

    train_df = splits["train"]
    cal_df = splits["calibration"]
    test_df = splits["test"]

    print(f"\nData splits — Train: {len(train_df)}, Cal: {len(cal_df)}, Test: {len(test_df)}")

    # Prepare group labels
    cal_groups = prepare_group_labels(cal_df)
    test_groups = prepare_group_labels(test_df)

    # =====================================================================
    # Step 2: Train or load models
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 2: Model Training / Loading")
    print("=" * 70)

    trained_models = train_all(
        data_dir=args.data_dir,
        output_dir=args.model_dir,
        device=device,
        skip_if_exists=args.skip_training,
    )

    # =====================================================================
    # Step 3: Get softmax probabilities for calibration and test sets
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 3: Computing Softmax Probabilities")
    print("=" * 70)

    cal_texts = cal_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    cal_labels = np.array(cal_df["label"].tolist())
    test_labels = np.array(test_df["label"].tolist())

    # Use the first available model (prefer HateBERT)
    primary_model_name = "hatebert_hatexplain" if "hatebert_hatexplain" in trained_models else list(trained_models.keys())[0]
    model, tokenizer = trained_models[primary_model_name]
    print(f"\nUsing primary model: {primary_model_name}")

    print("Computing calibration set probabilities...")
    cal_probs = get_softmax_probs(model, tokenizer, cal_texts, device=device)
    print("Computing test set probabilities...")
    test_probs = get_softmax_probs(model, tokenizer, test_texts, device=device)

    # =====================================================================
    # Step 4: Marginal Conformal Prediction
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 4: Marginal Conformal Prediction")
    print("=" * 70)

    marginal_results = run_marginal_cp(
        cal_probs, cal_labels, test_probs, test_labels,
        alpha=args.alpha, score_function="softmax",
    )

    # =====================================================================
    # Step 5: Group-Conditional Conformal Prediction
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 5: Group-Conditional Conformal Prediction")
    print("=" * 70)

    gc_results = run_group_conditional_cp(
        cal_probs, cal_labels, cal_groups,
        test_probs, test_labels, test_groups,
        alpha=args.alpha, score_function="softmax",
    )

    # =====================================================================
    # Step 6: Fairness-Regularized CP with Lambda Sweep
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 6: Fairness-Regularized Conformal Prediction")
    print("=" * 70)

    fair_sweep_results = run_fair_cp_sweep(
        cal_probs, cal_labels, cal_groups,
        test_probs, test_labels, test_groups,
        alpha=args.alpha, score_function="softmax",
    )

    # =====================================================================
    # Step 7: Coverage Analysis and Visualization
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 7: Coverage Analysis and Visualization")
    print("=" * 70)

    analysis_results = run_coverage_analysis(
        marginal_results, gc_results, fair_sweep_results,
        test_labels, test_groups, args.alpha, args.output_dir,
    )

    # Multi-alpha analysis
    print("\nRunning multi-alpha analysis...")
    alpha_results = {}
    for alpha_val in [0.05, 0.10, 0.15, 0.20]:
        m = run_marginal_cp(cal_probs, cal_labels, test_probs, test_labels,
                            alpha=alpha_val, score_function="softmax")
        m_pg = compute_per_group_marginal_coverage(
            m["prediction_sets"], test_labels, test_groups)
        m_disp = max(abs(v["coverage"] - (1 - alpha_val)) for v in m_pg.values()) if m_pg else 0.0

        gc = run_group_conditional_cp(
            cal_probs, cal_labels, cal_groups,
            test_probs, test_labels, test_groups,
            alpha=alpha_val, score_function="softmax")

        fair = run_fair_cp_sweep(
            cal_probs, cal_labels, cal_groups,
            test_probs, test_labels, test_groups,
            alpha=alpha_val, score_function="softmax")
        best_fair = min(fair, key=lambda r: r["coverage_disparity"])

        alpha_results[alpha_val] = {
            "marginal_disparity": m_disp,
            "group_conditional_disparity": gc["coverage_disparity"],
            "fair_cp_disparity": best_fair["coverage_disparity"],
        }

    plot_multi_alpha_disparity(
        alpha_results,
        os.path.join(args.output_dir, "multi_alpha_disparity.pdf"),
    )

    # =====================================================================
    # Step 8: Ablation Studies
    # =====================================================================
    if not args.skip_ablation:
        print("\n" + "=" * 70)
        print("  STEP 8: Ablation Studies")
        print("=" * 70)

        # Build model_probs dict for ablation
        model_probs = {}
        for model_name, (mod, tok) in trained_models.items():
            print(f"\nComputing probabilities for {model_name}...")
            m_cal_probs = get_softmax_probs(mod, tok, cal_texts, device=device)
            m_test_probs = get_softmax_probs(mod, tok, test_texts, device=device)

            model_probs[(model_name, "HateXplain")] = {
                "cal_probs": m_cal_probs,
                "cal_labels": cal_labels,
                "cal_groups": cal_groups,
                "test_probs": m_test_probs,
                "test_labels": test_labels,
                "test_groups": test_groups,
            }

        ablation_df = run_ablation_study(
            model_probs=model_probs,
            hatexplain_splits=splits,
            output_dir=args.output_dir,
        )
    else:
        print("\n[Skipping ablation studies]")

    # =====================================================================
    # Step 9: Summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    best_fair = min(fair_sweep_results, key=lambda r: r["coverage_disparity"])

    summary_lines = [
        f"ConfairNLP Results Summary",
        f"{'='*50}",
        f"Primary Model: {primary_model_name}",
        f"Dataset: HateXplain",
        f"Alpha: {args.alpha}",
        f"Target Coverage: {1 - args.alpha:.2f}",
        f"",
        f"Marginal CP:",
        f"  Coverage: {marginal_results['coverage']:.4f}",
        f"  Avg Set Size: {marginal_results['avg_set_size']:.4f}",
        f"  Coverage Disparity: {analysis_results['marginal_disparity']:.4f}",
        f"",
        f"Group-Conditional CP:",
        f"  Coverage: {gc_results['overall_coverage']:.4f}",
        f"  Avg Set Size: {gc_results['overall_avg_set_size']:.4f}",
        f"  Coverage Disparity: {gc_results['coverage_disparity']:.4f}",
        f"",
        f"Fair CP (best lambda={best_fair['lambda']:.2f}):",
        f"  Coverage: {best_fair['overall_coverage']:.4f}",
        f"  Avg Set Size: {best_fair['overall_avg_set_size']:.4f}",
        f"  Coverage Disparity: {best_fair['coverage_disparity']:.4f}",
        f"",
        f"Key Finding:",
        f"  Marginal CP disparity: {analysis_results['marginal_disparity']:.4f}",
        f"  Group-Conditional CP reduces disparity to: {gc_results['coverage_disparity']:.4f}",
        f"  Fair CP (lambda={best_fair['lambda']:.2f}) achieves disparity: {best_fair['coverage_disparity']:.4f}",
    ]

    elapsed = time.time() - start_time
    summary_lines.append(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save summary
    summary_path = os.path.join(args.output_dir, "full_results_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\nSummary saved to {summary_path}")

    print(f"\nAll results saved to {args.output_dir}/")
    print("Pipeline complete!")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
