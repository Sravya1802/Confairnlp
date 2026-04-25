"""
Single entry point for the ConfairNLP pipeline.

Pipeline:
1. Download/preprocess HateXplain into train/calibration/tuning/test splits
2. Train or load BERT and HateBERT classifiers
3. Calibrate marginal and group-conditional conformal predictors
4. Select Fair CP lambda on the tuning split
5. Evaluate all methods on the final test split
6. Generate coverage tables, plots, ablations, and a summary
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from conformal.fair_cp import result_for_lambda, run_fair_cp_sweep, select_lambda_by_tuning
from conformal.group_conditional_cp import compute_coverage_disparity, run_group_conditional_cp
from conformal.marginal_cp import run_marginal_cp
from data.download_data import download_hatexplain, upgrade_hatexplain_splits
from evaluation.ablation import run_ablation_study
from evaluation.coverage_analysis import (
    compute_per_group_marginal_coverage,
    plot_multi_alpha_disparity,
    run_coverage_analysis,
)
from models.train_classifier import get_softmax_probs, load_hatexplain_splits, train_all

SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ConfairNLP: Conformal Prediction for Equitable Hate Speech Detection"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Require existing saved models and do not train missing models",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain models even if saved model directories already exist",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Significance level; target coverage is 1-alpha (default: 0.10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu; auto-detects if omitted)",
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs for each classifier (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device training batch size (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning (default: 2e-5)",
    )
    parser.add_argument(
        "--lambda-steps",
        type=int,
        default=11,
        help="Number of lambda values to evaluate from 0 to 1 (default: 11)",
    )
    parser.add_argument(
        "--min-test-group-size",
        type=int,
        default=30,
        help="Minimum test examples for primary per-group disparity metrics",
    )
    args = parser.parse_args()
    if args.skip_training and args.force_retrain:
        parser.error("--skip-training and --force-retrain cannot be used together")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    if args.lambda_steps < 2:
        parser.error("--lambda-steps must be at least 2")
    if args.min_test_group_size < 1:
        parser.error("--min-test-group-size must be at least 1")
    return args


def prepare_group_labels(df):
    """Extract normalized group-label lists from a dataframe."""
    groups = []
    for _, row in df.iterrows():
        value = row.get("target_groups", [])
        if isinstance(value, str):
            sample_groups = [value]
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            sample_groups = list(value)
        elif value is None:
            sample_groups = []
        else:
            sample_groups = [str(value)]

        sample_groups = [
            str(group).strip()
            for group in sample_groups
            if group is not None and str(group).strip().lower() not in {"", "none", "nan"}
        ]
        groups.append(sample_groups or ["unknown"])
    return groups


def _load_or_create_splits(data_dir: str) -> dict:
    hatexplain_path = os.path.join(data_dir, "hatexplain_splits.pkl")
    if os.path.exists(hatexplain_path):
        print(f"Found existing HateXplain splits at {hatexplain_path}, loading...")
        splits = load_hatexplain_splits(data_dir)
        return upgrade_hatexplain_splits(splits, data_dir)
    return download_hatexplain(data_dir)


def _select_and_evaluate_fair_cp(
    cal_probs,
    cal_labels,
    cal_groups,
    tuning_probs,
    tuning_labels,
    tuning_groups,
    test_probs,
    test_labels,
    test_groups,
    alpha: float,
    lambda_steps: int,
    min_test_group_size: int,
    score_function: str = "softmax",
) -> tuple[list, float, dict]:
    print("\nSelecting Fair CP lambda on tuning split...")
    tuning_sweep = run_fair_cp_sweep(
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
    )
    selected_tuning = select_lambda_by_tuning(tuning_sweep, alpha)
    selected_lambda = selected_tuning["lambda"]
    print(f"Selected lambda from tuning split: {selected_lambda:.2f}")

    print("\nEvaluating Fair CP lambda sweep on final test split...")
    test_sweep = run_fair_cp_sweep(
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
    )
    return test_sweep, selected_lambda, result_for_lambda(test_sweep, selected_lambda)


def run_pipeline(args):
    """Execute the full ConfairNLP pipeline."""
    set_seed(SEED)
    start_time = time.time()

    print("=" * 70)
    print("  ConfairNLP: Conformal Prediction for Equitable Hate Speech Detection")
    print("=" * 70)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Alpha: {args.alpha}")
    print(f"Output directory: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  STEP 1: Data Download and Preprocessing")
    print("=" * 70)

    splits = _load_or_create_splits(args.data_dir)
    train_df = splits["train"]
    cal_df = splits["calibration"]
    tuning_df = splits["tuning"]
    test_df = splits["test"]

    print(
        f"\nData splits - Train: {len(train_df)}, Cal: {len(cal_df)}, "
        f"Tuning: {len(tuning_df)}, Test: {len(test_df)}"
    )

    cal_groups = prepare_group_labels(cal_df)
    tuning_groups = prepare_group_labels(tuning_df)
    test_groups = prepare_group_labels(test_df)

    print("\n" + "=" * 70)
    print("  STEP 2: Model Training / Loading")
    print("=" * 70)

    trained_models = train_all(
        data_dir=args.data_dir,
        output_dir=args.model_dir,
        device=device,
        skip_if_exists=not args.force_retrain,
        require_existing=args.skip_training,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print("\n" + "=" * 70)
    print("  STEP 3: Computing Softmax Probabilities")
    print("=" * 70)

    cal_texts = cal_df["text"].tolist()
    tuning_texts = tuning_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    cal_labels = np.array(cal_df["label"].tolist())
    tuning_labels = np.array(tuning_df["label"].tolist())
    test_labels = np.array(test_df["label"].tolist())

    primary_model_name = (
        "hatebert_hatexplain"
        if "hatebert_hatexplain" in trained_models
        else list(trained_models.keys())[0]
    )
    model, tokenizer = trained_models[primary_model_name]
    print(f"\nUsing primary model: {primary_model_name}")

    print("Computing calibration set probabilities...")
    cal_probs = get_softmax_probs(model, tokenizer, cal_texts, device=device)
    print("Computing tuning set probabilities...")
    tuning_probs = get_softmax_probs(model, tokenizer, tuning_texts, device=device)
    print("Computing test set probabilities...")
    test_probs = get_softmax_probs(model, tokenizer, test_texts, device=device)

    print("\n" + "=" * 70)
    print("  STEP 4: Marginal Conformal Prediction")
    print("=" * 70)

    marginal_results = run_marginal_cp(
        cal_probs,
        cal_labels,
        test_probs,
        test_labels,
        alpha=args.alpha,
        score_function="softmax",
    )

    print("\n" + "=" * 70)
    print("  STEP 5: Group-Conditional Conformal Prediction")
    print("=" * 70)

    gc_results = run_group_conditional_cp(
        cal_probs,
        cal_labels,
        cal_groups,
        test_probs,
        test_labels,
        test_groups,
        alpha=args.alpha,
        score_function="softmax",
        min_test_group_size=args.min_test_group_size,
    )

    print("\n" + "=" * 70)
    print("  STEP 6: Fairness-Regularized Conformal Prediction")
    print("=" * 70)

    fair_sweep_results, selected_lambda, selected_fair_test = _select_and_evaluate_fair_cp(
        cal_probs,
        cal_labels,
        cal_groups,
        tuning_probs,
        tuning_labels,
        tuning_groups,
        test_probs,
        test_labels,
        test_groups,
        alpha=args.alpha,
        lambda_steps=args.lambda_steps,
        min_test_group_size=args.min_test_group_size,
    )

    print("\n" + "=" * 70)
    print("  STEP 7: Coverage Analysis and Visualization")
    print("=" * 70)

    analysis_results = run_coverage_analysis(
        marginal_results,
        gc_results,
        fair_sweep_results,
        test_labels,
        test_groups,
        args.alpha,
        args.output_dir,
        selected_lambda=selected_lambda,
        min_test_group_size=args.min_test_group_size,
    )

    print("\nRunning multi-alpha analysis...")
    alpha_results = {}
    for alpha_val in [0.05, 0.10, 0.15, 0.20]:
        m = run_marginal_cp(
            cal_probs,
            cal_labels,
            test_probs,
            test_labels,
            alpha=alpha_val,
            score_function="softmax",
        )
        m_pg = compute_per_group_marginal_coverage(
            m["prediction_sets"],
            test_labels,
            test_groups,
            min_test_group_size=args.min_test_group_size,
        )
        m_disp = compute_coverage_disparity(
            m_pg,
            alpha_val,
            min_test_group_size=args.min_test_group_size,
        )

        gc = run_group_conditional_cp(
            cal_probs,
            cal_labels,
            cal_groups,
            test_probs,
            test_labels,
            test_groups,
            alpha=alpha_val,
            score_function="softmax",
            min_test_group_size=args.min_test_group_size,
        )

        fair_tuning = run_fair_cp_sweep(
            cal_probs,
            cal_labels,
            cal_groups,
            tuning_probs,
            tuning_labels,
            tuning_groups,
            alpha=alpha_val,
            score_function="softmax",
            lambda_steps=args.lambda_steps,
            min_test_group_size=args.min_test_group_size,
            verbose=False,
        )
        selected_for_alpha = select_lambda_by_tuning(fair_tuning, alpha_val)

        fair_test = run_fair_cp_sweep(
            cal_probs,
            cal_labels,
            cal_groups,
            test_probs,
            test_labels,
            test_groups,
            alpha=alpha_val,
            score_function="softmax",
            lambda_steps=args.lambda_steps,
            min_test_group_size=args.min_test_group_size,
            verbose=False,
        )
        selected_fair = result_for_lambda(fair_test, selected_for_alpha["lambda"])

        alpha_results[alpha_val] = {
            "marginal_disparity": m_disp,
            "group_conditional_disparity": gc["coverage_disparity"],
            "fair_cp_disparity": selected_fair["coverage_disparity"],
        }

    plot_multi_alpha_disparity(
        alpha_results,
        os.path.join(args.output_dir, "multi_alpha_disparity.pdf"),
    )

    if not args.skip_ablation:
        print("\n" + "=" * 70)
        print("  STEP 8: Ablation Studies")
        print("=" * 70)

        model_probs = {}
        for model_name, (mod, tok) in trained_models.items():
            print(f"\nComputing probabilities for {model_name}...")
            m_cal_probs = get_softmax_probs(mod, tok, cal_texts, device=device)
            m_tuning_probs = get_softmax_probs(mod, tok, tuning_texts, device=device)
            m_test_probs = get_softmax_probs(mod, tok, test_texts, device=device)

            model_probs[(model_name, "HateXplain")] = {
                "cal_probs": m_cal_probs,
                "cal_labels": cal_labels,
                "cal_groups": cal_groups,
                "tuning_probs": m_tuning_probs,
                "tuning_labels": tuning_labels,
                "tuning_groups": tuning_groups,
                "test_probs": m_test_probs,
                "test_labels": test_labels,
                "test_groups": test_groups,
                "lambda_steps": args.lambda_steps,
                "min_test_group_size": args.min_test_group_size,
            }

        run_ablation_study(model_probs=model_probs, output_dir=args.output_dir)
    else:
        print("\n[Skipping ablation studies]")

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    fair_summary = analysis_results["selected_fair_results"]

    summary_lines = [
        "ConfairNLP Results Summary",
        "=" * 50,
        f"Primary Model: {primary_model_name}",
        "Dataset: HateXplain",
        f"Alpha: {args.alpha}",
        f"Target Coverage: {1 - args.alpha:.2f}",
        f"Primary Disparity Metric: reliable groups with n_test >= {args.min_test_group_size}",
        "",
        "Marginal CP:",
        f"  Coverage: {marginal_results['coverage']:.4f}",
        f"  Avg Set Size: {marginal_results['avg_set_size']:.4f}",
        f"  Reliable-Group Coverage Disparity: {analysis_results['marginal_disparity_reliable']:.4f}",
        f"  All-Group Coverage Disparity: {analysis_results['marginal_disparity_all']:.4f}",
        "",
        "Group-Conditional CP:",
        f"  Coverage: {gc_results['overall_coverage']:.4f}",
        f"  Avg Set Size: {gc_results['overall_avg_set_size']:.4f}",
        f"  Reliable-Group Coverage Disparity: {gc_results['coverage_disparity_reliable']:.4f}",
        f"  All-Group Coverage Disparity: {gc_results['coverage_disparity_all']:.4f}",
        "",
        f"Fair CP (lambda={fair_summary['lambda']:.2f}, selected on tuning split):",
        f"  Coverage: {fair_summary['overall_coverage']:.4f}",
        f"  Avg Set Size: {fair_summary['overall_avg_set_size']:.4f}",
        f"  Reliable-Group Coverage Disparity: {fair_summary['coverage_disparity_reliable']:.4f}",
        f"  All-Group Coverage Disparity: {fair_summary['coverage_disparity_all']:.4f}",
        "",
        "Key Finding:",
        f"  Marginal CP reliable disparity: {analysis_results['marginal_disparity']:.4f}",
        f"  Group-Conditional CP reliable disparity: {gc_results['coverage_disparity']:.4f}",
        f"  Fair CP reliable disparity: {selected_fair_test['coverage_disparity']:.4f}",
    ]

    elapsed = time.time() - start_time
    summary_lines.append(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = os.path.join(args.output_dir, "full_results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\nSummary saved to {summary_path}")

    print(f"\nAll results saved to {args.output_dir}/")
    print("Pipeline complete!")


if __name__ == "__main__":
    run_pipeline(parse_args())
