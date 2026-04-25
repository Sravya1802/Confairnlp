"""
Shared setup for novelty modules.

Loads HateXplain splits, loads the primary trained classifier (HateBERT by
default), computes softmax probabilities on every split, runs the three CP
methods (marginal / group-conditional / fair), and returns everything the
novelty analyses need.

Softmax arrays and CP results are cached to results/_novelty_cache.pkl so
repeated novelty runs do not re-run BERT inference.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from conformal.fair_cp import result_for_lambda, run_fair_cp_sweep, select_lambda_by_tuning
from conformal.group_conditional_cp import run_group_conditional_cp
from conformal.marginal_cp import run_marginal_cp
from models.train_classifier import get_softmax_probs, load_hatexplain_splits

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PRIMARY_MODEL = "hatebert_hatexplain"
DEFAULT_CACHE = os.path.join(RESULTS_DIR, "_novelty_cache.pkl")
SEED = 42


def prepare_groups(df) -> list:
    """Normalize the target_groups column into a list-of-lists of clean strings."""
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
            str(g).strip()
            for g in sample_groups
            if g is not None and str(g).strip().lower() not in {"", "none", "nan"}
        ]
        groups.append(sample_groups or ["unknown"])
    return groups


def load_primary_setup(
    alpha: float = 0.10,
    min_test_group_size: int = 30,
    score_function: str = "softmax",
    data_dir: str = DATA_DIR,
    model_dir: str = MODEL_DIR,
    model_name: str = PRIMARY_MODEL,
    device: str | None = None,
    cache_path: str = DEFAULT_CACHE,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Return a dict of splits, probabilities, and CP results for novelty modules."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path) and not force_recompute:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        matches = (
            cached.get("alpha") == alpha
            and cached.get("model_name") == model_name
            and cached.get("score_function") == score_function
            and cached.get("min_test_group_size") == min_test_group_size
        )
        if matches:
            print(f"[novelty setup] Using cached artifacts from {cache_path}")
            cached["device"] = device
            return cached
        print(f"[novelty setup] Cache exists but config differs; recomputing.")

    splits = load_hatexplain_splits(data_dir)
    cal_df = splits["calibration"]
    tuning_df = splits["tuning"]
    test_df = splits["test"]

    model_path = os.path.join(model_dir, model_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Primary model directory not found at {model_path}. "
            f"Run 'python run_all.py' first to train HateBERT."
        )

    print(f"[novelty setup] Loading primary model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    cal_texts = cal_df["text"].tolist()
    tuning_texts = tuning_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    cal_labels = np.array(cal_df["label"].tolist())
    tuning_labels = np.array(tuning_df["label"].tolist())
    test_labels = np.array(test_df["label"].tolist())
    cal_groups = prepare_groups(cal_df)
    tuning_groups = prepare_groups(tuning_df)
    test_groups = prepare_groups(test_df)

    print("[novelty setup] Computing softmax on calibration split...")
    cal_probs = get_softmax_probs(model, tokenizer, cal_texts, device=device)
    print("[novelty setup] Computing softmax on tuning split...")
    tuning_probs = get_softmax_probs(model, tokenizer, tuning_texts, device=device)
    print("[novelty setup] Computing softmax on test split...")
    test_probs = get_softmax_probs(model, tokenizer, test_texts, device=device)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[novelty setup] Running Marginal CP...")
    marginal = run_marginal_cp(
        cal_probs, cal_labels, test_probs, test_labels,
        alpha=alpha, score_function=score_function,
    )
    print("[novelty setup] Running Group-Conditional CP...")
    group_cond = run_group_conditional_cp(
        cal_probs, cal_labels, cal_groups,
        test_probs, test_labels, test_groups,
        alpha=alpha, score_function=score_function,
        min_test_group_size=min_test_group_size,
    )
    print("[novelty setup] Running Fair CP (select lambda on tuning, evaluate on test)...")
    tuning_sweep = run_fair_cp_sweep(
        cal_probs, cal_labels, cal_groups,
        tuning_probs, tuning_labels, tuning_groups,
        alpha=alpha, score_function=score_function,
        min_test_group_size=min_test_group_size, verbose=False,
    )
    selected = select_lambda_by_tuning(tuning_sweep, alpha)
    test_sweep = run_fair_cp_sweep(
        cal_probs, cal_labels, cal_groups,
        test_probs, test_labels, test_groups,
        alpha=alpha, score_function=score_function,
        min_test_group_size=min_test_group_size, verbose=False,
    )
    fair = result_for_lambda(test_sweep, selected["lambda"])

    setup = {
        "alpha": alpha,
        "score_function": score_function,
        "model_name": model_name,
        "min_test_group_size": min_test_group_size,
        "cal_df": cal_df, "tuning_df": tuning_df, "test_df": test_df,
        "cal_probs": cal_probs, "tuning_probs": tuning_probs, "test_probs": test_probs,
        "cal_labels": cal_labels, "tuning_labels": tuning_labels, "test_labels": test_labels,
        "cal_groups": cal_groups, "tuning_groups": tuning_groups, "test_groups": test_groups,
        "marginal": marginal,
        "group_conditional": group_cond,
        "fair": fair,
        "fair_lambda": selected["lambda"],
        "fair_test_sweep": test_sweep,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(setup, f)
    print(f"[novelty setup] Cached artifacts to {cache_path}")

    setup["device"] = device
    return setup


def load_model_and_tokenizer(
    model_dir: str = MODEL_DIR,
    model_name: str = PRIMARY_MODEL,
    device: str | None = None,
):
    """Load the primary model separately (needed e.g. for counterfactual inference)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_dir, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer, device


def ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def append_novelty_summary(section_markdown: str, summary_path: str | None = None) -> str:
    """Append a section to results/NOVELTY_SUMMARY.md, creating it if needed."""
    ensure_results_dir()
    path = summary_path or os.path.join(RESULTS_DIR, "NOVELTY_SUMMARY.md")
    header = (
        "# ConfairNLP — Novelty Results Summary\n\n"
        "Each section below uses the 5-move discussion style:\n"
        "(1) decomposition, (2) attribution, (3) theoretical tie-back, "
        "(4) trade-off surfacing, (5) honest negative.\n\n"
    )
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
    with open(path, "a", encoding="utf-8") as f:
        f.write(section_markdown.rstrip() + "\n\n")
    return path
