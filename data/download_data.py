"""
Download and preprocess datasets for ConfairNLP.

The main experiment uses HateXplain. ToxiGen and Davidson helpers are kept for
future cross-dataset work, but they are not used by the default pipeline.

HateXplain is split into train/calibration/tuning/test partitions with ratios
60/20/10/10. The tuning split is used only to choose the Fair CP lambda, so the
test split remains a final holdout.
"""

from collections import Counter
import os
import pickle
import random
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SPLIT_RATIOS = {
    "train": 0.60,
    "calibration": 0.20,
    "tuning": 0.10,
    "test": 0.10,
}
LABEL_MAP = {"hatespeech": 0, "hate": 0, "offensive": 1, "normal": 2}

random.seed(SEED)
np.random.seed(SEED)


def _load_hf_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to download data. "
            "Install project dependencies with 'pip install -r requirements.txt'."
        ) from exc
    return load_dataset(*args, **kwargs)


def _normalize_label(label):
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized not in LABEL_MAP:
            raise ValueError(f"Unknown HateXplain label: {label!r}")
        return LABEL_MAP[normalized]
    return int(label)


def _majority_vote(labels: Iterable) -> int:
    normalized_labels = [_normalize_label(label) for label in labels]
    counts = Counter(normalized_labels)
    # Deterministic tie break: smaller label id wins.
    return max(sorted(counts), key=lambda label: counts[label])


def normalize_group_list(value) -> list[str]:
    """Return a clean list of group strings for one sample."""
    if isinstance(value, str):
        groups = [value]
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        groups = list(value)
    elif value is None or pd.isna(value):
        groups = []
    else:
        groups = [str(value)]

    cleaned = []
    for group in groups:
        if group is None:
            continue
        text = str(group).strip()
        if text and text.lower() not in {"none", "nan", "unknown/none"}:
            cleaned.append(text)
    return sorted(set(cleaned)) or ["unknown"]


def _primary_group(groups: list[str]) -> str:
    meaningful_groups = [group for group in groups if group != "unknown"]
    return meaningful_groups[0] if meaningful_groups else "unknown"


def _stratify_keys(df: pd.DataFrame) -> pd.Series | None:
    """Build label+primary-group strata with rare buckets collapsed.

    This keeps the split group-aware without making train_test_split fail on
    tiny group/label combinations. If the resulting strata are still too sparse,
    callers fall back to label-only or unstratified splitting.
    """
    labels = df["label"].astype(str)
    primary_groups = df["target_groups"].map(_primary_group)
    keys = labels + "__" + primary_groups

    counts = keys.value_counts()
    keys = keys.where(keys.map(counts) >= 2, labels + "__rare_group")

    counts = keys.value_counts()
    keys = keys.where(keys.map(counts) >= 2, labels)

    counts = keys.value_counts()
    if len(counts) <= 1 or counts.min() < 2:
        return None
    return keys


def _safe_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    split_description: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = _stratify_keys(df)
    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=SEED,
            stratify=stratify,
        )
    except ValueError as exc:
        print(
            f"[HateXplain] Group-aware stratification failed for "
            f"{split_description}: {exc}. Falling back to label-only split."
        )

    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=SEED,
            stratify=df["label"],
        )
    except ValueError as exc:
        print(
            f"[HateXplain] Label-only stratification failed for "
            f"{split_description}: {exc}. Falling back to random split."
        )
        return train_test_split(df, test_size=test_size, random_state=SEED)


def summarize_split_distributions(splits: dict, output_dir: str | None = None) -> pd.DataFrame:
    """Create label and group distribution diagnostics for each split."""
    rows = []
    for split_name, split_df in splits.items():
        total = len(split_df)
        for label, count in split_df["label"].value_counts().sort_index().items():
            rows.append(
                {
                    "split": split_name,
                    "kind": "label",
                    "value": label,
                    "count": int(count),
                    "proportion": count / total if total else np.nan,
                }
            )

        group_counts = Counter()
        for groups in split_df["target_groups"]:
            group_counts.update(normalize_group_list(groups))
        for group, count in sorted(group_counts.items()):
            rows.append(
                {
                    "split": split_name,
                    "kind": "group",
                    "value": group,
                    "count": int(count),
                    "proportion": count / total if total else np.nan,
                }
            )

    distribution_df = pd.DataFrame(rows)
    if output_dir is not None:
        path = os.path.join(output_dir, "hatexplain_split_distributions.csv")
        distribution_df.to_csv(path, index=False, float_format="%.6f")
        print(f"[HateXplain] Split distribution diagnostics saved to {path}")
    return distribution_df


def create_hatexplain_splits(df: pd.DataFrame, output_dir: str | None = None) -> dict:
    """Create deterministic 60/20/10/10 HateXplain splits."""
    df = df.copy().reset_index(drop=True)
    df["label"] = df["label"].map(_normalize_label)
    df["target_groups"] = df["target_groups"].map(normalize_group_list)

    train_df, remainder_df = _safe_train_test_split(
        df,
        test_size=1.0 - SPLIT_RATIOS["train"],
        split_description="train vs remainder",
    )
    remainder_ratio = (
        SPLIT_RATIOS["calibration"] + SPLIT_RATIOS["tuning"] + SPLIT_RATIOS["test"]
    )
    holdout_ratio_within_remainder = (
        SPLIT_RATIOS["tuning"] + SPLIT_RATIOS["test"]
    ) / remainder_ratio
    calibration_df, holdout_df = _safe_train_test_split(
        remainder_df,
        test_size=holdout_ratio_within_remainder,
        split_description="calibration vs tuning/test holdout",
    )

    tuning_df, test_df = _safe_train_test_split(
        holdout_df,
        test_size=0.5,
        split_description="tuning vs test",
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "calibration": calibration_df.reset_index(drop=True),
        "tuning": tuning_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    print(
        "[HateXplain] Split sizes - "
        f"Train: {len(splits['train'])}, "
        f"Calibration: {len(splits['calibration'])}, "
        f"Tuning: {len(splits['tuning'])}, "
        f"Test: {len(splits['test'])}"
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "hatexplain_splits.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(splits, f)
        print(f"[HateXplain] Saved splits to {save_path}")
        summarize_split_distributions(splits, output_dir)

    return splits


def upgrade_hatexplain_splits(splits: dict, output_dir: str | None = None) -> dict:
    """Rebuild older train/calibration/test splits with a tuning holdout."""
    if "tuning" in splits:
        return splits

    print("[HateXplain] Existing splits do not include tuning data; rebuilding splits.")
    combined = pd.concat(splits.values(), ignore_index=True)
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset="id")
    else:
        combined = combined.drop_duplicates()
    return create_hatexplain_splits(combined, output_dir)


def download_hatexplain(output_dir: str) -> dict:
    """Download and preprocess HateXplain into train/calibration/tuning/test."""
    print("[HateXplain] Loading dataset from HuggingFace...")
    dataset = _load_hf_dataset("hatexplain", trust_remote_code=True)

    records = []
    for split_name in dataset:
        for example in dataset[split_name]:
            target_groups = set()
            for annotator_targets in example["annotators"]["target"]:
                if annotator_targets is None:
                    continue
                for group in annotator_targets:
                    if group is None:
                        continue
                    group_text = str(group).strip()
                    if group_text and group_text.lower() not in {"none", "nan", "unknown/none"}:
                        target_groups.add(group_text)

            records.append(
                {
                    "id": example["id"],
                    "text": " ".join(example["post_tokens"]),
                    "label": _majority_vote(example["annotators"]["label"]),
                    "target_groups": sorted(target_groups) or ["unknown"],
                }
            )

    df = pd.DataFrame(records)
    print(f"[HateXplain] Total samples: {len(df)}")
    print(f"[HateXplain] Label distribution:\n{df['label'].value_counts().sort_index()}")

    all_groups = sorted({group for groups in df["target_groups"] for group in groups})
    print(f"[HateXplain] Target groups found: {all_groups}")
    return create_hatexplain_splits(df, output_dir)


def download_toxigen(output_dir: str, max_per_group: int = 10000) -> pd.DataFrame:
    """Download and preprocess ToxiGen for future cross-dataset experiments."""
    print("[ToxiGen] Loading dataset from HuggingFace...")
    dataset = _load_hf_dataset("skg/toxigen-data", "annotated", trust_remote_code=True)

    records = []
    for split_name in dataset:
        for example in dataset[split_name]:
            toxicity = (
                int(example["toxicity_human"] >= 2.5)
                if "toxicity_human" in example
                else int(example.get("label", 0))
            )
            records.append(
                {
                    "text": example["text"],
                    "label": toxicity,
                    "target_group": example.get("target_group", "unknown"),
                }
            )

    df = pd.DataFrame(records)
    print(f"[ToxiGen] Total samples before filtering: {len(df)}")

    balanced_dfs = []
    for group, group_df in df.groupby("target_group"):
        if len(group_df) > max_per_group:
            group_df = group_df.sample(n=max_per_group, random_state=SEED)
        balanced_dfs.append(group_df)
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)

    print(f"[ToxiGen] Total samples after balancing: {len(df_balanced)}")
    print(f"[ToxiGen] Groups: {df_balanced['target_group'].value_counts().to_dict()}")

    save_path = os.path.join(output_dir, "toxigen.pkl")
    df_balanced.to_pickle(save_path)
    print(f"[ToxiGen] Saved to {save_path}")
    return df_balanced


def download_davidson(output_dir: str) -> pd.DataFrame:
    """Download and preprocess Davidson for future model evaluation."""
    print("[Davidson] Downloading dataset...")
    url = (
        "https://raw.githubusercontent.com/t-davidson/"
        "hate-speech-and-offensive-language/master/data/labeled_data.csv"
    )
    df = pd.read_csv(url)
    df = df.rename(columns={"class": "label", "tweet": "text"})
    df = df[["text", "label"]].copy()

    print(f"[Davidson] Total samples: {len(df)}")
    print(f"[Davidson] Label distribution:\n{df['label'].value_counts().sort_index()}")

    save_path = os.path.join(output_dir, "davidson.pkl")
    df.to_pickle(save_path)
    print(f"[Davidson] Saved to {save_path}")
    return df


def download_all(output_dir: str | None = None) -> dict:
    """Download and preprocess all available datasets."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "hatexplain": download_hatexplain(output_dir),
        "toxigen": download_toxigen(output_dir),
        "davidson": download_davidson(output_dir),
    }

    print("\n[All datasets downloaded and preprocessed successfully!]")
    return results


if __name__ == "__main__":
    download_all()
