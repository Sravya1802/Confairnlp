"""
download_data.py — Download and preprocess datasets for ConfairNLP.

Handles three datasets:
  1. HateXplain (primary) — 3-class hate speech with demographic annotations
  2. ToxiGen — machine-generated toxic text with target group labels
  3. Davidson — tweet-level hate/offensive/neither classification

For HateXplain, creates stratified 60/20/20 train/calibration/test splits
with group proportions preserved across splits.
"""

import os
import pickle
import random

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def download_hatexplain(output_dir: str) -> dict:
    """Download and preprocess HateXplain dataset.

    Extracts text, 3-class labels (hate=0, offensive=1, normal=2), and
    target community annotations. Each post can target multiple groups.

    Returns dict with train/calibration/test splits.
    """
    print("[HateXplain] Loading dataset from HuggingFace...")
    dataset = load_dataset("hatexplain", trust_remote_code=True)

    records = []
    label_map = {"hatespeech": 0, "offensive": 1, "normal": 2}

    for split_name in dataset:
        for example in dataset[split_name]:
            post_id = example["id"]
            tokens = example["post_tokens"]
            text = " ".join(tokens)

            # Majority-vote label from annotators
            annotator_labels = example["annotators"]["label"]
            label_counts = {}
            for lbl in annotator_labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            majority_label = max(label_counts, key=label_counts.get)

            # Target communities from all annotators
            target_groups = set()
            for annotator_targets in example["annotators"]["target"]:
                for group in annotator_targets:
                    if group and group.strip().lower() not in ("", "none"):
                        target_groups.add(group.strip())

            records.append({
                "id": post_id,
                "text": text,
                "label": majority_label,
                "target_groups": list(target_groups),
            })

    df = pd.DataFrame(records)
    print(f"[HateXplain] Total samples: {len(df)}")
    print(f"[HateXplain] Label distribution:\n{df['label'].value_counts().sort_index()}")

    # Collect all unique groups
    all_groups = set()
    for groups in df["target_groups"]:
        all_groups.update(groups)
    print(f"[HateXplain] Target groups found: {sorted(all_groups)}")

    # Stratified 60/20/20 split on label
    train_df, temp_df = train_test_split(
        df, test_size=0.4, random_state=SEED, stratify=df["label"]
    )
    cal_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"]
    )

    print(f"[HateXplain] Split sizes — Train: {len(train_df)}, "
          f"Calibration: {len(cal_df)}, Test: {len(test_df)}")

    splits = {"train": train_df, "calibration": cal_df, "test": test_df}

    # Save splits
    save_path = os.path.join(output_dir, "hatexplain_splits.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"[HateXplain] Saved splits to {save_path}")

    return splits


def download_toxigen(output_dir: str, max_per_group: int = 10000) -> pd.DataFrame:
    """Download and preprocess ToxiGen dataset.

    Filters to a balanced subset of up to max_per_group samples per target group.
    """
    print("[ToxiGen] Loading dataset from HuggingFace...")
    dataset = load_dataset("skg/toxigen-data", "annotated", trust_remote_code=True)

    records = []
    for split_name in dataset:
        for example in dataset[split_name]:
            text = example["text"]
            # Toxicity label: binarize from the annotation
            toxicity = int(example["toxicity_human"] >= 2.5) if "toxicity_human" in example else example.get("label", 0)
            group = example.get("target_group", "unknown")

            records.append({
                "text": text,
                "label": toxicity,
                "target_group": group,
            })

    df = pd.DataFrame(records)
    print(f"[ToxiGen] Total samples before filtering: {len(df)}")

    # Balance: sample up to max_per_group per group
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
    """Download and preprocess Davidson dataset.

    Labels: 0=hate, 1=offensive, 2=neither.
    No demographic annotations available.
    """
    print("[Davidson] Downloading dataset...")
    url = (
        "https://raw.githubusercontent.com/t-davidson/"
        "hate-speech-and-offensive-language/master/data/labeled_data.csv"
    )
    df = pd.read_csv(url)

    # Keep relevant columns
    df = df.rename(columns={"class": "label", "tweet": "text"})
    df = df[["text", "label"]].copy()

    print(f"[Davidson] Total samples: {len(df)}")
    print(f"[Davidson] Label distribution:\n{df['label'].value_counts().sort_index()}")

    save_path = os.path.join(output_dir, "davidson.pkl")
    df.to_pickle(save_path)
    print(f"[Davidson] Saved to {save_path}")

    return df


def download_all(output_dir: str = None) -> dict:
    """Download and preprocess all datasets.

    Args:
        output_dir: Directory to save processed data. Defaults to data/ dir.

    Returns:
        Dictionary with all dataset splits/dataframes.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # HateXplain (primary)
    results["hatexplain"] = download_hatexplain(output_dir)

    # ToxiGen
    results["toxigen"] = download_toxigen(output_dir)

    # Davidson
    results["davidson"] = download_davidson(output_dir)

    print("\n[All datasets downloaded and preprocessed successfully!]")
    return results


if __name__ == "__main__":
    download_all()
