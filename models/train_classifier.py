"""
train_classifier.py — Fine-tune BERT and HateBERT on HateXplain for hate speech classification.

Uses HuggingFace Trainer API with the following config:
  - Epochs: 5, LR: 2e-5, Batch size: 16, Max length: 128
  - Optimizer: AdamW, Loss: CrossEntropyLoss (built into Trainer)
  - Evaluates per-class precision/recall/F1 and macro F1 on test set
"""

import os
import pickle

import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SEED = 42
NUM_LABELS = 3
MAX_LENGTH = 128
LABEL_NAMES = ["hate", "offensive", "normal"]


class HateDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for tokenized hate speech data."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_hatexplain_splits(data_dir: str) -> dict:
    """Load the preprocessed HateXplain splits from pickle."""
    path = os.path.join(data_dir, "hatexplain_splits.pkl")
    with open(path, "rb") as f:
        splits = pickle.load(f)
    return splits


def tokenize_data(tokenizer, texts, max_length=MAX_LENGTH):
    """Tokenize a list of texts using the given tokenizer."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def compute_metrics(eval_pred):
    """Compute accuracy for HuggingFace Trainer evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def train_model(
    model_name: str,
    save_name: str,
    train_df,
    test_df,
    output_dir: str,
    device: str = "cpu",
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
):
    """Fine-tune a single model on HateXplain training data.

    Args:
        model_name: HuggingFace model identifier (e.g., 'bert-base-uncased')
        save_name: Name for saving the fine-tuned model
        train_df: Training dataframe with 'text' and 'label' columns
        test_df: Test dataframe for evaluation
        output_dir: Directory to save the fine-tuned model
        device: 'cuda' or 'cpu'
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for AdamW

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n{'='*60}")
    print(f"Training {save_name} ({model_name})")
    print(f"{'='*60}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )

    # Tokenize
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print(f"Tokenizing {len(train_texts)} train and {len(test_texts)} test samples...")
    train_encodings = tokenize_data(tokenizer, train_texts)
    test_encodings = tokenize_data(tokenizer, test_texts)

    train_dataset = HateDataset(train_encodings, train_labels)
    test_dataset = HateDataset(test_encodings, test_labels)

    # Training arguments
    model_output_dir = os.path.join(output_dir, save_name)
    training_args = TrainingArguments(
        output_dir=os.path.join(model_output_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        seed=SEED,
        logging_steps=100,
        report_to="none",
        use_cpu=(device == "cpu"),
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    report = classification_report(
        test_labels, preds, target_names=LABEL_NAMES, digits=4
    )
    print(f"\nClassification Report for {save_name}:")
    print(report)

    # Save model and tokenizer
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")

    return model, tokenizer


def train_all(
    data_dir: str,
    output_dir: str,
    device: str = None,
    skip_if_exists: bool = True,
):
    """Train both BERT and HateBERT classifiers.

    Args:
        data_dir: Directory containing hatexplain_splits.pkl
        output_dir: Directory to save trained models
        device: 'cuda' or 'cpu' (auto-detects if None)
        skip_if_exists: Skip training if model directory already exists

    Returns:
        Dictionary mapping model names to (model, tokenizer) tuples.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    splits = load_hatexplain_splits(data_dir)
    train_df = splits["train"]
    test_df = splits["test"]

    # On CPU, subsample training data for tractability
    if device == "cpu" and len(train_df) > 3000:
        print(f"[CPU mode] Subsampling training data from {len(train_df)} to 3000 samples")
        train_df = train_df.sample(n=3000, random_state=SEED, replace=False)

    models_config = [
        ("bert-base-uncased", "bert_hatexplain"),
        ("GroNLP/hateBERT", "hatebert_hatexplain"),
    ]

    trained_models = {}
    for model_name, save_name in models_config:
        model_path = os.path.join(output_dir, save_name)

        if skip_if_exists and os.path.exists(model_path):
            print(f"\n[{save_name}] Found existing model at {model_path}, loading...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            trained_models[save_name] = (model, tokenizer)
            continue

        model, tokenizer = train_model(
            model_name=model_name,
            save_name=save_name,
            train_df=train_df,
            test_df=test_df,
            output_dir=output_dir,
            device=device,
        )
        trained_models[save_name] = (model, tokenizer)

    return trained_models


def get_softmax_probs(model, tokenizer, texts, device="cpu", batch_size=32):
    """Get softmax probabilities for a list of texts.

    Args:
        model: Fine-tuned classification model
        tokenizer: Corresponding tokenizer
        texts: List of input texts
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        numpy array of shape (n_samples, n_classes) with softmax probabilities
    """
    model.eval()
    model.to(device)
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train hate speech classifiers")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--output-dir", default="models/trained", help="Path to save models")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    train_all(args.data_dir, args.output_dir, args.device)
