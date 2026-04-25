"""
Fine-tune BERT and HateBERT on HateXplain for hate speech classification.

Default config:
- Epochs: 5
- Learning rate: 2e-5
- Batch size: 16
- Max sequence length: 128
"""

import inspect
import os
import pickle

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SEED = 42
NUM_LABELS = 3
MAX_LENGTH = 128
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 2e-5
LABEL_NAMES = ["hate", "offensive", "normal"]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HateDataset(torch.utils.data.Dataset):
    """Dataset wrapper for tokenized hate speech data."""

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
    """Load preprocessed HateXplain splits from pickle."""
    path = os.path.join(data_dir, "hatexplain_splits.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


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
    """Compute accuracy and macro F1 for HuggingFace Trainer evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def _training_args_kwargs(
    model_output_dir: str,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    kwargs = {
        "output_dir": os.path.join(model_output_dir, "checkpoints"),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": max(64, batch_size),
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "save_strategy": "no",
        "seed": SEED,
        "logging_steps": 100,
        "report_to": "none",
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "no"

    if "use_cpu" in signature:
        kwargs["use_cpu"] = device == "cpu"
    elif "no_cuda" in signature:
        kwargs["no_cuda"] = device == "cpu"

    return kwargs


def train_model(
    model_name: str,
    save_name: str,
    train_df,
    eval_df,
    output_dir: str,
    device: str = "cpu",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
):
    """Fine-tune a single model on HateXplain training data."""
    set_seed(SEED)

    print(f"\n{'=' * 60}")
    print(f"Training {save_name} ({model_name})")
    print(f"{'=' * 60}")
    print(
        f"Config: epochs={epochs}, batch_size={batch_size}, "
        f"learning_rate={learning_rate}, device={device}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
    )

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    eval_texts = eval_df["text"].tolist()
    eval_labels = eval_df["label"].tolist()

    print(f"Tokenizing {len(train_texts)} train and {len(eval_texts)} eval samples...")
    train_dataset = HateDataset(tokenize_data(tokenizer, train_texts), train_labels)
    eval_dataset = HateDataset(tokenize_data(tokenizer, eval_texts), eval_labels)

    model_output_dir = os.path.join(output_dir, save_name)
    training_args = TrainingArguments(
        **_training_args_kwargs(
            model_output_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("\nEvaluating classifier on tuning split...")
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    report = classification_report(
        eval_labels,
        preds,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    print(f"\nClassification report for {save_name}:")
    print(report)

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Model saved to {model_output_dir}")

    return model, tokenizer


def train_all(
    data_dir: str,
    output_dir: str,
    device: str = None,
    skip_if_exists: bool = True,
    require_existing: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
):
    """Train or load BERT and HateBERT classifiers."""
    set_seed(SEED)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    splits = load_hatexplain_splits(data_dir)
    train_df = splits["train"]
    eval_df = splits.get("tuning", splits["test"])

    if device == "cpu" and len(train_df) > 3000:
        print(f"[CPU mode] Subsampling training data from {len(train_df)} to 3000 samples")
        train_df = train_df.sample(n=3000, random_state=SEED, replace=False)

    models_config = [
        ("bert-base-uncased", "bert_hatexplain"),
        ("GroNLP/hateBERT", "hatebert_hatexplain"),
    ]

    trained_models = {}
    missing_models = []
    for model_name, save_name in models_config:
        model_path = os.path.join(output_dir, save_name)

        if skip_if_exists and os.path.exists(model_path):
            print(f"\n[{save_name}] Found existing model at {model_path}, loading...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            trained_models[save_name] = (model, tokenizer)
            continue

        if require_existing:
            missing_models.append(model_path)
            continue

        model, tokenizer = train_model(
            model_name=model_name,
            save_name=save_name,
            train_df=train_df,
            eval_df=eval_df,
            output_dir=output_dir,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        trained_models[save_name] = (model, tokenizer)

    if missing_models:
        missing = "\n  ".join(missing_models)
        raise FileNotFoundError(
            "--skip-training was set, but these saved model directories are missing:\n"
            f"  {missing}\n"
            "Run without --skip-training to train them, or set --model-dir correctly."
        )

    return trained_models


def get_softmax_probs(model, tokenizer, texts, device="cpu", batch_size=32):
    """Get softmax probabilities for a list of texts."""
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
        encodings = {key: val.to(device) for key, val in encodings.items()}

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
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    args = parser.parse_args()

    train_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
