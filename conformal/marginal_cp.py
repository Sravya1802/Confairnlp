"""
marginal_cp.py — Standard split conformal prediction.

Implements two nonconformity score functions:
  1. Softmax score: s(x, y) = 1 - softmax[true_label]
  2. APS (Adaptive Prediction Sets): cumulative softmax up to true label

Given a calibration set and significance level alpha, computes a quantile
threshold and constructs prediction sets for test samples.
"""

import numpy as np

SEED = 42
np.random.seed(SEED)


def softmax_nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute softmax-based nonconformity scores.

    Score for sample i = 1 - softmax_prob[true_label_i]

    Args:
        probs: Softmax probabilities, shape (n, num_classes)
        labels: True labels, shape (n,)

    Returns:
        Nonconformity scores, shape (n,)
    """
    n = len(labels)
    scores = 1.0 - probs[np.arange(n), labels]
    return scores


def aps_nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute APS (Adaptive Prediction Sets) nonconformity scores.

    For each sample, sort softmax probs in descending order and accumulate
    until the true label is included. The score is the cumulative sum
    up to and including the true label.

    Args:
        probs: Softmax probabilities, shape (n, num_classes)
        labels: True labels, shape (n,)

    Returns:
        APS nonconformity scores, shape (n,)
    """
    n, num_classes = probs.shape
    scores = np.zeros(n)

    for i in range(n):
        # Sort classes by probability descending
        sorted_indices = np.argsort(-probs[i])
        cumsum = 0.0
        for idx in sorted_indices:
            cumsum += probs[i, idx]
            if idx == labels[i]:
                scores[i] = cumsum
                break

    return scores


def compute_quantile_threshold(scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal quantile threshold.

    q_hat = quantile of scores at level ceil((n+1)(1-alpha)) / n

    Args:
        scores: Nonconformity scores from calibration set
        alpha: Significance level (e.g., 0.10 for 90% coverage)

    Returns:
        Quantile threshold q_hat
    """
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)  # Clamp to 1.0
    q_hat = np.quantile(scores, level, method="higher")
    return q_hat


def build_prediction_sets_softmax(probs: np.ndarray, q_hat: float) -> list:
    """Build prediction sets using softmax score threshold.

    C(x) = {y : softmax[y] >= 1 - q_hat}

    Args:
        probs: Softmax probabilities for test samples, shape (n, num_classes)
        q_hat: Quantile threshold from calibration

    Returns:
        List of prediction sets (each a list of class indices)
    """
    n, num_classes = probs.shape
    threshold = 1.0 - q_hat
    prediction_sets = []

    for i in range(n):
        pset = [y for y in range(num_classes) if probs[i, y] >= threshold]
        # Always include at least the argmax class
        if len(pset) == 0:
            pset = [np.argmax(probs[i])]
        prediction_sets.append(pset)

    return prediction_sets


def build_prediction_sets_aps(probs: np.ndarray, q_hat: float) -> list:
    """Build prediction sets using APS score threshold.

    Include classes in descending probability order until the cumulative
    sum exceeds q_hat.

    Args:
        probs: Softmax probabilities for test samples, shape (n, num_classes)
        q_hat: APS quantile threshold from calibration

    Returns:
        List of prediction sets (each a list of class indices)
    """
    n, num_classes = probs.shape
    prediction_sets = []

    for i in range(n):
        sorted_indices = np.argsort(-probs[i])
        cumsum = 0.0
        pset = []
        for idx in sorted_indices:
            cumsum += probs[i, idx]
            pset.append(int(idx))
            if cumsum >= q_hat:
                break
        prediction_sets.append(pset)

    return prediction_sets


def evaluate_prediction_sets(
    prediction_sets: list, true_labels: np.ndarray
) -> dict:
    """Evaluate prediction sets: coverage and average set size.

    Args:
        prediction_sets: List of prediction sets
        true_labels: True labels for test samples

    Returns:
        Dict with 'coverage', 'avg_set_size', 'set_sizes'
    """
    n = len(true_labels)
    covered = sum(
        1 for i in range(n) if true_labels[i] in prediction_sets[i]
    )
    set_sizes = [len(pset) for pset in prediction_sets]

    return {
        "coverage": covered / n,
        "avg_set_size": np.mean(set_sizes),
        "set_sizes": set_sizes,
    }


def run_marginal_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.10,
    score_function: str = "softmax",
) -> dict:
    """Run standard marginal conformal prediction.

    Args:
        cal_probs: Calibration set softmax probabilities
        cal_labels: Calibration set true labels
        test_probs: Test set softmax probabilities
        test_labels: Test set true labels
        alpha: Significance level
        score_function: 'softmax' or 'aps'

    Returns:
        Dict with q_hat, prediction_sets, coverage, avg_set_size, etc.
    """
    # Compute nonconformity scores on calibration set
    if score_function == "softmax":
        cal_scores = softmax_nonconformity_scores(cal_probs, cal_labels)
    elif score_function == "aps":
        cal_scores = aps_nonconformity_scores(cal_probs, cal_labels)
    else:
        raise ValueError(f"Unknown score function: {score_function}")

    # Compute threshold
    q_hat = compute_quantile_threshold(cal_scores, alpha)

    # Build prediction sets
    if score_function == "softmax":
        prediction_sets = build_prediction_sets_softmax(test_probs, q_hat)
    else:
        prediction_sets = build_prediction_sets_aps(test_probs, q_hat)

    # Evaluate
    results = evaluate_prediction_sets(prediction_sets, test_labels)
    results["q_hat"] = q_hat
    results["prediction_sets"] = prediction_sets
    results["alpha"] = alpha
    results["score_function"] = score_function

    print(f"[Marginal CP] alpha={alpha}, score={score_function}")
    print(f"  q_hat = {q_hat:.4f}")
    print(f"  Coverage = {results['coverage']:.4f} (target: {1 - alpha:.2f})")
    print(f"  Avg set size = {results['avg_set_size']:.4f}")

    return results
