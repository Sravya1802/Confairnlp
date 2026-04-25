"""
Standard split conformal prediction.

Implements two nonconformity score functions:
1. Softmax score: s(x, y) = 1 - softmax[true_label]
2. APS score: cumulative softmax probability up to the true label
"""

import numpy as np

SEED = 42
SUPPORTED_SCORE_FUNCTIONS = {"softmax", "aps"}
np.random.seed(SEED)


def validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha!r}")


def validate_score_function(score_function: str) -> None:
    if score_function not in SUPPORTED_SCORE_FUNCTIONS:
        supported = ", ".join(sorted(SUPPORTED_SCORE_FUNCTIONS))
        raise ValueError(
            f"Unknown score function: {score_function!r}. Supported values: {supported}."
        )


def validate_probability_inputs(probs: np.ndarray, labels: np.ndarray) -> None:
    if probs.ndim != 2:
        raise ValueError(f"probs must be a 2D array, got shape {probs.shape}")
    if len(labels) != len(probs):
        raise ValueError(
            f"labels length ({len(labels)}) must match probs rows ({len(probs)})"
        )
    if len(labels) == 0:
        raise ValueError("conformal prediction requires at least one sample")
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("labels contain class indices outside the probability array")


def softmax_nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Score for sample i = 1 - softmax_prob[true_label_i]."""
    validate_probability_inputs(probs, labels)
    return 1.0 - probs[np.arange(len(labels)), labels]


def aps_nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute APS nonconformity scores."""
    validate_probability_inputs(probs, labels)
    n, _ = probs.shape
    scores = np.zeros(n)

    for i in range(n):
        sorted_indices = np.argsort(-probs[i])
        cumsum = 0.0
        for idx in sorted_indices:
            cumsum += probs[i, idx]
            if idx == labels[i]:
                scores[i] = cumsum
                break

    return scores


def nonconformity_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    score_function: str,
) -> np.ndarray:
    validate_score_function(score_function)
    if score_function == "softmax":
        return softmax_nonconformity_scores(probs, labels)
    return aps_nonconformity_scores(probs, labels)


def compute_quantile_threshold(scores: np.ndarray, alpha: float) -> float:
    """Compute the finite-sample conformal quantile threshold."""
    validate_alpha(alpha)
    n = len(scores)
    if n == 0:
        raise ValueError("scores must contain at least one calibration sample")
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return np.quantile(scores, level, method="higher")


def build_prediction_sets_softmax(probs: np.ndarray, q_hat: float) -> list:
    """Build prediction sets using the softmax score threshold."""
    n, num_classes = probs.shape
    threshold = 1.0 - q_hat
    prediction_sets = []

    for i in range(n):
        pset = [int(y) for y in range(num_classes) if probs[i, y] >= threshold]
        if not pset:
            pset = [int(np.argmax(probs[i]))]
        prediction_sets.append(pset)

    return prediction_sets


def build_prediction_sets_aps(probs: np.ndarray, q_hat: float) -> list:
    """Build prediction sets using the APS score threshold."""
    n, _ = probs.shape
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


def build_prediction_sets(
    probs: np.ndarray,
    q_hat: float,
    score_function: str,
) -> list:
    validate_score_function(score_function)
    if score_function == "softmax":
        return build_prediction_sets_softmax(probs, q_hat)
    return build_prediction_sets_aps(probs, q_hat)


def evaluate_prediction_sets(prediction_sets: list, true_labels: np.ndarray) -> dict:
    """Evaluate prediction sets: coverage, count covered, and average set size."""
    n = len(true_labels)
    if n != len(prediction_sets):
        raise ValueError(
            f"prediction_sets length ({len(prediction_sets)}) must match "
            f"true_labels length ({n})"
        )
    if n == 0:
        return {
            "coverage": np.nan,
            "avg_set_size": np.nan,
            "set_sizes": [],
            "n": 0,
            "n_covered": 0,
        }

    n_covered = sum(1 for i in range(n) if true_labels[i] in prediction_sets[i])
    set_sizes = [len(pset) for pset in prediction_sets]

    return {
        "coverage": n_covered / n,
        "avg_set_size": float(np.mean(set_sizes)),
        "set_sizes": set_sizes,
        "n": n,
        "n_covered": n_covered,
    }


def run_marginal_cp(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.10,
    score_function: str = "softmax",
) -> dict:
    """Run standard marginal conformal prediction."""
    validate_alpha(alpha)
    validate_score_function(score_function)
    validate_probability_inputs(cal_probs, cal_labels)
    validate_probability_inputs(test_probs, test_labels)
    if cal_probs.shape[1] != test_probs.shape[1]:
        raise ValueError("calibration and test probabilities must have the same class count")

    cal_scores = nonconformity_scores(cal_probs, cal_labels, score_function)
    q_hat = compute_quantile_threshold(cal_scores, alpha)
    prediction_sets = build_prediction_sets(test_probs, q_hat, score_function)

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
