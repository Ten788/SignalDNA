"""Binary-classification metrics for SignalDNA experiments.

Positive labels represent waveforms containing an injected transient.
Metrics use a fixed decision threshold and require aligned binary labels.
"""
import numpy as np


def classification_metrics(probabilities, labels, threshold=0.5):
    """Return confusion counts, accuracy, precision, recall, and F1.

    Undefined precision, recall, or F1 is reported as zero when its
    denominator is zero. Inputs must be nonempty and finite.
    """
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1)
    if p.size == 0 or p.shape != y.shape:
        raise ValueError("Probabilities and labels must align and be nonempty")
    if not np.isfinite(p).all() or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels must be binary")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Threshold must be strictly between zero and one")
    predicted = p >= threshold
    positive = y == 1
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / p.size,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
def roc_auc(probabilities, labels):
    """Compute ROC-AUC using average ranks for tied prediction scores.

    AUC is the probability that a randomly selected positive example
    outranks a negative one, with ties contributing half credit.
    Both classes must be present.
    """
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels).reshape(-1)
    if scores.size == 0 or scores.shape != targets.shape:
        raise ValueError("Scores and labels must align and be nonempty")
    if not np.isfinite(scores).all() or not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError("Scores must be finite probabilities")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("Labels must be binary")
    positives = int(np.sum(targets == 1))
    negatives = scores.size - positives
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires both classes")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((np.sum(ranks[targets == 1]) - positives * (positives + 1) / 2) / (positives * negatives))
