"""Independent pairwise ROC-AUC and confusion-metric verification."""
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from evaluation import classification_metrics, roc_auc

rng = np.random.default_rng(2026)
cases = [
    (np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])),
    (np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])),
    (np.full(8, 0.5), np.array([0, 1] * 4)),
    (np.array([0.0, 0.5, 0.5, 1.0]), np.array([1, 0, 1, 0])),
]
for _ in range(100):
    n = int(rng.integers(4, 65))
    labels = rng.integers(0, 2, size=n)
    labels[0], labels[1] = 0, 1
    scores = rng.choice(np.linspace(0, 1, 21), size=n)

    cases.append((scores, labels))

maximum_auc_difference = 0.0
for scores, labels in cases:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    differences = positives[:, None] - negatives[None, :]
    expected_auc = (
        np.count_nonzero(differences > 0)
        + 0.5 * np.count_nonzero(differences == 0)
    ) / differences.size
    actual_auc = roc_auc(scores, labels)
    maximum_auc_difference = max(maximum_auc_difference, abs(expected_auc - actual_auc))
    assert np.isclose(actual_auc, expected_auc, atol=1e-12, rtol=0)

    predicted = scores >= 0.5
    tp = int(sum(bool(p) and bool(y) for p, y in zip(predicted, labels)))
    tn = int(sum(not bool(p) and not bool(y) for p, y in zip(predicted, labels)))
    fp = int(sum(bool(p) and not bool(y) for p, y in zip(predicted, labels)))
    fn = int(sum(not bool(p) and bool(y) for p, y in zip(predicted, labels)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    expected = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
    actual = classification_metrics(scores, labels)
    for key, value in expected.items():
        assert np.isclose(actual[key], value, atol=1e-12, rtol=0), (key, actual[key], value)

print(f"verified_cases={len(cases)}")
print(f"maximum_auc_difference={maximum_auc_difference:.12g}")
print("SIGNALDNA-INDEPENDENT-METRICS-204: PASS")