"""Explore operating thresholds on validation data, without retraining."""
import numpy as np
from pathlib import Path
from dataset import build_splits
from neural_network import forward, load_parameters
from evaluation import classification_metrics

root = Path(r"C:\Projects\SignalDNA")
parameters = load_parameters(root / "checkpoints/numpy_training_v1.npz")
inputs, labels = build_splits(seed=2026)["validation"]
probabilities, _ = forward(inputs, parameters)

best = None
for threshold in np.arange(0.1, 0.901, 0.05):
    metrics = classification_metrics(probabilities, labels, float(threshold))
    print(
        f"threshold={threshold:.2f} accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )
    if best is None or metrics["f1"] > best[1]["f1"]:
        best = (threshold, metrics)

print(f"Best_validation_F1_threshold={best[0]:.2f}")
print(f"Best_validation_F1={best[1]['f1']:.4f}")
print("SIGNALDNA-THRESHOLD-SWEEP-130: COMPLETE")