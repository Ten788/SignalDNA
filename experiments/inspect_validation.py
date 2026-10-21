"""Inspect the saved model on the established validation split only."""
from pathlib import Path
from dataset import build_splits
from neural_network import forward, load_parameters
from evaluation import classification_metrics

root = Path(r"C:\Projects\SignalDNA")
parameters = load_parameters(root / "checkpoints/numpy_training_v1.npz")
inputs, labels = build_splits(seed=2026)["validation"]
probabilities, _ = forward(inputs, parameters)
metrics = classification_metrics(probabilities, labels)

for name, value in metrics.items():
    print(f"{name}={value:.6f}" if isinstance(value, float) else f"{name}={value}")
print("SIGNALDNA-VALIDATION-METRICS-129: COMPLETE")