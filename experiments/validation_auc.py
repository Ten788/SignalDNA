"""Measure threshold-independent ranking on the validation split."""
from pathlib import Path
from dataset import build_splits
from neural_network import forward, load_parameters
from evaluation import classification_metrics, roc_auc

root = Path(r"C:\Projects\SignalDNA")
parameters = load_parameters(root / "checkpoints/numpy_training_v1.npz")
inputs, labels = build_splits(seed=2026)["validation"]
probabilities, _ = forward(inputs, parameters)
auc = roc_auc(probabilities, labels)
metrics = classification_metrics(probabilities, labels)
print(f"Validation_ROC_AUC={auc:.6f}")
print(f"Validation_accuracy={metrics['accuracy']:.6f}")
print(f"Validation_precision={metrics['precision']:.6f}")
print(f"Validation_recall={metrics['recall']:.6f}")
print(f"Validation_F1={metrics['f1']:.6f}")
print("SIGNALDNA-VALIDATION-AUC-134: COMPLETE")