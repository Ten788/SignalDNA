"""Measure one full-dataset CUDA epoch and compare NumPy predictions."""
import sys
import time
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from dataset import build_splits
from neural_network import initialize_parameters, forward
from gpu_training import train_epoch_cuda

splits = build_splits(seed=2026)
train_x, train_y = splits["train"]
val_x, val_y = splits["validation"]
initial = initialize_parameters(seed=2026)

started = time.perf_counter()
updated = train_epoch_cuda(
    train_x, train_y.reshape(-1, 1), initial,
    learning_rate=0.01, batch_size=64, seed=3000
)
elapsed = time.perf_counter() - started

initial_predictions, _ = forward(val_x, initial)
updated_predictions, _ = forward(val_x, updated)
initial_accuracy = float(np.mean((initial_predictions[:, 0] >= 0.5) == val_y))
updated_accuracy = float(np.mean((updated_predictions[:, 0] >= 0.5) == val_y))

print(f"training_examples={len(train_y)}")
print(f"validation_examples={len(val_y)}")
print(f"gpu_epoch_seconds={elapsed:.6f}")
print(f"initial_validation_accuracy={initial_accuracy:.6f}")
print(f"updated_validation_accuracy={updated_accuracy:.6f}")
assert all(np.isfinite(w).all() and np.isfinite(b).all() for w, b in updated)
print("SIGNALDNA-GPU-DATASET-PROBE-177: PASS")