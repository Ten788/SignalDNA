"""Test per-waveform normalization with unchanged NumPy training settings."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import build_splits
from neural_network import initialize_parameters, forward, binary_cross_entropy, train_epoch
from evaluation import roc_auc, classification_metrics


def normalize(inputs):
    """Remove each waveform's mean and scale its standard deviation."""
    mean = np.mean(inputs, axis=1, keepdims=True)
    scale = np.std(inputs, axis=1, keepdims=True)
    return (inputs - mean) / np.maximum(scale, 1e-8)


def main():
    started = time.perf_counter()
    splits = build_splits(seed=2026)
    train_x, train_y = splits["train"]
    val_x, val_y = splits["validation"]
    train_x = normalize(train_x)
    val_x = normalize(val_x)
    train_y = train_y.reshape(-1, 1)
    parameters = initialize_parameters(seed=2026)

    for epoch in range(11):
        probabilities, _ = forward(val_x, parameters)
        metrics = classification_metrics(probabilities, val_y)
        auc = roc_auc(probabilities, val_y)
        print(
            f"epoch={epoch:02d} val_acc={metrics['accuracy']:.4f} "
            f"val_auc={auc:.4f} val_f1={metrics['f1']:.4f}",
            flush=True,
        )
        if epoch < 10:
            parameters, _ = train_epoch(
                train_x, train_y, parameters,
                learning_rate=0.01, batch_size=64, seed=3000 + epoch
            )
    print(f"elapsed_seconds={time.perf_counter()-started:.3f}")
    print("SIGNALDNA-NORMALIZATION-135: COMPLETE")


if __name__ == "__main__":
    main()