"""Reproducible NumPy training experiment with held-out validation.

The test split is intentionally excluded from model development.
Metrics are saved for comparison across future experiments.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import build_splits
from neural_network import (
    binary_cross_entropy, forward, initialize_parameters, save_parameters, train_epoch
)


def evaluate(inputs, labels, parameters):
    """Return binary cross-entropy and threshold-0.5 accuracy."""
    probabilities, _ = forward(inputs, parameters)
    return {
        "loss": float(binary_cross_entropy(probabilities, labels)),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
    }


def main():
    """Train with fixed seeds and persist epoch-level metrics."""
    started = time.perf_counter()
    splits = build_splits(seed=2026)
    train_x, train_y = splits["train"]
    val_x, val_y = splits["validation"]
    train_y = train_y.reshape(-1, 1)
    val_y = val_y.reshape(-1, 1)

    parameters = initialize_parameters(seed=2026)
    history = []
    for epoch in range(11):
        record = {
            "epoch": epoch,
            "train": evaluate(train_x, train_y, parameters),
            "validation": evaluate(val_x, val_y, parameters),
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} "
            f"train_acc={record['train']['accuracy']:.3f} "
            f"val_acc={record['validation']['accuracy']:.3f} "
            f"val_loss={record['validation']['loss']:.4f}",
            flush=True,
        )
        if epoch < 10:
            parameters, _ = train_epoch(
                train_x, train_y, parameters,
                learning_rate=0.01, batch_size=64, seed=3000 + epoch
            )

    output = {
        "config": {
            "dataset_seed": 2026,
            "model_seed": 2026,
            "epochs": 10,
            "learning_rate": 0.01,
            "batch_size": 64,
        },
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    checkpoint = ROOT / "checkpoints" / "numpy_training_v1.npz"
    save_parameters(checkpoint, parameters)
    print(f"checkpoint={checkpoint}")
    destination = ROOT / "results" / "numpy_training_v1.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"results={destination}")
    print(f"elapsed_seconds={output['elapsed_seconds']:.3f}")


if __name__ == "__main__":
    main()