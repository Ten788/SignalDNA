"""Track training and validation generalization over 60 epochs."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import build_splits
from neural_network import initialize_parameters, forward, train_epoch, binary_cross_entropy
from evaluation import roc_auc, classification_metrics
from experiment_logger import ExperimentLogger


def evaluate(inputs, labels, parameters):
    """Calculate accuracy, loss, and ROC-AUC for one split."""
    probabilities, _ = forward(inputs, parameters)
    metrics = classification_metrics(probabilities, labels)
    return (
        metrics["accuracy"],
        binary_cross_entropy(probabilities, labels.reshape(-1, 1)),
        roc_auc(probabilities, labels),
    )


def main():
    started = time.perf_counter()
    splits = build_splits(seed=2026)
    train_x, train_y = splits["train"]
    val_x, val_y = splits["validation"]
    parameters = initialize_parameters(seed=2026)
    logger = ExperimentLogger("numpy_cpu_60_epochs", {"seed": 2026, "epochs": 60, "batch_size": 64, "learning_rate": 0.01, "train_count": 8400, "validation_count": 1800, "architecture": [1024, 64, 32, 1]})
    best = (float("inf"), None)

    for epoch in range(61):
        train_acc, train_loss, _ = evaluate(train_x, train_y, parameters)
        val_acc, val_loss, val_auc = evaluate(val_x, val_y, parameters)
        logger.log_epoch(epoch, training_accuracy=float(train_acc), training_loss=float(train_loss), validation_accuracy=float(val_acc), validation_loss=float(val_loss), validation_auc=float(val_auc))
        if val_loss < best[0]:
            best = (val_loss, epoch)
        if epoch % 5 == 0 or epoch == 60:
            print(
                f"epoch={epoch:02d} train_acc={train_acc:.4f} "
                f"val_acc={val_acc:.4f} val_loss={val_loss:.4f} "
                f"val_auc={val_auc:.4f} gap={train_acc-val_acc:.4f}",
                flush=True,
            )
        if epoch < 60:
            parameters, _ = train_epoch(
                train_x, train_y.reshape(-1, 1), parameters,
                learning_rate=0.01, batch_size=64, seed=3000 + epoch
            )

    print(f"best_validation_loss_epoch={best[1]}")
    logger.save(ROOT / "results" / "numpy_cpu_60_epochs_logged.json")
    print(f"elapsed_seconds={time.perf_counter()-started:.3f}")
    print("SIGNALDNA-LONGER-TRAINING: COMPLETE")


if __name__ == "__main__":
    main()
