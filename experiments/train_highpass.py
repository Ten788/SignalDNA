"""Compare high-pass residual inputs against raw-waveform training."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import build_splits
from neural_network import initialize_parameters, forward, train_epoch
from evaluation import roc_auc, classification_metrics


def highpass(inputs, window=129):
    """Subtract a centered moving average with edge-value padding."""
    left = window // 2
    right = window - left - 1
    padded = np.pad(inputs, ((0, 0), (left, right)), mode="edge")
    cumulative = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)))
    smooth = (cumulative[:, window:] - cumulative[:, :-window]) / window
    return inputs - smooth


def main():
    started = time.perf_counter()
    splits = build_splits(seed=2026)
    train_x, train_y = splits["train"]
    val_x, val_y = splits["validation"]
    train_x = highpass(train_x)
    val_x = highpass(val_x)
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
    print("SIGNALDNA-HIGHPASS-EXPERIMENT: COMPLETE")


if __name__ == "__main__":
    main()