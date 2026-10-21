"""Measure CUDA epoch setup, mini-batch training, and output transfer."""
import sys
import time
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from dataset import build_splits
from neural_network import initialize_parameters

splits = build_splits(seed=2026)
x, y = splits["train"]
initial = initialize_parameters(seed=2026)
device = "cuda"

def measure():
    torch.cuda.synchronize()
    start = time.perf_counter()
    tx = torch.tensor(x, dtype=torch.float64, device=device)
    ty = torch.tensor(y.reshape(-1, 1), dtype=torch.float64, device=device)
    layers = [
        (
            torch.tensor(w, dtype=torch.float64, device=device, requires_grad=True),
            torch.tensor(b, dtype=torch.float64, device=device, requires_grad=True),
        )
        for w, b in initial
    ]
    torch.cuda.synchronize()
    setup = time.perf_counter() - start

    order = np.random.default_rng(3000).permutation(len(y))
    start = time.perf_counter()
    for offset in range(0, len(y), 64):
        indices = order[offset:offset + 64]
        z = tx[indices]
        target = ty[indices]
        for w, b in layers[:-1]:
            z = torch.relu(z @ w + b)
        w, b = layers[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, target)
        loss.backward()
        with torch.no_grad():
            for w, b in layers:
                w -= 0.01 * w.grad
                b -= 0.01 * b.grad
                w.grad = None
                b.grad = None
    torch.cuda.synchronize()
    training = time.perf_counter() - start

    start = time.perf_counter()
    result = [(w.detach().cpu().numpy(), b.detach().cpu().numpy()) for w, b in layers]
    torch.cuda.synchronize()
    transfer = time.perf_counter() - start
    assert all(np.isfinite(w).all() and np.isfinite(b).all() for w, b in result)
    return setup, training, transfer

for _ in range(2):
    measure()
samples = np.array([measure() for _ in range(5)])
for index, name in enumerate(("setup_seconds", "training_seconds", "output_transfer_seconds")):
    print(f"{name}_median={np.median(samples[:, index]):.6f}")
print(f"total_seconds_median={np.median(samples.sum(axis=1)):.6f}")
print("SIGNALDNA-CUDA-PROFILE-184: PASS")