"""CUDA training using the NumPy reference model's SGD procedure."""
import numpy as np
import torch


def train_epoch_cuda(inputs, labels, parameters, learning_rate=0.01,
                     batch_size=64, seed=2026):
    """Return NumPy-compatible parameters after one CUDA training epoch."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 1024 or x.shape[0] == 0:
        raise ValueError("Invalid input shape")
    if y.shape != (x.shape[0], 1):
        raise ValueError("Invalid label shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite training data")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels must be binary")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("Invalid batch size")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid learning rate")

    device = "cuda"
    tx = torch.tensor(x, dtype=torch.float64, device=device)
    ty = torch.tensor(y, dtype=torch.float64, device=device)
    layers = [
        (
            torch.tensor(w, dtype=torch.float64, device=device, requires_grad=True),
            torch.tensor(b, dtype=torch.float64, device=device, requires_grad=True),
        )
        for w, b in parameters
    ]

    order = np.random.default_rng(seed).permutation(x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        indices = order[start:start + batch_size]
        z = tx[indices]
        target = ty[indices]
        for w, b in layers[:-1]:
            z = torch.relu(z @ w + b)
        w, b = layers[-1]
        logits = z @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        with torch.no_grad():
            for w, b in layers:
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad
                w.grad = None
                b.grad = None

    torch.cuda.synchronize()
    return [
        (w.detach().cpu().numpy(), b.detach().cpu().numpy())
        for w, b in layers
    ]