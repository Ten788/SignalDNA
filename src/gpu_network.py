"""CUDA forward-pass counterpart to the NumPy reference network."""
import numpy as np
import torch


def forward_cuda(inputs, parameters):
    """Return sigmoid probabilities using identical float64 weights on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    x = torch.as_tensor(np.asarray(inputs, dtype=np.float64), device="cuda")
    for weights, biases in parameters[:-1]:
        w = torch.as_tensor(weights, dtype=torch.float64, device="cuda")
        b = torch.as_tensor(biases, dtype=torch.float64, device="cuda")
        x = torch.relu(x @ w + b)
    weights, biases = parameters[-1]
    w = torch.as_tensor(weights, dtype=torch.float64, device="cuda")
    b = torch.as_tensor(biases, dtype=torch.float64, device="cuda")
    return torch.sigmoid(x @ w + b).cpu().numpy()