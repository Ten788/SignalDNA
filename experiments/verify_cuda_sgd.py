"""Verify a CUDA SGD step against the independent NumPy reference."""
import sys
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from neural_network import initialize_parameters, forward, backward, sgd_step

rng = np.random.default_rng(2026)
inputs = rng.normal(size=(16, 1024))
labels = rng.integers(0, 2, size=(16, 1)).astype(np.float64)
parameters = initialize_parameters(seed=2026)
learning_rate = 0.01

_, activations = forward(inputs, parameters)
gradients = backward(activations, parameters, labels)
reference = sgd_step(parameters, gradients, learning_rate)

x = torch.tensor(inputs, dtype=torch.float64, device="cuda")
y = torch.tensor(labels, dtype=torch.float64, device="cuda")
layers = []
for weights, biases in parameters:
    w = torch.tensor(weights, dtype=torch.float64, device="cuda", requires_grad=True)
    b = torch.tensor(biases, dtype=torch.float64, device="cuda", requires_grad=True)
    layers.append((w, b))

z = x
for w, b in layers[:-1]:
    z = torch.relu(z @ w + b)
w, b = layers[-1]
logits = z @ w + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
loss.backward()

with torch.no_grad():
    for w, b in layers:
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
torch.cuda.synchronize()

maximum = 0.0
for index, ((expected_w, expected_b), (w, b)) in enumerate(zip(reference, layers)):
    difference = max(
        float(np.max(np.abs(expected_w - w.detach().cpu().numpy()))),
        float(np.max(np.abs(expected_b - b.detach().cpu().numpy()))),
    )
    maximum = max(maximum, difference)
    print(f"layer_{index}_update_difference={difference:.12g}")
    assert np.allclose(expected_w, w.detach().cpu().numpy(), rtol=1e-10, atol=1e-12)
    assert np.allclose(expected_b, b.detach().cpu().numpy(), rtol=1e-10, atol=1e-12)

print(f"maximum_update_difference={maximum:.12g}")
print("SIGNALDNA-CUDA-SGD-172: PASS")