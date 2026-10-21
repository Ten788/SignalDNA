"""Compare CUDA autograd with independent NumPy backpropagation."""
import sys
from pathlib import Path
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from neural_network import initialize_parameters, forward, backward

rng = np.random.default_rng(2026)
inputs = rng.normal(size=(16, 1024))
labels = rng.integers(0, 2, size=(16, 1)).astype(np.float64)
parameters = initialize_parameters(seed=2026)

_, activations = forward(inputs, parameters)
reference = backward(activations, parameters, labels)

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
torch.cuda.synchronize()

maximum = 0.0
for index, ((dw, db), (w, b)) in enumerate(zip(reference, layers)):
    difference = max(
        float(np.max(np.abs(dw - w.grad.cpu().numpy()))),
        float(np.max(np.abs(db - b.grad.cpu().numpy()))),
    )
    maximum = max(maximum, difference)
    print(f"layer_{index}_gradient_difference={difference:.12g}")
    assert np.allclose(dw, w.grad.cpu().numpy(), rtol=1e-9, atol=1e-11)
    assert np.allclose(db, b.grad.cpu().numpy(), rtol=1e-9, atol=1e-11)

print(f"maximum_gradient_difference={maximum:.12g}")
print("SIGNALDNA-CUDA-GRADIENTS-171: PASS")