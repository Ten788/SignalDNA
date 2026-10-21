"""Check GPU predictions against the independent NumPy reference."""
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from neural_network import initialize_parameters, forward
from gpu_network import forward_cuda

rng = np.random.default_rng(2026)
inputs = rng.normal(size=(32, 1024))
parameters = initialize_parameters(seed=2026)
expected, _ = forward(inputs, parameters)
actual = forward_cuda(inputs, parameters)
difference = float(np.max(np.abs(expected - actual)))
print(f"maximum_absolute_difference={difference:.12g}")
assert actual.shape == expected.shape
assert np.allclose(expected, actual, rtol=1e-10, atol=1e-12)
print("SIGNALDNA-CUDA-FORWARD-170: PASS")