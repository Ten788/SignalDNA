"""Compare one complete CUDA training epoch against NumPy."""
import sys
import time
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from neural_network import initialize_parameters, train_epoch
from gpu_training import train_epoch_cuda

rng = np.random.default_rng(2026)
inputs = rng.normal(size=(256, 1024))
labels = rng.integers(0, 2, size=(256, 1)).astype(np.float64)
initial = initialize_parameters(seed=2026)

started = time.perf_counter()
cpu, _ = train_epoch(inputs, labels, initial, learning_rate=0.01,
                     batch_size=64, seed=3000)
cpu_seconds = time.perf_counter() - started

started = time.perf_counter()
gpu = train_epoch_cuda(inputs, labels, initial, learning_rate=0.01,
                       batch_size=64, seed=3000)
gpu_seconds = time.perf_counter() - started

maximum = 0.0
for index, ((cw, cb), (gw, gb)) in enumerate(zip(cpu, gpu)):
    difference = max(
        float(np.max(np.abs(cw - gw))),
        float(np.max(np.abs(cb - gb))),
    )
    maximum = max(maximum, difference)
    print(f"layer_{index}_parameter_difference={difference:.12g}")
    assert np.allclose(cw, gw, rtol=1e-9, atol=1e-11)
    assert np.allclose(cb, gb, rtol=1e-9, atol=1e-11)

print(f"maximum_parameter_difference={maximum:.12g}")
print(f"cpu_seconds={cpu_seconds:.6f}")
print(f"gpu_seconds={gpu_seconds:.6f}")
print("SIGNALDNA-CUDA-EPOCH-176: PASS")