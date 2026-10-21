"""Compare CUDA float64 and float32 SGD speed and numerical drift."""
import json
import statistics
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
order = np.random.default_rng(3000).permutation(len(y))

def epoch(dtype):
    tx = torch.tensor(x, dtype=dtype, device="cuda")
    ty = torch.tensor(y.reshape(-1, 1), dtype=dtype, device="cuda")
    layers = [
        (
            torch.tensor(w, dtype=dtype, device="cuda", requires_grad=True),
            torch.tensor(b, dtype=dtype, device="cuda", requires_grad=True),
        )
        for w, b in initial
    ]
    for start in range(0, len(y), 64):
        indices = order[start:start + 64]
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
    return [(w.detach().cpu().numpy(), b.detach().cpu().numpy()) for w, b in layers]

measurements = {"float64": [], "float32": []}
outputs = {}
configurations = (("float64", torch.float64), ("float32", torch.float32))
for round_index in range(7):
    sequence = configurations if round_index % 2 == 0 else configurations[::-1]
    for name, dtype in sequence:
        started = time.perf_counter()
        updated = epoch(dtype)
        elapsed = time.perf_counter() - started
        if round_index >= 2:
            measurements[name].append(elapsed)
        outputs[name] = updated

for name in measurements:
    print(f"{name}_seconds={measurements[name]}")
    print(f"{name}_median_seconds={statistics.median(measurements[name]):.6f}")

maximum = max(
    max(float(np.max(np.abs(a - b))) for a, b in zip(pair64, pair32))
    for pair64, pair32 in zip(outputs["float64"], outputs["float32"])
)
print(f"float32_vs_float64_max_parameter_difference={maximum:.12g}")
assert np.isfinite(maximum)
ratio = statistics.median(measurements["float64"]) / statistics.median(measurements["float32"])
print(f"float32_speedup_over_float64={ratio:.3f}")

report = {
    "schema_version": 1,
    "benchmark_type": "cuda_precision_alternating_replication",
    "batch_size": 64,
    "examples": len(y),
    "warmup_rounds": 2,
    "measured_rounds": 5,
    "alternating_order": True,
    "times_seconds": measurements,
    "float32_speedup_over_float64": ratio,
    "maximum_parameter_difference": maximum,
}
destination = root / "results" / "cuda_precision_alternating_v1.json"
with destination.open("x", encoding="utf-8") as output:
    json.dump(report, output, indent=2)
    output.write("\n")
print("SIGNALDNA-CUDA-PRECISION-REPLICATION-187: PASS")