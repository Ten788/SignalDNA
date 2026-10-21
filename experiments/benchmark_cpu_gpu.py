"""Exploratory warmed-up CPU/CUDA training benchmark."""
import json
import statistics
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from dataset import build_splits
from neural_network import initialize_parameters, train_epoch
from gpu_training import train_epoch_cuda

splits = build_splits(seed=2026)
x, y = splits["train"]
labels = y.reshape(-1, 1)
repeats = 5
warmups = 2
measurements = {}

for backend, function in (("numpy_cpu", train_epoch), ("pytorch_cuda", train_epoch_cuda)):
    times = []
    for index in range(warmups + repeats):
        parameters = initialize_parameters(seed=2026)
        started = time.perf_counter()
        function(x, labels, parameters, learning_rate=0.01,
                 batch_size=64, seed=3000)
        elapsed = time.perf_counter() - started
        if index >= warmups:
            times.append(elapsed)
    measurements[backend] = times
    print(f"{backend}_seconds={times}")
    print(f"{backend}_median_seconds={statistics.median(times):.6f}")

cpu = statistics.median(measurements["numpy_cpu"])
gpu = statistics.median(measurements["pytorch_cuda"])
print(f"gpu_over_cpu_time_ratio={gpu / cpu:.3f}")

report = {
    "schema_version": 1,
    "benchmark_type": "exploratory_training_epoch",
    "warmups": warmups,
    "measured_repeats": repeats,
    "examples": len(y),
    "batch_size": 64,
    "precision": "float64",
    "includes_parameter_initialization": False,
    "includes_gpu_tensor_allocation_and_transfers": True,
    "measurements_seconds": measurements,
    "median_gpu_over_cpu_time_ratio": gpu / cpu,
}
destination = root / "results" / "cpu_gpu_benchmark_v1.json"
with destination.open("x", encoding="utf-8") as output:
    json.dump(report, output, indent=2)
    output.write("\n")
print("SIGNALDNA-CPU-GPU-BENCHMARK-183: PASS")