"""Measure CUDA training-epoch scaling with mini-batch size."""
import json
import statistics
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from dataset import build_splits
from neural_network import initialize_parameters
from gpu_training import train_epoch_cuda

splits = build_splits(seed=2026)
x, y = splits["train"]
labels = y.reshape(-1, 1)
sizes = (32, 64, 128, 256, 512)
warmups = 2
repeats = 5
results = []

for size in sizes:
    samples = []
    for index in range(warmups + repeats):
        parameters = initialize_parameters(seed=2026)
        started = time.perf_counter()
        updated = train_epoch_cuda(
            x, labels, parameters,
            learning_rate=0.01, batch_size=size, seed=3000
        )
        elapsed = time.perf_counter() - started
        if index >= warmups:
            samples.append(elapsed)
    median = statistics.median(samples)
    results.append({
        "batch_size": size,
        "steps_per_epoch": (len(y) + size - 1) // size,
        "times_seconds": samples,
        "median_seconds": median,
    })
    print(f"batch={size} steps={results[-1]['steps_per_epoch']} median_seconds={median:.6f}")

baseline = next(row["median_seconds"] for row in results if row["batch_size"] == 64)
for row in results:
    row["speedup_vs_batch64"] = baseline / row["median_seconds"]
    print(f"batch={row['batch_size']} speedup_vs_batch64={row['speedup_vs_batch64']:.3f}")

report = {
    "schema_version": 1,
    "benchmark_type": "cuda_batch_size_scaling",
    "precision": "float64",
    "examples": len(y),
    "warmups": warmups,
    "repeats": repeats,
    "includes_tensor_setup_and_output_transfer": True,
    "results": results,
}
destination = root / "results" / "cuda_batch_scaling_v1.json"
with destination.open("x", encoding="utf-8") as output:
    json.dump(report, output, indent=2)
    output.write("\n")
print("SIGNALDNA-CUDA-BATCH-SCALING-185: PASS")