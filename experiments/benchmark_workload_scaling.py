"""Compare NumPy CPU and CUDA epoch times across dataset sizes."""
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
all_x, all_y = splits["train"]
sizes = (256, 1024, 4096, 8400)
warmups = 2
repeats = 5
results = []

for size in sizes:
    x = all_x[:size]
    y = all_y[:size].reshape(-1, 1)
    row = {"examples": size, "batch_size": 64}
    for name, function in (
        ("numpy_cpu", train_epoch),
        ("pytorch_cuda", train_epoch_cuda),
    ):
        samples = []
        for index in range(warmups + repeats):
            parameters = initialize_parameters(seed=2026)
            started = time.perf_counter()
            function(
                x, y, parameters,
                learning_rate=0.01,
                batch_size=64,
                seed=3000,
            )
            elapsed = time.perf_counter() - started
            if index >= warmups:
                samples.append(elapsed)
        row[name + "_seconds"] = samples
        row[name + "_median_seconds"] = statistics.median(samples)
    row["gpu_over_cpu_ratio"] = (
        row["pytorch_cuda_median_seconds"] /
        row["numpy_cpu_median_seconds"]
    )
    results.append(row)
    print(
        f"examples={size} "
        f"cpu_ms={1000 * row['numpy_cpu_median_seconds']:.3f} "
        f"gpu_ms={1000 * row['pytorch_cuda_median_seconds']:.3f} "
        f"gpu_over_cpu={row['gpu_over_cpu_ratio']:.3f}"
    )

report = {
    "schema_version": 1,
    "benchmark_type": "cpu_cuda_dataset_size_scaling",
    "precision": "float64",
    "batch_size": 64,
    "warmups": warmups,
    "repeats": repeats,
    "includes_gpu_setup_and_output_transfer": True,
    "backend_order": ["numpy_cpu", "pytorch_cuda"],
    "results": results,
}
destination = root / "results" / "cpu_gpu_workload_scaling_v1.json"
with destination.open("x", encoding="utf-8") as output:
    json.dump(report, output, indent=2)
    output.write("\n")
print("SIGNALDNA-WORKLOAD-SCALING-195: PASS")