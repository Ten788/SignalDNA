"""Compare complete CPU and CUDA learning curves without test-set access."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
results = root / "results"
cpu = json.loads((results / "numpy_cpu_60_epochs_logged.json").read_text())
gpu = json.loads((results / "pytorch_cuda_60_epochs_logged.json").read_text())

metrics = (
    "training_accuracy",
    "training_loss",
    "validation_accuracy",
    "validation_loss",
    "validation_auc",
)
assert len(cpu["epochs"]) == len(gpu["epochs"]) == 61
maximum = {metric: 0.0 for metric in metrics}
for index, (a, b) in enumerate(zip(cpu["epochs"], gpu["epochs"])):
    assert a["epoch"] == b["epoch"] == index
    for metric in metrics:
        difference = abs(a[metric] - b[metric])
        maximum[metric] = max(maximum[metric], difference)

for metric, difference in maximum.items():
    print(f"{metric}_max_difference={difference:.12g}")
    assert difference <= 1e-9, f"CPU/GPU disagreement: {metric}"

report = {
    "schema_version": 1,
    "cpu_record": "results/numpy_cpu_60_epochs_logged.json",
    "gpu_record": "results/pytorch_cuda_60_epochs_logged.json",
    "epochs_compared": 61,
    "maximum_absolute_differences": maximum,
    "acceptance_tolerance": 1e-9,
    "result": "PASS",
}
destination = results / "cpu_gpu_curve_comparison.json"
with destination.open("x", encoding="utf-8") as output:
    json.dump(report, output, indent=2)
    output.write("\n")
print("SIGNALDNA-CPU-GPU-CURVES-181: PASS")