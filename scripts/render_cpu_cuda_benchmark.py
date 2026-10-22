import json
import math
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.loads(Path("results/cpu_gpu_benchmark_v1.json").read_text(encoding="utf-8"))
measurements = data["measurements_seconds"]
count = int(data["measured_repeats"])
assert count == 5, "Unexpected benchmark repeat count"

series = []
for key, label, color in [
    ("numpy_cpu", "NumPy CPU", "#1764A0"),
    ("pytorch_cuda", "PyTorch CUDA", "#E08025"),
]:
    values = measurements[key]
    assert isinstance(values, list) and len(values) == count
    assert all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values)
    series.append((label, color, [v * 1000 for v in values]))

cpu = statistics.median(series[0][2])
gpu = statistics.median(series[1][2])
ratio = gpu / cpu
assert abs(ratio - float(data["median_gpu_over_cpu_time_ratio"])) < 0.02

fig, ax = plt.subplots(figsize=(9, 4.8))
for y, (label, color, values) in enumerate(series):
    median = statistics.median(values)
    ax.barh(y, median, color=color, height=0.48, alpha=0.85)
    ax.scatter(values, [y] * len(values), marker="D", color="#222222", s=30, zorder=5)
    ax.text(max(values) + 10, y, f"{median:.1f} ms", va="center", fontsize=11, weight="bold")

ax.set_yticks([0, 1], [s[0] for s in series])
ax.invert_yaxis()
ax.set_xlim(0, max(max(s[2]) for s in series) * 1.40)
ax.set_xlabel("Measured epoch duration (milliseconds)")
ax.set_title("SignalDNA | CPU vs. CUDA Epoch Performance", loc="left", weight="bold", pad=16)
ax.set_axisbelow(True)
ax.grid(axis="x", alpha=0.2)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.text(0.01, -0.25, f"CUDA / CPU median time: {ratio:.2f}x | Five measured runs per backend",
        transform=ax.transAxes, fontsize=10)
fig.tight_layout(pad=2)

out = Path("docs/figures")
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"cpu_cuda_epoch_benchmark.{ext}", dpi=240, bbox_inches="tight")
plt.close(fig)

print(f"CPU median: {cpu:.3f} ms")
print(f"CUDA median: {gpu:.3f} ms")
print(f"CUDA/CPU ratio: {ratio:.3f}x")
print("SIGNALDNA-CPU-CUDA-FIGURE-236: PASS")
