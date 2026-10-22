import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

source = Path("results/numpy_cpu_60_epochs_logged.json")
records = json.loads(source.read_text(encoding="utf-8"))["epochs"]
assert len(records) == 61, "Expected 61 archived checkpoints"
epochs = [int(r["epoch"]) for r in records]
assert epochs == sorted(set(epochs)), "Invalid epoch ordering"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "savefig.facecolor": "white",
})

output = Path("docs/figures")
output.mkdir(parents=True, exist_ok=True)

for name, field, ylabel, scale in [
    ("learning_accuracy", "accuracy", "Accuracy (%)", 100),
    ("learning_loss", "loss", "Binary cross-entropy loss", 1),
]:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for prefix, label, color in [
        ("training", "Training", "#1764A0"),
        ("validation", "Validation", "#E08025"),
    ]:
        values = [float(r[prefix + "_" + field]) * scale for r in records]
        assert all(math.isfinite(v) for v in values)
        ax.plot(epochs, values, color=color, linewidth=2.3,
                label=f"{label} (final: {values[-1]:.2f})")
        ax.scatter([epochs[-1]], [values[-1]], color=color, s=35, zorder=4)
    ax.set_title("SignalDNA | " + ("Classification Accuracy" if scale == 100 else "Training Loss"),
                 loc="left", weight="bold", pad=15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    ax.set_xlim(min(epochs), max(epochs))
    if scale == 100:
        ax.set_ylim(0, 105)
    fig.tight_layout(pad=2)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"{name}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"CREATED: {name}.pdf / {name}.png")

print("SIGNALDNA-PUBLICATION-CURVES-235: PASS")
