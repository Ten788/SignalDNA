"""Compare saved NumPy checkpoints and archive validation evidence."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import build_splits
from neural_network import forward, load_parameters, binary_cross_entropy
from evaluation import classification_metrics, roc_auc


def main():
    signals, labels = build_splits(seed=2026)["validation"]
    results = {}

    for name, filename, epoch in (
        ("best_validation_loss", "numpy_best_validation.npz", 48),
        ("final_60", "numpy_final_60.npz", 60),
    ):
        parameters = load_parameters(ROOT / "checkpoints" / filename)
        probabilities, _ = forward(signals, parameters)
        metrics = classification_metrics(probabilities, labels)
        results[name] = {
            "checkpoint": filename,
            "epoch": epoch,
            "validation_loss": float(
                binary_cross_entropy(probabilities, labels.reshape(-1, 1))
            ),
            "validation_auc": roc_auc(probabilities, labels),
            "validation_accuracy": metrics["accuracy"],
            "validation_precision": metrics["precision"],
            "validation_recall": metrics["recall"],
            "validation_f1": metrics["f1"],
        }
        print(
            f"{name}: epoch={epoch} "
            f"loss={results[name]['validation_loss']:.6f} "
            f"accuracy={metrics['accuracy']:.6f} "
            f"auc={results[name]['validation_auc']:.6f} "
            f"f1={metrics['f1']:.6f}"
        )

    comparison = ROOT / "results" / "checkpoint_comparison.json"
    if comparison.exists():
        raise FileExistsError(comparison)
    comparison.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    archive = ROOT / "results" / "research_study.json"
    study = json.loads(archive.read_text(encoding="utf-8-sig"))
    if "checkpoint_comparison" in study:
        raise ValueError("Research archive already contains checkpoint comparison")
    study["checkpoint_comparison"] = results
    archive.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    print("SIGNALDNA-CHECKPOINT-COMPARISON-149: PASS")


if __name__ == "__main__":
    main()