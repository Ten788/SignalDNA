"""Single-use held-out evaluation of the preapproved SignalDNA primary model."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from dataset import DatasetConfig, build_dataset
from neural_network import load_parameters, forward, binary_cross_entropy
from evaluation import classification_metrics, roc_auc

expected_hash = "846137EE4333F7FAC7990E29FD6F7D0EAA59E6D1153F9A6BF6775F9C21A72975"
checkpoint = root / "checkpoints" / "numpy_final_60.npz"
pretest_path = root / "results" / "pretest_checkpoint_verification_v1.json"
protocol = root / "docs" / "FINAL_EVALUATION_PROTOCOL.md"
lock = root / "results" / "heldout_evaluation_attempt_v1.json"
destination = root / "results" / "heldout_final_primary_v1.json"

assert protocol.is_file(), "Locked protocol missing"
assert not destination.exists(), "Final result already exists"
assert not lock.exists(), "Previous evaluation attempt exists"
actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
assert actual_hash == expected_hash, "Primary checkpoint identity mismatch"

pretest = json.loads(pretest_path.read_text(encoding="utf-8"))
assert pretest["test_set_accessed"] is False
assert pretest["classification_threshold"] == 0.5
primary = [c for c in pretest["candidates"] if c["role"] == "primary"]
assert len(primary) == 1 and primary[0]["epoch"] == 60
assert primary[0]["sha256"].upper() == expected_hash
assert primary[0]["checkpoint"] == "checkpoints/numpy_final_60.npz"

parameters = load_parameters(checkpoint)
attempt = {
    "purpose": "One authorized final held-out evaluation",
    "authorized_by": "Project owner in ChatGPT conversation, 2026-10-21",
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint_sha256": expected_hash,
    "status": "attempt_started_no_automatic_retry",
}
with lock.open("x", encoding="utf-8") as output:
    json.dump(attempt, output, indent=2)
    output.write("\n")

# The held-out dataset is constructed only after the attempt lock exists.
child = np.random.SeedSequence(2026).spawn(3)[2]
test_seed = int(child.generate_state(1, dtype=np.uint32)[0])
x, y = build_dataset(DatasetConfig(count=1800, seed=test_seed))
assert x.shape == (1800, 1024) and y.shape == (1800,)
assert int(np.sum(y)) == 900

probabilities, _ = forward(x, parameters)
metrics = classification_metrics(probabilities, y, threshold=0.5)
metrics["roc_auc"] = float(roc_auc(probabilities, y))
metrics["binary_cross_entropy"] = float(
    binary_cross_entropy(probabilities, y.reshape(-1, 1))
)
assert sum(metrics[k] for k in ("tp", "tn", "fp", "fn")) == 1800

validation = primary[0]["validation_metrics"]
report = {
    "schema_version": 1,
    "evaluation_type": "final_heldout_primary_single_attempt",
    "checkpoint": "checkpoints/numpy_final_60.npz",
    "checkpoint_sha256": expected_hash,
    "selected_epoch": 60,
    "selection_basis": "maximum validation ROC-AUC; earliest tie",
    "threshold": 0.5,
    "test_count": 1800,
    "positive_count": 900,
    "negative_count": 900,
    "parent_seed": 2026,
    "test_child_seed": test_seed,
    "validation_metrics": validation,
    "test_metrics": metrics,
    "test_minus_validation": {
        "accuracy": metrics["accuracy"] - validation["validation_accuracy"],
        "roc_auc": metrics["roc_auc"] - validation["validation_auc"],
        "loss": metrics["binary_cross_entropy"] - validation["validation_loss"],
    },
    "test_informed_tuning_performed": False,
    "attempt_lock": "results/heldout_evaluation_attempt_v1.json",
}
with destination.open("x", encoding="utf-8") as output:
    json.dump(report, output, indent=2, allow_nan=False)
    output.write("\n")

for key, value in metrics.items():
    print(f"{key}={value}")
print(f"result_file={destination.relative_to(root)}")
print("SIGNALDNA-FINAL-HELDOUT-209: PASS")