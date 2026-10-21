"""Verify locked checkpoint identities using validation data only."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from dataset import DatasetConfig, build_dataset
from neural_network import load_parameters, forward, binary_cross_entropy
from evaluation import classification_metrics, roc_auc

expected_hashes = {
    "primary": "846137EE4333F7FAC7990E29FD6F7D0EAA59E6D1153F9A6BF6775F9C21A72975",
    "secondary": "1D1D72CD6D68659AE9DD4001AC21842A59921CDF9807910966B03EC2B938A243",
}

curve = json.loads(
    (root / "results" / "numpy_cpu_60_epochs_logged.json").read_text()
)
epochs = {int(row["epoch"]): row for row in curve["epochs"]}

# Derive the validation stream without constructing the test dataset.
children = np.random.SeedSequence(2026).spawn(3)
validation_seed = int(
    children[1].generate_state(1, dtype=np.uint32)[0]
)
x, y = build_dataset(DatasetConfig(count=1800, seed=validation_seed))

candidates = [
    ("primary", 60, "numpy_final_60.npz"),
    ("secondary", 48, "numpy_best_validation.npz"),
]
report = {
    "schema_version": 1,
    "purpose": "Locked pre-test checkpoint validation",
    "test_set_accessed": False,
    "selection_rule": "Primary: maximum validation ROC-AUC, earliest tie",
    "classification_threshold": 0.5,
    "candidates": [],
}

for role, epoch, filename in candidates:
    path = root / "checkpoints" / filename
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    assert actual_hash == expected_hashes[role], (
        role, "Checkpoint hash mismatch"
    )

    parameters = load_parameters(path)
    probabilities, _ = forward(x, parameters)
    metrics = classification_metrics(probabilities, y, threshold=0.5)

    observed = {
        "validation_accuracy": float(metrics["accuracy"]),
        "validation_loss": float(
            binary_cross_entropy(probabilities, y.reshape(-1, 1))
        ),
        "validation_auc": float(roc_auc(probabilities, y)),
    }
    recorded = epochs[epoch]

    for metric, value in observed.items():
        assert np.isclose(
            value, recorded[metric], atol=1e-12, rtol=0
        ), (role, metric, value, recorded[metric])

    report["candidates"].append({
        "role": role,
        "epoch": epoch,
        "checkpoint": "checkpoints/" + filename,
        "sha256": actual_hash,
        "validation_metrics": observed,
    })
    print(
        f"{role} epoch={epoch} "
        f"accuracy={observed['validation_accuracy']:.6f} "
        f"loss={observed['validation_loss']:.6f} "
        f"auc={observed['validation_auc']:.6f} PASS"
    )

destination = root / "results" / "pretest_checkpoint_verification_v1.json"
with destination.open("x", encoding="utf-8") as output:
    json.dump(report, output, indent=2)
    output.write("\n")

print("held_out_test_accessed=False")
print("SIGNALDNA-PRETEST-CHECKPOINTS-207: PASS")