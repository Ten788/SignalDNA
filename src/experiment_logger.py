"""Write reproducible experiment records without overwriting prior runs."""

import json
from datetime import datetime, timezone
from pathlib import Path


class ExperimentLogger:
    """Collect configuration and epoch metrics for a single experiment."""

    def __init__(self, name, configuration, backend="numpy_cpu"):
        if not name or not isinstance(configuration, dict):
            raise ValueError("Experiment name and configuration are required")
        self.record = {
            "schema_version": 1,
            "name": name,
            "backend": backend,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": dict(configuration),
            "epochs": [],
        }

    def log_epoch(self, epoch, **metrics):
        """Record one unique epoch with finite numeric metrics."""
        import math

        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        if any(item["epoch"] == epoch for item in self.record["epochs"]):
            raise ValueError("Duplicate epoch")
        if not metrics:
            raise ValueError("At least one metric is required")
        for name, value in metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid metric: {name}")
        self.record["epochs"].append({"epoch": epoch, **metrics})

    def save(self, path):
        """Write JSON exclusively so existing experiment evidence is preserved."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as output:
            json.dump(self.record, output, indent=2)
            output.write("\n")
        return destination