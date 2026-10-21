"""Regression tests for reproducible experiment recording."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment_logger import ExperimentLogger


class ExperimentLoggerTests(unittest.TestCase):
    def test_configuration_and_epoch_roundtrip(self):
        logger = ExperimentLogger("cpu_baseline", {"seed": 2026, "batch_size": 64})
        logger.log_epoch(0, validation_accuracy=0.5, loss=0.693)
        logger.log_epoch(1, validation_accuracy=0.6, loss=0.65)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "experiment.json"
            logger.save(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["backend"], "numpy_cpu")
        self.assertEqual(saved["configuration"]["seed"], 2026)
        self.assertEqual(len(saved["epochs"]), 2)
        self.assertEqual(saved["epochs"][1]["epoch"], 1)
        self.assertAlmostEqual(saved["epochs"][1]["validation_accuracy"], 0.6)

    def test_duplicate_epoch_rejected(self):
        logger = ExperimentLogger("duplicate", {"seed": 1})
        logger.log_epoch(0, accuracy=0.5)
        with self.assertRaises(ValueError):
            logger.log_epoch(0, accuracy=0.6)

    def test_invalid_metrics_rejected(self):
        logger = ExperimentLogger("invalid", {"seed": 1})
        for value in (math.nan, math.inf, -math.inf, "invalid"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    logger.log_epoch(0, accuracy=value)

    def test_existing_record_not_overwritten(self):
        logger = ExperimentLogger("safe", {"seed": 1})
        logger.log_epoch(0, accuracy=0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            logger.save(path)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                logger.save(path)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)