"""Deterministic regression tests for binary-classification metrics."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        probabilities = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        labels = np.array([1, 1, 0, 0, 0, 1])
        result = classification_metrics(probabilities, labels)
        self.assertEqual(
            (result["tp"], result["tn"], result["fp"], result["fn"]),
            (2, 2, 1, 1),
        )
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_zero_denominator_handling(self):
        result = classification_metrics([0.1, 0.2], [0, 0])
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)

    def test_invalid_inputs(self):
        cases = [
            ([], []),
            ([0.5], [0, 1]),
            ([1.2], [1]),
            ([np.nan], [0]),
            ([0.5], [2]),
        ]
        for probabilities, labels in cases:
            with self.subTest(probabilities=probabilities, labels=labels):
                with self.assertRaises(ValueError):
                    classification_metrics(probabilities, labels)


class RocAucTests(unittest.TestCase):
    def test_perfect_ranking(self):
        from evaluation import roc_auc
        self.assertAlmostEqual(roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)

    def test_reversed_ranking(self):
        from evaluation import roc_auc
        self.assertAlmostEqual(roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_tied_scores(self):
        from evaluation import roc_auc
        self.assertAlmostEqual(roc_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]), 0.5)

    def test_single_class_rejected(self):
        from evaluation import roc_auc
        with self.assertRaises(ValueError):
            roc_auc([0.2, 0.8], [1, 1])

if __name__ == "__main__":
    unittest.main(verbosity=2)