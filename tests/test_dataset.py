import sys
import hashlib
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import DatasetConfig, build_dataset, build_splits
from signal_generator import SAMPLE_COUNT

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = build_dataset(DatasetConfig(count=20, seed=42))

    def test_dimensions(self):
        self.assertEqual(self.x.shape, (20, SAMPLE_COUNT))
        self.assertEqual(self.y.shape, (20,))

    def test_balance(self):
        self.assertEqual(int(np.sum(self.y)), 10)

    def test_finite(self):
        self.assertTrue(np.isfinite(self.x).all())

    def test_reproducibility(self):
        x, y = build_dataset(DatasetConfig(count=20, seed=42))
        self.assertTrue(np.array_equal(self.x, x))
        self.assertTrue(np.array_equal(self.y, y))

    def test_seed_independence(self):
        x, _ = build_dataset(DatasetConfig(count=20, seed=43))
        self.assertFalse(np.array_equal(self.x, x))

class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = build_splits(seed=123, counts=(20, 10, 10))

    def test_split_sizes_and_balance(self):
        for name, count in (("train", 20), ("validation", 10), ("test", 10)):
            x, y = self.splits[name]
            self.assertEqual(x.shape, (count, SAMPLE_COUNT))
            self.assertEqual(int(np.sum(y)), count // 2)

    def test_split_reproducibility(self):
        again = build_splits(seed=123, counts=(20, 10, 10))
        for name in self.splits:
            self.assertTrue(np.array_equal(self.splits[name][0], again[name][0]))
            self.assertTrue(np.array_equal(self.splits[name][1], again[name][1]))

    def test_no_cross_split_duplicates(self):
        seen = set()
        for x, _ in self.splits.values():
            hashes = {hashlib.sha256(row.tobytes()).hexdigest() for row in x}
            self.assertEqual(len(hashes), len(x))
            self.assertTrue(seen.isdisjoint(hashes))
            seen.update(hashes)

if __name__ == "__main__":
    unittest.main(verbosity=2)

