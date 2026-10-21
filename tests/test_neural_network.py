import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neural_network import initialize_parameters, LAYER_SIZES, forward, binary_cross_entropy

class InitializationTests(unittest.TestCase):
    def test_shapes(self):
        params = initialize_parameters()
        self.assertEqual(len(params), 3)
        for (w, b), a, c in zip(params, LAYER_SIZES[:-1], LAYER_SIZES[1:]):
            self.assertEqual(w.shape, (a, c))
            self.assertEqual(b.shape, (c,))

    def test_reproducibility(self):
        a = initialize_parameters(seed=42)
        b = initialize_parameters(seed=42)
        for (wa, ba), (wb, bb) in zip(a, b):
            self.assertTrue(np.array_equal(wa, wb))
            self.assertTrue(np.array_equal(ba, bb))

    def test_seed_independence(self):
        a = initialize_parameters(seed=42)
        b = initialize_parameters(seed=43)
        self.assertFalse(np.array_equal(a[0][0], b[0][0]))

    def test_finite_and_zero_biases(self):
        for w, b in initialize_parameters():
            self.assertTrue(np.isfinite(w).all())
            self.assertTrue(np.array_equal(b, np.zeros_like(b)))

class ForwardTests(unittest.TestCase):
    def test_output_shape(self):
        p, a = forward(np.zeros((4, 1024)), initialize_parameters())
        self.assertEqual(p.shape, (4, 1))
        self.assertEqual([v.shape for v in a], [(4, 1024), (4, 64), (4, 32), (4, 1)])

    def test_probability_bounds(self):
        x = np.random.default_rng(42).normal(size=(8, 1024))
        p, _ = forward(x, initialize_parameters())
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.all((p >= 0) & (p <= 1)))

    def test_deterministic_predictions(self):
        x = np.ones((2, 1024))
        params = initialize_parameters(seed=7)
        a, _ = forward(x, params)
        b, _ = forward(x, params)
        self.assertTrue(np.array_equal(a, b))

    def test_invalid_input(self):
        params = initialize_parameters()
        for x in (np.zeros((2, 100)), np.zeros(1024), np.full((2, 1024), np.nan)):
            with self.assertRaises(ValueError):
                forward(x, params)
class LossTests(unittest.TestCase):
    def test_known_value(self):
        p = np.array([[0.5], [0.5]])
        y = np.array([[0], [1]])
        self.assertAlmostEqual(binary_cross_entropy(p, y), np.log(2), places=12)

    def test_better_prediction_lower_loss(self):
        y = np.array([[1], [0]])
        good = np.array([[0.9], [0.1]])
        bad = np.array([[0.1], [0.9]])
        self.assertLess(binary_cross_entropy(good, y), binary_cross_entropy(bad, y))

    def test_extreme_probabilities_finite(self):
        p = np.array([[0.0], [1.0]])
        y = np.array([[1], [0]])
        self.assertTrue(np.isfinite(binary_cross_entropy(p, y)))

    def test_invalid_inputs(self):
        for p, y in ((np.ones((2, 1)), np.ones((2,))), (np.array([[1.2]]), np.array([[1]])), (np.array([[0.5]]), np.array([[2]]))):
            with self.assertRaises(ValueError):
                binary_cross_entropy(p, y)
if __name__ == "__main__":
    unittest.main(verbosity=2)
