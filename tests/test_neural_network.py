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
class BackwardTests(unittest.TestCase):
    def test_gradient_shapes_and_finiteness(self):
        from neural_network import backward
        parameters = initialize_parameters(seed=71)
        inputs = np.random.default_rng(72).normal(size=(4, 1024))
        labels = np.array([[0.0], [1.0], [1.0], [0.0]])
        _, activations = forward(inputs, parameters)
        gradients = backward(activations, parameters, labels)
        self.assertEqual(len(gradients), len(parameters))
        for (dw, db), (weights, biases) in zip(gradients, parameters):
            self.assertEqual(dw.shape, weights.shape)
            self.assertEqual(db.shape, biases.shape)
            self.assertTrue(np.isfinite(dw).all())
            self.assertTrue(np.isfinite(db).all())

class GradientCheckTests(unittest.TestCase):
    def test_finite_difference_gradients(self):
        from neural_network import backward
        rng = np.random.default_rng(81)
        params = initialize_parameters(seed=82)
        x = rng.normal(0.0, 0.2, size=(3, 1024))
        y = np.array([[0.0], [1.0], [0.0]])
        _, cache = forward(x, params)
        gradients = backward(cache, params, y)
        step = 1e-5

        def loss():
            probabilities, _ = forward(x, params)
            return binary_cross_entropy(probabilities, y)

        for layer, ((weights, biases), (dw, db)) in enumerate(zip(params, gradients)):
            for values, analytical in ((weights, dw), (biases, db)):
                for index in list(np.ndindex(values.shape))[:4]:
                    original = values[index]
                    values[index] = original + step
                    plus = loss()
                    values[index] = original - step
                    minus = loss()
                    values[index] = original
                    numerical = (plus - minus) / (2 * step)
                    self.assertAlmostEqual(
                        analytical[index], numerical, delta=1e-5,
                        msg=f"Gradient mismatch at layer {layer}, index {index}"
                    )

class BatchGradientTests(unittest.TestCase):
    def test_batch_duplication_invariance(self):
        from neural_network import backward
        params = initialize_parameters(seed=101)
        x = np.random.default_rng(102).normal(0, 0.2, size=(3, 1024))
        y = np.array([[0.0], [1.0], [0.0]])
        _, single_cache = forward(x, params)
        _, repeated_cache = forward(np.tile(x, (2, 1)), params)
        single = backward(single_cache, params, y)
        repeated = backward(repeated_cache, params, np.tile(y, (2, 1)))
        for (dw1, db1), (dw2, db2) in zip(single, repeated):
            np.testing.assert_allclose(dw1, dw2, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(db1, db2, rtol=1e-12, atol=1e-12)

class SgdTests(unittest.TestCase):
    def test_step_reduces_loss_without_mutation(self):
        from neural_network import backward, sgd_step
        params = initialize_parameters(seed=111)
        original = [(w.copy(), b.copy()) for w, b in params]
        x = np.random.default_rng(112).normal(0, 0.2, size=(8, 1024))
        y = np.array([[0], [1], [0], [1], [0], [1], [0], [1]])
        predictions, cache = forward(x, params)
        before = binary_cross_entropy(predictions, y)
        gradients = backward(cache, params, y)
        updated = sgd_step(params, gradients, learning_rate=0.01)
        after = binary_cross_entropy(forward(x, updated)[0], y)
        self.assertLess(after, before)
        for (w, b), (ow, ob) in zip(params, original):
            np.testing.assert_array_equal(w, ow)
            np.testing.assert_array_equal(b, ob)

class TrainingEpochTests(unittest.TestCase):
    def test_epoch_learning_and_reproducibility(self):
        from neural_network import train_epoch
        x = np.random.default_rng(121).normal(0, 0.2, size=(32, 1024))
        y = np.array([[i % 2] for i in range(32)], dtype=np.float64)
        initial = initialize_parameters(seed=122)
        before = binary_cross_entropy(forward(x, initial)[0], y)
        first, first_loss = train_epoch(x, y, initial, learning_rate=0.01, batch_size=8, seed=123)
        second, second_loss = train_epoch(x, y, initial, learning_rate=0.01, batch_size=8, seed=123)
        self.assertLess(first_loss, before)
        self.assertEqual(first_loss, second_loss)
        for (w1, b1), (w2, b2) in zip(first, second):
            np.testing.assert_array_equal(w1, w2)
            np.testing.assert_array_equal(b1, b2)

class CheckpointTests(unittest.TestCase):
    def test_save_load_round_trip(self):
        import tempfile
        from neural_network import save_parameters, load_parameters
        params = initialize_parameters(seed=131)
        inputs = np.random.default_rng(132).normal(size=(4, 1024))
        before, _ = forward(inputs, params)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_parameters(path, params)
            restored = load_parameters(path)
        after, _ = forward(inputs, restored)
        np.testing.assert_array_equal(before, after)
        for (w1, b1), (w2, b2) in zip(params, restored):
            np.testing.assert_array_equal(w1, w2)
            np.testing.assert_array_equal(b1, b2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
