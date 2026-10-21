import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from signal_generator import generate_base_signal, generate_waveform, add_baseline_drift, add_transient, SAMPLE_COUNT, SAMPLE_RATE

class SignalGeneratorTests(unittest.TestCase):
    def test_shape_and_finite(self):
        x = generate_base_signal()
        self.assertEqual(x.shape, (SAMPLE_COUNT,))
        self.assertTrue(np.isfinite(x).all())

    def test_determinism(self):
        self.assertTrue(np.array_equal(generate_base_signal(), generate_base_signal()))

    def test_amplitude(self):
        self.assertAlmostEqual(np.max(np.abs(generate_base_signal(50, 1.0))), 1.0, places=10)

    def test_frequency(self):
        x = generate_base_signal(50)
        peak = np.argmax(np.abs(np.fft.rfft(x)))
        self.assertAlmostEqual(peak * SAMPLE_RATE / SAMPLE_COUNT, 50.0)

    def test_invalid_parameters(self):
        for f, a in [(19, 1), (201, 1), (50, 0.4), (50, 1.6)]:
            with self.assertRaises(ValueError):
                generate_base_signal(f, a)

class WaveformTests(unittest.TestCase):
    def test_seed_reproducibility(self):
        a = generate_waveform(noise_fraction=0.1, seed=42)
        b = generate_waveform(noise_fraction=0.1, seed=42)
        self.assertTrue(np.array_equal(a, b))

    def test_different_seeds(self):
        a = generate_waveform(noise_fraction=0.1, seed=42)
        b = generate_waveform(noise_fraction=0.1, seed=43)
        self.assertFalse(np.array_equal(a, b))

    def test_second_harmonic(self):
        x = generate_waveform(frequency=50, harmonics=(0.5,))
        spectrum = np.abs(np.fft.rfft(x))
        self.assertGreater(spectrum[50], 0.4 * spectrum[25])

    def test_invalid_waveform_parameters(self):
        for kwargs in ({"noise_fraction": 0.2}, {"harmonics": (1, 1, 1, 1)}, {"harmonics": (-0.1,)}):
            with self.assertRaises(ValueError):
                generate_waveform(**kwargs)

class DriftTests(unittest.TestCase):
    def test_zero_drift(self):
        x = generate_base_signal()
        self.assertTrue(np.array_equal(x, add_baseline_drift(x)))

    def test_deterministic_drift(self):
        x = generate_base_signal()
        a = add_baseline_drift(x, 0.2, 1.0, 0.3)
        b = add_baseline_drift(x, 0.2, 1.0, 0.3)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, x))

    def test_invalid_drift(self):
        x = generate_base_signal()
        for kwargs in ({"drift_amplitude": -0.1}, {"drift_frequency": 8.0}, {"drift_phase": float("nan")}):
            with self.assertRaises(ValueError):
                add_baseline_drift(x, **kwargs)

class TransientTests(unittest.TestCase):
    def test_peak_location(self):
        x = add_transient(np.zeros(SAMPLE_COUNT), position=0.4)
        self.assertLessEqual(abs(np.argmax(x) / SAMPLE_COUNT - 0.4), 1 / SAMPLE_COUNT)

    def test_polarity(self):
        z = np.zeros(SAMPLE_COUNT)
        a = add_transient(z, polarity=1)
        b = add_transient(z, polarity=-1)
        self.assertTrue(np.allclose(a, -b))

    def test_determinism(self):
        z = np.zeros(SAMPLE_COUNT)
        self.assertTrue(np.array_equal(add_transient(z), add_transient(z)))

    def test_invalid_transient(self):
        z = np.zeros(SAMPLE_COUNT)
        for kwargs in ({"position": 1.0}, {"width": 0.0}, {"magnitude": -1.0}, {"polarity": 0}):
            with self.assertRaises(ValueError):
                add_transient(z, **kwargs)

class PhaseTests(unittest.TestCase):
    def test_phase_changes_waveform(self):
        a = generate_waveform(50, 1.0, phase=0.0)
        b = generate_waveform(50, 1.0, phase=np.pi / 2)
        self.assertFalse(np.array_equal(a, b))

    def test_phase_reproducibility(self):
        a = generate_waveform(50, 1.0, phase=1.234)
        b = generate_waveform(50, 1.0, phase=1.234)
        self.assertTrue(np.array_equal(a, b))
if __name__ == "__main__":
    unittest.main(verbosity=2)




