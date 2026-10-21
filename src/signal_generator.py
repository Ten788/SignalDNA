import numpy as np

SAMPLE_RATE = 2048
SAMPLE_COUNT = 1024

def generate_base_signal(frequency=50.0, amplitude=1.0, phase=0.0):
    """Generate a deterministic half-second sine wave."""
    if not (20 <= frequency <= 200 and 0.5 <= amplitude <= 1.5):
        raise ValueError("Signal parameters outside approved ranges")
    t = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

def generate_waveform(frequency=50.0, amplitude=1.0, harmonics=(), noise_fraction=0.0, seed=0, phase=0.0):
    """Generate a deterministic waveform with harmonics and Gaussian noise."""
    if len(harmonics) > 3 or any(not np.isfinite(h) or h < 0 or h > 1 for h in harmonics):
        raise ValueError("Invalid harmonic strengths")
    if not np.isfinite(noise_fraction) or not 0 <= noise_fraction <= 0.15:
        raise ValueError("Invalid noise fraction")
    x = generate_base_signal(frequency, amplitude, phase)
    t = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE
    for order, strength in enumerate(harmonics, start=2):
        x = x + amplitude * strength * np.sin(2 * np.pi * order * frequency * t)
    if noise_fraction:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0.0, amplitude * noise_fraction, SAMPLE_COUNT)
    return x

def add_baseline_drift(signal, drift_amplitude=0.0, drift_frequency=1.0, drift_phase=0.0):
    """Add deterministic low-frequency baseline drift."""
    x = np.asarray(signal, dtype=np.float64)
    if x.shape != (SAMPLE_COUNT,) or not np.isfinite(x).all():
        raise ValueError("Invalid input signal")
    if not np.isfinite(drift_amplitude) or not 0 <= drift_amplitude <= 0.3:
        raise ValueError("Invalid drift amplitude")
    if not np.isfinite(drift_frequency) or not 0.1 <= drift_frequency <= 5.0:
        raise ValueError("Invalid drift frequency")
    if not np.isfinite(drift_phase):
        raise ValueError("Invalid drift phase")
    t = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE
    return x + drift_amplitude * np.sin(2 * np.pi * drift_frequency * t + drift_phase)

def add_transient(signal, position=0.5, width=0.02, magnitude=0.5, polarity=1):
    """Add a Gaussian-shaped transient to a waveform."""
    x = np.asarray(signal, dtype=np.float64)
    if x.shape != (SAMPLE_COUNT,) or not np.isfinite(x).all():
        raise ValueError("Invalid input signal")
    if not np.isfinite(position) or not 0.1 <= position <= 0.9:
        raise ValueError("Invalid transient position")
    if not np.isfinite(width) or not 0.005 <= width <= 0.1:
        raise ValueError("Invalid transient width")
    if not np.isfinite(magnitude) or not 0.05 <= magnitude <= 2.0:
        raise ValueError("Invalid transient magnitude")
    if polarity not in (-1, 1):
        raise ValueError("Invalid transient polarity")
    t = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_COUNT
    pulse = polarity * magnitude * np.exp(-0.5 * ((t - position) / width) ** 2)
    return x + pulse

