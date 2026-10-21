"""Deterministic labeled synthetic-signal dataset builder."""

from dataclasses import dataclass
import numpy as np
from signal_generator import SAMPLE_COUNT, generate_waveform, add_baseline_drift, add_transient

@dataclass(frozen=True)
class DatasetConfig:
    count: int = 12000
    seed: int = 2026
    positive_fraction: float = 0.5

def build_dataset(config: DatasetConfig):
    """Return (signals, labels); implementation follows verified generator integration."""
    if config.count < 2 or config.count % 2 or config.positive_fraction != 0.5:
        raise ValueError("Initial dataset requires even count and balanced classes")
    rng = np.random.default_rng(config.seed)
    labels = np.array([0, 1] * (config.count // 2), dtype=np.int8)
    rng.shuffle(labels)
    signals = np.empty((config.count, SAMPLE_COUNT), dtype=np.float64)
    for i, label in enumerate(labels):
        f = rng.uniform(20, 200)
        a = rng.uniform(0.5, 1.5)
        harmonics = tuple(rng.uniform(0, 0.5, size=int(rng.integers(0, 4))))
        noise = rng.uniform(0, 0.15)
        seed = int(rng.integers(0, 2**32))
        phase = rng.uniform(0, 2*np.pi)
        x = generate_waveform(f, a, harmonics, noise, seed, phase=phase)
        x = add_baseline_drift(x, rng.uniform(0, 0.3), rng.uniform(0.1, 5), rng.uniform(0, 2*np.pi))
        if label:
            x = add_transient(x, rng.uniform(0.1, 0.9), rng.uniform(0.005, 0.05), rng.uniform(0.2, 1.5), int(rng.choice([-1, 1])))
        signals[i] = x
    return signals, labels


def build_splits(seed=2026, counts=(8400, 1800, 1800)):
    """Generate independent, reproducible train/validation/test splits."""
    if len(counts) != 3 or any(not isinstance(n, int) or n < 2 or n % 2 for n in counts):
        raise ValueError("Split counts must contain three positive even integers")
    seeds = np.random.SeedSequence(seed).spawn(3)
    result = {}
    for name, count, child in zip(("train", "validation", "test"), counts, seeds):
        child_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        result[name] = build_dataset(DatasetConfig(count=count, seed=child_seed))
    return result

