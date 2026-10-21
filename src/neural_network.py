"""Reference binary classifier implemented directly with NumPy.

Architecture: 1024 inputs -> 64 ReLU -> 32 ReLU -> 1 sigmoid output.
This module is the mathematical reference for later CPU/CUDA comparisons.
"""

import numpy as np

LAYER_SIZES = (1024, 64, 32, 1)

def initialize_parameters(seed=2026):
    """Initialize deterministic weights and biases for each dense layer.

    Args:
        seed: Seed for an isolated NumPy random generator.

    Returns:
        A list of (weights, biases) pairs. Weight shapes follow LAYER_SIZES;
        biases start at zero.

    Notes:
        He scaling, sqrt(2 / fan_in), helps preserve activation variance
        through ReLU layers. The output layer uses the same initialization
        convention for a simple, reproducible starting point.
    """
    rng = np.random.default_rng(seed)
    parameters = []
    for fan_in, fan_out in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:]):
        weights = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        biases = np.zeros(fan_out, dtype=np.float64)
        parameters.append((weights, biases))
    return parameters

def forward(inputs, parameters):
    """Compute binary-class probabilities and cache activations.

    Args:
        inputs: Finite float-compatible batch shaped (N, 1024).
        parameters: Dense-layer (weights, biases) pairs from initialization.

    Returns:
        Probabilities shaped (N, 1), followed by activations shaped
        (N, 1024), (N, 64), (N, 32), and (N, 1).

    Raises:
        ValueError: If inputs are not a finite two-dimensional batch
            with 1024 features.

    Notes:
        logaddexp evaluates sigmoid in a form that avoids overflow for
        large-magnitude logits. Parameter-shape validation is not yet
        performed by this function.
    """
    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != LAYER_SIZES[0] or not np.isfinite(x).all():
        raise ValueError("Expected finite input matrix with 1024 features")
    activations = [x]
    for weights, biases in parameters[:-1]:
        x = np.maximum(x @ weights + biases, 0.0)
        activations.append(x)
    weights, biases = parameters[-1]
    logits = x @ weights + biases
    probabilities = np.exp(-np.logaddexp(0.0, -logits))
    activations.append(probabilities)
    return probabilities, activations

def binary_cross_entropy(probabilities, labels):
    """Compute mean binary cross-entropy for binary classification.

    Args:
        probabilities: Finite predictions shaped (N, 1), within [0, 1].
        labels: Matching binary targets shaped (N, 1).

    Returns:
        Mean loss as a Python float.

    Raises:
        ValueError: If shapes, labels, or probabilities violate the contract.

    Notes:
        Clipping avoids log(0) at exact probability endpoints. It also means
        endpoint losses are finite approximations rather than mathematical
        infinities.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 1 or p.shape[0] == 0:
        raise ValueError("Probabilities must have shape (batch, 1)")
    if y.shape != p.shape or not np.isfinite(y).all() or not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels must be binary and match probability shape")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    eps = np.finfo(np.float64).eps
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
