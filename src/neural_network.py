"""Reference binary classifier implemented directly with NumPy.

Architecture: 1024 inputs -> 64 ReLU -> 32 ReLU -> 1 sigmoid output.
This module is the mathematical reference for later CPU/CUDA comparisons.
"""
from pathlib import Path

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
def backward(activations, parameters, labels):
    """Compute mean-loss gradients for each dense layer.

    Args:
        activations: Forward cache ordered input, hidden layers, output.
        parameters: Layer (weights, biases) pairs matching the cache.
        labels: Binary targets shaped (batch, 1).

    Returns:
        List of (weight_gradient, bias_gradient) pairs in layer order.

    Notes:
        Sigmoid with binary cross-entropy yields output delta p - y.
        ReLU propagates gradients only through positive activations.
        Gradients are averaged over the batch.
    """
    p = np.asarray(activations[-1], dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 1 or p.shape[0] == 0:
        raise ValueError("Invalid cached output shape")
    if y.shape != p.shape or not np.isfinite(y).all() or not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels must be binary and match output shape")
    if len(parameters) != len(LAYER_SIZES) - 1 or len(activations) != len(LAYER_SIZES):
        raise ValueError("Layer count mismatch")
    if not all(np.isfinite(a).all() for a in activations):
        raise ValueError("Non-finite cached activation")
    delta = (p - y) / p.shape[0]
    gradients = [None] * len(parameters)
    for layer in range(len(parameters) - 1, -1, -1):
        weights, biases = parameters[layer]
        previous = activations[layer]
        if weights.shape != (LAYER_SIZES[layer], LAYER_SIZES[layer + 1]) or biases.shape != (LAYER_SIZES[layer + 1],):
            raise ValueError("Parameter shape mismatch")
        gradients[layer] = (previous.T @ delta, np.sum(delta, axis=0))
        if layer:
            delta = (delta @ weights.T) * (activations[layer] > 0)
    return gradients
def sgd_step(parameters, gradients, learning_rate):
    """Return updated parameters after one gradient-descent step.

    Each parameter is reduced by the learning rate times its gradient.
    The original parameter arrays are preserved.

    Raises:
        ValueError: For invalid learning rates, gradients, or shapes.
    """
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    if len(parameters) != len(gradients):
        raise ValueError("Parameter and gradient counts differ")
    updated = []
    for (weights, biases), (dw, db) in zip(parameters, gradients):
        if dw.shape != weights.shape or db.shape != biases.shape:
            raise ValueError("Gradient shape mismatch")
        if not np.isfinite(dw).all() or not np.isfinite(db).all():
            raise ValueError("Non-finite gradient")
        updated.append((weights - learning_rate * dw, biases - learning_rate * db))
    return updated
def train_epoch(inputs, labels, parameters, learning_rate=0.01, batch_size=64, seed=2026):
    """Train for one shuffled epoch and return updated parameters.

    The seed controls example order, making each epoch reproducible.
    Mini-batch gradients use mean binary cross-entropy loss.
    The returned loss is evaluated after all parameter updates.
    """
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != LAYER_SIZES[0] or x.shape[0] == 0:
        raise ValueError("Invalid training input shape")
    if y.shape != (x.shape[0], 1):
        raise ValueError("Training labels must have shape (batch, 1)")
    if not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, (bool, np.bool_)) or batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.all((y == 0) | (y == 1)):
        raise ValueError("Training data must be finite with binary labels")
    order = np.random.default_rng(seed).permutation(x.shape[0])
    current = parameters
    for start in range(0, x.shape[0], batch_size):
        indices = order[start:start + batch_size]
        _, cache = forward(x[indices], current)
        gradients = backward(cache, current, y[indices])
        current = sgd_step(current, gradients, learning_rate)
    probabilities, _ = forward(x, current)
    return current, binary_cross_entropy(probabilities, y)
def save_parameters(path, parameters):
    """Save dense-layer parameters in NumPy's non-pickle NPZ format.

    Layer arrays are stored under stable, indexed keys for portability.
    Reject malformed or nonfinite parameters before writing.
    """
    if len(parameters) != len(LAYER_SIZES) - 1:
        raise ValueError("Unexpected parameter count")
    arrays = {}
    for index, ((weights, biases), fan_in, fan_out) in enumerate(
        zip(parameters, LAYER_SIZES[:-1], LAYER_SIZES[1:])
    ):
        if weights.shape != (fan_in, fan_out) or biases.shape != (fan_out,):
            raise ValueError("Parameter shape mismatch")
        if not np.isfinite(weights).all() or not np.isfinite(biases).all():
            raise ValueError("Non-finite parameter")
        arrays[f"weights_{index}"] = weights
        arrays[f"biases_{index}"] = biases
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
def load_parameters(path):
    """Load and validate a NumPy model checkpoint without pickle.

    Only the expected dense-layer arrays are accepted. Invalid
    shapes, nonfinite values, or extra archive entries are rejected.
    """
    expected = {
        f"{kind}_{index}"
        for index in range(len(LAYER_SIZES) - 1)
        for kind in ("weights", "biases")
    }
    parameters = []
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("Checkpoint array names do not match architecture")
        for index, (fan_in, fan_out) in enumerate(zip(LAYER_SIZES[:-1], LAYER_SIZES[1:])):
            weights = np.asarray(archive[f"weights_{index}"], dtype=np.float64)
            biases = np.asarray(archive[f"biases_{index}"], dtype=np.float64)
            if weights.shape != (fan_in, fan_out) or biases.shape != (fan_out,):
                raise ValueError("Checkpoint parameter shape mismatch")
            if not np.isfinite(weights).all() or not np.isfinite(biases).all():
                raise ValueError("Non-finite checkpoint parameter")
            parameters.append((weights.copy(), biases.copy()))
    return parameters
