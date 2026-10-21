# SignalDNA - Architecture

Status: M0 proposal

## Model
Input: 1024 normalized signal samples.
Hidden 1: 64 ReLU neurons.
Hidden 2: 32 ReLU neurons.
Output: 1 sigmoid probability.
Loss: binary cross-entropy.
Optimizer: mini-batch gradient descent.
Implementation: original NumPy forward and backward propagation.

## Engineering Constraints
No pretrained models or external training services.
Deterministic initialization, shuffling, and dataset generation.
Training-derived fixed scaling; prevent validation/test leakage.
Numerical gradient verification before training acceptance.
Architecture alternatives evaluated using validation data only.

## Pending Decisions
Batch size, learning-rate schedule, initialization, stopping policy, and conventional baseline.

## Initial Training Configuration
Initialization: He for ReLU layers.
Batch size: 64.
Initial learning rate: 0.01.
Optimizer: mini-batch SGD.
Maximum epochs: 100.
Early stopping: validation loss, patience 12.
Checkpoint: minimum validation loss.
Random seeds: explicit and recorded.
Loss: numerically stable binary cross-entropy.

## Gradient Verification
Compare analytical gradients with central finite differences on a reduced network.
Avoid ReLU nondifferentiable points.
Check absolute and relative errors, accounting for near-zero gradients.
Block training acceptance if gradient verification fails.

## Model Selection
Tune using training and validation data only.
Evaluate the selected model once on the held-out test set.

## CPU/GPU Comparative Study
Status: Approved M0 scope addition.
NumPy CPU network remains the original mathematical reference.
PyTorch CPU and CUDA implementations provide comparison paths.
Compare identical architectures, weights, inputs, outputs, and gradients within documented numerical tolerances.
Benchmark training and inference separately across batch sizes.
Record hardware, software versions, warmup, repeated timings, transfer overhead, and GPU memory usage.
Report accuracy, reproducibility, performance, and limitations without assuming GPU superiority.
CPU-only operation remains supported; no paid APIs or cloud services.
Publish reproducible methodology and results only after verification.
