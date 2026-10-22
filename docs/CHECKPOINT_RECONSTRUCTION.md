# SignalDNA - Checkpoint Reconstruction

Checkpoint archives are excluded from Git and can be regenerated using the deterministic NumPy training procedure.

## Prerequisites

Use Python 3.13 and install the CPU dependencies from requirements.lock.txt.

## Safe procedure

Use a separate disposable checkout, never the original research workspace.

1. Clone the published repository into a new directory.
2. Create a Python 3.13 virtual environment.
3. Install requirements.lock.txt into that environment.
4. Run: python experiments/train_best_checkpoint.py
5. Compute SHA-256 hashes of both generated checkpoint files.

The training script creates checkpoints/numpy_final_60.npz and checkpoints/numpy_best_validation.npz. It may overwrite those files if they already exist.

## Expected SHA-256

Final epoch-60 checkpoint: 846137EE4333F7FAC7990E29FD6F7D0EAA59E6D1153F9A6BF6775F9C21A72975
Best-validation-loss checkpoint: 1D1D72CD6D68659AE9DD4001AC21842A59921CDF9807910966B03EC2B938A243

Both checkpoints were previously reconstructed byte-for-byte in an isolated worktree. Reproduction on other software environments is not guaranteed.

## Scientific safeguards

The primary evaluated model is the epoch-60 checkpoint. The best-validation-loss checkpoint is a separate artifact.
Do not run experiments/evaluate_heldout_once.py. The final held-out evaluation has already been completed once.
Do not overwrite original checkpoints, archived measurements, logs, or evidence.
Reconstruction is not a new held-out evaluation and does not require CUDA.
