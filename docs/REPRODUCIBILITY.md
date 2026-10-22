# SignalDNA — Reproducibility Guide

Status: Pre-publication draft. Original research evidence must remain preserved.

## Recorded environment

Current environment snapshot: results/runtime_environment_v1.json

- Windows; Python 3.13.15
- NumPy 2.5.4
- PyTorch 2.11.0+cu128; CUDA build 12.8
- CPU: Intel Core i9-13900H
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU

The environment snapshot was captured after the principal experimental runs. Exact historical environment identity has not been independently established.

## CPU installation

From the project root, use PowerShell:

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt

Do not install dependencies globally or modify an unrelated managed Python environment.

The current lockfile does not include PyTorch. GPU research requires a separately installed compatible PyTorch build. Verify Python, CUDA, and driver compatibility before installation.

## Safe verification commands

Unit tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

Independent metric verification:

    .\.venv\Scripts\python.exe experiments\verify_evaluation_independent.py

The current workspace passed 55 automated unit tests and 104 independent metric-verification cases.

## Archived experiment results

- results/numpy_cpu_60_epochs_logged.json
- results/pytorch_cuda_60_epochs_logged.json
- results/cpu_gpu_curve_comparison.json
- results/cpu_gpu_benchmark_v1.json
- results/heldout_final_primary_v1.json
- results/research_evidence_register_v1.json

The primary checkpoint SHA-256 is documented in docs/FINAL_EVALUATION_PROTOCOL.md. Saved checkpoint binaries are excluded from normal Git tracking.

## Important execution boundaries

DO NOT run experiments/evaluate_heldout_once.py as a normal setup, test, or benchmark command. It represents the completed single-use final held-out evaluation.

Some training and analysis scripts write to fixed paths and may replace existing artifacts. Run evidence-generating experiments only in isolated, disposable workspaces after inspecting their outputs.

Repeated benchmarks may produce different timings. Do not overwrite original experimental records with reproduction results.

New experiments informed by the final held-out result must be identified as exploratory, not another untouched final evaluation.

## Publication limitations

The project uses synthetic signal classification, one principal neural-network architecture and training seed, and the tested local hardware configuration. It has not established real-world embedded or FPGA deployment performance.

Clean-checkout installation and reproduction instructions still require final verification before publication.
## Verified isolated checkpoint reconstruction

On 2026-10-21, both checkpoints were regenerated from committed source in an isolated detached Git worktree without pre-existing checkpoint files. The existing project Python environment was reused; this was not a clean dependency-installation test.

- Source baseline: commit 1a23cca.
- Final 60-epoch checkpoint SHA-256: 846137EE4333F7FAC7990E29FD6F7D0EAA59E6D1153F9A6BF6775F9C21A72975.
- Best-validation checkpoint SHA-256: 1D1D72CD6D68659AE9DD4001AC21842A59921CDF9807910966B03EC2B938A243.
- Both regenerated archives matched their original SHA-256 fingerprints exactly.
- The isolated checkout initially lacked the Git-ignored checkpoints directory. Creating that directory enabled successful reconstruction.
- Commit 108bf1a subsequently updated save_parameters to create the destination directory automatically.
- All 55 existing unit tests passed after the directory-creation change.

The final held-out evaluation was not repeated. Fresh-environment installation verification remains outstanding.