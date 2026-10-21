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