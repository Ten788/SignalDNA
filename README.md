# SignalDNA

**Learning neural networks from first principles through reproducible signal-classification experiments.**

SignalDNA is an independent educational research project exploring neural networks as a foundation for future work in embedded systems, FPGA design, verification, and engineering analysis.

The project began with an original NumPy implementation to understand forward propagation, backpropagation, and optimization before investigating GPU acceleration. It has been an enjoyable and enlightening engineering experiment, with plenty still to learn.

## What was built

- Original 1024-64-32-1 NumPy neural network, using ReLU, sigmoid, binary cross-entropy, and mini-batch SGD.
- Deterministic, balanced synthetic transient-detection dataset.
- PyTorch CUDA training implementation and numerical-equivalence checks.
- Automated tests, archived learning curves, and performance measurements.
- Locked checkpoint selection and one authorized final held-out evaluation.

## Final research results

The epoch-60 model was selected by validation ROC-AUC before test access and evaluated once on 1,800 held-out synthetic waveforms.

| Metric | Held-out result |
|---|---:|
| Accuracy | 78.78% |
| Precision | 79.37% |
| Recall | 77.78% |
| F1 | 0.7856 |
| ROC-AUC | 0.8613 |
| True positives | 700 |
| True negatives | 718 |
| False positives | 182 |
| False negatives | 200 |

CPU and CUDA training curves agreed across 61 recorded checkpoints, with a maximum absolute metric difference of approximately 2.22e-16.

In the five-run warmed-up benchmark, CPU median epoch time was approximately 98.7 ms versus 167.4 ms for CUDA. This is a workload-specific observation, not evidence of general CPU superiority.

## Getting started on Windows

Install Python 3.13, then open PowerShell in the repository root.

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt

Run the automated tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

Verify the evaluation metrics using synthetic test vectors:

    .\.venv\Scripts\python.exe experiments\verify_evaluation_independent.py

The provided dependency lockfile covers NumPy and plotting libraries, but not PyTorch. GPU experiments require a separately installed compatible PyTorch CUDA build. See docs/REPRODUCIBILITY.md and docs/GPU_SETUP.md.

## Documentation

- Research manuscript: docs/RESEARCH_PAPER.md
- Final held-out results: docs/FINAL_RESULTS.md
- Locked evaluation protocol: docs/FINAL_EVALUATION_PROTOCOL.md
- Verification scope: docs/PUBLIC_VERIFICATION_SCOPE.md
- Reproducibility guide: docs/REPRODUCIBILITY.md
- Experimental records: results/

## Limitations and future work

Results apply to synthetic signals generated from the documented distribution. The model has a substantial training-to-test accuracy gap and produces both false alarms and missed transients.

No completed standalone classical-DSP detector comparison, real-sensor validation, FPGA implementation, embedded deployment, or hardware-engineering productivity improvement is claimed.

Future work may investigate conventional signal-processing baselines, real sensor recordings, quantization, inference latency, FPGA resource requirements, and power consumption.

## Status and license

Initial research study completed; manuscript and release package under review. Not peer reviewed.

The project is not yet publicly released. Selected license: Apache License 2.0 (see LICENSE). Public release remains pending.