# SignalDNA — Public Verification and Research Scope

Status: Automated unit tests passed; publication review remains open.
Date: 2026-10-21.

## Verified work
- Original NumPy neural network and deterministic synthetic dataset.
- CPU/CUDA numerical equivalence and archived learning curves.
- Independent evaluation-metric verification on 104 synthetic cases.
- Locked primary checkpoint and one authorized held-out evaluation.
- Held-out accuracy 78.78% and ROC-AUC 0.8613.
- Archived CPU/GPU performance measurements, including negative findings.

## Exploratory work
- High-pass and normalized-input neural-network experiments.
- Validation-only classification-threshold exploration.
- CUDA batch-size, precision, TF32, and workload-scaling experiments.

## Not established
- A controlled classical DSP detector comparison.
- Performance on real sensor recordings.
- Generalization across multiple independent training seeds.
- FPGA deployment, embedded inference, power or resource measurements.
- Demonstrated improvements in hardware design or verification workflows.

## Publication boundaries
- The balanced dataset has a 50% majority-class accuracy reference, not a measured classical DSP baseline.
- High-pass preprocessing followed by neural-network training is not a conventional standalone detector.
- The completed final held-out evaluation is preserved and must not be repeated to improve results.
- Unit-test success supports tested behaviors, not universal correctness.
- Release blockers include installation instructions, licensing, references and repository hygiene.