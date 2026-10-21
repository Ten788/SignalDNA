# SignalDNA - Verification Plan

Status: M0 proposal

V01: Signal generator - dimensions, finite values, parameter bounds, deterministic seeds.
V02: Dataset integrity - independent splits, class balance, duplicate detection.
V03: Neural mathematics - finite-difference gradient checks.
V04: Training - controlled learnable fixture demonstrates decreasing loss.
V05: Reproducibility - repeatable results within documented tolerances.
V06: Classical baseline - validation-tuned residual/local-energy detector.
V07: Generalization - held-out test and controlled mutation experiments.
V08: Reporting - configurations, metrics, plots, and limitations.

## Fair Comparison
Use identical input waveforms and evaluation splits.
Select hyperparameters and detector thresholds without test-set access.
Report majority-class reference and conventional detector results.

## Release Gate
All required deterministic tests pass; failures and limitations documented.
