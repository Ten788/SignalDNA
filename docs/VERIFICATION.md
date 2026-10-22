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

## Final Verification Reconciliation

Status: Evidence-based scope reconciliation; the preceding M0 verification plan remains historical.

- V01: Signal generation properties and deterministic behavior covered by passing unit tests.
- V02: Dataset dimensions, balance, reproducibility, and cross-split duplicate checks covered by passing unit tests.
- V03: Analytical gradients checked against finite differences in passing unit tests.
- V04: Learning behavior and reproducible training epochs covered by passing unit tests.
- V05: Deterministic tests, archived CPU/CUDA equivalence, and checkpoint reconstruction verified within documented configurations.
- V06: Controlled standalone classical DSP detector baseline not established; the balanced 50% reference is not a DSP benchmark.
- V07: One authorized final held-out evaluation completed; broad mutation robustness and multi-seed generalization not established.
- V08: Research results, experimental configurations, plots, limitations, and publication drafts archived.
- The 55 discovered CPU unit tests passed using freshly installed dependencies in both an isolated older checkout and the current source revision.
- These results do not imply universal correctness, real-world generalization, or FPGA/embedded performance.
