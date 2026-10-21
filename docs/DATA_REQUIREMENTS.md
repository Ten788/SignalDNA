# SignalDNA - Data Requirements

Status: M0 proposal

## Task
Binary classification: detect transient disturbances in synthetic signals.

## Signal Variables
Fundamental frequency, harmonic content, amplitude, Gaussian noise, baseline drift, transient presence and characteristics.

## Data Integrity
Deterministic generation with recorded seeds and parameters; independent training, validation, and test examples; no duplicated waveforms across splits.

## Evaluation
Precision, recall, F1, confusion matrix, and comparison against a conventional detector.

## Robustness
Controlled signal mutation and held-out parameter ranges; report failures and limitations.

## Restrictions
Synthetic data only; no real-world diagnostic accuracy claims.

## Pending M0 Decisions
Sampling rate, sample length, parameter distributions, transient definition, split policy, and acceptance thresholds.

## Proposed Initial Signal Specification
Sampling rate: 2048 Hz
Duration: 0.5 seconds
Samples per signal: 1024
Fundamental frequency: 20-200 Hz
Amplitude: 0.5-1.5
Harmonics: 0-3 with randomized strengths
Gaussian noise: 0-15% of nominal amplitude
Baseline drift: low-frequency variation
Transient: Gaussian-shaped pulse with variable location, width, amplitude, and polarity
Class balance: 50% transient, 50% normal

## Dataset Design Safeguards
Avoid label leakage through generation artifacts or fixed transient locations.
Vary nuisance parameters independently of class labels.
Reserve harder transient strengths and parameter ranges for robustness evaluation.
Verify that conventional baselines and the neural model receive comparable information.

## Dataset Partition
Initial dataset: 12000 independently generated signals.
Training: 8400 (70%).
Validation: 1800 (15%).
Test: 1800 (15%).
Separate robustness suite with held-out signal conditions.
No duplicate samples or tuning against the final test set.

## Acceptance Gates
Gradient relative error below 1e-5 on representative differentiable cases.
Reproducible data and training within documented numerical tolerances.
Neural model exceeds balanced-class chance performance on the held-out test set.
Compare against at least one conventional transient detector.
Report precision, recall, F1, confusion matrix, and robustness results.
Document failures, limitations, and baseline wins honestly.
