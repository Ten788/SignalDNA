# SignalDNA - Milestones

Status: M0 proposal

M0 - Requirements, architecture, verification, environment readiness.
M1 - Deterministic signal generator and dataset integrity tests.
M2 - Original neural-network engine and gradient verification.
M3 - Training pipeline, checkpoints, and learning curves.
M4 - Conventional baseline and fair model comparison.
M5 - Signal Mutation experiments and robustness analysis.
M6 - Superseded by approved CPU/GPU study revision below.

## Implementation Boundaries
Use isolated Python environment; do not modify Espressif tooling.
Prefer NumPy and lightweight plotting; defer PyTorch comparison.
No GUI framework until core experiments are validated.
No production implementation before M0 review and approval.

## Completion Principle
Each milestone must produce a measurable deliverable and verification evidence.

## Approved CPU/GPU Study Revision
M6 - PyTorch CPU/CUDA implementation, numerical equivalence checks, training and inference benchmarks.
M7 - Reproducible visualizations, technical study, and public demonstration.
The original M6 visualization and documentation deliverables move to M7.
Benchmark GPU compute and end-to-end runtime separately, including transfer overhead.
CPU remains the reference implementation and supported fallback.
This scope revision does not authorize production implementation.

