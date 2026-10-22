# SignalDNA - Release Artifact Classification

Status: Pre-publication classification; no release authorized.

## Proposed public additions
- docs/PUBLIC_VERIFICATION_SCOPE.md: documented scientific verification boundaries.
- results/runtime_environment_v1.json: later environment snapshot; explicitly not proof of historical benchmark conditions.
- experiments/capture_environment.py: optional environment-capture utility requiring PyTorch.
- results/numpy_training_v1.json: historical exploratory 10-epoch training record; not the primary 60-epoch result.

## Preserve locally; exclude from public release
- Root-level execution stdout/stderr logs and installation diagnostics.
- Exploratory root-level smoke scripts, including standalone DSP detector prototypes.
- Optional checkpoint and CUDA inspection utilities pending individual review.
- Pre-246 script backups.
- Rendered PDF QA pages and temporary brief-page images.

## Safeguards
- Do not delete untracked files during release preparation.
- Do not confuse exploratory experiments with the final primary evaluation.
- Do not represent the later environment snapshot as historical benchmark provenance.
- Do not imply a validated classical DSP baseline exists.
- No Git staging, commit, remote push, or publication is authorized by this manifest.
