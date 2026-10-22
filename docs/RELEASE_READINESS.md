# SignalDNA — Public Release Readiness Register

Status: Pre-publication audit. No public release authorized.
Audit date: 2026-10-21.

## Verified completed research

- Original NumPy feedforward neural-network implementation.
- Deterministic balanced synthetic-signal dataset.
- NumPy/CUDA numerical equivalence testing.
- Archived training, validation, and performance experiments.
- Single authorized final held-out evaluation.
- Locked evaluation protocol and preserved result fingerprints.
- Consolidated research manuscript, pending final review.

## Release blockers

### R01 — Dependencies

The existing requirements.lock.txt lists NumPy and plotting dependencies but does not include PyTorch. Define and verify installation procedures for CPU-only use and optional CUDA experiments. Verify packaging and platform compatibility rather than assuming universal support.

### R02 — README

README.md is present and Git-tracked, covering motivation, architecture, results, limitations, tests, and reproduction instructions. Content audit completed; final publication review remains.

### R03 — License

Apache License 2.0 is present in the Git-tracked LICENSE file and identified in README.md. Complete the third-party code and asset obligations audit before public distribution.

### R04 — Checkpoint availability

.gitignore excludes checkpoint archives. Both original checkpoints were reconstructed byte-for-byte from committed source in an isolated worktree, matching their SHA-256 fingerprints. The directory-creation defect was fixed in commit 108bf1a. Public reconstruction instructions and fresh-install verification remain to be completed; never overwrite archived evidence.

### R05 — Stale architecture document

docs/ARCHITECTURE.md describes normalized inputs, early stopping, and planning-stage checkpoint selection. Audit against the implemented experimental procedure and mark original proposals separately from completed work.

### R06 — Original verification scope

docs/VERIFICATION.md specifies a classical baseline and additional verification objectives. Establish which objectives were completed, deferred, or superseded using actual test and experiment evidence. Do not claim unverified completion.

### R07 — Working-tree hygiene

Numerous untracked diagnostic files exist. Review and classify them before public release; do not delete blindly or include them automatically.

### R08 — Research publication

Integrate verified technical references, capture reproducibility commands, check benchmark methodology, and perform manuscript consistency review. Include research motivation and both positive and negative findings.

### R09 — Public assets

Prepare a reviewed one-page PDF brief, final manuscript, and platform-specific Reddit/LinkedIn drafts. Publishing or posting requires explicit owner approval.

## Scientific integrity safeguards

- Preserve the first authorized held-out result unchanged.
- Do not run additional held-out evaluations as release preparation.
- Do not adjust public claims to make negative results disappear.
- Distinguish source-derived facts, original measurements, exploratory experiments, and future research.
- Changes to approved VERGE governance or project requirements require the appropriate explicit review and approval.

## Release state

Research experiment: complete.
Publication package: in preparation.
Open-source release: NOT AUTHORIZED.
Social-media publication: NOT AUTHORIZED.