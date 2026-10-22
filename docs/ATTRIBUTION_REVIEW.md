# SignalDNA — Attribution and Licensing Review

Status: LOCAL PREPARATION ONLY. Public release not authorized.

Selected license: Apache License, Version 2.0.
Official source: https://www.apache.org/licenses/LICENSE-2.0.txt
SPDX identifier: Apache-2.0.

## Required pre-publication checks

- Owner-selected attribution: Copyright 2026 David Boring.
- Confirm ownership and rights for all published materials before release.
- Verify that every published source file is original or appropriately licensed.
- Review third-party code, assets, figures, datasets, and documentation.
- Preserve applicable third-party copyright and attribution notices.
- Identify bundled dependencies separately from packages installed by users.
- Review whether additional NOTICE entries are legally required.
- Verify the full LICENSE file against the authoritative Apache text.
- Confirm that public materials do not include secrets or private files.
- Obtain explicit project-owner approval before publication.

The NOTICE file provides project attribution. Apache 2.0 does not
universally require an original project to create a NOTICE file, but
when included, applicable notices must be preserved by downstream
redistributors under Section 4(d).

No third-party attribution requirements have yet been independently
cleared by this review.
## Dependency metadata audit - Step 322

- All 11 pinned CPU dependencies returned installed package license metadata.
- Reported licenses include BSD, MIT, Apache, and package-specific permissive arrangements.
- NumPy reports multiple component licenses; Matplotlib metadata includes bundled third-party component notices.
- These packages are installed separately by users; the planned source release does not bundle their installed distributions.
- PyTorch/CUDA is optional and outside the pinned CPU dependency lockfile.
- Project-generated research figures have corresponding tracked generation scripts.
- License metadata inspection is evidence collection, not independent legal clearance.

Remaining checks:
- Confirm original ownership or authorized licensing of every distributed source file and asset.
- Verify the project Apache-2.0 LICENSE against the authoritative text.
- Confirm dependency attribution documentation and whether project NOTICE additions are necessary.
- Perform final release-content and sensitive-file review.
- Obtain explicit owner approval before any public distribution.

## Publication font evidence - Step 331

- Earlier superseded drafts embedded Microsoft Segoe UI and Georgia fonts; the current reviewed PDFs use DejaVu fonts.
- DejaVu Sans and Serif are available through the pinned Matplotlib installation.
- The complete DejaVu license is preserved at docs/licenses/LICENSE_DEJAVU.
- DejaVu licensing permits redistribution subject to its copyright, trademark, permission, modification, and other conditions.
- Completed: both publication generators use DejaVu fonts; regenerated PDFs were visually reviewed, and the complete DejaVu license is staged for distribution.
- Font replacement completed; broader third-party licensing review and public-release authorization remain open.
