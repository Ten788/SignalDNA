# SignalDNA — Technical Reference Register

Status: Authoritative sources identified; manuscript citation integration pending.
Verified: 2026-10-21.

## [1] Rectifier initialization

He, K., Zhang, X., Ren, S., and Sun, J. (2015).
"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification."
Proceedings of the IEEE International Conference on Computer Vision.
https://arxiv.org/abs/1502.01852

Use: Background for He-style variance-scaled weight initialization.
SignalDNA-specific implementation remains established by src/neural_network.py.
Do not imply the paper validates SignalDNA's architecture or reported performance.

## [2] NumPy random-stream generation

NumPy Developers. "numpy.random.SeedSequence."
https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html

Use: Description of reproducible spawned random streams.
Independent streams do not mathematically guarantee unique generated waveforms.
Dataset implementation: src/dataset.py.

## [3] PyTorch CUDA execution

PyTorch Contributors. "CUDA semantics."
https://docs.pytorch.org/docs/stable/notes/cuda.html

Use: CUDA asynchronous execution, synchronization, and timing methodology.
The SignalDNA benchmark implementation and results require separate source audit.
Do not claim that CUDA must be faster or that a particular GPU kernel was used.

## [4] ROC-AUC definition

scikit-learn Developers. "sklearn.metrics.roc_auc_score."
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

Use: Standard ROC-AUC interpretation and terminology.
SignalDNA does not depend on scikit-learn for its reported ROC-AUC.
Its implementation is src/evaluation.py and has independent pairwise verification.

## Publication safeguards

- Separate external methodological references from original experimental evidence.
- Verify exact package and driver versions locally before final manuscript release.
- Avoid claiming cross-platform bitwise reproducibility.
- Identify exploratory measurements and non-controlled comparisons explicitly.
- Inspect bibliographic completeness and link availability during final review.
- Do not alter original experimental results to match external sources.