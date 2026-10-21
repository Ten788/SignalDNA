# SignalDNA: Reproducible Neural-Network Training and CPU/GPU Performance Characterization

**Status:** Research manuscript in preparation. Not peer reviewed.

## Abstract

Pending final evaluation and manuscript review. No held-out test results are reported yet.

## 1. Introduction

- Motivation: learning neural-network fundamentals through an independently implemented NumPy reference.
- Research questions: computational reproducibility, generalization, and CPU/GPU performance.
- Scope: synthetic transient-signal binary classification.

## 2. Methods

### 2.1 Dataset

- Synthetic binary classification task.
- Input length: 1,024 samples.
- Sampling rate: 2,048 Hz.
- Training: 8,400 examples.
- Validation: 1,800 examples.
- Held-out test: 1,800 examples, evaluation pending.
- Document generation procedure, class balance, and split isolation before submission.

### 2.2 Network and optimization

- Fully connected architecture: 1024-64-32-1.
- ReLU hidden activations and sigmoid output.
- Mini-batch stochastic gradient descent.
- Batch size 64; learning rate 0.01; 60 epochs.
- Seed 2026 for initialization and dataset generation.
- Document precise loss, initialization, and shuffle implementation.

### 2.3 Computational backends

- Independent NumPy forward and backward implementation.
- PyTorch CUDA automatic differentiation.
- Float64 numerical-equivalence reference.
- Hardware: NVIDIA GeForce RTX 4070 Laptop GPU.
- Record software environment and complete reproduction commands.

### 2.4 Experimental protocol

- Numerical equivalence across layers, SGD updates, and training epochs.
- Complete 61-checkpoint learning-curve comparison.
- Warmed-up CPU/GPU timing comparisons.
- Batch-size, precision, TF32, and dataset-size experiments.
- Explicitly distinguish exploratory timing from controlled inference.

## 3. Results

### 3.1 Learning and generalization

- Final training accuracy: 0.9975.
- Final validation accuracy: 0.7955555555555556.
- Final validation ROC-AUC: 0.8741864197530864.
- Discuss training-validation generalization gap.
- Best validation-loss checkpoint: epoch 48.

### 3.2 Computational reproducibility

- All 61 CPU/GPU learning-curve checkpoints passed numerical comparison.
- Maximum observed metric difference: 2.2204460492503131e-16.
- Same dataset, initialization, and training procedure.

### 3.3 Performance characterization

- CPU faster than CUDA at all four tested dataset sizes with batch 64 and float64.
- Larger CUDA mini-batches reduced measured epoch time.
- Float32 slower than float64 in both original and alternating-order measurements.
- PyTorch high-precision-mode experiment did not resolve the slowdown.
- Include exact distributions and experimental limitations.

## 4. Discussion

- Numerical agreement supports implementation correctness for tested conditions.
- Matching learning curves do not independently establish generalization.
- GPU performance depends on workload and implementation characteristics.
- Faster epochs with larger batches do not guarantee faster convergence.
- Distinguish measured findings from possible hardware explanations.

## 5. Limitations

- Synthetic data and one network architecture.
- Single GPU and host system.
- Limited benchmark repetitions and uncontrolled environmental variation.
- Backend implementations have different runtime overhead.
- Hyperparameter selection and validation reuse require documentation.
- Held-out test evaluation pending.

## 6. Reproducibility and Evidence

- Primary register: results/research_evidence_register_v1.json.
- Archived CPU and CUDA learning curves.
- Versioned benchmark scripts and JSON results.
- Preserve negative results and relevant execution failures.
- Record final Git revision and dependency versions.

## 7. Conclusion

Pending final evaluation, evidence audit, and manuscript review.

## References

Pending verification of authoritative technical references.