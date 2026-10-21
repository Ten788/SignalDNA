"""Consolidate SignalDNA research manuscript from verified evidence."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
paper = root / "docs/RESEARCH_PAPER.md"
methods_file = root / "docs/METHODS_DRAFT.md"
results_file = root / "docs/FINAL_RESULTS.md"

for path in (paper, methods_file, results_file):
    if not path.is_file():
        raise RuntimeError(f"Missing required source: {path}")

status = subprocess.run(
    ["git", "status", "--porcelain", "--", "docs/RESEARCH_PAPER.md"],
    cwd=root, capture_output=True, text=True, check=True
)
if status.stdout.strip():
    raise RuntimeError("Manuscript has uncommitted changes; refusing overwrite")

original = paper.read_text(encoding="utf-8")
marker = "## Research Motivation and Personal Learning Objectives"
if original.count(marker) != 1:
    raise RuntimeError("Expected one personal motivation section")
if "Pending final evaluation" not in original:
    raise RuntimeError("Unexpected manuscript state; refusing overwrite")

motivation = original.split(marker, 1)[1].strip()

methods = methods_file.read_text(encoding="utf-8")
start = "## Dataset construction"
end = "## Remaining methodology verification"
if start not in methods or end not in methods:
    raise RuntimeError("Unexpected Methods structure")
method_body = methods.split(start, 1)[1].split(end, 1)[0]
method_body = ("### Dataset construction" + method_body).strip()
method_body = method_body.replace("\n## ", "\n### ")

results = results_file.read_text(encoding="utf-8")
if "## Final results" not in results or "## Validation comparison" not in results:
    raise RuntimeError("Unexpected final-results structure")
result_table = results.split("## Final results", 1)[1].split(
    "## Validation comparison", 1
)[0].strip()

content = f"""# SignalDNA: Reproducible Neural-Network Training and CPU/GPU Performance Characterization

**Status:** Completed initial experimental study; manuscript under review. Not peer reviewed.

## Abstract

SignalDNA investigates reproducible neural-network training, generalization, and computational performance through synthetic transient detection. A 67,713-parameter feedforward neural network was implemented from first principles in NumPy and compared with a PyTorch CUDA training implementation. CPU and CUDA learning metrics agreed across 61 recorded checkpoints, with a maximum absolute difference of approximately 2.22e-16. The locked epoch-60 model achieved 78.78% accuracy and 0.8613 ROC-AUC in one authorized held-out evaluation of 1,800 waveforms. In the measured reference workload, CPU training was faster than CUDA. The study documents successful numerical replication, meaningful classification errors, a substantial generalization gap, and workload-specific computational trade-offs. Its broader purpose is to establish a foundation for future neural-network investigations in embedded systems and FPGA engineering.

## 1. Introduction

Neural networks may offer useful approaches to signal interpretation, hardware testing, design assistance, and engineering analysis. SignalDNA was undertaken to explore these possibilities by first understanding the underlying computational methods.

Instead of beginning with a pretrained neural network, the project developed a NumPy reference implementation with explicit forward propagation, backpropagation, and parameter updates. GPU acceleration was investigated after establishing and verifying this mathematical foundation.

The research questions concern numerical reproducibility, held-out generalization on synthetic signals, and CPU/GPU performance under controlled implementation comparisons.

### 1.1 Research motivation and personal learning objectives

{motivation}

## 2. Methods

{method_body}

### Evaluation and checkpoint selection

The classification threshold was fixed at 0.5. ROC-AUC was calculated using average ranks, with tied positive-negative score pairs receiving half credit. The evaluation implementation agreed with an independent pairwise reference across 104 synthetic verification cases.

The primary model was selected at epoch 60 for the highest recorded validation ROC-AUC, with earliest epoch breaking ties. A secondary epoch-48 checkpoint minimized validation loss. Both checkpoints reproduced archived validation metrics before the held-out evaluation.

A final evaluation protocol was approved and committed before the held-out test was accessed. The epoch-60 primary model was then evaluated once without test-informed model selection, retraining, or threshold adjustment.

## 3. Results

### 3.1 Final held-out classification

The primary model was evaluated on 1,800 balanced synthetic signals.

{result_table}

Among 900 transient-containing waveforms, 700 were correctly detected and 200 were missed. Among 900 background waveforms, 718 were correctly rejected and 182 generated false alarms.

Accuracy decreased from 79.56% on validation to 78.78% on the held-out test, a decline of 0.78 percentage points. ROC-AUC declined from 0.8742 to 0.8613, while binary cross-entropy increased from 0.5745 to 0.6112.

Training accuracy reached approximately 99.75%, substantially exceeding held-out accuracy. This generalization gap indicates that the learned training performance did not fully transfer to independently generated examples.

Evidence: `results/heldout_final_primary_v1.json` and `docs/FINAL_RESULTS.md`.

### 3.2 Numerical reproducibility

The CPU and CUDA implementations agreed across all 61 recorded training checkpoints. The maximum absolute metric difference was 2.2204460492503131e-16. Additional comparisons covered forward propagation, gradients, SGD updates, and training epochs.

Both training harnesses used the common NumPy evaluation implementation for logged metrics. Agreement of those logged metrics does not constitute an independent verification of every evaluation calculation.

### 3.3 CPU/GPU performance

The five-run warmed-up benchmark produced median epoch times of approximately 98.7 ms for CPU and 167.4 ms for CUDA. CUDA was approximately 1.70 times slower for this configuration.

CPU execution was faster at all four measured dataset sizes using float64 and batch size 64. Increasing the CUDA batch size reduced measured epoch time but changed the number of optimization updates.

CUDA float32 execution was slower than float64 in the tested precision experiments, including alternating-order measurements. A change to PyTorch matrix-multiplication precision settings did not eliminate the observed slowdown. These measurements do not establish the precise underlying kernel-level cause.

The archived JSON benchmark records contain the supporting observations.

## 4. Discussion

The experiments demonstrate reproducible training behavior across the two tested implementations and predictive discrimination on independently generated synthetic signals.

The classification results are encouraging but imperfect. False alarms, missed transients, and the large training-to-test gap indicate substantial room for improvement.

The computational results also challenge a simplistic assumption that a GPU must outperform a CPU. Performance depends on workload size, numerical precision, launch and transfer overhead, implementation details, and optimization strategy.

A major practical lesson is the importance of verification discipline: comparing independent implementations, maintaining immutable evidence, separating validation from held-out testing, and preserving unexpected negative results.

## 5. Limitations

The dataset is entirely synthetic and represents a constrained family of waveform-generating processes. The findings do not establish performance on real sensor recordings, changing acquisition conditions, or unfamiliar noise distributions.

The study used one principal network architecture, one principal training seed, and a single final held-out evaluation. Cross-seed uncertainty and confidence intervals were not established. Validation data was consulted repeatedly during exploratory development.

Performance findings apply only to the tested hardware, software, batch sizes, and workload configurations. No FPGA inference implementation, embedded deployment, power measurement, or hardware-design productivity improvement has been demonstrated.

## 6. Reproducibility and Evidence

The repository contains source code, experiments, checkpoints, archived JSON results, and a research evidence register at `results/research_evidence_register_v1.json`.

The final primary checkpoint and held-out results have recorded SHA-256 fingerprints. The evaluation attempt was locked against automatic repetition.

Before public release, environment documentation, executable reproduction instructions, third-party licensing, repository hygiene, and external references require review.

## 7. Future Work

Possible next steps include stronger classical signal-processing baselines, cross-seed analysis on development data, regularization experiments, failure-case analysis, and investigation using representative real-world sensor signals.

Future embedded and FPGA experiments may investigate quantization, inference latency, resource utilization, power consumption, streaming data paths, and whether neural networks provide measurable improvements over established engineering methods.

Any work informed by the completed held-out evaluation must be identified as exploratory rather than another untouched final evaluation.

## 8. Conclusion

SignalDNA provided an independently implemented neural-network reference, closely matching CPU/CUDA training results, and a documented held-out assessment of synthetic transient detection. The primary model reached 78.78% held-out accuracy and 0.8613 ROC-AUC while revealing substantial generalization and performance limitations.

Beyond these numerical results, the project established a reproducible learning and experimentation process. It is an initial foundation for further investigation of neural networks in embedded systems and FPGA engineering, not proof of benefits in those applications.

## References

Authoritative references and software citations require verification before publication.
"""

if content.count("## References") != 1:
    raise RuntimeError("Unexpected references structure")
if "Pending final evaluation" in content or "evaluation pending" in content:
    raise RuntimeError("Stale evaluation placeholder")
if "SignalDNA began as a practical exploration" not in content:
    raise RuntimeError("Personal motivation was not preserved")

paper.write_text(content, encoding="utf-8")
print(f"manuscript_lines={len(content.splitlines())}")
print(f"manuscript_words={len(content.split())}")
print("motivation_integrated=True")
print("heldout_results_integrated=True")
print("SIGNALDNA-MANUSCRIPT-CONSOLIDATION-213R: PASS")