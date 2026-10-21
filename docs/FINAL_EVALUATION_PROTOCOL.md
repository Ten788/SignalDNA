# SignalDNA Final Evaluation Protocol — LOCKED

Status: Approved by project owner on 2026-10-21. Held-out evaluation NOT YET AUTHORIZED.

## Primary model
- Checkpoint: checkpoints/numpy_final_60.npz
- Epoch: 60
- Selection: highest validation ROC-AUC; earliest epoch breaks ties
- SHA256: 846137EE4333F7FAC7990E29FD6F7D0EAA59E6D1153F9A6BF6775F9C21A72975

## Secondary model
- Checkpoint: checkpoints/numpy_best_validation.npz
- Epoch: 48
- Selection: minimum validation loss
- SHA256: 1D1D72CD6D68659AE9DD4001AC21842A59921CDF9807910966B03EC2B938A243
- Report as a validation-based comparison; do not substitute for the primary after observing test results.

## Final held-out evaluation
- Dataset: 1,800 held-out synthetic examples, parent seed 2026
- Classification threshold: 0.5, fixed
- Primary metrics: accuracy, precision, recall, F1, ROC-AUC, TP, TN, FP, FN
- Report validation-to-test differences and methodological limitations
- Evaluate the locked primary model once after separate explicit authorization
- Do not tune thresholds, select checkpoints, retrain, or rerun to improve the test result
- Any subsequent test-informed work must be clearly identified as exploratory

## Pre-test evidence
- results/pretest_checkpoint_verification_v1.json
- experiments/verify_pretest_checkpoints.py
- Both checkpoints reproduced archived validation metrics within 1e-12
- The verification generated validation examples only; held-out examples were not constructed

## Gate
The project owner's approval of this protocol authorizes pre-test verification and documentation only. A separate explicit decision is required before held-out evaluation.