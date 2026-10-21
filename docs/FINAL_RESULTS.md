# SignalDNA — Final Held-Out Results

Status: First authorized held-out evaluation, preserved without model or threshold tuning.

## Locked evaluation

- Primary model: epoch 60, selected by maximum validation ROC-AUC.
- Checkpoint SHA-256: 846137EE4333F7FAC7990E29FD6F7D0EAA59E6D1153F9A6BF6775F9C21A72975
- Classification threshold: 0.5.
- Dataset: 1,800 balanced synthetic waveforms (900 background, 900 transient).
- Result source: results\heldout_final_primary_v1.json
- Result SHA-256: B8DC3B0EDF1054248F8245184B56FB67BB1DC5F452B770FDF710D5B34F4F7CA8
- Attempt record: results\heldout_evaluation_attempt_v1.json

## Final results

| Metric | Held-out value |
|---|---:|
| Accuracy | 0.7877777777777778 |
| Precision | 0.7936507936507936 |
| Recall | 0.7777777777777778 |
| F1 | 0.7856341189674523 |
| ROC-AUC | 0.8612641975308643 |
| Binary cross-entropy | 0.6111594047177068 |
| True positives | 700 |
| True negatives | 718 |
| False positives | 182 |
| False negatives | 200 |

## Validation comparison

- Validation accuracy: 0.7955555555555556
- Validation ROC-AUC: 0.8741864197530864
- Validation loss: 0.5745085134980361
- Held-out accuracy decrease: 0.7778 percentage points
- Held-out ROC-AUC decrease: 0.012922

## Interpretation boundaries

Results apply to the specified synthetic signal distribution and the single approved evaluation. They do not establish real-world sensor performance, production reliability, or benefits for embedded/FPGA engineering workflows. The large gap between training and held-out accuracy warrants discussion. No confidence intervals, cross-seed replication, or hardware deployment results are claimed.

The epoch-48 secondary model was selected using validation loss and was not evaluated on the held-out set in this final primary-model test.