# Linear Regression

Branch: `ml/linear-regression`

## Protocol
5-fold GroupKFold by `subject_id` (one id per video), scaler fit on training folds, video-level median of window predictions, accuracy = share of videos within ±2 SpO2 points.

Data: `dataset.csv`, 1,376 five-second windows from 171 videos (2-second step), windowed from the 30-second fingertip videos.

## Accuracy

Documented: **91.04%**, from Project_Journey_Documentation.md, Phase 3 (Linear Regression, 18 features, video-level median). Not re-measured.

Measured with scikit-learn:

| Model | MAE | R2 | Accuracy (±2 SpO2) |
|---|---|---|---|
| Linear Regression | _pending_ | _pending_ | _pending_ |

The measured row stays `pending` until `benchmark_linear_regression.py` has been run. It writes `results/*/metrics.json`, which replaces the pending files in this branch.

## Notes
- Same model and protocol as the Phase 3 result, so it shares that documented figure.
- `subject_id` is the video file name, so the folds split by video. If one person appears in several videos, the same person can sit in both train and test folds.
- The ±2 accuracy here is not the same metric as the formula method's ~90% (see `method/formula-based`).
