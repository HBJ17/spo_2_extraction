# Random Forest Classifier (depth 10)

Branch: `ml/random-forest-classifier`

## Protocol
5-fold GroupKFold by `subject_id` (one id per video), scaler fit on training folds, video-level median of window predictions, accuracy = share of videos within ±2 SpO2 points.

Data: `dataset.csv`, 1,376 five-second windows from 171 videos (2-second step), windowed from the 30-second fingertip videos.

## Accuracy

Documented: No figure was documented for this configuration.

Measured with scikit-learn:

| Model | MAE | R2 | Accuracy (±2 SpO2) |
|---|---|---|---|
| Random Forest Classifier | _pending_ | _pending_ | _pending_ |

The measured row stays `pending` until `benchmark_random_forest_classifier.py` has been run. It writes `results/*/metrics.json`, which replaces the pending files in this branch.

## Notes
- Treats each integer SpO2 value as a class. Configuration copied from `train_model.py`; no score was recorded.
- `subject_id` is the video file name, so the folds split by video. If one person appears in several videos, the same person can sit in both train and test folds.
- The ±2 accuracy here is not the same metric as the formula method's ~90% (see `method/formula-based`).
