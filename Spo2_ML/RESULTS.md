# Phase 3: video-level median + formula injection

Branch: `ml/phase3-formula-injection-groupkfold`

## Protocol
5-fold GroupKFold by `subject_id` (one id per video), scaler fit on training folds, video-level median of window predictions, accuracy = share of videos within ±2 SpO2 points; 17 features + the formula estimate as an 18th feature.

Data: `dataset.csv`, 1,376 five-second windows from 171 videos (2-second step), windowed from the 30-second fingertip videos.

## Accuracy

Documented: **91.04%**, from Project_Journey_Documentation.md, Phase 3. Not re-measured.

Measured with scikit-learn:

| Model | MAE | R2 | Accuracy (±2 SpO2) |
|---|---|---|---|
| Linear Regression | _pending_ | _pending_ | _pending_ |

The measured row stays `pending` until `benchmark_phase3_formula_injection.py` has been run. It writes `results/*/metrics.json`, which replaces the pending files in this branch.

## Notes
- The script has `--no-formula` and `--no-aggregation` switches so each Phase 3 change can be switched off to see its own effect.
- `subject_id` is the video file name, so the folds split by video. If one person appears in several videos, the same person can sit in both train and test folds.
- The ±2 accuracy here is not the same metric as the formula method's ~90% (see `method/formula-based`).
