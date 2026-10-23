# Phase 2: window-level Random Forest / SVR

Branch: `ml/phase2-window-level`

## Protocol
5-fold GroupKFold by `subject_id`, scaler fit on training folds, every 5-second window scored on its own (no video-level median), accuracy = share of windows within ±2 SpO2 points; 17 features, no formula feature.

Data: `dataset.csv`, 1,376 five-second windows from 171 videos (2-second step), windowed from the 30-second fingertip videos.

## Accuracy

Documented: **83.0%**, from Project_Journey_Documentation.md, Phase 2 (stated as ~83%). Not re-measured.

Measured with scikit-learn:

| Model | MAE | R2 | Accuracy (±2 SpO2) |
|---|---|---|---|
| Random Forest Regressor | _pending_ | _pending_ | _pending_ |
| Support Vector Regressor | _pending_ | _pending_ | _pending_ |

The measured row stays `pending` until `benchmark_phase2_window_level.py` has been run. It writes `results/*/metrics.json`, which replaces the pending files in this branch.

## Notes
- Reconstruction. The original Phase 2 script was not kept, so hyperparameters and the exact accuracy definition are assumptions.
- `subject_id` is the video file name, so the folds split by video. If one person appears in several videos, the same person can sit in both train and test folds.
- The ±2 accuracy here is not the same metric as the formula method's ~90% (see `method/formula-based`).
