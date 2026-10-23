# Phase 4: deep Random Forest, calibrated on the full dataset

Branch: `ml/phase4-deep-rf-calibrated`

## Model
`RandomForestRegressor(n_estimators=200, random_state=42)` with no depth limit, on 18 features (17 signal features plus the formula estimate), after `StandardScaler`. `train_model.py` fits it on all 1,376 windows and saves it, with the scaler and feature list, to `best_spo2_model.pkl`. `predict.py` extracts windows from a video, predicts each one and reports the median.

## Accuracy

| Measure | Value |
|---|---|
| Documented final accuracy | **99.25%** (±2 SpO2, per video, `Project_Journey_Documentation.md` Phase 4) |
| Re-measured | No |

## Read this before quoting 99.25%
The model is trained on every video and then scored on those same videos. An unconstrained forest can nearly memorise its training data, so this number measures fit, not accuracy on a new video.

The same script also runs the grouped cross-validation on four models before the final fit. That run is the honest estimate. Its per-model results are not stored. They are what the other `ml/*` branches re-measure, one model per branch.

## Reproduce
```bash
cd Spo2_ML
pip install -r requirements-ml.txt
python train_model.py            # prints cross-validation scores, then the calibrated accuracy
python predict.py <video.mp4>    # needs opencv-python for feature extraction
```
`train_model.py` also saves `spo2_performance_plot.png`.
