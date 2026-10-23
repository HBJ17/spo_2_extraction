"""Shared helpers for the SpO2 ML benchmarks.

Every benchmark branch evaluates one model on the same windowed rPPG feature
table (dataset.csv) with the same protocol, so the numbers are comparable:

  * StandardScaler is fit on the training folds only.
  * 5-fold GroupKFold, grouped by `subject_id` (one id per video).
  * Optional video-level aggregation: the median of a video's window
    predictions, rounded to the nearest integer.
  * Metrics: MAE, R2 and accuracy within +/-2 SpO2 points.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))

BASE_FEATURES = [
    'dc_r', 'dc_g', 'dc_b', 'ac_r', 'ac_g', 'ac_b',
    'ratio_r', 'ratio_g', 'ratio_b', 'RoR_g', 'RoR_b',
    'hr_r', 'hr_g', 'skew_r', 'kurt_r', 'skew_g', 'kurt_g',
]
FORMULA_FEATURE = 'formula_spo2'


def load_dataset(dataset_csv=None, use_formula_feature=True):
    """Return X, y, groups and the feature names used."""
    dataset_csv = dataset_csv or os.path.join(HERE, 'dataset.csv')
    df = pd.read_csv(dataset_csv).dropna()

    # The non-ML formula (110 - 25 * R) as an extra feature, so the model can
    # learn residuals on top of the baseline.
    df[FORMULA_FEATURE] = 110 - 25 * df['RoR_g']

    features = [f for f in BASE_FEATURES if f in df.columns]
    if use_formula_feature:
        features.append(FORMULA_FEATURE)

    X = df[features].values
    y = np.round(df['true_spo2'].values).astype(int)
    groups = df['subject_id'].values
    return X, y, groups, features


def score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'accuracy_within_2': float(np.mean(np.abs(y_true - y_pred) <= 2.0) * 100),
        'n': int(len(y_true)),
    }


def cross_validate(model_factory, X, y, groups, n_splits=5, aggregate=True):
    """Grouped K-fold. Scores per window, or per video when `aggregate`."""
    true_parts, pred_parts, group_parts = [], [], []

    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model = model_factory()
        model.fit(X_train, y[train_idx])

        true_parts.append(y[test_idx])
        pred_parts.append(model.predict(X_test))
        group_parts.append(groups[test_idx])

    frame = pd.DataFrame({
        'group': np.concatenate(group_parts),
        'true': np.concatenate(true_parts),
        'pred': np.concatenate(pred_parts),
    })

    if aggregate:
        per_video = frame.groupby('group').agg(true=('true', 'first'), pred=('pred', 'median'))
        return score(per_video['true'], np.round(per_video['pred']))
    return score(frame['true'], np.round(frame['pred']))


def run_benchmark(name, model_factory, dataset_csv=None, use_formula_feature=True,
                  aggregate=True, write=True):
    """Run one benchmark, print the scores and write results/<name>/metrics.json."""
    X, y, groups, features = load_dataset(dataset_csv, use_formula_feature)
    print(f"[{name}] {len(X)} windows, {len(np.unique(groups))} videos, {len(features)} features")
    print(f"[{name}] formula feature: {use_formula_feature}, video-level median: {aggregate}")

    metrics = cross_validate(model_factory, X, y, groups, aggregate=aggregate)
    unit = 'videos' if aggregate else 'windows'
    print(f"[{name}] scored on {metrics['n']} {unit}")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  R2:  {metrics['r2']:.3f}")
    print(f"  Accuracy (within +/-2 SpO2): {metrics['accuracy_within_2']:.2f}%")

    if write:
        import sklearn
        out_dir = os.path.join(HERE, 'results', name)
        os.makedirs(out_dir, exist_ok=True)
        record = {
            'benchmark': name,
            'status': 'measured',
            'measured_on': time.strftime('%Y-%m-%d'),
            'sklearn_version': sklearn.__version__,
            'protocol': {
                'cv': 'GroupKFold(5) by subject_id',
                'video_level_median': aggregate,
                'formula_feature': use_formula_feature,
                'n_features': len(features),
            },
            'metrics': metrics,
        }
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(record, f, indent=2)
        print(f"  Wrote {os.path.join(out_dir, 'metrics.json')}")
    return metrics
