"""Benchmark: Phase 2: window-level Random Forest / SVR

Protocol: 5-fold GroupKFold by `subject_id`, scaler fit on training folds, every 5-second window scored on its own (no video-level median), accuracy = share of windows within ±2 SpO2 points; 17 features, no formula feature.

Run from the Spo2_ML folder:
    python benchmark_phase2_window_level.py
"""
import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from benchmark_common import run_benchmark

MODELS = {
    "random_forest_regressor": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    "svr": lambda: SVR(),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: window-level Random Forest / SVR benchmark")
    parser.add_argument("--dataset", default=None, help="Path to dataset.csv (default: next to this script)")
    parser.add_argument("--no-write", action="store_true", help="Print scores without writing results/")
    args = parser.parse_args()

    for key, factory in MODELS.items():
        run_benchmark("phase2_window_level__" + key, factory, dataset_csv=args.dataset,
                      use_formula_feature=False, aggregate=False, write=not args.no_write)
