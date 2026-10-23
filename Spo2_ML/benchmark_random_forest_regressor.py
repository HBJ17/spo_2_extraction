"""Benchmark: Random Forest Regressor (depth 5)

Protocol: 5-fold GroupKFold by `subject_id` (one id per video), scaler fit on training folds, video-level median of window predictions, accuracy = share of videos within ±2 SpO2 points.

Run from the Spo2_ML folder:
    python benchmark_random_forest_regressor.py
"""
import argparse

from sklearn.ensemble import RandomForestRegressor
from benchmark_common import run_benchmark

BENCHMARK = "random_forest_regressor"


def build_model():
    return RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest Regressor (depth 5) benchmark")
    parser.add_argument("--dataset", default=None, help="Path to dataset.csv (default: next to this script)")
    parser.add_argument("--no-write", action="store_true", help="Print scores without writing results/")
    args = parser.parse_args()

    run_benchmark(BENCHMARK, build_model, dataset_csv=args.dataset,
                  use_formula_feature=True, aggregate=True, write=not args.no_write)
