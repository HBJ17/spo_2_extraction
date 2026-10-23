"""Benchmark: Random Forest Classifier (depth 10)

Protocol: 5-fold GroupKFold by `subject_id` (one id per video), scaler fit on training folds, video-level median of window predictions, accuracy = share of videos within ±2 SpO2 points.

Run from the Spo2_ML folder:
    python benchmark_random_forest_classifier.py
"""
import argparse

from sklearn.ensemble import RandomForestClassifier
from benchmark_common import run_benchmark

BENCHMARK = "random_forest_classifier"


def build_model():
    return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest Classifier (depth 10) benchmark")
    parser.add_argument("--dataset", default=None, help="Path to dataset.csv (default: next to this script)")
    parser.add_argument("--no-write", action="store_true", help="Print scores without writing results/")
    args = parser.parse_args()

    run_benchmark(BENCHMARK, build_model, dataset_csv=args.dataset,
                  use_formula_feature=True, aggregate=True, write=not args.no_write)
