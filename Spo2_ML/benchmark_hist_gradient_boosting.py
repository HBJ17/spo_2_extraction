"""Benchmark: Histogram Gradient Boosting Classifier

Protocol: 5-fold GroupKFold by `subject_id` (one id per video), scaler fit on training folds, video-level median of window predictions, accuracy = share of videos within ±2 SpO2 points.

Run from the Spo2_ML folder:
    python benchmark_hist_gradient_boosting.py
"""
import argparse

from sklearn.ensemble import HistGradientBoostingClassifier
from benchmark_common import run_benchmark

BENCHMARK = "hist_gradient_boosting"


def build_model():
    return HistGradientBoostingClassifier(random_state=42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram Gradient Boosting Classifier benchmark")
    parser.add_argument("--dataset", default=None, help="Path to dataset.csv (default: next to this script)")
    parser.add_argument("--no-write", action="store_true", help="Print scores without writing results/")
    args = parser.parse_args()

    run_benchmark(BENCHMARK, build_model, dataset_csv=args.dataset,
                  use_formula_feature=True, aggregate=True, write=not args.no_write)
