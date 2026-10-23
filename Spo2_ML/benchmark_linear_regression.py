"""Benchmark: Linear Regression

Protocol: 5-fold GroupKFold by `subject_id` (one id per video), scaler fit on training folds, video-level median of window predictions, accuracy = share of videos within ±2 SpO2 points.

Run from the Spo2_ML folder:
    python benchmark_linear_regression.py
"""
import argparse

from sklearn.linear_model import LinearRegression
from benchmark_common import run_benchmark

BENCHMARK = "linear_regression"


def build_model():
    return LinearRegression()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Regression benchmark")
    parser.add_argument("--dataset", default=None, help="Path to dataset.csv (default: next to this script)")
    parser.add_argument("--no-write", action="store_true", help="Print scores without writing results/")
    args = parser.parse_args()

    run_benchmark(BENCHMARK, build_model, dataset_csv=args.dataset,
                  use_formula_feature=True, aggregate=True, write=not args.no_write)
