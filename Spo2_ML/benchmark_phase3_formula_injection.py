"""Benchmark: Phase 3: video-level median + formula injection

Protocol: 5-fold GroupKFold by `subject_id` (one id per video), scaler fit on training folds, video-level median of window predictions, accuracy = share of videos within ±2 SpO2 points; 17 features + the formula estimate as an 18th feature.

Run from the Spo2_ML folder:
    python benchmark_phase3_formula_injection.py
"""
import argparse

from sklearn.linear_model import LinearRegression
from benchmark_common import run_benchmark

BENCHMARK = "phase3_formula_injection"


def build_model():
    return LinearRegression()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: video-level median + formula injection benchmark")
    parser.add_argument("--dataset", default=None, help="Path to dataset.csv (default: next to this script)")
    parser.add_argument("--no-write", action="store_true", help="Print scores without writing results/")
    parser.add_argument("--no-formula", action="store_true", help="Drop the formula estimate feature")
    parser.add_argument("--no-aggregation", action="store_true", help="Score windows instead of the per-video median")
    args = parser.parse_args()

    run_benchmark(BENCHMARK, build_model, dataset_csv=args.dataset,
                  use_formula_feature=not args.no_formula, aggregate=not args.no_aggregation, write=not args.no_write and not (args.no_formula or args.no_aggregation))
