# Spo2_ML: shared ML pipeline

Everything the benchmark branches build on.

| File | Role |
|---|---|
| `convert_excel_to_csv.py` | Turns the reference spreadsheet into `ground_truth.csv` (`filename`, `spo2`) |
| `extract_features.py` | Reads a video, cuts it into 5-second windows (2-second step) and computes 17 features per window: DC and AC for R/G/B, normalised ratios, ratio of ratios, heart rate, skewness and kurtosis |
| `build_dataset.py` | Runs the extractor over every video and joins the reference SpO2, producing `dataset.csv` |
| `dataset.csv` | 1,376 windows from 171 videos, ready for training |
| `benchmark_common.py` | Loading, grouped cross-validation, scoring and result writing shared by every benchmark |

## Rebuild the dataset
```bash
python build_dataset.py --video_dir <folder with .mp4 files> --gt_csv ground_truth.csv --out_csv dataset.csv
```
Videos are not stored in git.

## Evaluation protocol
All benchmarks use 5-fold `GroupKFold` grouped by `subject_id`, which is the video file name. Folds therefore split by video. If one person recorded several videos, they can appear in both train and test folds.

Benchmarks live on their own branches. See the `README.md` on `main` for the list.


## Branch: `ml/hist-gradient-boosting`
Benchmark: **Histogram Gradient Boosting Classifier**. See `RESULTS.md` for the accuracy and `benchmark_hist_gradient_boosting.py` to reproduce it.

```bash
cd Spo2_ML
pip install -r requirements-ml.txt
python benchmark_hist_gradient_boosting.py
```
