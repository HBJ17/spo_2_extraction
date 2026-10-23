# Formula-based method: results

Source: `results/spo2_comparison_results.csv`, the formula output compared with the reference SpO2 for 152 videos.

| Measure | Value |
|---|---|
| Videos | 152 |
| Mean absolute error | 8.86 SpO2 points |
| Mean percentage error | 9.09% |
| Accuracy as 100 − mean % error | 90.9% |
| Within ±2 SpO2 points | 12 of 152 (7.9%) |
| Within ±5 SpO2 points | 32 of 152 (21.1%) |
| Largest error | 24.85 points |
| Reference range | 89 to 100 |
| Highest formula output | 109.99 |

## What this says
- The "~90% accuracy" quoted in `Project_Journey_Documentation.md` matches 100 − mean % error.
- The formula reads high. Its output sits near 105–110 for videos whose reference is 95–99, so the error is a bias, not noise. A single offset or a calibrated fit would remove much of it.
- The ML branches report the share of predictions within ±2 SpO2 points. That is a stricter measure, so their figures cannot be compared directly with 90%.

Figures were computed from the CSV in this branch. The formula itself was not re-run.
