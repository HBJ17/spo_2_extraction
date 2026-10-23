# Formula-based SpO2 (non-ML baseline)

Estimates SpO2 from a fingertip video without any trained model.

1. `extract_signal.py` averages the red, green and blue channels of every frame.
2. `bandpass_filter.py` applies a Butterworth band-pass filter to isolate the pulse.
3. `calculate_spo2.py` takes AC (standard deviation of the filtered signal) and DC (mean) for red and green, forms the ratio of ratios `R = (AC_red / DC_red) / (AC_green / DC_green)`, and returns `SpO2 = 110 - 25 * R`.
4. `main.py` runs the steps on one video and prints the result and a status line.

Per-module notes are in `report/`. Measured error against pulse-oximeter readings is in `RESULTS.md`.

## Run
```bash
pip install -r requirements-formula.txt
python main.py
```
`VIDEO_PATH` at the top of `main.py` points at the video to analyse. Edit it before running.
