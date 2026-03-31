# 📄 bandpass_filter.py

> **Purpose**: Applies a Butterworth bandpass filter to isolate heart rate signals from raw RGB data.

---

## 🛠️ Overview
This module is responsible for cleaning the raw PPG signals. It removes frequencies outside the typical human heart rate range (0.7 Hz to 4.0 Hz), ensuring that subsequent SpO2 calculations are based on actual pulsatile blood flow rather than noise (like motion or lighting changes).

## ⚡ Key Highlights
- **Input**: Raw 1D signal (array), sampling frequency (`fs`), and frequency boundaries.
- **Output**: A filtered, zero-phase shifted signal ready for pulse detection.
- **Algorithm**: 4th Order Butterworth Filter with `filtfilt` for zero-phase distortion.

---

## 🧠 Theoretical Background

### Why Bandpass Filtering?
Raw optical signals captured by a camera are extremely noisy. They contain:
1.  **DC Component**: The steady tissue/bone reflectance.
2.  **Low-Frequency Noise**: Breathing (0.1–0.5 Hz) and body sway.
3.  **High-Frequency Noise**: Fluorescent light flickering and camera sensor jitter.

By applying a **Bandpass Filter** (0.7 Hz to 4.0 Hz), we isolate the frequency band corresponding to **42 BPM to 240 BPM**, which is the valid range for human heart rates.

### The Butterworth Filter
We use a **Butterworth Filter** because it has a "maximally flat" frequency response in the passband. This means it doesn't distort the amplitude of the pulse signal within our range of interest, which is critical for accurate AC/DC ratio calculations.

### Zero-Phase Distortion (`filtfilt`)
Standard digital filters often introduce a "phase shift" or time delay. By using `filtfilt`, the code processes the data twice: once forward and once backward. This cancels out the phase shift, ensuring the peaks in the filtered signal align perfectly in time with the pulses in the raw video.

---

## 🔍 Line-by-Line Breakdown

### I. Dependencies & Imports
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 1 | `import numpy as np` | Provides fundamental array structures for signal data. |
| 2 | `from scipy.signal import butter, filtfilt` | Loads standard tools for designing and applying digital filters. |

### II. Function Configuration
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 5 | `def bandpass_filter(...)` | Defines typical pulse boundaries: 0.7Hz (~42 bpm) to 4.0Hz (~240 bpm). |
| 10 | `nyq = 0.5 * fs` | Calculates the Nyquist frequency (max frequency for the given `fs`). |
| 12-13| `low = ... / high = ...` | Normalizes cutoffs into the [0, 1] range required by digital filters. |
| 15 | `high = min(high, 0.99)` | Prevents illegal frequency values that would crash the filter design. |

### III. Processing & Execution
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 17 | `b, a = butter(...)` | Computes the mathematical filter coefficients based on the order and limits. |
| 19 | `filtered = filtfilt(...)` | Applies the filter forward and backward to eliminate time delays (phase lag). |
| 21 | `return filtered` | Outputs the refined signal for precise mathematical analysis. |
