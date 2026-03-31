# 📄 calculate_spo2.py

> **Purpose**: Implements the "Ratio of Ratios" algorithm to estimate blood oxygen levels (SpO2).

---

## 🛠️ Overview
This module takes raw color intensity signals and converts them into a meaningful health metric. By comparing the pulsating (AC) and steady (DC) components of Red and Green light, it determines how much oxygen is being carried by the blood.

## ⚡ Key Highlights
- **Input**: Raw Red/Green arrays and sampling frequency.
- **Output**: SpO2 percentage and the underlying calculated components.
- **Core Formula**: `SpO2 = 110 - 25 * R` (where R is the Ratio of Ratios).

---

## 🧠 Theoretical Background

### The Physics of Light Absorption
Blood oxygen saturation (SpO2) is calculated based on the different absorption characteristics of **Oxygenated Hemoglobin (HbO2)** and **Deoxygenated Hemoglobin (Hb)**:
-   **Red Light (~660nm)**: Absorbed more by deoxygenated blood.
-   **Infrared/Green Light (~530-940nm)**: Absorbed more by oxygenated blood.

### The "Ratio of Ratios" Method
Pulse oximeters use two wavelengths to calculate a ratio `R`:
$$R = \frac{AC_{\text{Red}} / DC_{\text{Red}}}{AC_{\text{Green}} / DC_{\text{Green}}}$$
-   **DC Component**: Represents light absorbed by tissue, bone, and non-pulsating blood.
-   **AC Component**: Represents the change in absorption caused by the "pulse" of new arterial blood during a heartbeat.

By dividing AC by DC, we "normalize" the signal, making it independent of the skin thickness or the initial brightness of the video.

### The Empirical Formula
The value of `R` is then mapped to SpO2 using an empirical formula (typically calibrated against medical grade sensors). In this project, we use the common approximation:
`SpO2 = 110 - 25 * R`
A higher `R` value indicates lower oxygen levels (usually because the Red pulse is larger relative to the Green pulse).

---

## 🔍 Line-by-Line Breakdown

### I. Signal Preparation
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 1 | `import numpy as np` | Used for calculating standard deviation (AC) and mean (DC). |
| 2 | `from bandpass_filter import ...` | Imports the filtering tool to remove noise before calculation. |
| 7-8 | `dc_red / dc_green = np.mean(...)` | Captures the base light intensity (non-pulsating part). |
| 10-11| `r_filtered / g_filtered = ...` | Cleans raw signals to extract only heart-related pulses. |

### II. Component Extraction (AC/DC)
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 13-14| `ac_red / ac_green = np.std(...)` | Measures the variance (pulse amplitude) of the cleaned signals. |
| 16-17| `if dc_red == 0 or ...` | Safety check: Prevents division by zero if the signal is invalid. |

### III. The SpO2 Algorithm
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 19 | `R = (ac_red / dc_red) / (ac_green / dc_green)` | Normalizes the relative absorption change across wavelengths. |
| 21 | `spo2 = 110 - 25 * R` | Converts the physics of light absorption into a medical percentage. |
| 23 | `return R, spo2, ...` | Outputs the result along with all data needed for verification. |
