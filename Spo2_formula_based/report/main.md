# 📄 main.py

> **Purpose**: Serves as the central coordinator for extraction, calculation, and reporting of SpO2 data.

---

## 🛠️ Overview
`main.py` is the conductor of the entire analysis orchestration. it directs the flow from video-based signal extraction to physiological calculation, and finally interprets those results into a health status report for the user.

## ⚡ Key Highlights
- **Role**: Application entry point and orchestrator.
- **Workflow**: 1. Signal Extraction → 2. SpO2 Calculation → 3. Status Assessment.
- **Data Path**: Video File ➡ Raw RGB Signals ➡ Cleaned AC/DC Components ➡ Medical Estimate.

---

## 🧠 Theoretical Background

### The System-Level Pipeline
This project implements a multi-stage **Digital Signal Processing (DSP)** pipeline to bridge the gap between a standard video file and a medical health indicator.

1.  **Phase 1: Transduction (Optical to Digital)**
    In `extract_signal.py`, the system converts spatial video information into temporal signal information. This is synonymous with how a light sensor on a smartwatch works.
2.  **Phase 2: Denoising & Feature Extraction**
    In `calculate_spo2.py` and `bandpass_filter.py`, the system separates the "useful data" (the heartbeat pulse) from the "static data" (tissue reflectance). This is a classical signal-processing challenge where we must isolate a very small pulsating signal (AC) on top of a very large constant signal (DC).
3.  **Phase 3: Clinical Calibration**
    The final logic in `main.py` maps the mathematical result (`R`) to a clinical grade:
    -   **95% - 100%**: Normal oxygen saturation.
    -   **90% - 94%**: Mild Hypoxemia (low oxygen).
    -   **< 90%**: Critical level requiring medical intervention.

By integrating these specialized modules, the software acts as a "Virtual Pulse Oximeter."

---

## 🔍 Line-by-Line Breakdown

### I. Dependencies & Pathing
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 1-2 | `from ... import ...` | Links the extraction and calculation modules into one workflow. |
| 5 | `VIDEO_PATH = "..."` | Configures the source video for the analysis attempt. |

### II. Phase 1: Signal Retrieval
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 15 | `extract_signals(VIDEO_PATH)` | Triggers the module that converts pixels into mathematical signals. |
| 18 | `calculate_spo2(...)` | Passes the extracted raw signals to the SpO2 algorithm. |

### III. Phase 2: User Reporting
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 24-29| `print(...)` | Presents a breakdown of the AC, DC, and final SpO2 values. |
| 31-36| `if spo2 >= 95...` | Categorizes the resulting number into a health-grade diagnostic. |
| 38 | `print(f"Status : {status}")` | Final output showing whether the data indicates a healthy level. |
