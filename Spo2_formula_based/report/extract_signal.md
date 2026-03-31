# 📄 extract_signal.py

> **Purpose**: Uses OpenCV to process video files and extract raw RGB intensity time-series data.

---

## 🛠️ Overview
This module acts as the "Optical Sensor" of the project. It reads a video file frame-by-frame and calculates the average brightness of the Red, Green, and Blue channels for each frame. These averages create the "Raw PPG Signal" used for health analysis.

## ⚡ Key Highlights
- **Input**: Path to a video file (e.g., finger over a camera lens).
- **Output**: Three NumPy arrays (R, G, B raw) and the video's Frame Rate (FPS).
- **Tech Stack**: OpenCV (`cv2`) for frame access and NumPy for spatial averaging.

---

## 🧠 Theoretical Background

### Remote Photoplethysmography (rPPG)
**Photoplethysmography (PPG)** is an optical technique used to detect volumetric changes in blood in the microvascular bed of tissue. While medical pulse oximeters use LEDs, **Remote PPG (rPPG)** uses ambient light and a camera sensor.
-   When the heart beats, a pulse of blood enters the capillaries.
-   This pulse increases the blood volume in the tissue, which increases light absorption.
-   The camera sensor detects this as a microscopic decrease in skin brightness.

### Why Green Light?
While pulse oximetry uses Red and Infrared, consumer cameras are most sensitive to **Green Light** because its wavelength penetrates human tissue just enough to reach the capillary bed without being absorbed too quickly by deeper layers. This makes the Green channel the most reliable "oscillator" for detecting heart rate and SpO2.

### Spatial Averaging for Noise Reduction
Instead of looking at a single pixel, the code calculates the **Mean Intensity** across the entire frame. This "Spatial Averaging" dramatically improves the Signal-to-Noise Ratio (SNR) by averaging out the electronic noise of individual pixels and small sensor artifacts.

---

## 🔍 Line-by-Line Breakdown

### I. Initialization & Metadata
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 1 | `import cv2 / import numpy as np` | Dependencies for video decoding and mathematical averaging. |
| 12 | `cap = cv2.VideoCapture(...)` | Opens the video stream for reading. |
| 17 | `fs = cap.get(...)` | Detects the video's FPS, which dictates the timing of the signals. |
| 23-25| `print(...)` | Informative console logging for the user during the extraction. |

### II. The Extraction Loop
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 30 | `while True:` | Iterates through every single frame in the video file. |
| 31 | `ret, frame = cap.read()` | Fetches the next image frame. `ret` confirms if reading succeeded. |
| 35-37| `np.mean(frame[:, :, 0...2])` | Computes the "Spatial Mean" of Blue, Green, and Red channels. |
| 39-41| `r_raw.append(...)` | Stores the average intensity as a point in the time-series signal. |

### III. Cleanup & Output
| Line | Code | Why it's happening |
| :--- | :--- | :--- |
| 45 | `cap.release()` | Explicitly releases the video file handle to free system RAM. |
| 49 | `return np.array(...), fs` | Returns ready-to-use arrays and the sampling frequency. |
