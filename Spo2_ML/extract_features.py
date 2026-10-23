import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis

def bandpass_filter(signal, lowcut=0.7, highcut=4.0, fs=30, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    high = min(high, 0.99)
    b, a = butter(order, [low, high], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    return filtered

def calculate_hr_from_signal(signal, fs):
    # Calculate heart rate from the frequency domain using Welch's method
    f, Pxx = welch(signal, fs, nperseg=len(signal))
    # Restrict to realistic HR range: 42 to 240 BPM (0.7 Hz to 4.0 Hz)
    valid_idx = np.where((f >= 0.7) & (f <= 4.0))[0]
    if len(valid_idx) == 0:
        return 0
    f_valid = f[valid_idx]
    Pxx_valid = Pxx[valid_idx]
    peak_freq = f_valid[np.argmax(Pxx_valid)]
    return peak_freq * 60  # BPM

def extract_window_features(r_raw, g_raw, b_raw, fs):
    # DC components (mean of raw signal)
    dc_r, dc_g, dc_b = np.mean(r_raw), np.mean(g_raw), np.mean(b_raw)
    
    # Filter signals
    r_filt = bandpass_filter(r_raw, fs=fs)
    g_filt = bandpass_filter(g_raw, fs=fs)
    b_filt = bandpass_filter(b_raw, fs=fs)
    
    # AC components (std of filtered signal)
    ac_r, ac_g, ac_b = np.std(r_filt), np.std(g_filt), np.std(b_filt)
    
    # Handle zeros to avoid division by zero
    dc_r = max(dc_r, 1e-6)
    dc_g = max(dc_g, 1e-6)
    dc_b = max(dc_b, 1e-6)
    ac_g = max(ac_g, 1e-6)
    ac_b = max(ac_b, 1e-6)
    
    # Normalized ratios
    ratio_r = ac_r / dc_r
    ratio_g = ac_g / dc_g
    ratio_b = ac_b / dc_b
    
    # Ratio of Ratios
    RoR_g = ratio_r / ratio_g
    RoR_b = ratio_r / ratio_b
    
    # HR Estimation
    hr_r = calculate_hr_from_signal(r_filt, fs)
    hr_g = calculate_hr_from_signal(g_filt, fs)
    
    # Statistical features on filtered signals
    skew_r, kurt_r = skew(r_filt), kurtosis(r_filt)
    skew_g, kurt_g = skew(g_filt), kurtosis(g_filt)
    
    return {
        'dc_r': dc_r, 'dc_g': dc_g, 'dc_b': dc_b,
        'ac_r': ac_r, 'ac_g': ac_g, 'ac_b': ac_b,
        'ratio_r': ratio_r, 'ratio_g': ratio_g, 'ratio_b': ratio_b,
        'RoR_g': RoR_g, 'RoR_b': RoR_b,
        'hr_r': hr_r, 'hr_g': hr_g,
        'skew_r': skew_r, 'kurt_r': kurt_r,
        'skew_g': skew_g, 'kurt_g': kurt_g
    }

def process_video(video_path, window_sec=5, step_sec=2):
    """
    Reads a video and extracts features for multiple overlapping windows.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fs = cap.get(cv2.CAP_PROP_FPS)
    if fs <= 0: fs = 30
    
    r_raw, g_raw, b_raw = [], [], []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Crop to center to avoid background (simple ROI)
        h, w = frame.shape[:2]
        ch, cw = h//2, w//2
        # Use a 100x100 region in the center
        roi = frame[max(0, ch-50):ch+50, max(0, cw-50):cw+50]
        
        b_mean = np.mean(roi[:, :, 0])
        g_mean = np.mean(roi[:, :, 1])
        r_mean = np.mean(roi[:, :, 2])
        
        b_raw.append(b_mean)
        g_raw.append(g_mean)
        r_raw.append(r_mean)
        
    cap.release()
    
    r_raw = np.array(r_raw)
    g_raw = np.array(g_raw)
    b_raw = np.array(b_raw)
    
    total_frames = len(r_raw)
    window_frames = int(window_sec * fs)
    step_frames = int(step_sec * fs)
    
    features_list = []
    
    if total_frames < window_frames:
        # If video is shorter than window, process whole video as one window
        if total_frames > int(fs): # at least 1 second
            feat = extract_window_features(r_raw, g_raw, b_raw, fs)
            features_list.append(feat)
    else:
        for start in range(0, total_frames - window_frames + 1, step_frames):
            end = start + window_frames
            feat = extract_window_features(r_raw[start:end], g_raw[start:end], b_raw[start:end], fs)
            features_list.append(feat)
            
    return features_list

if __name__ == "__main__":
    # Test on a dummy signal if run directly
    print("Feature extraction script ready.")
