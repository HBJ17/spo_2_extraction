import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, lowcut=0.7, highcut=4.0, fs=30, order=4):
    """
    Applies a Butterworth bandpass filter.
    """

    nyq = 0.5 * fs

    low = lowcut / nyq
    high = highcut / nyq

    high = min(high, 0.99)

    b, a = butter(order, [low, high], btype='bandpass')

    filtered = filtfilt(b, a, signal)

    return filtered