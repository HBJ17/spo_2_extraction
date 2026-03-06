import numpy as np
from bandpass_filter import bandpass_filter


def calculate_spo2(r_raw, g_raw, fs):

    dc_red = np.mean(r_raw)
    dc_green = np.mean(g_raw)

    r_filtered = bandpass_filter(r_raw, fs=fs)
    g_filtered = bandpass_filter(g_raw, fs=fs)

    ac_red = np.std(r_filtered)
    ac_green = np.std(g_filtered)

    if dc_red == 0 or dc_green == 0 or ac_green == 0:
        raise ValueError("Signal is flat or empty — check your video.")

    R = (ac_red / dc_red) / (ac_green / dc_green)

    spo2 = 110 - 25 * R

    return R, spo2, ac_red, dc_red, ac_green, dc_green