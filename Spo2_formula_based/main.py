from extract_signal import extract_signals
from calculate_spo2 import calculate_spo2


VIDEO_PATH = r"D:\Studies\ML\DATA\Video\2510138.mp4"


if __name__ == "__main__":

    print()
    print("  SpO2 Analyzer — Terminal Mode")
    print()

    print("\n[1/3] Extracting RGB signals from video...")
    r_raw, g_raw, b_raw, fs = extract_signals(VIDEO_PATH)

    print("\n[2/3] Computing SpO2...")
    R, spo2, ac_r, dc_r, ac_g, dc_g = calculate_spo2(r_raw, g_raw, fs)

    print()
    print("  RESULTS")
    print()

    print(f"  DC_red         : {dc_r:.4f}")
    print(f"  AC_red         : {ac_r:.4f}")
    print(f"  DC_green       : {dc_g:.4f}")
    print(f"  AC_green       : {ac_g:.4f}")
    print(f"  R value        : {R:.4f}")
    print(f"  SpO2 estimate  : {spo2:.2f} %")

    if spo2 >= 95:
        status = "Normal (healthy oxygen level)"
    elif spo2 >= 90:
        status = "Mild Hypoxemia — monitor closely"
    else:
        status = "Low SpO2 — seek medical attention"

    print(f"  Status         : {status}")