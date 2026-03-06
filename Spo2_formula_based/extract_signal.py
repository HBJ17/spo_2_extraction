import cv2
import numpy as np


def extract_signals(video_path):
    """
    Opens the video and reads every frame.
    Computes the spatial mean of the Red, Green, and Blue
    channels per frame to get the raw PPG signal.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fs = cap.get(cv2.CAP_PROP_FPS)
    if fs <= 0:
        fs = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video opened  : {video_path}")
    print(f"  Frame rate    : {fs:.2f} fps")
    print(f"  Total frames  : {total_frames}")

    r_raw, g_raw, b_raw = [], [], []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        b_mean = np.mean(frame[:, :, 0])
        g_mean = np.mean(frame[:, :, 1])
        r_mean = np.mean(frame[:, :, 2])

        b_raw.append(b_mean)
        g_raw.append(g_mean)
        r_raw.append(r_mean)

        frame_count += 1

    cap.release()

    print(f"  Frames read   : {frame_count}")

    return np.array(r_raw), np.array(g_raw), np.array(b_raw), fs