"""
build_frame_dataset_from_videos.py
------------------------------------
Walks every session/condition folder under dataset/, extracts several
evenly-spaced frames from each .MOV, detects + crops the hand with
MediaPipe (same crop style as livepredict.py), and saves the results
into an image folder organized one-folder-per-sign-label.

Run in your mediapipe-enabled environment:
    source venv_mp/bin/activate
    python build_frame_dataset_from_videos.py
"""

import os
import re
import cv2
import numpy as np
import mediapipe as mp

DATASET_ROOT = "dataset"
OUTPUT_PATH = "dataset_frames"
FRAMES_PER_VIDEO = 8          # how many frames to sample from each video
CROP_PADDING = 0.25           # must match livepredict.py's CROP_PADDING

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)


def extract_label(filename):
    """'S1_DHA.MOV' -> 'DHA'. Adjust if your filenames differ."""
    name = os.path.splitext(filename)[0]
    match = re.search(r"S\d+_(.+)", name)
    return match.group(1) if match else name


def get_hand_bbox(hand_landmarks, frame_w, frame_h, padding=CROP_PADDING):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = x_max - x_min
    box_h = y_max - y_min

    x_min -= box_w * padding
    x_max += box_w * padding
    y_min -= box_h * padding
    y_max += box_h * padding

    x1 = max(int(x_min * frame_w), 0)
    y1 = max(int(y_min * frame_h), 0)
    x2 = min(int(x_max * frame_w), frame_w)
    y2 = min(int(y_max * frame_h), frame_h)

    return x1, y1, x2, y2


def process_video(video_path, label, out_dir, video_index):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    frame_numbers = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO).astype(int)
    saved = 0

    for i, frame_number in enumerate(frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if not success:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = get_hand_bbox(results.multi_hand_landmarks[0], w, h)
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                out_name = f"{label}_{video_index}_{i}.jpg"
                cv2.imwrite(os.path.join(out_dir, out_name), crop)
                saved += 1

    cap.release()
    return saved


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    video_paths = []
    for root, _, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.lower().endswith(".mov"):
                video_paths.append(os.path.join(root, f))

    print(f"Found {len(video_paths)} video files across all sessions/conditions.")

    total_saved = 0
    for idx, video_path in enumerate(sorted(video_paths)):
        label = extract_label(os.path.basename(video_path))
        out_dir = os.path.join(OUTPUT_PATH, label)
        os.makedirs(out_dir, exist_ok=True)

        saved = process_video(video_path, label, out_dir, idx)
        total_saved += saved

        if idx % 20 == 0:
            print(f"[{idx}/{len(video_paths)}] processed, {total_saved} frames saved so far")

    print(f"\nDone. Total frames saved: {total_saved}")


if __name__ == "__main__":
    main()