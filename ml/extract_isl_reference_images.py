"""
extract_isl_reference_images.py
─────────────────────────────────────────────────────────────────
One-off: pick one clean reference photo per ISL class out of the frame
dataset and write it into the frontend's static assets.

Learning mode and Text-to-Sign both look up SIGN_IMAGES[label] and fall back to
a plain text glyph when there's no photo. We already have thousands of cropped
hand frames, so there's no reason to show "KHA" as text when we can show the
actual handshape.

Selection: prefer a MIDDLE frame of a recording (the sampled frames run
0..7 across a video, so the first and last tend to catch the hand entering or
leaving the pose), and require MediaPipe to find a hand in it - the same check
that decided whether the frame was worth training on.

Run:  python extract_isl_reference_images.py
"""

import json
import os

import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from train_isl_model import FRAMES_DIR, LANDMARKER, parse_frame_name

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(BASE_DIR, "exported_model", "isl_labels.json")
OUT_DIR = os.path.join(BASE_DIR, "..", "frontend", "dristi", "src", "static", "isl")

MAX_DIM = 256          # these render at ~160px; anything larger is wasted bytes
JPEG_QUALITY = 85
PREFERRED_FRAMES = (3, 4, 2, 5)   # middle of the 0..7 sampled range, outward


def frame_sort_key(fname):
    """Middle frames first, then everything else."""
    stem = os.path.splitext(fname)[0]
    frame_idx = int(stem.rsplit("_", 1)[1])
    rank = PREFERRED_FRAMES.index(frame_idx) if frame_idx in PREFERRED_FRAMES else len(PREFERRED_FRAMES)
    return (rank, fname)


def main():
    labels = json.load(open(LABELS_PATH))
    os.makedirs(OUT_DIR, exist_ok=True)

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.4,
        )
    )

    written, missing = 0, []
    for label in labels:
        class_dir = os.path.join(FRAMES_DIR, label)
        if not os.path.isdir(class_dir):
            missing.append(label)
            continue

        candidates = sorted(
            (f for f in os.listdir(class_dir)
             if f.lower().endswith(".jpg") and parse_frame_name(f)),
            key=frame_sort_key,
        )

        for fname in candidates:
            path = os.path.join(class_dir, fname)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=np.array(img, dtype=np.uint8)))
            if not res.hand_landmarks:
                continue
            img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
            img.save(os.path.join(OUT_DIR, f"{label}.jpg"), "JPEG", quality=JPEG_QUALITY)
            written += 1
            break
        else:
            missing.append(label)

    detector.close()
    print(f"Wrote {written}/{len(labels)} reference images -> {OUT_DIR}")
    if missing:
        print(f"No usable frame for: {', '.join(missing)} (these fall back to a text glyph)")


if __name__ == "__main__":
    main()
