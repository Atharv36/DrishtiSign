"""
generate_word_poses.py
─────────────────────────────────────────────────────────────────
One-off script: extracts avatar pose data for the 8 word signs
(Hello, Thankyou, Bye, Deaf, NotOk, Pen, Please, Yes) from the same
sign_language_dataset1/ images already used to train the classifier,
so Hand_avatar.py can play them back as whole-word signs instead of
falling back to fingerspelling.

Reuses the same MediaPipe extraction approach as record_gesture.py
(which only ever covered the alphabet) and writes {Word}.json files
in the exact 21-point [x, y] format Hand_avatar.py already reads for
the letters — run once, output sits next to A.json, B.json, etc.

Usage:  python generate_word_poses.py
"""

import os, json, glob
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LANDMARKER = os.path.join(BASE_DIR, "hand_landmarker.task")
WORDS_DIR  = os.path.join(BASE_DIR, "sign_language_dataset1")

WORD_CLASSES     = ["Bye", "Deaf", "Hello", "NotOk", "Pen", "Please", "Thankyou", "Yes"]
FRAMES_PER_WORD  = 32   # matches the letter JSONs (see A.json)


def build_detector():
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(detector, img_path):
    """Same shape as record_gesture.py: [[x, y], ...] x 21, or None."""
    import cv2
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    try:
        result = detector.detect(mp_img)
    except Exception:
        return None
    if not result or not result.hand_landmarks:
        return None
    return [[float(p.x), float(p.y)] for p in result.hand_landmarks[0]]


def main():
    print("Initialising MediaPipe Hand Landmarker...")
    detector = build_detector()
    print("Detector ready.\n")

    for word in WORD_CLASSES:
        word_dir = os.path.join(WORDS_DIR, word)
        if not os.path.isdir(word_dir):
            print(f"  [ skip ]  '{word}' — folder not found: {word_dir}")
            continue

        img_paths = sorted(
            glob.glob(os.path.join(word_dir, "*.jpg"))  +
            glob.glob(os.path.join(word_dir, "*.jpeg")) +
            glob.glob(os.path.join(word_dir, "*.png"))
        )
        if not img_paths:
            print(f"  [ skip ]  '{word}' — no images found")
            continue

        step    = max(1, len(img_paths) // FRAMES_PER_WORD)
        sampled = img_paths[::step][:FRAMES_PER_WORD]

        frames, missed = [], 0
        for img_path in sampled:
            lm = extract_landmarks(detector, img_path)
            if lm:
                frames.append(lm)
            else:
                missed += 1

        if len(frames) < 2:
            print(f"  [ skip ]  '{word}' — only {len(frames)} hands detected ({missed} missed)")
            continue

        out_path = os.path.join(BASE_DIR, f"{word}.json")
        with open(out_path, "w") as f:
            json.dump(frames, f)

        print(f"  [  ok  ]  {word:10s} — {len(frames):3d} frames ({missed} missed)  →  {word}.json")

    detector.close()
    print("\nDone. Word signs will now animate as whole gestures instead of fingerspelling.")


if __name__ == "__main__":
    main()
