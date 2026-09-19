"""
audit_two_handed.py
─────────────────────────────────────────────────────────────────
Determines which vocabulary words are genuinely two-handed IN OUR OWN DATA,
rather than guessing from memory of ASL. Our extraction currently only reads
one hand (num_hands=1) - so a two-handed sign is losing half its information,
which plausibly explains some of the confusion/inaccuracy.

For each word, runs num_hands=2 detection over its clips and measures, per
frame with a detection: how often TWO distinct hands are found, and how far
apart they are (a second hand only barely in frame, or a tracking artifact,
shouldn't count - genuine bimanual signs have both hands clearly separated
and both moving).

Run:  python audit_two_handed.py
"""

import glob
import json
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from word_sequences import VOCABULARY

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LANDMARKER = os.path.join(BASE_DIR, "hand_landmarker.task")
CLIPS_DIR  = os.path.join(BASE_DIR, "word_clips")

SEPARATION_THRESHOLD = 0.15   # min wrist distance (normalized image coords) to count as "two hands genuinely apart"
TWO_HAND_FRAME_RATIO = 0.30   # fraction of detected frames needing 2 separated hands to call a word two-handed


def build_detector():
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.4,
        )
    )


def analyze_clip(detector, path, sample_every=3):
    cap = cv2.VideoCapture(path)
    total = two_hand = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            except Exception:
                res = None
            if res and res.hand_landmarks:
                total += 1
                if len(res.hand_landmarks) == 2:
                    w0 = res.hand_landmarks[0][0]
                    w1 = res.hand_landmarks[1][0]
                    dist = ((w0.x - w1.x) ** 2 + (w0.y - w1.y) ** 2) ** 0.5
                    if dist >= SEPARATION_THRESHOLD:
                        two_hand += 1
        i += 1
    cap.release()
    return total, two_hand


def main():
    detector = build_detector()
    results = {}

    print(f"{'word':12s} {'clips':>6s} {'frames':>7s} {'2-hand%':>8s}")
    for word in VOCABULARY:
        clips = sorted(glob.glob(os.path.join(CLIPS_DIR, word, "*.mp4")))[:6]  # sample a few clips/word
        if not clips:
            continue
        total = two_hand = 0
        for c in clips:
            t, h = analyze_clip(detector, c)
            total += t
            two_hand += h
        ratio = two_hand / total if total else 0.0
        results[word] = {"clips_checked": len(clips), "frames": total, "two_hand_ratio": ratio}
        flag = " <- TWO-HANDED" if ratio >= TWO_HAND_FRAME_RATIO else ""
        print(f"{word:12s} {len(clips):6d} {total:7d} {ratio:7.1%}{flag}")

    detector.close()

    two_handed = sorted(w for w, r in results.items() if r["two_hand_ratio"] >= TWO_HAND_FRAME_RATIO)
    print(f"\nTwo-handed in our data ({len(two_handed)}): {two_handed}")

    out = os.path.join(BASE_DIR, "exported_model", "two_hand_audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"threshold": TWO_HAND_FRAME_RATIO, "results": results, "two_handed_words": two_handed},
              open(out, "w"), indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
