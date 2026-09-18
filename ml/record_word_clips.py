"""
record_word_clips.py
─────────────────────────────────────────────────────────────────
Records short webcam clips of word signs into ml/word_clips/{word}/,
the training data for the word model (train_word_model.py).

Why record rather than download: no dataset dependency or link rot, the
signs are guaranteed to match how YOU sign them (which is what the model
will see at inference), and the same clips double as reference clips for
Text->Sign.

It shows live MediaPipe detection while you record and reports how many
frames actually contained a visible hand - a clip where the hand wasn't
tracked is useless as training data, so you want to catch that immediately
rather than at training time.

Controls
    SPACE   record a clip of the current word
    N / P   next / previous word
    D       delete the last clip recorded for this word
    Q       quit

Run:  python record_word_clips.py
"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from word_sequences import VOCABULARY

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LANDMARKER = os.path.join(BASE_DIR, "hand_landmarker.task")
CLIPS_DIR  = os.path.join(BASE_DIR, "word_clips")

CLIP_SECONDS   = 2.0    # long enough for one full sign
COUNTDOWN      = 3      # seconds before recording starts
FPS            = 20
TARGET_CLIPS   = 10     # per word, for a usable model

GREEN, RED, WHITE, AMBER = (0, 220, 120), (60, 60, 235), (255, 255, 255), (0, 200, 255)


def clip_count(word):
    folder = os.path.join(CLIPS_DIR, word)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".mp4")])


def next_clip_path(word):
    folder = os.path.join(CLIPS_DIR, word)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{word}_{int(time.time()*1000)}.mp4")


def delete_last_clip(word):
    folder = os.path.join(CLIPS_DIR, word)
    if not os.path.isdir(folder):
        return None
    clips = sorted(f for f in os.listdir(folder) if f.endswith(".mp4"))
    if not clips:
        return None
    victim = os.path.join(folder, clips[-1])
    os.remove(victim)
    return os.path.basename(victim)


def has_hand(detector, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    except Exception:
        return False, None
    if res and res.hand_landmarks:
        return True, res.hand_landmarks[0]
    return False, None


def draw_landmarks(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    bones = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
             (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),
             (18,19),(19,20),(5,9),(9,13),(13,17)]
    for a, b in bones:
        cv2.line(frame, pts[a], pts[b], GREEN, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, WHITE, -1, cv2.LINE_AA)


def banner(frame, word, idx, status, status_color):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 92), (20, 20, 25), -1)
    n = clip_count(word)
    cv2.putText(frame, f"{word.upper()}", (16, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2, cv2.LINE_AA)
    tick = GREEN if n >= TARGET_CLIPS else AMBER
    cv2.putText(frame, f"{n}/{TARGET_CLIPS} clips", (16, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, tick, 2, cv2.LINE_AA)
    cv2.putText(frame, f"word {idx+1}/{len(VOCABULARY)}", (w - 190, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (w - 190, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 25), -1)
    cv2.putText(frame, "SPACE record   N/P word   D delete last   Q quit",
                (16, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 180), 1, cv2.LINE_AA)


def record_clip(cap, detector, word, size):
    """Countdown, then capture CLIP_SECONDS of video. Returns (path, hand_ratio)."""
    # ── countdown ─────────────────────────────────────────────
    start = time.time()
    while (elapsed := time.time() - start) < COUNTDOWN:
        ok, frame = cap.read()
        if not ok:
            return None, 0.0
        frame = cv2.flip(frame, 1)
        left = COUNTDOWN - int(elapsed)
        cv2.putText(frame, str(left), (size[0] // 2 - 40, size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, AMBER, 6, cv2.LINE_AA)
        banner(frame, word, VOCABULARY.index(word), "GET READY", AMBER)
        cv2.imshow("DrishtiSign - record word clips", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None, 0.0

    # ── record ────────────────────────────────────────────────
    path = next_clip_path(word)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
    frames_total = frames_with_hand = 0
    start = time.time()
    while time.time() - start < CLIP_SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        writer.write(frame)                      # save the clean frame
        frames_total += 1

        shown = frame.copy()
        found, lm = has_hand(detector, frame)
        if found:
            frames_with_hand += 1
            draw_landmarks(shown, lm)
        cv2.circle(shown, (size[0] - 40, 120), 12, RED, -1)
        banner(shown, word, VOCABULARY.index(word), "RECORDING", RED)
        cv2.imshow("DrishtiSign - record word clips", shown)
        cv2.waitKey(1)

    writer.release()
    ratio = frames_with_hand / max(frames_total, 1)
    return path, ratio


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.4,
        )
    )

    idx = 0
    status, status_color = "READY", GREEN
    print("Recording to:", CLIPS_DIR)
    print(f"Aim for {TARGET_CLIPS}+ clips per word, varying speed and distance.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        word = VOCABULARY[idx]

        found, lm = has_hand(detector, frame)
        if found:
            draw_landmarks(frame, lm)
        banner(frame, word, idx,
               status if status != "READY" else ("HAND OK" if found else "NO HAND"),
               status_color if status != "READY" else (GREEN if found else RED))
        cv2.imshow("DrishtiSign - record word clips", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = (idx + 1) % len(VOCABULARY)
            status, status_color = "READY", GREEN
        elif key == ord('p'):
            idx = (idx - 1) % len(VOCABULARY)
            status, status_color = "READY", GREEN
        elif key == ord('d'):
            removed = delete_last_clip(word)
            print(f"deleted {removed}" if removed else f"no clips to delete for '{word}'")
            status, status_color = "DELETED", AMBER
        elif key == ord(' '):
            path, ratio = record_clip(cap, detector, word, size)
            if path is None:
                break
            if ratio < 0.5:
                # Keep it, but make the problem obvious now rather than at training.
                print(f"  ! weak clip for '{word}': hand visible in only {ratio:.0%} "
                      f"of frames - consider deleting with D and redoing")
                status, status_color = "WEAK CLIP", RED
            else:
                print(f"  saved {os.path.basename(path)}  (hand visible {ratio:.0%})")
                status, status_color = "SAVED", GREEN

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\nClips per word:")
    total = 0
    for w in VOCABULARY:
        n = clip_count(w)
        total += n
        if n:
            flag = "" if n >= TARGET_CLIPS else "  (needs more)"
            print(f"  {w:12s} {n:3d}{flag}")
    print(f"\n{total} clips total. Train with:  python train_word_model.py")


if __name__ == "__main__":
    main()
