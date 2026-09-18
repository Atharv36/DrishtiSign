"""
build_sign_poses.py
─────────────────────────────────────────────────────────────────
Rebuilds every {sign}.json the avatar plays back, fixing two things
the original recording approach got wrong:

1. FLAT HANDS.  The old extraction saved only [x, y] image coordinates,
   so every landmark had z = 0 - a flat cardboard hand that looked like
   a garbled sliver as soon as the 3D camera rotated around it. MediaPipe
   also returns `hand_world_landmarks`: true metric 3D (meters, wrist-
   relative, real depth). We use those instead.

2. WORD SIGNS HAD NO MOTION.  A real ASL word sign is handshape +
   location + MOVEMENT. "Hello" is a flat hand at the temple arcing
   outward like a salute - a single frozen frame can never be "Hello".
   So for word signs we take the real handshape extracted from the
   dataset photos and animate it along the actual ASL trajectory,
   emitting keyframes the avatar interpolates between.

Letters stay single-frame, because fingerspelled letters genuinely ARE
static handshapes (J and Z are the exceptions - they involve motion -
and are handled with trajectories too).

Run:  python build_sign_poses.py
"""

import glob
import json
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LANDMARKER = os.path.join(BASE_DIR, "hand_landmarker.task")
WORDS_DIR  = os.path.join(BASE_DIR, "sign_language_dataset1")
ALPHABET_DIR = os.path.expanduser(
    "~/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/"
    "asl_alphabet_train/asl_alphabet_train"
)

SAMPLES_PER_SIGN = 18   # images sampled per sign to choose the best handshape from

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")      # J, Z handled as motion signs
WORD_CLASSES = ["Bye", "Deaf", "Hello", "NotOk", "Pen", "Please", "Thankyou", "Yes"]


# ── extraction ────────────────────────────────────────────────
def build_detector():
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
    )


def extract_world_landmarks(detector, img_path):
    """True 3D (metric, wrist-relative) landmarks -> (21, 3) array, or None."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    except Exception:
        return None
    if not res or not res.hand_world_landmarks:
        return None
    return np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]], dtype=np.float64)


def canonicalize(shape):
    """
    Rotate a handshape into a consistent, readable orientation.

    MediaPipe's world landmarks are camera-aligned, so each sign inherits
    whatever angle its source photo happened to be shot at - which is why
    signs came out lying sideways or tipped over at random. We rebuild a
    hand-local frame from the anatomy itself:

        up     = wrist -> middle-finger knuckle   (fingers point up)
        across = index knuckle -> pinky knuckle   (hand spreads sideways)
        normal = up x across                      (palm faces the viewer)

    and rotate the hand so those land on +Y, +X and +Z. Every sign then
    renders upright and palm-forward regardless of the original photo.
    Left hands are mirrored to right so the whole set matches.
    """
    p = shape - shape[0]

    up = p[9]
    up = up / (np.linalg.norm(up) or 1.0)

    across = p[17] - p[5]
    across = across - np.dot(across, up) * up          # orthogonalise against up
    across = across / (np.linalg.norm(across) or 1.0)

    normal = np.cross(across, up)
    normal = normal / (np.linalg.norm(normal) or 1.0)

    R = np.stack([across, up, normal])                 # hand frame -> world axes
    out = p @ R.T

    # Mirror left hands so every sign is shown as a right hand.
    if np.dot(np.cross(out[5] - out[0], out[17] - out[0]), np.array([0, 0, 1.0])) < 0:
        out[:, 0] = -out[:, 0]

    return out


def best_handshape(detector, folder, limit=SAMPLES_PER_SIGN):
    """
    Sample images from a sign's folder and return the single most
    representative 3D handshape (the medoid - minimises total distance to
    all the others), so one bad/odd detection can't define the sign.
    """
    paths = sorted(
        glob.glob(os.path.join(folder, "*.jpg"))
        + glob.glob(os.path.join(folder, "*.jpeg"))
        + glob.glob(os.path.join(folder, "*.png"))
    )
    if not paths:
        return None
    step = max(1, len(paths) // limit)
    paths = paths[::step][:limit]

    shapes = []
    for p in paths:
        lm = extract_world_landmarks(detector, p)
        if lm is not None:
            shapes.append(canonicalize(lm))

    if not shapes:
        return None
    if len(shapes) == 1:
        return shapes[0]

    flat = np.array([s.flatten() for s in shapes])
    dists = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
    return shapes[int(np.argmin(dists.sum(axis=1)))]


# ── motion authoring ──────────────────────────────────────────
def _rot(ax=0.0, ay=0.0, az=0.0):
    """Rotation matrix from degrees about x, y, z."""
    ax, ay, az = np.radians([ax, ay, az])
    cx, sx, cy, sy, cz, sz = (np.cos(ax), np.sin(ax), np.cos(ay),
                              np.sin(ay), np.cos(az), np.sin(az))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def animate(shape, keyframes):
    """
    Apply a trajectory to a handshape.

    `shape` is the (21, 3) base handshape. Each keyframe is
    (dx, dy, dz, rx, ry, rz): a translation in hand-widths and a rotation
    in degrees. Returns a list of (21, 3) frames the avatar tweens between.
    """
    hand_size = float(np.linalg.norm(shape - shape[0], axis=1).max()) or 1.0
    frames = []
    for dx, dy, dz, rx, ry, rz in keyframes:
        moved = (shape - shape[0]) @ _rot(rx, ry, rz).T + shape[0]
        moved = moved + np.array([dx, dy, dz]) * hand_size
        frames.append(moved)
    return frames


# Real ASL movement paths. Coordinates are in hand-widths relative to the
# resting pose, using the avatar's view: +x right, +y up, +z toward viewer.
TRAJECTORIES = {
    # Flat hand at the temple, arcing outward and away - a salute.
    "Hello":   [(-0.35,  0.55, 0.0,  0,  0, -20),
                (-0.10,  0.60, 0.0,  0,  0, -10),
                ( 0.30,  0.45, 0.1,  0,  0,  10),
                ( 0.55,  0.25, 0.2,  0,  0,  20)],

    # Flat hand from the chin, moving forward and down toward the listener.
    "Thankyou": [(0.0,  0.55, -0.05, -10, 0, 0),
                 (0.0,  0.35,  0.15,   0, 0, 0),
                 (0.0,  0.10,  0.35,  10, 0, 0)],

    # Raised hand waving side to side.
    "Bye":     [( 0.00, 0.45, 0.0, 0,   0, -18),
                ( 0.12, 0.45, 0.0, 0,   0,  18),
                (-0.12, 0.45, 0.0, 0,   0, -18),
                ( 0.12, 0.45, 0.0, 0,   0,  18)],

    # Fist nodding up and down, like a head saying yes.
    "Yes":     [(0.0, 0.25, 0.0, -28, 0, 0),
                (0.0, 0.22, 0.0,  14, 0, 0),
                (0.0, 0.25, 0.0, -28, 0, 0),
                (0.0, 0.22, 0.0,  14, 0, 0)],

    # Flat hand circling on the chest.
    "Please":  [( 0.00, 0.10, 0.15, 0, 0, 0),
                ( 0.18, 0.25, 0.15, 0, 0, 0),
                ( 0.00, 0.40, 0.15, 0, 0, 0),
                (-0.18, 0.25, 0.15, 0, 0, 0)],

    # Index finger touching near the ear, then moving to the mouth.
    "Deaf":    [( 0.35, 0.60, 0.0, 0, 0,   0),
                ( 0.20, 0.40, 0.1, 0, 0, -15),
                ( 0.10, 0.20, 0.1, 0, 0, -30)],

    # Small definite outward push.
    "NotOk":   [(0.0, 0.25, 0.00, 0, 0,   0),
                (0.0, 0.28, 0.20, 0, 0, -12)],

    # Writing motion across the page.
    "Pen":     [(-0.20, 0.15, 0.1, 0, 0, -8),
                ( 0.00, 0.22, 0.1, 0, 0,  0),
                ( 0.20, 0.15, 0.1, 0, 0,  8)],

    # J: pinky traces a hook downward.
    "J":       [( 0.15, 0.25, 0.0, 0, 0,   0),
                ( 0.12, 0.05, 0.0, 0, 0,  15),
                (-0.05,-0.10, 0.0, 0, 0,  35),
                (-0.20,-0.05, 0.0, 0, 0,  45)],

    # Z: index finger draws a zig-zag.
    "Z":       [(-0.20, 0.35, 0.0, 0, 0, 0),
                ( 0.20, 0.35, 0.0, 0, 0, 0),
                (-0.20, 0.00, 0.0, 0, 0, 0),
                ( 0.20, 0.00, 0.0, 0, 0, 0)],
}

# Which handshape each motion sign borrows, when its own folder has none
# (J uses the "I" pinky handshape, Z uses the "D" index point).
HANDSHAPE_SOURCE = {"J": ("letter", "I"), "Z": ("letter", "D")}


def main():
    print("Initialising MediaPipe...")
    detector = build_detector()
    print("Detector ready.\n")

    handshapes = {}

    # ── letters (static handshapes) ───────────────────────────
    print("Extracting letter handshapes (true 3D):")
    for letter in LETTERS:
        folder = os.path.join(ALPHABET_DIR, letter)
        shape = best_handshape(detector, folder)
        if shape is None:
            print(f"  [ skip ]  {letter} — no hand detected")
            continue
        handshapes[letter] = shape
        json.dump([shape.tolist()], open(os.path.join(BASE_DIR, f"{letter}.json"), "w"))
        print(f"  [  ok  ]  {letter}")

    # ── word handshapes ───────────────────────────────────────
    print("\nExtracting word handshapes (true 3D):")
    for word in WORD_CLASSES:
        shape = best_handshape(detector, os.path.join(WORDS_DIR, word))
        if shape is None:
            print(f"  [ skip ]  {word} — no hand detected")
            continue
        handshapes[word] = shape
        print(f"  [  ok  ]  {word}")

    # ── apply motion trajectories ─────────────────────────────
    print("\nAnimating motion signs:")
    for name, keyframes in TRAJECTORIES.items():
        shape = handshapes.get(name)
        if shape is None:
            kind, src = HANDSHAPE_SOURCE.get(name, (None, None))
            shape = handshapes.get(src) if kind == "letter" else None
        if shape is None:
            print(f"  [ skip ]  {name} — no handshape available")
            continue

        frames = animate(shape, keyframes)
        json.dump([f.tolist() for f in frames],
                  open(os.path.join(BASE_DIR, f"{name}.json"), "w"))
        print(f"  [  ok  ]  {name:9s} — {len(frames)} keyframes of real movement")

    detector.close()
    print("\nDone. Letters are true-3D static handshapes; word signs now move.")


if __name__ == "__main__":
    main()
