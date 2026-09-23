"""
Shared feature extraction for the gesture model.

Imported by BOTH the trainer (mltrainercode.py) and the inference server
(server.py) so the two can never disagree about how a hand pose becomes a
feature vector. Depends only on numpy.

Two representations:
  • normalize_flat(raw)  -> 63 dims : the original wrist-relative, scale-
                                      normalized coordinates.
  • extract_features(raw) -> RICH_DIM : the above 63 PLUS engineered features
                                      (finger joint angles + fingertip
                                      distances) that make handshape easier to
                                      tell apart from a single frame.

The server auto-detects which one to use from the loaded checkpoint's input
size, so an old 63-dim model and a new rich model both keep working.
"""

import numpy as np

BASIC_DIM = 63

# Each finger as a 5-point chain starting at the wrist, so the interior joints
# give us curl angles (wrist→MCP→PIP→DIP→TIP).
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]
FINGERTIPS = [4, 8, 12, 16, 20]


def _normalize(raw):
    """Wrist-relative, scale-normalized (21, 3) points."""
    pts = np.array(raw, dtype=np.float64).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 0:
        pts = pts / scale
    return pts


def normalize_flat(raw):
    """63-dim: the original normalized coordinates."""
    return _normalize(raw).flatten().tolist()


def _angle(a, b, c):
    """Angle at vertex b formed by points a-b-c, in radians. Robust to
    zero-length segments (returns 0)."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosv = np.dot(v1, v2) / (n1 * n2)
    return float(np.arccos(np.clip(cosv, -1.0, 1.0)))


def extract_features(raw):
    """
    Rich single-frame feature vector:
      • 63 normalized coordinates
      • 15 finger joint angles (curl at each interior joint) — rotation-
           invariant, so they describe handshape directly
      • 10 pairwise fingertip distances — capture finger spread / contact
    """
    pts = _normalize(raw)
    feats = list(pts.flatten())

    for chain in FINGER_CHAINS:
        for j in range(1, 4):                       # interior joints
            feats.append(_angle(pts[chain[j - 1]], pts[chain[j]], pts[chain[j + 1]]))

    for i in range(len(FINGERTIPS)):
        for k in range(i + 1, len(FINGERTIPS)):
            feats.append(float(np.linalg.norm(pts[FINGERTIPS[i]] - pts[FINGERTIPS[k]])))

    return feats


# Single source of truth for the rich feature length (63 + 15 + 10 = 88).
RICH_DIM = len(extract_features([0.0] * 63))


def features_for_dim(raw, in_features):
    """Pick the representation matching a model's expected input size."""
    return extract_features(raw) if in_features == RICH_DIM else normalize_flat(raw)


# ── Training-time augmentation ────────────────────────────────
# Lives here (rather than in one trainer) so every letter-model trainer
# augments identically - the ASL trainer and the ISL trainer must not teach
# the model two different ideas of what "the same pose, slightly moved" means.

def _rotation_matrix(ax, ay, az):
    """Compose a 3D rotation from small pitch/yaw/roll angles (radians)."""
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def augment_landmarks(raw):
    """
    Randomly perturb a raw hand pose so the model sees the kind of variation a
    real webcam produces - training images are unnaturally clean, which is why
    an un-augmented model "works sometimes" live. Training set only.

      • small 3D rotation  -> hand tilted at different angles
      • horizontal mirror  -> left/right hand + the mirrored webcam feed
      • scale jitter       -> hand nearer/further from the camera
      • gaussian noise     -> MediaPipe landmark jitter
    """
    pts = np.array(raw, dtype=np.float64).reshape(21, 3)
    pts = pts - pts[0]                       # rotate/mirror about the wrist

    ax = np.random.uniform(-0.17, 0.17)      # ≈ ±10° pitch
    ay = np.random.uniform(-0.17, 0.17)      # ≈ ±10° yaw
    az = np.random.uniform(-0.35, 0.35)      # ≈ ±20° in-plane roll
    pts = pts @ _rotation_matrix(ax, ay, az).T

    if np.random.rand() < 0.5:               # mirror handedness / feed flip
        pts[:, 0] = -pts[:, 0]

    pts *= np.random.uniform(0.9, 1.1)       # scale jitter
    pts += np.random.normal(0, 0.01, pts.shape)  # landmark noise

    return pts.flatten().tolist()
