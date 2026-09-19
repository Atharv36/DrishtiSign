"""
word_sequences.py
─────────────────────────────────────────────────────────────────
Shared helpers for the WORD model (motion signs), kept separate from the
letter model on purpose: a word sign is a movement over time, so its input
is a SEQUENCE of frames, not the single frame the letter MLP takes.

A clip becomes a fixed-size (SEQ_LEN, FEATURE_DIM) array:
  * MediaPipe hand landmarks are read per frame
  * each frame is turned into the same feature vector the letter model uses
    (feature_utils.extract_features - normalized coords + joint angles +
    fingertip distances, so handshape is described rotation-invariantly)
  * the clip is resampled to SEQ_LEN frames so clips of different lengths
    all become the same shape

Used by both record_word_clips.py (your own recordings) and
train_word_model.py (any folder of per-word videos, e.g. WLASL).
"""

import numpy as np
import torch
import torch.nn as nn

from feature_utils import extract_features, RICH_DIM

SEQ_LEN = 32                    # frames every clip is resampled to
HAND_FEATURE_DIM = RICH_DIM     # 88 per hand
FEATURE_DIM = HAND_FEATURE_DIM * 2  # 176: both hands, left-then-right, zero-filled
                                     # if a hand is absent - see two_hand_features().
                                     #
                                     # A single hand's classifier can only classify
                                     # each hand independently (fine for letters,
                                     # which are spelled one hand at a time). A
                                     # two-handed WORD sign's meaning is in the
                                     # RELATIONSHIP between both hands moving
                                     # together, which needs both encoded into one
                                     # feature per frame - not two separate
                                     # classifications. One-handed signs still work
                                     # fine this way: their second-hand slot is
                                     # just zeros, which LayerNorm handles cleanly.

# Everyday signs worth supporting. Kept small and frequent on purpose: a
# compact reliable vocabulary beats a large flaky one, and any word outside it
# still works via fingerspelling - which is what real signers do for names and
# out-of-vocabulary terms. Shared by the recorder and the trainer so the two
# can't drift apart.
#
# The two-handed words previously excluded here (audit_two_handed.py found
# them under-served by a single-hand feature vector) are back in - see
# two_hand_features() above. Extraction now captures both hands, ordered
# Left-then-Right, so a genuinely bimanual sign is no longer missing half its
# information.
#
# "good" stays excluded: it and "thankyou" are a well-known ASL beginner
# minimal pair (both a flat hand near the chin moving forward/down) and
# consistently confused each other across every evaluation run - a genuine
# sign-similarity collision that adding a second hand's data doesn't fix.
VOCABULARY = [
    "hello", "bye", "please", "thankyou", "sorry", "yes", "no",
    "need", "bad", "love", "eat", "drink", "water", "home", "work",
    "understand", "where", "who", "you", "me",
    "family", "friend", "go", "happy", "how", "learn", "more", "name",
    "sad", "want", "what", "help", "school", "stop",
]

EXCLUDED_SIMILAR = ["good"]  # collides with "thankyou"


class WordSignGRU(nn.Module):
    """
    Bidirectional GRU over a clip, then classify.

    Lives here rather than in the trainer so the training script and the
    inference server share ONE definition - if they drifted apart, the saved
    weights would silently stop matching the architecture loading them.

    Bidirectional because a sign's meaning depends on the whole movement:
    where it ends matters as much as where it starts.
    """

    def __init__(self, num_classes, feature_dim=FEATURE_DIM, hidden=128):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.gru = nn.GRU(feature_dim, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.3)
        # hidden*2 final states + hidden*2 mean-pooled = hidden*4
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):                       # x: (B, SEQ_LEN, FEATURE_DIM)
        out, h_n = self.gru(self.norm(x))

        # Mean-pooling ALONE is direction-blind: a movement outward and the
        # same movement inward average to the same thing, and for sign
        # language that difference is meaning. So combine the final forward
        # and backward hidden states (which encode where the motion ended up)
        # with the mean (which is robust to dead frames at the clip edges).
        final = torch.cat([h_n[-2], h_n[-1]], dim=1)
        pooled = torch.cat([final, out.mean(dim=1)], dim=1)
        return self.head(pooled)


def frame_features(hand_landmarks):
    """One detected hand's 21 landmarks -> (HAND_FEATURE_DIM,) feature vector."""
    raw = [v for p in hand_landmarks for v in (p.x, p.y, p.z)]
    return np.array(extract_features(raw), dtype=np.float32)


def two_hand_features(hand_landmarks_list, handedness_list):
    """
    Build one (FEATURE_DIM,) vector per frame from however many hands
    MediaPipe detected (0, 1, or 2), ordered Left-then-Right so the model
    always sees a consistent layout regardless of which hand happens to be
    "hand_landmarks[0]" for that frame. A missing hand is zero-filled rather
    than omitted, so the sequence length isn't affected by a hand briefly
    leaving frame.
    """
    left = np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    right = np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    for lm, handed in zip(hand_landmarks_list, handedness_list):
        if not handed:
            continue
        side = handed[0].category_name
        feats = frame_features(lm)
        if side == "Left":
            left = feats
        elif side == "Right":
            right = feats
    return np.concatenate([left, right])


def resample(seq, n=SEQ_LEN):
    """
    Resample a variable-length sequence to exactly n frames.

    Signs are performed at different speeds, so we stretch/squash in time
    rather than padding - the model then learns the SHAPE of the motion
    instead of how fast the signer happened to move.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == 0:
        return None
    if len(seq) == n:
        return seq
    idx = np.linspace(0, len(seq) - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    t = (idx - lo)[:, None]
    return seq[lo] * (1 - t) + seq[hi] * t


def sequence_from_video(path, detector, min_frames=6, stride=1):
    """
    Read a video file and return a (SEQ_LEN, FEATURE_DIM) array, or None if
    too few frames contained a detectable hand.

    Works for your own recordings and for dataset clips (WLASL etc.) alike -
    anything OpenCV can open. `detector` should be configured with
    num_hands=2 so two-handed signs are captured correctly.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    feats, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            except Exception:
                res = None
            if res and res.hand_landmarks:
                feats.append(two_hand_features(res.hand_landmarks, res.handedness))
        i += 1
    cap.release()

    if len(feats) < min_frames:
        return None
    return resample(feats)
