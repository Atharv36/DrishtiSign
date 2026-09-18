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

SEQ_LEN = 32          # frames every clip is resampled to
FEATURE_DIM = RICH_DIM  # 88 per frame

# Everyday signs worth supporting. Kept small and frequent on purpose: a
# compact reliable vocabulary beats a large flaky one, and any word outside it
# still works via fingerspelling - which is what real signers do for names and
# out-of-vocabulary terms. Shared by the recorder and the trainer so the two
# can't drift apart.
VOCABULARY = [
    "hello", "bye", "please", "thankyou", "sorry", "yes", "no",
    "help", "want", "need", "good", "bad", "love", "friend", "family",
    "eat", "drink", "water", "home", "work", "school", "learn",
    "understand", "what", "where", "who", "how", "more", "stop", "go",
    "happy", "sad", "name", "you", "me",
]


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
    """One MediaPipe hand result -> (FEATURE_DIM,) feature vector."""
    raw = [v for p in hand_landmarks for v in (p.x, p.y, p.z)]
    return np.array(extract_features(raw), dtype=np.float32)


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
    anything OpenCV can open.
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
                feats.append(frame_features(res.hand_landmarks[0]))
        i += 1
    cap.release()

    if len(feats) < min_frames:
        return None
    return resample(feats)
