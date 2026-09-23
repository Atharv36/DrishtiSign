"""
letter_model.py
─────────────────────────────────────────────────────────────────
The single-frame LETTER classifier architecture, shared by every script
that trains or loads one.

Kept in one place for the same reason WordSignGRU lives in
word_sequences.py: the trainer and the inference server must agree on the
architecture exactly, and three drifting copies of an nn.Module is how
saved weights silently stop matching the code loading them.

Used by:
  * server.py             - live inference (ASL and ISL letter models)
  * mltrainercode.py      - trains the ASL letter model
  * train_isl_model.py    - trains the ISL consonant model

in_features is a parameter rather than a constant so a checkpoint trained on
the older 63-dim coordinate features and one trained on the 88-dim rich
features both load without a code change - the caller reads the dimension off
the checkpoint and passes it in.
"""

import torch.nn as nn

from feature_utils import RICH_DIM


class GestureClassifier(nn.Module):
    def __init__(self, num_classes: int, in_features: int = RICH_DIM):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Skip connection projection (in_features -> 128). Deep stacks blur the
        # small landmark differences that separate similar handshapes (A/S/T),
        # so the raw features get a direct path to the classifier too.
        self.skip = nn.Linear(in_features, 128)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        skip     = self.skip(x)          # residual shortcut
        out      = features + skip       # merge
        return self.classifier(out)
