"""
train_word_model.py
─────────────────────────────────────────────────────────────────
Trains the WORD model: a temporal (GRU) network over landmark sequences.

This is deliberately a SEPARATE model from the letter classifier. A letter is
a static handshape (one frame -> MLP); a word sign is a movement (a sequence
-> GRU). One model cannot represent both - that limitation is exactly why
"Hello" could never work as a static pose and why J/Z had to be dropped from
the alphabet. With this model those become learnable.

Input layout - one folder per word, any videos inside:

    word_clips/
        hello/     clip1.mp4 clip2.mp4 ...
        please/    ...
        thankyou/  ...

Works with your own recordings (record_word_clips.py) or a dataset like
WLASL, as long as it's arranged that way.

Only words listed in VOCABULARY are trained, so rare signs don't dilute the
model - edit that list to control scope.

Run:  python train_word_model.py
"""

import json
import os
import glob
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from word_sequences import SEQ_LEN, FEATURE_DIM, VOCABULARY, sequence_from_video

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LANDMARKER = os.path.join(BASE_DIR, "hand_landmarker.task")
CLIPS_DIR  = os.path.join(BASE_DIR, "word_clips")
EXPORT_DIR = os.path.join(BASE_DIR, "exported_model")
CACHE_PATH = os.path.join(BASE_DIR, "word_sequences_cache.npz")

EPOCHS        = 120
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
EARLY_STOP    = 20

# ── model ─────────────────────────────────────────────────────
class WordSignGRU(nn.Module):
    """
    Bidirectional GRU over the clip, then classify.

    Bidirectional because a sign's meaning depends on the whole movement -
    where it ends matters as much as where it starts - so the model should
    see the sequence from both directions before deciding.
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


class SeqDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            # Same idea as the letter model's augmentation: real webcam input
            # is noisier and slower/faster than training clips.
            x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
            if np.random.rand() < 0.5:          # mild time jitter
                shift = np.random.randint(-2, 3)
                x = np.roll(x, shift, axis=0)
        return torch.tensor(x), torch.tensor(self.y[i])


# ── data ──────────────────────────────────────────────────────
def build_dataset():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached sequences from {CACHE_PATH}")
        d = np.load(CACHE_PATH, allow_pickle=True)
        return d["X"], d["y"], list(d["labels"])

    if not os.path.isdir(CLIPS_DIR):
        raise SystemExit(
            f"No clips found at {CLIPS_DIR}\n"
            "Record some with:  python record_word_clips.py\n"
            "or point CLIPS_DIR at a dataset arranged as one folder per word."
        )

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.4,
        )
    )

    folders = sorted(
        d for d in os.listdir(CLIPS_DIR)
        if os.path.isdir(os.path.join(CLIPS_DIR, d)) and d.lower() in VOCABULARY
    )
    if not folders:
        raise SystemExit(f"No folders in {CLIPS_DIR} match VOCABULARY.")

    labels, X, y = sorted(folders), [], []
    print(f"Extracting sequences for {len(labels)} words...\n")

    for word in labels:
        vids = []
        for ext in ("*.mp4", "*.mov", "*.avi", "*.webm", "*.mkv"):
            vids += glob.glob(os.path.join(CLIPS_DIR, word, ext))
        ok = 0
        for v in sorted(vids):
            seq = sequence_from_video(v, detector)
            if seq is not None:
                X.append(seq)
                y.append(labels.index(word))
                ok += 1
        print(f"  {word:12s} {ok:3d} usable clips (of {len(vids)})")

    detector.close()
    if not X:
        raise SystemExit("No usable clips - was a hand visible in them?")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(CACHE_PATH, X=X, y=y, labels=np.array(labels))
    print(f"\nCached {len(X)} sequences -> {CACHE_PATH}")
    return X, y, labels


def main():
    X, y, labels = build_dataset()
    print(f"\nDataset: {X.shape[0]} clips, {len(labels)} words, "
          f"shape {X.shape[1:]} per clip")

    counts = Counter(y.tolist())
    if min(counts.values()) < 3:
        print("\n! Some words have <3 clips. Aim for 10+ per word for a usable model.")

    # Three-way split. The test set is held back and never used for training
    # OR for choosing the best checkpoint - because the val set IS used for
    # model selection, val accuracy is an optimistic estimate. The test number
    # is the one to quote in a report.
    idx = np.random.RandomState(42).permutation(len(X))
    n_val = max(1, int(VAL_SPLIT * len(X)))
    n_test = max(1, int(TEST_SPLIT * len(X)))
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]

    train_dl = DataLoader(SeqDataset(X[train_idx], y[train_idx], augment=True),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(SeqDataset(X[val_idx], y[val_idx]), batch_size=BATCH_SIZE)
    test_dl = DataLoader(SeqDataset(X[test_idx], y[test_idx]), batch_size=BATCH_SIZE)
    print(f"Split - train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WordSignGRU(len(labels)).to(device)
    print(f"Training on {device} | {sum(p.numel() for p in model.parameters()):,} params\n")

    w = torch.tensor([len(y) / (len(labels) * counts.get(c, 1)) for c in range(len(labels))],
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best, since_best = 0.0, 0
    os.makedirs(EXPORT_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        sched.step()
        train_acc = correct / max(total, 1)

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vc += (model(xb).argmax(1) == yb).sum().item()
                vt += len(yb)
        val_acc = vc / max(vt, 1)

        improved = val_acc > best
        if improved:
            best, since_best = val_acc, 0
            torch.save(model.state_dict(), os.path.join(EXPORT_DIR, "word_model.pth"))
            json.dump(labels, open(os.path.join(EXPORT_DIR, "word_labels.json"), "w"), indent=2)
            json.dump({"seq_len": SEQ_LEN, "feature_dim": FEATURE_DIM,
                       "num_classes": len(labels)},
                      open(os.path.join(EXPORT_DIR, "word_config.json"), "w"), indent=2)
        else:
            since_best += 1

        if epoch % 5 == 0 or improved:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_acc:.1%}  val={val_acc:.1%}"
                  + ("  <- best" if improved else ""))

        if since_best >= EARLY_STOP:
            print(f"\nNo improvement for {EARLY_STOP} epochs - stopping.")
            break

    print(f"\nBest val accuracy: {best:.1%}  (used for model selection - optimistic)")

    # ── honest evaluation on the held-out test set ────────────
    model.load_state_dict(torch.load(os.path.join(EXPORT_DIR, "word_model.pth"),
                                     map_location=device))
    model.eval()

    preds, actual = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds += model(xb.to(device)).argmax(1).cpu().tolist()
            actual += yb.tolist()

    if preds:
        test_acc = sum(p == a for p, a in zip(preds, actual)) / len(preds)
        print(f"TEST accuracy:     {test_acc:.1%}  <- quote this one\n")

        # Per-word accuracy, worst first - this is your recording to-do list.
        per_word = defaultdict(lambda: [0, 0])
        for p, a in zip(preds, actual):
            per_word[a][1] += 1
            if p == a:
                per_word[a][0] += 1

        print("Per-word accuracy (worst first - record more clips for these):")
        rows = sorted(((labels[c], ok, n) for c, (ok, n) in per_word.items()),
                      key=lambda r: (r[1] / r[2], -r[2]))
        for word, ok, n in rows:
            bar = "#" * int(10 * ok / n)
            print(f"  {word:12s} {ok}/{n:<3d} {bar}")

        # Confusion pairs: which signs the model actually mixes up.
        confusions = Counter((labels[a], labels[p])
                             for p, a in zip(preds, actual) if p != a)
        if confusions:
            print("\nMost common confusions (true -> predicted):")
            for (t, p), n in confusions.most_common(8):
                print(f"  {t:12s} -> {p:12s} x{n}")

        json.dump({"test_accuracy": test_acc,
                   "best_val_accuracy": best,
                   "per_word": {labels[c]: {"correct": ok, "total": n}
                                for c, (ok, n) in per_word.items()},
                   "confusions": [{"true": t, "predicted": p, "count": n}
                                  for (t, p), n in confusions.most_common()]},
                  open(os.path.join(EXPORT_DIR, "word_eval.json"), "w"), indent=2)
        print(f"\nFull results -> {EXPORT_DIR}/word_eval.json")

    print(f"Saved -> {EXPORT_DIR}/word_model.pth, word_labels.json, word_config.json")


if __name__ == "__main__":
    main()
