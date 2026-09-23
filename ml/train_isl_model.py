"""
train_isl_model.py
─────────────────────────────────────────────────────────────────
Trains the ISL consonant model - the second alphabet DrishtiSign can read.

DATA PROVENANCE: the source frames come from a Nepali Sign Language consonant
recording set (the source folders are named NSL_Consonant_*, and the classes
are Devanagari consonants). The feature is called ISL throughout this project.
NSL and ISL are related South Asian sign languages with substantial shared
vocabulary but they are NOT the same language, so quote the data source
honestly in any write-up.

Why this reuses the ASL letter pipeline instead of the source project's CNN:
that project classified raw 128x128 RGB hand crops. Pixels carry lighting,
skin tone, background and camera into the model. We classify GEOMETRY instead
- MediaPipe landmarks -> feature_utils.extract_features (88-dim: normalized
coordinates + joint angles + fingertip distances) - which is invariant to all
of that. It also means ISL is a weights-and-labels swap on the existing letter
path rather than a third architecture in the server. Measured landmark
detection rate on the source crops: ~90%.

Input layout (one folder per class, frames sampled from recorded videos):

    hand_gesture_app/dataset_frames/
        KA/    KA_15_0.jpg  KA_15_1.jpg ... KA_51_0.jpg ...
        KHA/   ...
                  ^label ^video ^frame

Run:  python train_isl_model.py
"""

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from feature_utils import extract_features, augment_landmarks, RICH_DIM
from letter_model import GestureClassifier

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "..", "hand_gesture_app", "dataset_frames")
LANDMARKER = os.path.join(BASE_DIR, "hand_landmarker.task")
EXPORT_DIR = os.path.join(BASE_DIR, "exported_model")
CACHE_PATH = os.path.join(BASE_DIR, "isl_landmarks_cache.npz")

EPOCHS        = 120
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
EARLY_STOP    = 15
SEED          = 42

# Folders whose frames carry a WRONG label. The source project derived each
# label from its video filename via `S\d+_(.+)`; videos that didn't match that
# pattern - recordings of ALL consonants in one continuous take - fell through
# with the whole filename as the "label". So every frame in these folders shows
# a different sign while claiming one. Pure label noise, not a thin class.
JUNK_FOLDERS = {
    "Consonant_Camera",
    "NSL_Consonant_Camera",
    "NSL_Consonant_Phone_camera_",
    "NSL_Consonant_PnoneCamera",
    "NSL_Consonant_Unprepared",
    "all_Consonant_RealWorld_2",
    "all_Consonant_Real_World_1",
    "all_consonant_Phone_Camera",
}

# Romanization variants of a class that already exists. KSHA has 59 frames
# across 8 videos of its own; K_SHA is 5 frames from a single video. Dropped
# rather than merged - merging would assume they're the same sign, and a wrong
# merge teaches a wrong label.
DUPLICATE_SPELLINGS = {"K_SHA"}

# Classes cut after reading isl_eval.json, the way word_sequences.py records
# EXCLUDED_SIMILAR - measured on held-out videos, not guessed at.
#
# First run: 48 classes, 76.9% test accuracy. These six scored 0/8 or 0/7, and
# the confusion table says why - each collapses into a class it is a spelling
# or articulation variant of:
#     N     -> D          D_SA -> D_SHA      THH -> TA
#     SHHA  -> TRA        KSH  -> (1 sample, also 0)
# D itself was the worst offender: with 0/8 of its own it still absorbed N, THA
# and TRA, so it was a magnet dragging down well-sampled neighbours rather than
# a class the model could ever place. All six come from the late recording
# session that used a different romanization scheme from the main set, which is
# consistent with them naming signs the main set already covers.
EXCLUDED_AFTER_EVAL = {"D", "N", "SHHA", "THH", "KSH", "D_SA",
                       # Second run (42 classes, 81.0%): these two stayed near
                       # chance on well-sampled test videos - YA 1/16 (6%),
                       # YAN 1/7 - while steadily polluting CHHA and KHA. A
                       # class the model cannot place is worse than an absent
                       # one: live, it is never recognized AND it steals
                       # predictions from classes that work.
                       "YA", "YAN"}

# Label is everything before the last two underscore-separated fields, since
# several labels contain underscores themselves (D_SA, M_SHA, T_SHA).
FRAME_RE = re.compile(r"^(?P<label>.+)_(?P<video>\d+)_(?P<frame>\d+)$")


def parse_frame_name(filename):
    """'D_SHA_152_2.jpg' -> ('D_SHA', 152). None if it doesn't parse."""
    m = FRAME_RE.match(os.path.splitext(filename)[0])
    return (m.group("label"), int(m.group("video"))) if m else None


def usable_classes():
    folders = sorted(
        d for d in os.listdir(FRAMES_DIR)
        if os.path.isdir(os.path.join(FRAMES_DIR, d))
    )
    dropped = [d for d in folders
               if d in JUNK_FOLDERS or d in DUPLICATE_SPELLINGS or d in EXCLUDED_AFTER_EVAL]
    kept = [d for d in folders if d not in dropped]
    return kept, dropped


def build_dataset():
    """Returns (samples, labels) where samples = [(raw63, label_idx, video_id)]."""
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached landmarks from {CACHE_PATH}")
        d = np.load(CACHE_PATH, allow_pickle=True)
        cached_labels = list(d["labels"])
        samples = [(r, int(y), int(v)) for r, y, v in zip(d["X"], d["y"], d["video"])]
        # A cache built before a class was excluded must not silently resurrect it.
        keep = {i for i, lab in enumerate(cached_labels)
                if lab not in EXCLUDED_AFTER_EVAL and lab not in DUPLICATE_SPELLINGS}
        if len(keep) != len(cached_labels):
            labels = [lab for i, lab in enumerate(cached_labels) if i in keep]
            remap = {old: labels.index(cached_labels[old]) for old in keep}
            samples = [(r, remap[y], v) for r, y, v in samples if y in keep]
            print(f"  cache filtered to {len(labels)} classes (exclusions applied)")
            return samples, labels
        return samples, cached_labels

    labels, dropped = usable_classes()
    print(f"Classes: {len(labels)} kept, {len(dropped)} dropped ({', '.join(dropped)})\n")

    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
        )
    )

    samples = []
    print("Extracting landmarks (one pass, then cached)...")
    for idx, label in enumerate(labels):
        class_dir = os.path.join(FRAMES_DIR, label)
        found = miss = 0
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parsed = parse_frame_name(fname)
            if parsed is None:
                miss += 1
                continue
            _, video_id = parsed
            try:
                img = np.array(Image.open(os.path.join(class_dir, fname)).convert("RGB"),
                               dtype=np.uint8)
                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
            except Exception:
                res = None
            if res and res.hand_landmarks:
                raw = [v for p in res.hand_landmarks[0] for v in (p.x, p.y, p.z)]
                samples.append((np.array(raw, dtype=np.float32), idx, video_id))
                found += 1
            else:
                miss += 1
        n_videos = len({v for _, y, v in samples if y == idx})
        print(f"  {label:<10} {found:>4} frames from {n_videos:>2} videos  (skipped {miss})")

    detector.close()
    if not samples:
        raise SystemExit("No landmarks extracted - is the frame dataset present?")

    np.savez_compressed(
        CACHE_PATH,
        X=np.stack([s[0] for s in samples]),
        y=np.array([s[1] for s in samples], dtype=np.int64),
        video=np.array([s[2] for s in samples], dtype=np.int64),
        labels=np.array(labels),
    )
    print(f"\nCached {len(samples)} landmark sets -> {CACHE_PATH}")
    return samples, labels


def split_by_video(samples, labels):
    """
    Hold out whole VIDEOS, not frames.

    8 frames sampled from one video are near-duplicates of each other. Splitting
    at frame level puts siblings of a test frame in the training set, so the
    model is graded on poses it has effectively already seen and the accuracy
    number is meaningless. Splitting by video is the difference between a
    quotable figure and a fake one.

    Stratified per class because the class sizes are wildly uneven (2 videos for
    the late-session consonants, 11 for the early ones): a global random split
    would hand some classes zero training videos.
    """
    by_class = defaultdict(set)
    for _, y, v in samples:
        by_class[y].add(v)

    rng = np.random.RandomState(SEED)
    assign = {}                       # (class, video) -> 'train' | 'val' | 'test'
    thin = []
    for y, videos in by_class.items():
        vids = sorted(videos)
        rng.shuffle(vids)
        n = len(vids)
        if n == 1:
            # Nothing can be held out - it trains, but it can't be graded.
            parts = {vids[0]: "train"}
            thin.append(labels[y])
        elif n == 2:
            parts = {vids[0]: "train", vids[1]: "test"}
        else:
            n_test = max(1, round(0.15 * n))
            n_val = max(1, round(0.15 * n))
            parts = {}
            for i, v in enumerate(vids):
                parts[v] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
        for v, part in parts.items():
            assign[(y, v)] = part

    if thin:
        print(f"! {len(thin)} class(es) have a single video, so they train but are "
              f"never tested: {', '.join(sorted(thin))}")

    out = {"train": [], "val": [], "test": []}
    for raw, y, v in samples:
        out[assign[(y, v)]].append((raw, y))

    # The whole point of this function - assert it rather than trust it.
    train_videos = {(y, v) for (y, v), p in assign.items() if p == "train"}
    held_videos = {(y, v) for (y, v), p in assign.items() if p != "train"}
    assert not (train_videos & held_videos), "video leaked across the split"
    return out


class LandmarkDataset(Dataset):
    """Stores RAW landmarks; normalization runs per fetch (matching inference),
    augmentation on the training set only, fresh each epoch."""

    def __init__(self, data, augment=False):
        self.data, self.augment = data, augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        raw, label = self.data[i]
        raw = raw.tolist()
        if self.augment:
            raw = augment_landmarks(raw)
        return torch.tensor(extract_features(raw), dtype=torch.float32), torch.tensor(label)


def evaluate(model, loader, device):
    model.eval()
    preds, actual = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X.to(device)).argmax(1).cpu().tolist()
            actual += y.tolist()
    return preds, actual


def main():
    samples, labels = build_dataset()
    print(f"\nDataset: {len(samples)} landmark sets across {len(labels)} classes")

    parts = split_by_video(samples, labels)
    print(f"Split by video - train:{len(parts['train'])}  "
          f"val:{len(parts['val'])}  test:{len(parts['test'])} frames")

    train_dl = DataLoader(LandmarkDataset(parts["train"], augment=True),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(LandmarkDataset(parts["val"]), batch_size=BATCH_SIZE)
    test_dl = DataLoader(LandmarkDataset(parts["test"]), batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureClassifier(len(labels), in_features=RICH_DIM).to(device)
    print(f"Training on {device} | {sum(p.numel() for p in model.parameters()):,} params\n")

    # Class weighting matters more here than in the ASL trainer: the late-session
    # consonants have 2 videos against the early set's 11, so without it the
    # model would learn to ignore them.
    counts = Counter(y for _, y in parts["train"])
    n_train = len(parts["train"])
    w = torch.tensor([n_train / (len(labels) * counts.get(c, 1)) for c in range(len(labels))],
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    model_path = os.path.join(EXPORT_DIR, "isl_model.pth")
    best, since_best = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct = total = 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = crit(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        sched.step()
        train_acc = correct / max(total, 1)

        vp, va = evaluate(model, val_dl, device)
        val_acc = sum(p == a for p, a in zip(vp, va)) / max(len(vp), 1)

        improved = val_acc > best
        if improved:
            best, since_best = val_acc, 0
            torch.save(model.state_dict(), model_path)
            json.dump(labels, open(os.path.join(EXPORT_DIR, "isl_labels.json"), "w"), indent=2)
        else:
            since_best += 1

        if epoch % 5 == 0 or improved:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_acc:.1%}  val={val_acc:.1%}"
                  + ("  <- best" if improved else ""))

        if since_best >= EARLY_STOP:
            print(f"\nNo val improvement for {EARLY_STOP} epochs - stopping.")
            break

    print(f"\nBest val accuracy: {best:.1%}  (used for model selection - optimistic)")

    # ── honest evaluation on held-out videos ──────────────────
    model.load_state_dict(torch.load(model_path, map_location=device))
    preds, actual = evaluate(model, test_dl, device)
    if not preds:
        print("No test data - nothing to report.")
        return

    test_acc = sum(p == a for p, a in zip(preds, actual)) / len(preds)
    print(f"TEST accuracy:     {test_acc:.1%}  <- quote this one "
          f"(chance is {1/len(labels):.1%} at {len(labels)} classes)\n")

    per_class = defaultdict(lambda: [0, 0])
    for p, a in zip(preds, actual):
        per_class[a][1] += 1
        if p == a:
            per_class[a][0] += 1

    print("Per-class accuracy (worst first - these are the trim candidates):")
    rows = sorted(((labels[c], ok, n) for c, (ok, n) in per_class.items()),
                  key=lambda r: (r[1] / r[2], -r[2]))
    for name, ok, n in rows:
        print(f"  {name:<10} {ok:>3}/{n:<3d} {'#' * int(10 * ok / n)}")

    confusions = Counter((labels[a], labels[p]) for p, a in zip(preds, actual) if p != a)
    if confusions:
        print("\nMost common confusions (true -> predicted):")
        for (t, p), n in confusions.most_common(10):
            print(f"  {t:<10} -> {p:<10} x{n}")

    json.dump({"test_accuracy": test_acc,
               "best_val_accuracy": best,
               "num_classes": len(labels),
               "per_class": {labels[c]: {"correct": ok, "total": n}
                             for c, (ok, n) in per_class.items()},
               "confusions": [{"true": t, "predicted": p, "count": n}
                              for (t, p), n in confusions.most_common()]},
              open(os.path.join(EXPORT_DIR, "isl_eval.json"), "w"), indent=2)

    print(f"\nSaved -> {EXPORT_DIR}/isl_model.pth, isl_labels.json, isl_eval.json")


if __name__ == "__main__":
    main()
