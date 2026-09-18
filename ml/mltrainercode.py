# ============================================================
#  Hand Sign Detection — Train & Export (Improved Accuracy)
#  Compatible: VS Code · Python 3.9–3.13 · CPU or CUDA GPU
# ============================================================

import os
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import kagglehub

from feature_utils import extract_features, RICH_DIM

print("⏳ Downloading/Locating Kaggle Dataset automatically...")
base_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
DATASET_PATH  = os.path.join(base_path, "asl_alphabet_train", "asl_alphabet_train")

# Local, self-collected word dataset (static pose per word). We pull the
# ALPHABET (A-Z + del/nothing/space) from Kaggle and the WORD classes from
# here, so we end up with one model that knows both letters and words.
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
WORDS_DATASET_PATH = os.path.join(BASE_DIR, "sign_language_dataset1")
WORD_CLASSES       = ["Bye", "Deaf", "Hello", "NotOk", "Pen", "Please", "Thankyou", "Yes"]

# Cap images sampled per class. Kaggle has ~3000/letter which would make
# landmark extraction take hours and drown out the words (which have <100
# each). Sampling evenly to this cap keeps extraction fast and the classes
# reasonably balanced.
MAX_PER_CLASS = 300

EXPORT_DIR    = "exported_model"
EPOCHS        = 80          # upper bound; early stopping ends it once val plateaus
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.1
TEST_SPLIT    = 0.1
NUM_PREVIEW   = 0           # set to 5 if you want previews
# ──────────────────────────────────────────────────────────────


# Motion-based letters: their meaning is a movement, so a single frame just
# looks like another letter (J≈I, Z≈D/pointing). Training on them adds label
# noise that hurts I and D, so we drop them from the model entirely.
EXCLUDED_ALPHABET = {"J", "Z"}

# ── 1. Discover Labels & build source list ───────────────────
# Alphabet + meta classes come from the Kaggle folders...
alphabet_labels = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d)) and d not in EXCLUDED_ALPHABET
])
assert alphabet_labels, f"No sub-folders found in {DATASET_PATH}"

# ...and word classes come from the local dataset (skip any that are missing).
word_labels = []
sources = [(lab, os.path.join(DATASET_PATH, lab)) for lab in alphabet_labels]
for w in WORD_CLASSES:
    wdir = os.path.join(WORDS_DATASET_PATH, w)
    if os.path.isdir(wdir):
        word_labels.append(w)
        sources.append((w, wdir))
    else:
        print(f"  ⚠️  word folder not found, skipping: {wdir}")

labels = sorted(alphabet_labels + word_labels)
print(f"✅ Found {len(labels)} labels "
      f"({len(alphabet_labels)} alphabet/meta + {len(word_labels)} words): {labels}")


# ── 2. MediaPipe landmark extractor ──────────────────────────
LANDMARKER_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL)
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector     = vision.HandLandmarker.create_from_options(options)


def _rotation_matrix(ax, ay, az):
    """Compose a 3D rotation from small pitch/yaw/roll angles (radians)."""
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def augment_landmarks(raw: list) -> list:
    """
    Randomly perturb a raw hand pose so the model sees the kind of variation a
    real webcam produces — the training images are unnaturally clean, which is
    why detection "works sometimes" live. Applied only to the training set.

      • small 3D rotation  → hand tilted at different angles
      • horizontal mirror  → left/right hand + the mirrored webcam feed
      • scale jitter       → hand nearer/further from the camera
      • gaussian noise     → MediaPipe landmark jitter
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


def extract_landmarks(image_path: str):
    """Return RAW 63-float landmark list (x,y,z per point) or None.
    Normalization + augmentation happen later, per-sample, in the Dataset."""
    try:
        img    = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        result = detector.detect(mp_img)
        if result.hand_landmarks:
            lm  = result.hand_landmarks[0]
            return [val for pt in lm for val in (pt.x, pt.y, pt.z)]
    except Exception:
        pass
    return None


# ── 3. Extract landmarks ──────────────────────────────────────
print("\n⏳ Extracting hand landmarks...")
all_data = []
skipped  = 0

for class_name, class_dir in sources:
    label_idx = labels.index(class_name)
    found, miss = 0, 0

    imgs = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    # Evenly sample up to MAX_PER_CLASS so big Kaggle folders don't dominate.
    if len(imgs) > MAX_PER_CLASS:
        step = len(imgs) // MAX_PER_CLASS
        imgs = imgs[::step][:MAX_PER_CLASS]

    for fname in imgs:
        landmarks = extract_landmarks(os.path.join(class_dir, fname))
        if landmarks:
            all_data.append((landmarks, label_idx))
            found += 1
        else:
            miss += 1

    print(f"  {class_name:<20} detected={found:>4}  skipped={miss:>4}")
    skipped += miss

print(f"\n✅ Total usable : {len(all_data)}")
print(f"⚠️  Skipped      : {skipped}")
assert len(all_data) > 0, "No landmarks extracted."


# ── 4. Dataset ────────────────────────────────────────────────
# Stores RAW landmarks; normalization runs on every fetch (matching inference),
# and augmentation is applied to the training set only, fresh each epoch.
class LandmarkDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw, label = self.data[idx]
        if self.augment:
            raw = augment_landmarks(raw)
        x = extract_features(raw)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label)


# Deterministic shuffled split (so train gets augmentation, val/test don't).
import random as _random
_idxs = list(range(len(all_data)))
_random.Random(42).shuffle(_idxs)

n       = len(all_data)
n_val   = max(1, int(VAL_SPLIT  * n))
n_test  = max(1, int(TEST_SPLIT * n))
val_data   = [all_data[i] for i in _idxs[:n_val]]
test_data  = [all_data[i] for i in _idxs[n_val:n_val + n_test]]
train_data = [all_data[i] for i in _idxs[n_val + n_test:]]
n_train    = len(train_data)

train_ds = LandmarkDataset(train_data, augment=True)
val_ds   = LandmarkDataset(val_data,   augment=False)
test_ds  = LandmarkDataset(test_data,  augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
print(f"\nSplit — train:{n_train}  val:{n_val}  test:{n_test}")

# Class-balancing weights: the words have <100 samples each while letters have
# hundreds, so without this the model would happily ignore the rare classes.
_counts  = Counter(lbl for _, lbl in train_data)
_weights = [n_train / (len(labels) * _counts.get(c, 1)) for c in range(len(labels))]
class_weights = torch.tensor(_weights, dtype=torch.float32)


# ── 5. Improved Model ─────────────────────────────────────────
# Deeper network with residual-style skip connection
# Much better at distinguishing similar signs (A vs S vs T etc.)
class GestureClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Linear(RICH_DIM, 512),
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

        # Skip connection projection (RICH_DIM -> 128)
        self.skip = nn.Linear(RICH_DIM, 128)

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GestureClassifier(num_classes=len(labels)).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n🖥️  Training on: {device}  |  Parameters: {total_params:,}")


# ── 6. Training with label smoothing + class weights ─────────
# Label smoothing reduces overconfidence; class weights stop the rare word
# classes from being drowned out by the letters.
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

EARLY_STOP_PATIENCE = 12   # stop if val accuracy hasn't improved in this many epochs

train_acc_history = []
val_acc_history   = []
best_val_acc      = 0.0
epochs_since_best = 0

print("\n── Training ──────────────────────────────────────────────")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct = 0.0, 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        # Gradient clipping — prevents unstable training spikes
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()

    scheduler.step()
    train_acc = correct / n_train

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            val_correct += (model(X).argmax(1) == y).sum().item()

    val_acc = val_correct / n_val
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        epochs_since_best = 0
        os.makedirs(EXPORT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(EXPORT_DIR, "best_checkpoint.pth"))
    else:
        epochs_since_best += 1

    print(
        f"Epoch {epoch:02d}/{EPOCHS}  "
        f"loss={total_loss/len(train_loader):.4f}  "
        f"train={train_acc:.2%}  "
        f"val={val_acc:.2%}"
        + ("  ← best" if improved else "")
    )

    # Augmentation means val accuracy wanders a bit; only stop after a real
    # plateau, and keep the best checkpoint regardless.
    if epochs_since_best >= EARLY_STOP_PATIENCE:
        print(f"\n⏹️  No val improvement for {EARLY_STOP_PATIENCE} epochs — stopping early.")
        break

print(f"\n🏆 Best val accuracy: {best_val_acc:.2%}")


# ── 7. Learning curves ────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(train_acc_history, label='Train')
plt.plot(val_acc_history,   label='Val')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Learning Curves'); plt.legend(); plt.tight_layout()
os.makedirs(EXPORT_DIR, exist_ok=True)
plt.savefig(os.path.join(EXPORT_DIR, "learning_curves.png"), dpi=150)
plt.show()


# ── 8. Test accuracy ──────────────────────────────────────────
model.load_state_dict(
    torch.load(os.path.join(EXPORT_DIR, "best_checkpoint.pth"), map_location=device)
)
model.eval()
test_correct = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        test_correct += (model(X).argmax(1) == y).sum().item()

print(f"\n✅ Test Accuracy: {test_correct/n_test:.2%}")


# ── 9. Export ─────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(EXPORT_DIR, "gesture_recognizer.pth"))

with open(os.path.join(EXPORT_DIR, "labels.json"), "w") as f:
    json.dump(labels, f, indent=2)

# Save normalization flag so detect_live knows to normalize
with open(os.path.join(EXPORT_DIR, "config.json"), "w") as f:
    json.dump({"normalize": True, "num_classes": len(labels)}, f, indent=2)

scripted = torch.jit.script(model)
scripted.save(os.path.join(EXPORT_DIR, "gesture_recognizer_scripted.pt"))

print(f"\n📦 Saved to '{EXPORT_DIR}/':")
for fname in sorted(os.listdir(EXPORT_DIR)):
    size = os.path.getsize(os.path.join(EXPORT_DIR, fname))
    print(f"   {fname}  ({size/1024:.1f} KB)")