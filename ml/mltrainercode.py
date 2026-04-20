# ============================================================
#  Hand Sign Detection — Train & Export (Improved Accuracy)
#  Compatible: VS Code · Python 3.9–3.13 · CPU or CUDA GPU
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── CONFIG ────────────────────────────────────────────────────
DATASET_PATH  = "/Users/dikshitbhusal/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train"
EXPORT_DIR    = "exported_model"
EPOCHS        = 50          # more epochs for better accuracy
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.1
TEST_SPLIT    = 0.1
NUM_PREVIEW   = 0           # set to 5 if you want previews
# ──────────────────────────────────────────────────────────────


# ── 1. Discover Labels ────────────────────────────────────────
labels = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])
assert labels, f"No sub-folders found in {DATASET_PATH}"
print(f"✅ Found {len(labels)} labels: {labels}")


# ── 2. MediaPipe landmark extractor ──────────────────────────
LANDMARKER_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL)
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector     = vision.HandLandmarker.create_from_options(options)


def normalize_landmarks(raw: list) -> list:
    """
    KEY ACCURACY FIX — normalize landmarks relative to wrist.
    Raw x/y/z are absolute positions in the image frame.
    This means the same sign at different positions/scales
    looks completely different to the model.
    Normalizing makes position and hand size irrelevant —
    only the SHAPE of the sign matters.
    """
    pts = np.array(raw).reshape(21, 3)

    # Translate: make wrist (landmark 0) the origin
    pts = pts - pts[0]

    # Scale: divide by the distance from wrist to middle finger MCP (landmark 9)
    # This makes hand size invariant
    scale = np.linalg.norm(pts[9])
    if scale > 0:
        pts = pts / scale

    return pts.flatten().tolist()


def extract_landmarks(image_path: str):
    """Return 63-float NORMALIZED list or None if no hand detected."""
    try:
        img    = Image.open(image_path).convert('RGB')
        img_np = np.array(img, dtype=np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        result = detector.detect(mp_img)
        if result.hand_landmarks:
            lm  = result.hand_landmarks[0]
            raw = [val for pt in lm for val in (pt.x, pt.y, pt.z)]
            return normalize_landmarks(raw)
    except Exception:
        pass
    return None


# ── 3. Extract landmarks ──────────────────────────────────────
print("\n⏳ Extracting hand landmarks...")
all_data = []
skipped  = 0

for label in labels:
    label_dir = os.path.join(DATASET_PATH, label)
    label_idx = labels.index(label)
    found, miss = 0, 0

    for fname in os.listdir(label_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        landmarks = extract_landmarks(os.path.join(label_dir, fname))
        if landmarks:
            all_data.append((landmarks, label_idx))
            found += 1
        else:
            miss += 1

    print(f"  {label:<20} detected={found:>4}  skipped={miss:>4}")
    skipped += miss

print(f"\n✅ Total usable : {len(all_data)}")
print(f"⚠️  Skipped      : {skipped}")
assert len(all_data) > 0, "No landmarks extracted."


# ── 4. Dataset ────────────────────────────────────────────────
class LandmarkDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        landmarks, label = self.data[idx]
        return torch.tensor(landmarks, dtype=torch.float32), torch.tensor(label)


dataset = LandmarkDataset(all_data)
n       = len(dataset)
n_val   = max(1, int(VAL_SPLIT  * n))
n_test  = max(1, int(TEST_SPLIT * n))
n_train = n - n_val - n_test

train_ds, val_ds, test_ds = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
print(f"\nSplit — train:{n_train}  val:{n_val}  test:{n_test}")


# ── 5. Improved Model ─────────────────────────────────────────
# Deeper network with residual-style skip connection
# Much better at distinguishing similar signs (A vs S vs T etc.)
class GestureClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Linear(63, 512),
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

        # Skip connection projection (63 -> 128)
        self.skip = nn.Linear(63, 128)

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


# ── 6. Training with label smoothing ─────────────────────────
# Label smoothing reduces overconfidence — model generalises better
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_acc_history = []
val_acc_history   = []
best_val_acc      = 0.0

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

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(EXPORT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(EXPORT_DIR, "best_checkpoint.pth"))

    print(
        f"Epoch {epoch:02d}/{EPOCHS}  "
        f"loss={total_loss/len(train_loader):.4f}  "
        f"train={train_acc:.2%}  "
        f"val={val_acc:.2%}"
        + ("  ← best" if val_acc == best_val_acc else "")
    )

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