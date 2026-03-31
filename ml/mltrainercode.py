# ✅ VS Code + Python 3.13 compatible
# First, install dependencies in your terminal:
# pip install mediapipe opencv-python torch torchvision torchaudio matplotlib Pillow

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import json

# ── 1. Set your dataset path ────────────────────────────────────
my_folder_path = r'C:\Users\harsh\DrishtiSign\ml\sign_language_dataset'  # ← Change this
# For Mac/Linux: my_folder_path = '/home/yourname/sign_language_dataset'

# ── 2. Load Labels ──────────────────────────────────────────────
labels = sorted([
    i for i in os.listdir(my_folder_path)
    if os.path.isdir(os.path.join(my_folder_path, i))
])
print(f"Found {len(labels)} labels: {labels}")

# ── 3. Preview Examples ─────────────────────────────────────────
NUM_EXAMPLES = 10
for label in labels:
    label_dir = os.path.join(my_folder_path, label)
    example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
    fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10, 2))
    for i in range(NUM_EXAMPLES):
        axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {label}')
    plt.show()

# ── 4. Extract Hand Landmarks using MediaPipe ───────────────────
mp_hands = mp.solutions.hands

def extract_landmarks(image_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(img_np)
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            return [val for point in lm for val in (point.x, point.y, point.z)]
    return None

# ── 5. Build Landmark Dataset ───────────────────────────────────
class LandmarkDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        landmarks, label = self.data[idx]
        return torch.tensor(landmarks, dtype=torch.float32), label

print("Extracting hand landmarks (this may take a few minutes)...")
all_data = []
for label in labels:
    label_dir = os.path.join(my_folder_path, label)
    for fname in os.listdir(label_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(label_dir, fname)
            landmarks = extract_landmarks(fpath)
            if landmarks:
                all_data.append((landmarks, labels.index(label)))

print(f"✅ Extracted landmarks from {len(all_data)} images")

# ── 6. Split Dataset ────────────────────────────────────────────
dataset    = LandmarkDataset(all_data)
train_size = int(0.8 * len(dataset))
rest_size  = len(dataset) - train_size
train_dataset, rest_dataset = random_split(dataset, [train_size, rest_size])

val_size  = rest_size // 2
test_size = rest_size - val_size
val_dataset, test_dataset = random_split(rest_dataset, [val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32)
test_loader  = DataLoader(test_dataset,  batch_size=32)

# ── 7. Define Model ─────────────────────────────────────────────
class GestureClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GestureClassifier(num_classes=len(labels)).to(device)
print(f"Training on: {device}")

# ── 8. Train ────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
EPOCHS    = 30

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == y).sum().item()

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            val_correct += (model(X).argmax(1) == y).sum().item()

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Loss: {total_loss/len(train_loader):.4f} | "
          f"Train Acc: {correct/len(train_dataset):.2%} | "
          f"Val Acc: {val_correct/len(val_dataset):.2%}")

# ── 9. Test Accuracy ────────────────────────────────────────────
model.eval()
test_correct = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        test_correct += (model(X).argmax(1) == y).sum().item()
print(f"\n✅ Test Accuracy: {test_correct/len(test_dataset):.2%}")

# ── 10. Export Model ────────────────────────────────────────────
os.makedirs("exported_model", exist_ok=True)

torch.save(model.state_dict(), "exported_model/gesture_recognizer.pth")

with open("exported_model/labels.json", "w") as f:
    json.dump(labels, f)

scripted = torch.jit.script(model)
scripted.save("exported_model/gesture_recognizer_scripted.pt")

print("📦 Model saved to exported_model/")
print(os.listdir("exported_model"))