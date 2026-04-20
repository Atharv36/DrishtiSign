import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import json
import time
import os

# 1. Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define the relative path to the folder
REL_PATH = os.path.join("ml", "exported_model")

# 3. Combine them
MODEL_DIR = os.path.join(SCRIPT_DIR, REL_PATH)

# --- Load labels with fallback ---
labels_path = os.path.join(MODEL_DIR, "labels.json")

if not os.path.exists(labels_path):
    print(f"❌ Still can't find it at: {labels_path}")
    fallback_path = os.path.join(os.getcwd(), "ml", "exported_model", "labels.json")
    if os.path.exists(fallback_path):
        print(f"✅ Found it in fallback path!")
        labels_path = fallback_path
    else:
        print("❌ Folder structure check: Please ensure 'ml' folder is in the same place as this script.")
        exit()

# FIX 1: Only load labels ONCE, using the resolved path
with open(labels_path, "r") as f:
    labels = json.load(f)

# FIX 2: Use MODEL_PATH variable (was defined but ignored before)
MODEL_PATH = os.path.join(os.path.dirname(labels_path), "gesture_recognizer.pth")

# --- Model Definition ---
class GestureClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureClassifier(num_classes=len(labels)).to(device)

# FIX 2 (continued): Load weights from resolved MODEL_PATH, not hardcoded string
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded | {len(labels)} gestures: {labels}")

# --- Setup MediaPipe & Camera ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

def run_accuracy_test(target_label_idx):
    """Tests accuracy for a specific gesture over 5 seconds."""
    target_name = labels[target_label_idx]
    correct_frames = 0
    total_valid_frames = 0
    start_time = time.time()
    duration = 5  # Seconds per test

    print(f"\n--- Testing Gesture: {target_name.upper()} ---")
    print("Get ready...")
    time.sleep(2)

    with mp_hands.Hands(min_detection_confidence=0.7) as hands:
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb_frame)  # FIX 3: renamed to avoid any future clash

            current_pred = None

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    landmarks = [v for p in hand_landmarks.landmark for v in (p.x, p.y, p.z)]
                    input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        current_pred = torch.softmax(output, dim=1).argmax(1).item()

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                total_valid_frames += 1
                if current_pred == target_label_idx:
                    correct_frames += 1

            # UI Overlay
            remaining = int(duration - (time.time() - start_time))
            cv2.putText(frame, f"HOLD GESTURE: {target_name}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {remaining}s", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Accuracy Tester", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    accuracy = (correct_frames / total_valid_frames * 100) if total_valid_frames > 0 else 0
    return accuracy

# --- Main Testing Logic ---
print("Available gestures to test:", labels)
results_summary = {}

try:
    for idx, name in enumerate(labels):
        acc = run_accuracy_test(idx)
        results_summary[name] = acc
        print(f"Result for {name}: {acc:.2f}% accuracy")

    print("\n" + "="*30)
    print("FINAL ACCURACY REPORT")
    print("="*30)
    for gesture, score in results_summary.items():
        print(f"{gesture.ljust(15)}: {score:.2f}%")
    print("="*30)

finally:
    cap.release()
    cv2.destroyAllWindows()