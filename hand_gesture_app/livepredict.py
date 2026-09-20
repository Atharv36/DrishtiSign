"""
livepredict.py
---------------
Opens the webcam, runs MediaPipe hand tracking + your trained model,
and builds a "sentence" out of held signs.

Run with:
    python livepredict.py

Make sure nepali_char_model.pth (produced by train.py) is in the
same folder, or update MODEL_PATH below.
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque, Counter

from model import MYNN

# ----------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------
IMG_SIZE = 128
DATASET_PATH = "dataset"
MODEL_PATH = "nepali_char_model.pth"

CONFIDENCE_THRESHOLD = 55.0      # % — predictions below this are ignored
HOLD_FRAMES_TO_ADD = 15          # ~0.5-1 sec depending on your webcam fps
CROP_PADDING = 0.25
MAX_NUM_HANDS = 2

# --- Smoothing settings ---
# EMA (exponential moving average) smooths the model's probability
# vector over time instead of voting on raw discrete labels. Lower
# PROB_EMA_ALPHA = smoother but slower to react; higher = snappier
# but jitterier. 0.25-0.35 is a good starting range.
PROB_EMA_ALPHA = 0.3
# Same idea, applied to the hand's bounding box coordinates, so the
# crop fed to the model doesn't jump around frame to frame.
BBOX_EMA_ALPHA = 0.4

# --- Landmark drawing style ---
# Ocean blue, thin lines/dots instead of MediaPipe's default style.
# Note: color tuples here are BGR (OpenCV convention), since we draw
# directly onto the BGR webcam frame.
OCEAN_BLUE = (182, 119, 0)       # BGR for RGB (0, 119, 182)
LANDMARK_SPEC = None              # set after mp_drawing is imported below
CONNECTION_SPEC = None

CLASS_NAMES_PATH = "class_names.txt"
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

# ----------------------------------------------------------------------
# 2. LOAD MODEL
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MYNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print(f"Model loaded on {device}. {len(CLASS_NAMES)} classes.")


# ----------------------------------------------------------------------
# 3. MEDIAPIPE HANDS SETUP — tracks up to two hands
# ----------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LANDMARK_SPEC = mp_drawing.DrawingSpec(color=OCEAN_BLUE, thickness=1, circle_radius=2)
CONNECTION_SPEC = mp_drawing.DrawingSpec(color=OCEAN_BLUE, thickness=1)

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)


# ----------------------------------------------------------------------
# 4. HELPERS
# ----------------------------------------------------------------------
def get_hand_bbox(hand_landmarks, frame_w, frame_h, padding=CROP_PADDING):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_w = x_max - x_min
    box_h = y_max - y_min

    x_min -= box_w * padding
    x_max += box_w * padding
    y_min -= box_h * padding
    y_max += box_h * padding

    x1 = max(int(x_min * frame_w), 0)
    y1 = max(int(y_min * frame_h), 0)
    x2 = min(int(x_max * frame_w), frame_w)
    y2 = min(int(y_max * frame_h), frame_h)

    return x1, y1, x2, y2


def preprocess_crop(crop_bgr):
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    tensor = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)
    return tensor.to(device)


def predict_probs(crop_bgr):
    """Returns the full softmax probability vector (numpy array), not just top-1."""
    tensor = preprocess_crop(crop_bgr)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
    return probs.squeeze(0).cpu().numpy()


def draw_sentence_bar(frame, sentence_words):
    h, w = frame.shape[:2]
    bar_height = 60
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    sentence_text = " ".join(sentence_words) if sentence_words else "(sentence empty)"
    cv2.putText(
        frame, sentence_text, (15, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )


class HandTrackState:
    """
    Independent hold-to-add + smoothing state for ONE hand (Left/Right).

    Smoothing is done with an exponential moving average (EMA) over the
    model's full probability vector, and over the bounding box coords,
    rather than voting on raw discrete labels. This makes both the
    displayed confidence and the crop location glide instead of jump.
    """
    def __init__(self, num_classes):
        self.smoothed_probs = None            # EMA of probability vector
        self.smoothed_bbox = None             # EMA of (x1, y1, x2, y2)
        self.hold_label = None
        self.hold_count = 0
        self.already_added = False
        self.current_label = None
        self.current_conf = 0.0

    def reset_if_absent(self):
        self.smoothed_probs = None
        self.smoothed_bbox = None
        self.hold_label = None
        self.hold_count = 0
        self.already_added = False
        self.current_label = None
        self.current_conf = 0.0

    def smooth_bbox(self, raw_bbox):
        if self.smoothed_bbox is None:
            self.smoothed_bbox = np.array(raw_bbox, dtype=np.float32)
        else:
            raw = np.array(raw_bbox, dtype=np.float32)
            self.smoothed_bbox = (
                BBOX_EMA_ALPHA * raw + (1 - BBOX_EMA_ALPHA) * self.smoothed_bbox
            )
        return tuple(int(v) for v in self.smoothed_bbox)

    def update(self, raw_probs):
        if self.smoothed_probs is None:
            self.smoothed_probs = raw_probs
        else:
            self.smoothed_probs = (
                PROB_EMA_ALPHA * raw_probs + (1 - PROB_EMA_ALPHA) * self.smoothed_probs
            )

        best_idx = int(np.argmax(self.smoothed_probs))
        self.current_label = CLASS_NAMES[best_idx]
        self.current_conf = float(self.smoothed_probs[best_idx]) * 100

    def handle_hold(self, sentence_words, hand_tag):
        if self.current_label is not None and self.current_conf >= CONFIDENCE_THRESHOLD:
            if self.current_label == self.hold_label:
                self.hold_count += 1
            else:
                self.hold_label = self.current_label
                self.hold_count = 1
                self.already_added = False

            if self.hold_count >= HOLD_FRAMES_TO_ADD and not self.already_added:
                sentence_words.append(self.hold_label)
                self.already_added = True
                print(f"Added ({hand_tag}):", self.hold_label, "-> Sentence:", " ".join(sentence_words))
        else:
            self.hold_label = None
            self.hold_count = 0
            self.already_added = False


# ----------------------------------------------------------------------
# 5. LIVE WEBCAM LOOP
# ----------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    hand_states = {
        "Left": HandTrackState(len(CLASS_NAMES)),
        "Right": HandTrackState(len(CLASS_NAMES)),
    }

    sentence_words = []

    print("Controls: SPACE=add current sign  b=backspace  c=clear  s=save  q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)

        seen_this_frame = set()

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                hand_tag = handedness.classification[0].label  # "Left" or "Right"
                seen_this_frame.add(hand_tag)
                state = hand_states[hand_tag]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=LANDMARK_SPEC,
                    connection_drawing_spec=CONNECTION_SPEC,
                )

                raw_bbox = get_hand_bbox(hand_landmarks, frame_w, frame_h)
                x1, y1, x2, y2 = state.smooth_bbox(raw_bbox)

                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        raw_probs = predict_probs(crop)
                        state.update(raw_probs)

                        box_color = OCEAN_BLUE if state.current_conf >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(
                            frame,
                            f"{hand_tag}: {state.current_label} ({state.current_conf:.0f}%)",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA
                        )

        for hand_tag, state in hand_states.items():
            if hand_tag not in seen_this_frame:
                state.reset_if_absent()

        for hand_tag, state in hand_states.items():
            if hand_tag in seen_this_frame:
                state.handle_hold(sentence_words, hand_tag)

        active_states = [
            (tag, s) for tag, s in hand_states.items()
            if s.hold_label is not None and not s.already_added
        ]
        if active_states:
            tag, state = max(active_states, key=lambda t: t[1].hold_count)
            progress = min(state.hold_count / HOLD_FRAMES_TO_ADD, 1.0)
            bar_w = int(200 * progress)
            cv2.rectangle(frame, (20, 60), (220, 80), (80, 80, 80), 2)
            cv2.rectangle(frame, (20, 60), (20 + bar_w, 80), OCEAN_BLUE, -1)
            cv2.putText(
                frame, f"{tag} hold to add...", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        draw_sentence_bar(frame, sentence_words)

        cv2.imshow("Hand Sign Sentence Builder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            if sentence_words:
                removed = sentence_words.pop()
                print("Removed:", removed, "-> Sentence:", " ".join(sentence_words))
        elif key == ord('c'):
            sentence_words.clear()
            print("Sentence cleared.")
        elif key == ord('s'):
            with open("sentence_output.txt", "a") as f:
                f.write(" ".join(sentence_words) + "\n")
            print("Saved to sentence_output.txt:", " ".join(sentence_words))
        elif key == ord(' '):
            chosen = None
            if hand_states["Right"].current_label is not None:
                chosen = hand_states["Right"].current_label
            elif hand_states["Left"].current_label is not None:
                chosen = hand_states["Left"].current_label

            if chosen is not None:
                sentence_words.append(chosen)
                print("Manually added:", chosen, "-> Sentence:", " ".join(sentence_words))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()