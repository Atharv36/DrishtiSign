import os, json
import numpy as np, cv2, torch, torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from collections import deque, Counter

# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "exported_model", "gesture_recognizer.pth")
LABELS_PATH = os.path.join(BASE_DIR, "exported_model", "labels.json")
LANDMARKER  = os.path.join(BASE_DIR, "hand_landmarker.task")

CONFIDENCE_MIN   = 0.6
SMOOTH_FRAMES    = 15
HOLD_FRAMES_NEED = 20     # frames a stable sign must hold before it's committed
COOLDOWN_FRAMES  = 15     # frames of "no sign / different sign" before next commit allowed

labels = json.load(open(LABELS_PATH))

# ── MODEL (matches mltrainercode.py's deeper residual arch) ──
class GestureClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(63, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.skip = nn.Linear(63, 128)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        feat = self.features(x)
        skip = self.skip(x)
        return self.classifier(feat + skip)

model = GestureClassifier(len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

def normalize_landmarks(raw):
    pts = np.array(raw).reshape(21, 3)
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9])
    if scale > 0: pts /= scale
    return pts.flatten()

def predict(raw):
    x = torch.tensor(normalize_landmarks(raw), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        conf, idx = probs.max(dim=1)
    return labels[idx.item()], conf.item()

# ── DRAW 2D SKELETON (same as detect_live.py) ─────────────────
CONNECTION_COLORS = {
    (0,1):(0,200,255),(1,2):(0,200,255),(2,3):(0,200,255),(3,4):(0,200,255),
    (0,5):(0,255,100),(5,6):(0,255,100),(6,7):(0,255,100),(7,8):(0,255,100),
    (0,9):(255,180,0),(9,10):(255,180,0),(10,11):(255,180,0),(11,12):(255,180,0),
    (0,13):(200,0,255),(13,14):(200,0,255),(14,15):(200,0,255),(15,16):(200,0,255),
    (0,17):(0,100,255),(17,18):(0,100,255),(18,19):(0,100,255),(19,20):(0,100,255),
    (5,9):(200,200,200),(9,13):(200,200,200),(13,17):(200,200,200),
}
FINGERTIPS = [4, 8, 12, 16, 20]

def draw_landmarks(frame, lm_list):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
    for (a, b), color in CONNECTION_COLORS.items():
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        r = 8 if i in FINGERTIPS else (7 if i == 0 else 5)
        cv2.circle(frame, pt, r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, r, (0, 200, 100), 1,  cv2.LINE_AA)

# ── HAND STATE (temporal smoothing) ────────────────────────────
class HandState:
    def __init__(self):
        self.hist  = deque(maxlen=SMOOTH_FRAMES)
        self.label = ""
        self.conf  = 0.0

    def update(self, l, c):
        if c < CONFIDENCE_MIN: return
        self.hist.append((l, c))
        if len(self.hist) < 5: return
        votes = Counter([x[0] for x in self.hist])
        best  = votes.most_common(1)[0][0]
        confs = [x[1] for x in self.hist if x[0] == best]
        self.label = best
        self.conf  = sum(confs) / len(confs)

hand = HandState()

# ── SENTENCE BUILDER STATE ─────────────────────────────────────
sentence      = ""
hold_counter  = 0
cooldown      = 0
last_label    = ""

def commit(label):
    """Apply a confirmed gesture (a letter) to the sentence buffer."""
    global sentence
    if label == "del":
        sentence = sentence[:-1]
    elif label == "nothing":
        pass
    else:
        sentence += label

# ── MEDIAPIPE ─────────────────────────────────────────────────
latest = [None]
def cb(res, img, t): latest[0] = res

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        result_callback=cb
    )
)

cap = cv2.VideoCapture(0)
frame_count = 0

print("Sign-to-Text running. Hold a sign steady to commit it.")
print("SPACEBAR = space | BACKSPACE = delete | C = clear | Q = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_count += 1

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detector.detect_async(mp_img, frame_count)

    res      = latest[0]
    detected = False

    if res and res.hand_landmarks:
        lm = res.hand_landmarks[0]
        draw_landmarks(frame, lm)
        raw = [v for p in lm for v in (p.x, p.y, p.z)]
        label, conf = predict(raw)
        hand.update(label, conf)
        detected = True

    if not detected:
        hand.label, hand.conf = "", 0.0

    current = hand.label

    # ── Commit logic: hold a stable sign, then require a
    # cooldown gap before the same sign can commit again ──────
    if cooldown > 0:
        cooldown -= 1
        hold_counter = 0
        if current != last_label:
            cooldown = 0
    elif current and current != "nothing":
        if current == last_label:
            hold_counter += 1
        else:
            hold_counter = 1
            last_label = current

        if hold_counter >= HOLD_FRAMES_NEED:
            commit(current)
            hold_counter = 0
            cooldown     = COOLDOWN_FRAMES
    else:
        hold_counter = 0
        last_label   = ""

    # ── TOP: camera feed with overlay ──────────────────────────
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 48), (20, 20, 20), -1)
    cv2.putText(frame, "DrishtiSign  |  Sign to Text",
                (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 120), 2, cv2.LINE_AA)

    status = current if current else "—"
    cv2.putText(frame, f"Sign: {status}", (w - 260, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2, cv2.LINE_AA)

    bar_w = w - 30
    fill  = int(bar_w * hold_counter / HOLD_FRAMES_NEED)
    cv2.rectangle(frame, (15, h-20), (15+bar_w, h-10), (50,50,60), -1)
    cv2.rectangle(frame, (15, h-20), (15+fill,  h-10), (0,200,120), -1)

    # ── BOTTOM: translated sentence panel ──────────────────────
    panel_h = 140
    panel   = np.full((panel_h, w, 3), (18, 18, 25), dtype=np.uint8)
    cv2.line(panel, (0, 0), (w, 0), (50, 50, 60), 1)
    cv2.putText(panel, "TRANSLATED TEXT", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 140), 1, cv2.LINE_AA)

    display_text = sentence if sentence else "(start signing...)"
    max_chars = 48
    lines = [display_text[i:i+max_chars] for i in range(0, len(display_text), max_chars)] or [""]
    y = 70
    for line in lines[-2:]:
        cv2.putText(panel, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        y += 40

    cv2.putText(panel, "SPACEBAR = space | BACKSPACE = delete | C = clear | Q = quit",
                (20, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140,140,140), 1, cv2.LINE_AA)

    combined = np.vstack((frame, panel))
    cv2.imshow("DrishtiSign — Sign to Text", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
    elif key == 32:              # spacebar
        sentence += " "
    elif key in (8, 127):        # backspace / delete
        sentence = sentence[:-1]

cap.release()
cv2.destroyAllWindows()