import os, json, glob, numpy as np, cv2, torch, torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from collections import deque, Counter
 
from Hand_avatar import HandAvatar3D
 
# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "exported_model", "gesture_recognizer.pth")
LABELS_PATH = os.path.join(BASE_DIR, "exported_model", "labels.json")
LANDMARKER  = os.path.join(BASE_DIR, "hand_landmarker.task")
 
CONFIDENCE_MIN = 0.6
SMOOTH_FRAMES  = 15
 
labels = json.load(open(LABELS_PATH))
 
# ── CHECK: are gesture JSONs present? ────────────────────────
def check_gestures():
    exclude = {"labels.json"}
    jsons   = [f for f in glob.glob(os.path.join(BASE_DIR, "*.json"))
               if os.path.basename(f) not in exclude]
    if not jsons:
        print("=" * 60)
        print("  No gesture JSON files found!")
        print("  Run this first:")
        print("      python setup_asl_gestures.py")
        print("=" * 60)
        print("  Continuing — avatar will show nothing until setup runs.\n")
        return
    names = sorted([os.path.splitext(os.path.basename(f))[0] for f in jsons])
    print(f"Gesture files loaded: {names}\n")
 
check_gestures()
 
# ── MODEL ─────────────────────────────────────────────────────
class GestureClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x): return self.net(x)
 
model = GestureClassifier(len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
 
# ── NORMALIZE + PREDICT ───────────────────────────────────────
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
 
# ── DRAW 2D SKELETON ON CAMERA FEED ──────────────────────────
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
 
# ── HAND STATE (temporal smoothing) ───────────────────────────
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
 
hands = [HandState(), HandState()]   # [0]=LEFT  [1]=RIGHT
 
# ── 3D AVATAR ─────────────────────────────────────────────────
avatar = HandAvatar3D(width=400, height=400)
 
# ── MEDIAPIPE ─────────────────────────────────────────────────
latest = [None]
def cb(res, img, t): latest[0] = res
 
detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        result_callback=cb
    )
)
 
# ── CAMERA ────────────────────────────────────────────────────
cap        = cv2.VideoCapture(0)
frame_count = 0
last_label  = ""     # only call load_gesture when label changes
 
print("DrishtiSign running. Press Q to quit.\n")
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    frame = cv2.flip(frame, 1)
    frame_count += 1
 
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detector.detect_async(mp_img, frame_count)
 
    res            = latest[0]
    left_detected  = False
    right_detected = False
 
    if res and res.hand_landmarks:
        for i, lm in enumerate(res.hand_landmarks):
            # Flip handedness because frame is mirrored
            raw_side   = res.handedness[i][0].category_name
            handedness = "Right" if raw_side == "Left" else "Left"
 
            draw_landmarks(frame, lm)
 
            raw         = [v for p in lm for v in (p.x, p.y, p.z)]
            label, conf = predict(raw)
 
            if handedness == "Left":
                hands[0].update(label, conf)
                left_detected = True
            else:
                hands[1].update(label, conf)
                right_detected = True
 
    # ── Pick active label → load avatar only when it changes ──
    active_label = ""
    if left_detected  and hands[0].label:
        active_label = hands[0].label
    elif right_detected and hands[1].label:
        active_label = hands[1].label
 
    if active_label and active_label != last_label:
        avatar.load_gesture(active_label)   # HandAvatar3D.load_gesture()
        last_label = active_label
 
    # Clear missing hands
    if not left_detected:
        hands[0].label, hands[0].conf = "", 0.0
    if not right_detected:
        hands[1].label, hands[1].conf = "", 0.0
 
    # ── UI LAYOUT ─────────────────────────────────────────────
    h, w    = frame.shape[:2]
    panel_h = h // 2
    panel   = np.full((panel_h, w, 3), (18, 18, 25), dtype=np.uint8)
    cv2.line(panel, (0, 0), (w, 0), (50, 50, 60), 1)
 
    # 3D Avatar — centre half of panel
    avatar_x = w // 4
    avatar_w = w // 2
    avatar.draw(panel[:, avatar_x : avatar_x + avatar_w])
 
    cv2.line(panel, (avatar_x, 0),            (avatar_x, panel_h),            (50,50,60), 1)
    cv2.line(panel, (avatar_x + avatar_w, 0), (avatar_x + avatar_w, panel_h), (50,50,60), 1)
 
    # ── LEFT hand info ────────────────────────────────────────
    info_w = avatar_x - 30
    lbl_l  = hands[0].label or "—"
    conf_l = int(hands[0].conf * 100)
    cv2.putText(panel, "LEFT HAND",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,140), 1, cv2.LINE_AA)
    cv2.putText(panel, lbl_l,
                (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(panel, f"{conf_l}%",
                (15, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,120), 2, cv2.LINE_AA)
    cv2.rectangle(panel, (15, 132), (15 + info_w, 140),             (50,50,60),  -1)
    cv2.rectangle(panel, (15, 132), (15 + int(info_w*hands[0].conf), 140), (0,200,120), -1)
 
    # ── RIGHT hand info ───────────────────────────────────────
    rx     = avatar_x + avatar_w + 15
    lbl_r  = hands[1].label or "—"
    conf_r = int(hands[1].conf * 100)
    cv2.putText(panel, "RIGHT HAND",
                (rx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,140), 1, cv2.LINE_AA)
    cv2.putText(panel, lbl_r,
                (rx, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(panel, f"{conf_r}%",
                (rx, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,120), 2, cv2.LINE_AA)
    cv2.rectangle(panel, (rx, 132), (rx + info_w, 140),             (50,50,60),  -1)
    cv2.rectangle(panel, (rx, 132), (rx + int(info_w*hands[1].conf), 140), (0,200,120), -1)
 
    # ── Top bar on camera ─────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 48), (20, 20, 20), -1)
    cv2.putText(frame, "DrishtiSign  |  Live Detection",
                (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,220,120), 2, cv2.LINE_AA)
 
    # Active gesture label — top right of camera feed
    if active_label:
        tag = f"Gesture: {active_label}"
        (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(frame, tag, (w - tw - 15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2, cv2.LINE_AA)
 
    combined = np.vstack((frame, panel))
    cv2.imshow("DrishtiSign", combined)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
avatar.close()
 