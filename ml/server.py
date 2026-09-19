import eventlet
eventlet.monkey_patch()

import os, json, glob, time, signal, numpy as np, cv2, torch, torch.nn as nn
import base64
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from collections import deque, Counter
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from feature_utils import features_for_dim, BASIC_DIM
from text_to_sign import translate as translate_text_to_sign
from word_sequences import (SEQ_LEN, FEATURE_DIM, WordSignGRU,
                            frame_features, resample)

# Force an immediate exit on Ctrl+C / kill. Eventlet's greenlet scheduling can
# swallow or delay the default signal handling, so an explicit os._exit() keeps
# stopping the server reliable.
# (This originally worked around the 3D avatar's SDL/OpenGL window turning the
# process into a macOS Cocoa app that ignored SIGTERM entirely. The avatar has
# since been removed - no OpenGL, no window - but reliable Ctrl+C is worth
# keeping regardless.)
def _force_exit(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    os._exit(0)

signal.signal(signal.SIGINT, _force_exit)
signal.signal(signal.SIGTERM, _force_exit)

# ── CONFIG ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "exported_model", "gesture_recognizer.pth")
LABELS_PATH = os.path.join(BASE_DIR, "exported_model", "labels.json")
LANDMARKER  = os.path.join(BASE_DIR, "hand_landmarker.task")

CONFIDENCE_MIN = 0.6
SMOOTH_FRAMES  = 15

try:
    labels = json.load(open(LABELS_PATH))
except FileNotFoundError:
    print(f"Error: Could not find labels file at {LABELS_PATH}")
    labels = []

# ── MODEL ─────────────────────────────────────────────────────
class GestureClassifier(nn.Module):
    def __init__(self, num_classes: int, in_features: int):
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

        # Skip connection projection (in_features -> 128)
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


# Auto-detect the model's expected input size from the checkpoint so an old
# 63-dim model and a new rich model both load without any code change here.
MODEL_IN_FEATURES = BASIC_DIM
model = None
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location='cpu')
    MODEL_IN_FEATURES = state['features.0.weight'].shape[1]
    model = GestureClassifier(len(labels), MODEL_IN_FEATURES)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model: {len(labels)} classes, {MODEL_IN_FEATURES}-dim input.")
else:
    print(f"Warning: model file not found at {MODEL_PATH}")

# ── PREDICT ───────────────────────────────────────────────────
def predict(raw):
    if model is None or len(labels) == 0:
        return "Unknown", 0.0
    # Use whichever feature representation matches the loaded model.
    feats = features_for_dim(raw, MODEL_IN_FEATURES)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
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

# ── SIGN-TO-TEXT (continuous fingerspelling) ──────────────────
# Word breaks and corrections are driven by the keyboard (space /
# backspace) rather than the "space"/"del" gesture classes, so only
# actual letters get auto-committed here.
LETTER_SET = set(labels) - {"del", "nothing", "space"}

TEXT_HOLD_FRAMES_NEED = 14   # frames a stable letter must hold before it's committed
TEXT_COOLDOWN_FRAMES  = 12   # frames of "no sign / different sign" before the same letter can repeat

class TextSession:
    def __init__(self):
        self.hist         = deque(maxlen=SMOOTH_FRAMES)
        self.label        = ""
        self.conf         = 0.0
        self.sentence     = ""
        self.hold_counter = 0
        self.cooldown     = 0
        self.last_label   = ""

    def update(self, l, c):
        if c < CONFIDENCE_MIN: return
        self.hist.append((l, c))
        if len(self.hist) < 5: return
        votes = Counter([x[0] for x in self.hist])
        best  = votes.most_common(1)[0][0]
        confs = [x[1] for x in self.hist if x[0] == best]
        self.label = best
        self.conf  = sum(confs) / len(confs)

# Per-connection state, keyed by socket id, so concurrent users don't
# share a sentence buffer.
text_sessions = {}

# Per-connection Text->Sign playback state, keyed by socket id. Each entry
# is {'stop': bool} - a background task checks this flag every frame so a
# new translation (or a disconnect) can cancel a running sequence.
sign_sessions = {}

# ── WORD MODEL (motion signs) ─────────────────────────────────
# Loaded alongside the letter model, never replacing it. A letter is a static
# handshape (one frame -> MLP); a word sign is a movement (a sequence -> GRU),
# so they are genuinely different models and the letter path stays untouched.
WORD_MODEL_PATH  = os.path.join(BASE_DIR, "exported_model", "word_model.pth")
WORD_LABELS_PATH = os.path.join(BASE_DIR, "exported_model", "word_labels.json")

WORD_CONFIDENCE_MIN = 0.55   # below this we say nothing rather than guess
WORD_STABLE_NEEDED  = 4      # consecutive agreeing predictions before committing

word_model, word_labels = None, []
try:
    if os.path.exists(WORD_MODEL_PATH) and os.path.exists(WORD_LABELS_PATH):
        word_labels = json.load(open(WORD_LABELS_PATH))
        word_model = WordSignGRU(len(word_labels))
        word_model.load_state_dict(torch.load(WORD_MODEL_PATH, map_location='cpu'))
        word_model.eval()
        print(f"Loaded word model: {len(word_labels)} words.")
    else:
        print("No word model found - word mode disabled (train_word_model.py).")
except Exception as e:
    print("Failed to load word model:", e)
    word_model = None


class WordSession:
    """
    Per-connection state for word recognition.

    Holds a rolling window of the most recent SEQ_LEN frames, so the model can
    be run continuously over live video rather than needing a pre-cut clip.
    """

    def __init__(self):
        self.buffer = deque(maxlen=SEQ_LEN)
        self.sentence = ""
        self.candidate = ""
        self.stable = 0

    def reset_after_commit(self):
        # Clear the window so the sign just committed can't immediately
        # re-trigger from the frames still sitting in the buffer.
        self.buffer.clear()
        self.candidate, self.stable = "", 0


word_sessions = {}

TEXT_TO_SIGN_HOLD_SECONDS = 1.3   # how long each sign stays on screen
TEXT_TO_SIGN_FRAME_INTERVAL = 0.08  # ~12fps pushed to the client

# Setup Flask server
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global state for instances
detector = None
last_label = ""
hands = [HandState(), HandState()]

def init_globals():
    global detector
    try:
        detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
    except Exception as e:
        print("Failed to initialize Mediapipe:", e)

init_globals()

@app.route('/')
def index():
    return "DrishtiSign ML Web Socket Server is running."

@app.route('/word-labels')
def get_word_labels():
    """The word model's vocabulary, so Learning/Practice can offer a word mode
    without hard-coding a list that could drift from the trained model."""
    return jsonify(word_labels)

@app.route('/labels')
def get_labels():
    """The model's real vocabulary, so the frontend never hard-codes a sign
    list that can drift out of sync with what the model was trained on."""
    return jsonify(labels)

@socketio.on('connect')
def test_connect():
    print('Client connected')
    # Clear any smoothing state left over from a previous session so a stale
    # label can't linger into the first few frames of a fresh modal.
    global last_label
    last_label = ""
    for h in hands:
        h.hist.clear()
        h.label, h.conf = "", 0.0

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')
    text_sessions.pop(request.sid, None)
    word_sessions.pop(request.sid, None)
    session = sign_sessions.pop(request.sid, None)
    if session:
        session['stop'] = True

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Receives base64 encoded jpeg from frontend.
    Runs Mediapipe, predicts the gesture, draws the overlay, returns base64 jpeg.
    """
    global last_label, detector
    
    if not data.startswith("data:image"):
        return
        
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return

    # Frame is already flipped by client usually, but we mirror it correctly if needed.
    # The client side canvas draws directly the mirror string.
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    res = None
    if detector:
        res = detector.detect(mp_img)

    left_detected  = False
    right_detected = False

    if res and res.hand_landmarks:
        for i, lm in enumerate(res.hand_landmarks):
            # If the user mirrors the frontend feed, MediaPipe might get it backwards.
            raw_side   = res.handedness[i][0].category_name
            handedness = raw_side

            draw_landmarks(frame, lm)

            raw         = [v for p in lm for v in (p.x, p.y, p.z)]
            label, conf = predict(raw)

            if handedness == "Left":
                hands[0].update(label, conf)
                left_detected = True
            else:
                hands[1].update(label, conf)
                right_detected = True

    active_label = ""
    if left_detected and hands[0].label:
        active_label = hands[0].label
    elif right_detected and hands[1].label:
        active_label = hands[1].label

    if active_label and active_label != last_label:
        last_label = active_label

    if not left_detected:
        hands[0].label, hands[0].conf = "", 0.0
    if not right_detected:
        hands[1].label, hands[1].conf = "", 0.0

    # ── UI LAYOUT ─────────────────────────────────────────────
    h, w    = frame.shape[:2]
    panel_h = h // 2
    if panel_h == 0:
        panel_h = 240
    panel   = np.full((panel_h, w, 3), (18, 18, 25), dtype=np.uint8)
    cv2.line(panel, (0, 0), (w, 0), (50, 50, 60), 1)

    mid_x = w // 4
    mid_w = w // 2

    cv2.line(panel, (mid_x, 0),            (mid_x, panel_h),            (50,50,60), 1)
    cv2.line(panel, (mid_x + mid_w, 0), (mid_x + mid_w, panel_h), (50,50,60), 1)

    info_w = max(mid_x - 30, 10)
    lbl_l  = hands[0].label or "—"
    conf_l = int(hands[0].conf * 100)
    cv2.putText(panel, "LEFT HAND", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,140), 1, cv2.LINE_AA)
    cv2.putText(panel, lbl_l, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(panel, f"{conf_l}%", (15, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,120), 2, cv2.LINE_AA)
    cv2.rectangle(panel, (15, 132), (15 + info_w, 140), (50,50,60), -1)
    cv2.rectangle(panel, (15, 132), (15 + int(info_w*hands[0].conf), 140), (0,200,120), -1)

    rx     = mid_x + mid_w + 15
    lbl_r  = hands[1].label or "—"
    conf_r = int(hands[1].conf * 100)
    cv2.putText(panel, "RIGHT HAND", (rx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,140), 1, cv2.LINE_AA)
    cv2.putText(panel, lbl_r, (rx, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(panel, f"{conf_r}%", (rx, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,120), 2, cv2.LINE_AA)
    cv2.rectangle(panel, (rx, 132), (rx + info_w, 140), (50,50,60), -1)
    cv2.rectangle(panel, (rx, 132), (rx + int(info_w*hands[1].conf), 140), (0,200,120), -1)

    cv2.rectangle(frame, (0, 0), (w, 48), (20, 20, 20), -1)
    cv2.putText(frame, "DrishtiSign  |  Live Detection", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,220,120), 2, cv2.LINE_AA)

    if active_label:
        tag = f"Gesture: {active_label}"
        (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(frame, tag, (w - tw - 15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2, cv2.LINE_AA)

    combined = np.vstack((frame, panel))
    
    # Encode as max-quality jpeg
    _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
    out_b64 = base64.b64encode(buffer).decode('utf-8')
    final_output = f"data:image/jpeg;base64,{out_b64}"
    
    emit('processed_frame', {
        'image': final_output,
        'label': active_label if active_label else "",
        'confidence': max(hands[0].conf, hands[1].conf) if active_label else 0.0
    })

@socketio.on('text_frame')
def handle_text_frame(data):
    """
    Continuous fingerspelling mode. Holds a stable letter for
    TEXT_HOLD_FRAMES_NEED frames before committing it to the session's
    sentence buffer. Word breaks / corrections come from the keyboard
    (see handle_text_key), not from gesture classes.
    """
    if not data.startswith("data:image"):
        return

    sid = request.sid
    session = text_sessions.setdefault(sid, TextSession())

    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    res = detector.detect(mp_img) if detector else None

    detected = False
    if res and res.hand_landmarks:
        lm = res.hand_landmarks[0]
        draw_landmarks(frame, lm)
        raw         = [v for p in lm for v in (p.x, p.y, p.z)]
        label, conf = predict(raw)
        session.update(label, conf)
        detected = True

    if not detected:
        session.label, session.conf = "", 0.0

    current = session.label

    # Hold-to-commit: a letter must stay steady for a stretch of frames,
    # then a cooldown gap is required before the same letter can repeat
    # (so one held sign doesn't spam the same letter over and over).
    if session.cooldown > 0:
        session.cooldown -= 1
        session.hold_counter = 0
        if current != session.last_label:
            session.cooldown = 0
    elif current and current in LETTER_SET:
        if current == session.last_label:
            session.hold_counter += 1
        else:
            session.hold_counter = 1
            session.last_label = current

        if session.hold_counter >= TEXT_HOLD_FRAMES_NEED:
            session.sentence += current
            session.hold_counter = 0
            session.cooldown     = TEXT_COOLDOWN_FRAMES
    else:
        session.hold_counter = 0
        session.last_label   = ""

    hold_progress = min(1.0, session.hold_counter / TEXT_HOLD_FRAMES_NEED)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(frame, "DrishtiSign  |  Sign to Text", (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2, cv2.LINE_AA)

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    out_b64 = base64.b64encode(buffer).decode('utf-8')

    emit('text_processed_frame', {
        'image': f"data:image/jpeg;base64,{out_b64}",
        'label': current,
        'confidence': session.conf,
        'holdProgress': hold_progress,
        'sentence': session.sentence,
    })

@socketio.on('text_key')
def handle_text_key(data):
    """Keyboard-driven word breaks / corrections: {action: 'space' | 'backspace' | 'clear'}."""
    sid = request.sid
    session = text_sessions.setdefault(sid, TextSession())
    action = (data or {}).get('action')

    if action == 'space':
        session.sentence += ' '
    elif action == 'backspace':
        session.sentence = session.sentence[:-1]
    elif action == 'clear':
        session.sentence = ''

    emit('text_processed_frame', {'sentence': session.sentence})

# ── WORD RECOGNITION (motion signs -> text) ─────────────────────
# Same shape as text_frame, but instead of classifying each frame on its own
# it keeps a rolling window of the last SEQ_LEN frames and runs the temporal
# model over the whole window - which is what lets it recognise a movement
# rather than a pose.
@socketio.on('word_frame')
def handle_word_frame(data):
    if not data.startswith("data:image"):
        return

    sid = request.sid
    session = word_sessions.setdefault(sid, WordSession())

    if word_model is None:
        emit('word_processed_frame', {'error': 'Word model not loaded on the server.'})
        return

    nparr = np.frombuffer(base64.b64decode(data.split(',')[1]), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)) if detector else None

    if res and res.hand_landmarks:
        lm = res.hand_landmarks[0]
        draw_landmarks(frame, lm)
        session.buffer.append(frame_features(lm))
    # A gap with no hand is meaningful - it separates one sign from the next -
    # so we simply stop feeding the buffer rather than clearing it outright.

    label, confidence = "", 0.0
    committed = None

    if len(session.buffer) == SEQ_LEN:
        seq = resample(np.array(session.buffer))
        with torch.no_grad():
            probs = torch.softmax(
                word_model(torch.tensor(seq, dtype=torch.float32).unsqueeze(0)), dim=1)
            conf, idx = probs.max(dim=1)
        label, confidence = word_labels[idx.item()], float(conf.item())

        # Dwell before committing, mirroring the fingerspelling logic: the same
        # word has to win several predictions in a row, so a single frame of a
        # half-formed sign can't type a word.
        if confidence >= WORD_CONFIDENCE_MIN:
            if label == session.candidate:
                session.stable += 1
            else:
                session.candidate, session.stable = label, 1

            if session.stable >= WORD_STABLE_NEEDED:
                session.sentence += (" " if session.sentence else "") + label
                committed = label
                session.reset_after_commit()
        else:
            session.candidate, session.stable = "", 0

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    emit('word_processed_frame', {
        'image': f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}",
        'label': label,
        'confidence': confidence,
        'bufferProgress': len(session.buffer) / SEQ_LEN,
        'stableProgress': min(1.0, session.stable / WORD_STABLE_NEEDED),
        'committed': committed,
        'sentence': session.sentence,
    })


@socketio.on('word_key')
def handle_word_key(data):
    """Corrections for word mode: {action: 'backspace' | 'clear'}."""
    sid = request.sid
    session = word_sessions.setdefault(sid, WordSession())
    action = (data or {}).get('action')

    if action == 'backspace':
        session.sentence = " ".join(session.sentence.split()[:-1])
    elif action == 'clear':
        session.sentence = ''

    emit('word_processed_frame', {'sentence': session.sentence})

# ── TEXT -> SIGN (typed sentence -> ordered sign queue) ─────────
# The server only does the language work: English -> ASL gloss -> an ordered
# queue of signs to show. Playback is the frontend's job, so this path needs
# no rendering, no OpenGL and no background streaming task.
MAX_SIGN_QUEUE = 60  # ~1.5 min of playback at TEXT_TO_SIGN_HOLD_SECONDS each

@socketio.on('translate_text')
def handle_translate_text(data):
    sid = request.sid
    text = (data or {}).get('text', '').strip()
    use_llm = bool((data or {}).get('useLLM', False))

    # A new translation cancels whatever this client was already playing.
    old = sign_sessions.get(sid)
    if old:
        old['stop'] = True

    if not text:
        emit('sign_translate_error', {'message': 'Please enter some text.'})
        return
    if len(text) > 300:
        emit('sign_translate_error', {'message': 'That sentence is too long — please shorten it.'})
        return

    result = translate_text_to_sign(text, use_llm=use_llm)
    queue = result['queue'][:MAX_SIGN_QUEUE]

    if not queue:
        emit('sign_translate_error', {
            'message': "Couldn't find any signable words in that sentence.",
            'simplified': result['simplified'],
            'gloss': result['gloss'],
        })
        return

    session = {'stop': False}
    sign_sessions[sid] = session

    emit('sign_translate_started', {
        'simplified': result['simplified'],
        'gloss': result['gloss'],
        'queue': queue,
    })



@socketio.on('stop_sign_sequence')
def handle_stop_sign_sequence():
    session = sign_sessions.get(request.sid)
    if session:
        session['stop'] = True

if __name__ == '__main__':
    print("Starting DrishtiSign ML Socket Server on port 5002...")
    socketio.run(app, host='0.0.0.0', port=5002)
