import os, sys, json, glob
import numpy as np
import cv2
import kagglehub
from Hand_avatar import HandAvatar3D
# ── Config ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LANDMARKER = os.path.join(BASE_DIR, "hand_landmarker.task")
 
# Letters to process
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]
 
# Images sampled per letter (40 = good balance of quality vs speed)
IMAGES_PER_LETTER = 40
 
# Warn if fewer than this many hands detected per letter
MIN_VALID = 10
 
 
# ── Find dataset folder ───────────────────────────────────────
def find_dataset(base_path):
    """
    Walk base_path looking for a folder that contains A/, B/, C/
    subdirectories — that is the asl_alphabet_train root.
    """
    for root, dirs, _ in os.walk(base_path):
        dirs_upper = [d.upper() for d in dirs]
        if "A" in dirs_upper and "B" in dirs_upper and "C" in dirs_upper:
            return root
    return None
 
 
def ask_dataset_path():
    print("\n" + "=" * 60)
    print("  DrishtiSign — ASL Gesture Setup")
    print("=" * 60)
    print("\nDownload the dataset from Kaggle first (no API needed):")
    print("  https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
    print("  → Click Download → Unzip → note the folder path\n")
    print("Or via terminal (Kaggle CLI):")
    print("  kaggle datasets download -d grassknoted/asl-alphabet")
    print("  unzip asl-alphabet.zip -d asl_dataset\n")
 
    while True:
        path = input("Enter path to unzipped dataset folder: ").strip().strip('"').strip("'")
 
        if not path:
            print("  Path cannot be empty. Try again.")
            continue
 
        path = os.path.expanduser(path)   # handle ~/... paths
 
        if not os.path.isdir(path):
            print(f"  Folder not found: {path}")
            print("  Check the path and try again.")
            continue
 
        train_dir = find_dataset(path)
        if train_dir:
            print(f"\n  Found dataset at: {train_dir}\n")
            return train_dir
 
        print(f"  Could not find A/, B/, C/ subfolders inside: {path}")
        print("  Make sure you unzipped the dataset correctly.")
        print("  Expected structure:  <your_folder>/A/*.jpg")
 
 
# ── MediaPipe detector ────────────────────────────────────────
def build_detector():
    if not os.path.exists(LANDMARKER):
        print(f"\n❌ hand_landmarker.task not found at:\n   {LANDMARKER}")
        print("\nDownload it with:")
        print("  wget https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        sys.exit(1)
 
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
 
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER),
            running_mode=vision.RunningMode.IMAGE,   # synchronous = frame-accurate
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"❌ MediaPipe init failed: {e}")
        sys.exit(1)
 
 
def extract_landmarks(detector, img_path):
    """
    Detect hand landmarks in one image.
    Returns [[x, y], ...] for 21 points, or None if no hand found.
    """
    import mediapipe as mp
 
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
 
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
 
    try:
        result = detector.detect(mp_img)
    except Exception:
        return None
 
    if not result or not result.hand_landmarks:
        return None
 
    return [[float(p.x), float(p.y)] for p in result.hand_landmarks[0]]
 
 
# ── Main ──────────────────────────────────────────────────────
def main():
    train_dir = ask_dataset_path()
 
    print("Initialising MediaPipe Hand Landmarker...")
    detector = build_detector()
    print("Detector ready.\n")
 
    print(f"Processing {len(LETTERS)} gesture classes "
          f"({IMAGES_PER_LETTER} images each)...\n")
 
    total_saved = 0
    skipped     = []
 
    for letter in LETTERS:
        # Find the matching subfolder (case-insensitive)
        letter_dir = None
        for d in os.listdir(train_dir):
            if d.upper() == letter.upper() or d == letter:
                letter_dir = os.path.join(train_dir, d)
                break
 
        if not letter_dir or not os.path.isdir(letter_dir):
            print(f"  [ skip ]  '{letter}' — folder not found")
            skipped.append(letter)
            continue
 
        # Collect all images
        img_paths = sorted(
            glob.glob(os.path.join(letter_dir, "*.jpg"))  +
            glob.glob(os.path.join(letter_dir, "*.jpeg")) +
            glob.glob(os.path.join(letter_dir, "*.png"))
        )
 
        if not img_paths:
            print(f"  [ skip ]  '{letter}' — no images found")
            skipped.append(letter)
            continue
 
        # Evenly sample IMAGES_PER_LETTER images from the full set
        step    = max(1, len(img_paths) // IMAGES_PER_LETTER)
        sampled = img_paths[::step][:IMAGES_PER_LETTER]
 
        frames   = []
        detected = 0
        missed   = 0
 
        for img_path in sampled:
            lm = extract_landmarks(detector, img_path)
            if lm:
                frames.append(lm)
                detected += 1
            else:
                missed += 1
 
        if len(frames) < 2:
            print(f"  [ skip ]  '{letter}' — only {len(frames)} hands detected "
                  f"({missed} missed). Skipping.")
            skipped.append(letter)
            continue
 
        # Save JSON next to this script (where Hand_avatar.py looks for them)
        out_path = os.path.join(BASE_DIR, f"{letter}.json")
        with open(out_path, "w") as f:
            json.dump(frames, f)
 
        warn = " (low)" if detected < MIN_VALID else ""
        print(f"  [  ok  ]  {letter:8s} — {detected:3d} frames"
              f"{warn}  ({missed} missed)  →  {letter}.json")
        total_saved += 1
 
    detector.close()
 
    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Done!  {total_saved} gesture JSON files saved to:")
    print(f"  {BASE_DIR}")
    if skipped:
        print(f"\n  Skipped ({len(skipped)}): {', '.join(skipped)}")
    print("=" * 55)
    print("\n  You can now launch:  python detect_live.py")
    print("  The 3D avatar will auto-animate each detected gesture.\n")
 
 
if __name__ == "__main__":
    main()
    
    
    
    