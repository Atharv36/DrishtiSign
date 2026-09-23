"""
train.py
--------
Training pipeline for the NSL Consonant video dataset.

Dataset structure:

dataset/
├── S1_NSL_Consonant_Bright/
│   ├── S1_WA.MOV
│   ├── S1_DHA.MOV
│   ├── S1_BA.MOV
│   └── ...
├── S2_NSL_Consonant_Bright/
├── S3_NSL_Consonant_Prepared/
└── ...

Each .MOV filename contains the sign label.

Example:
    S1_BA.MOV       -> BA
    S1_DHA.MOV      -> DHA
    S1_D_SHA.MOV    -> D_SHA

Pipeline:

    MOV videos
        ↓
    video-level train/test split
        ↓
    sample frames from videos
        ↓
    MediaPipe hand detection + crop  <-- NEW: matches live webcam framing
        ↓
    resize to 128x128
        ↓
    SVD compression
        ↓
    PyTorch MYNN
        ↓
    nepali_char_model.pth

Run (must be in the mediapipe-enabled environment):
    source venv_mp/bin/activate
    python train.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from model import MYNN


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 128

LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 32

MODEL_SAVE_PATH = "nepali_char_model.pth"
CLASS_NAMES_PATH = "class_names.txt"

# Back to the raw video folders -- this script extracts + crops frames
# from the .MOV files directly, it does not read from dataset_frames.
DATASET_PATH = "dataset"

# Number of frames sampled from EACH video.
# Start with 10 to keep training manageable.
FRAMES_PER_VIDEO = 10

# SVD rank.
SVD_RANK = 25

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Must match livepredict.py's CROP_PADDING so training crops look like
# the crops the model will see live.
CROP_PADDING = 0.25


# ============================================================
# MEDIAPIPE HAND DETECTION (for cropping training frames)
# ============================================================

mp_hands = mp.solutions.hands

hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)


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


# ============================================================
# SVD
# ============================================================

def svd_compress_image(img_rgb, rank=SVD_RANK):
    """
    Compress an RGB image using truncated SVD.

    Input:
        H x W x 3 image

    Output:
        List containing SVD factors for each RGB channel.
    """

    factors = []

    for c in range(3):

        channel = img_rgb[:, :, c].astype(np.float32)

        U, S, Vt = np.linalg.svd(
            channel,
            full_matrices=False
        )

        actual_rank = min(rank, len(S))

        factors.append(
            (
                U[:, :actual_rank].copy(),
                S[:actual_rank].copy(),
                Vt[:actual_rank, :].copy()
            )
        )

    return factors


def svd_reconstruct_image(factors):
    """
    Reconstruct an RGB image from SVD factors.
    """

    channels = []

    for U_k, S_k, Vt_k in factors:

        channel = (U_k * S_k) @ Vt_k

        channel = np.clip(
            channel,
            0,
            255
        )

        channels.append(channel)

    return np.stack(
        channels,
        axis=-1
    ).astype(np.float32)


# ============================================================
# DATASET CLASS
# ============================================================

class CustomDataset(Dataset):

    def __init__(self, compressed_factors, labels):

        self.compressed_factors = compressed_factors

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.compressed_factors)

    def __getitem__(self, idx):

        img = svd_reconstruct_image(
            self.compressed_factors[idx]
        )

        # Normalize 0-255 -> 0-1
        img = img / 255.0

        tensor = torch.tensor(
            img,
            dtype=torch.float32
        )

        return tensor, self.labels[idx]


# ============================================================
# EXTRACT LABEL FROM VIDEO NAME
# ============================================================

def get_label_from_filename(filename):
    """
    Convert:

        S1_BA.MOV       -> BA
        S1_DHA.MOV      -> DHA
        S1_D_SHA.MOV    -> D_SHA

    The first part (S1, S2, etc.) represents the subject
    and is removed.
    """

    name = os.path.splitext(
        os.path.basename(filename)
    )[0]

    parts = name.split("_", 1)

    if len(parts) == 2:

        label = parts[1]

    else:

        label = name

    return label


# ============================================================
# FIND ALL VIDEOS
# ============================================================

def build_dataframe():

    data = []

    print("\nSearching for NSL videos...\n")

    for root, dirs, files in os.walk(DATASET_PATH):

        for filename in files:

            if filename.lower().endswith(
                (".mov", ".mp4", ".avi", ".mkv")
            ):

                video_path = os.path.join(
                    root,
                    filename
                )

                label = get_label_from_filename(
                    filename
                )

                data.append(
                    [
                        video_path,
                        label
                    ]
                )

    df = pd.DataFrame(
        data,
        columns=[
            "video_path",
            "label"
        ]
    )

    print("Total videos:", len(df))

    print("\nFirst few videos:")
    print(df.head())

    print("\nNumber of classes:")
    print(df["label"].nunique())

    print("\nClasses:")
    print(sorted(df["label"].unique()))

    return df


# ============================================================
# SAMPLE FRAMES FROM ONE VIDEO (now with MediaPipe hand cropping)
# ============================================================

def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            f"WARNING: Could not open video: {video_path}"
        )

        return []

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        cap.release()

        return []

    # Choose evenly spaced frames.
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        min(num_frames, total_frames),
        dtype=int
    )

    frames = []

    for frame_index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = cap.read()

        if not success:
            continue

        # BGR -> RGB
        rgb_full = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect the hand in the FULL frame first, so the crop matches
        # the same style livepredict.py uses on the live webcam feed.
        results = hands_detector.process(rgb_full)

        if not results.multi_hand_landmarks:
            # No hand detected in this frame -- skip it rather than
            # training on a frame with no hand in it.
            continue

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = get_hand_bbox(
            results.multi_hand_landmarks[0], w, h
        )

        if x2 <= x1 or y2 <= y1:
            continue

        crop = rgb_full[y1:y2, x1:x2]

        # Resize the CROP (not the full frame) to IMG_SIZE
        crop = cv2.resize(
            crop,
            (IMG_SIZE, IMG_SIZE)
        )

        frames.append(crop)

    cap.release()

    return frames


# ============================================================
# PROCESS VIDEOS
# ============================================================

def load_videos(video_paths, labels, rank=SVD_RANK):

    compressed = []
    processed_labels = []

    total_videos = len(video_paths)

    for video_number, (video_path, label) in enumerate(
        zip(video_paths, labels),
        start=1
    ):

        print(
            f"\rProcessing video "
            f"{video_number}/{total_videos}",
            end=""
        )

        frames = extract_frames(
            video_path,
            FRAMES_PER_VIDEO
        )

        if len(frames) == 0:

            continue

        for frame in frames:

            factors = svd_compress_image(
                frame,
                rank=rank
            )

            compressed.append(
                factors
            )

            processed_labels.append(
                label
            )

    print()

    return compressed, processed_labels


# ============================================================
# SAVE SAMPLE FRAME
# ============================================================

def save_sample_frame(df):

    if len(df) == 0:
        return

    sample = df.iloc[0]

    frames = extract_frames(
        sample["video_path"],
        num_frames=1
    )

    if len(frames) == 0:
        return

    frame = frames[0]

    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(frame)

    plt.axis("off")

    plt.title(
        sample["label"]
    )

    plt.savefig(
        "sample_frame.png",
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved sample frame to sample_frame.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NSL CONSONANT TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Find videos
    # --------------------------------------------------------

    df = build_dataframe()

    if len(df) == 0:

        print(
            "\nERROR: No video files found."
        )

        print(
            f"Check that your dataset exists at: {DATASET_PATH}"
        )

        return

    # --------------------------------------------------------
    # 2. Save sample frame
    # --------------------------------------------------------

    save_sample_frame(df)

    # --------------------------------------------------------
    # 3. Show class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    print(counts)

    # --------------------------------------------------------
    # 4. Remove classes with fewer than 2 VIDEOS
    # --------------------------------------------------------

    rare_classes = counts[
        counts < 2
    ]

    if len(rare_classes) > 0:

        print(
            "\nClasses with fewer than 2 videos:"
        )

        print(rare_classes)

        valid_labels = counts[
            counts >= 2
        ].index

        df = df[
            df["label"].isin(valid_labels)
        ].reset_index(drop=True)

    print(
        "\nVideos after filtering:",
        len(df)
    )

    # --------------------------------------------------------
    # 5. Video-level train/test split
    # --------------------------------------------------------

    X = df["video_path"]
    y = df["label"]

    print("\nSplitting videos into train/test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        "Training videos:",
        len(X_train)
    )

    print(
        "Testing videos:",
        len(X_test)
    )

    # --------------------------------------------------------
    # 6. Label encoding
    # --------------------------------------------------------

    le = LabelEncoder()

    le.fit(y)

    y_train_encoded = le.transform(
        y_train
    )

    y_test_encoded = le.transform(
        y_test
    )

    num_classes = len(
        le.classes_
    )

    print(
        "\nNumber of classes:",
        num_classes
    )

    print(
        "Classes:",
        le.classes_
    )

    # --------------------------------------------------------
    # 7. Process training videos
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROCESSING TRAINING VIDEOS")
    print("=" * 60)

    X_train_compressed, y_train_frames = load_videos(
        X_train.tolist(),
        y_train.tolist(),
        rank=SVD_RANK
    )

    # --------------------------------------------------------
    # 8. Process testing videos
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROCESSING TEST VIDEOS")
    print("=" * 60)

    X_test_compressed, y_test_frames = load_videos(
        X_test.tolist(),
        y_test.tolist(),
        rank=SVD_RANK
    )

    # Encode frame labels
    y_train_frames_encoded = le.transform(
        y_train_frames
    )

    y_test_frames_encoded = le.transform(
        y_test_frames
    )

    print(
        "\nTrain frames:",
        len(X_train_compressed)
    )

    print(
        "Test frames:",
        len(X_test_compressed)
    )

    # --------------------------------------------------------
    # 9. Create PyTorch datasets
    # --------------------------------------------------------

    train_dataset = CustomDataset(
        X_train_compressed,
        y_train_frames_encoded
    )

    test_dataset = CustomDataset(
        X_test_compressed,
        y_test_frames_encoded
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # 10. Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nTraining on:",
        device
    )

    # --------------------------------------------------------
    # 11. Create YOUR existing model
    # --------------------------------------------------------

    model = MYNN(
        num_classes
    ).to(device)

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # 12. Loss + optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # 13. Training
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):

        model.train()

        total_epoch_loss = 0.0

        correct = 0
        total = 0

        for batch_features, batch_labels in train_loader:

            # H,W,C -> C,H,W
            batch_features = (
                batch_features
                .permute(0, 3, 1, 2)
                .to(device)
            )

            batch_labels = (
                batch_labels
                .to(device)
            )

            # Clear gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(
                batch_features
            )

            # Loss
            loss = criterion(
                outputs,
                batch_labels
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            total_epoch_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += batch_labels.size(0)

            correct += (
                predicted == batch_labels
            ).sum().item()

        avg_loss = (
            total_epoch_loss /
            len(train_loader)
        )

        train_accuracy = (
            100.0 * correct / total
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        test_correct = 0
        test_total = 0

        with torch.no_grad():

            for batch_features, batch_labels in test_loader:

                batch_features = (
                    batch_features
                    .permute(0, 3, 1, 2)
                    .to(device)
                )

                batch_labels = (
                    batch_labels
                    .to(device)
                )

                outputs = model(
                    batch_features
                )

                _, predicted = torch.max(
                    outputs,
                    1
                )

                test_total += (
                    batch_labels.size(0)
                )

                test_correct += (
                    predicted == batch_labels
                ).sum().item()

        test_accuracy = (
            100.0 * test_correct / test_total
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Test Acc: {test_accuracy:.2f}%"
        )

    # --------------------------------------------------------
    # 14. Save model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        MODEL_SAVE_PATH
    )

    print(
        f"\nSaved trained model to: "
        f"{MODEL_SAVE_PATH}"
    )

    # --------------------------------------------------------
    # 15. Save class names
    # --------------------------------------------------------

    with open(
        CLASS_NAMES_PATH,
        "w"
    ) as f:

        for cls in le.classes_:

            f.write(
                cls + "\n"
            )

    print(
        f"Saved class names to: "
        f"{CLASS_NAMES_PATH}"
    )

    # --------------------------------------------------------
    # 16. Save class distribution graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 5)
    )

    counts.plot(
        kind="bar"
    )

    plt.xlabel(
        "Gesture"
    )

    plt.ylabel(
        "Number of videos"
    )

    plt.title(
        "NSL Consonant Dataset Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        "class_distribution.png"
    )

    plt.close()

    print(
        "Saved class distribution to: "
        "class_distribution.png"
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()