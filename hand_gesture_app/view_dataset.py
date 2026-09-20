import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Root folder to scan recursively for .MOV files
DATASET_ROOT = "dataset"

OUTPUT_PATH = "dataset_overview.png"


def extract_label(filename):
    """
    Turns 'S1_DHA.MOV' -> 'DHA'
    Adjust this if your filenames follow a different pattern.
    """
    name = os.path.splitext(filename)[0]
    match = re.search(r"S\d+_(.+)", name)
    return match.group(1) if match else name


def get_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    middle = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
    success, frame = cap.read()
    cap.release()

    if not success:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    video_paths = []
    for root, _, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.lower().endswith(".mov"):
                video_paths.append(os.path.join(root, f))

    print(f"Found {len(video_paths)} video files.")

    # Keep only ONE video per unique label, so each sign shows once
    seen_labels = {}
    for path in sorted(video_paths):
        label = extract_label(os.path.basename(path))
        if label not in seen_labels:
            seen_labels[label] = path

    labels_sorted = sorted(seen_labels.keys())
    print(f"Unique signs found: {len(labels_sorted)}")

    # Figure out a roughly square grid layout
    n = len(labels_sorted)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(-1)  # flatten in case of 1 row/col

    for i, label in enumerate(labels_sorted):
        frame = get_middle_frame(seen_labels[label])
        ax = axes[i]
        if frame is not None:
            ax.imshow(frame)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    # Hide any unused grid cells
    for j in range(len(labels_sorted), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()
    print(f"Saved overview to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()