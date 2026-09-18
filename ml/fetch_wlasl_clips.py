"""
fetch_wlasl_clips.py
─────────────────────────────────────────────────────────────────
Downloads WLASL word-sign clips into ml/word_clips/{word}/, the layout
train_word_model.py expects - so the word model can be trained from a real
published dataset instead of only your own recordings.

Source: the Hugging Face mirror (Voxel51/WLASL), which hosts the actual
video files. That matters: the original WLASL distribution is a list of
URLs to third-party sites and many have since rotted, whereas the mirror
serves the videos directly.

This downloads SELECTIVELY - only the glosses in VOCABULARY (a few hundred
clips, tens of MB) rather than the full 11,980-video / 5.5 GB dataset.
Re-running skips files already fetched, so it's safe to interrupt.

WLASL is released for academic / non-commercial use - cite it:
  Li et al., "Word-level Deep Sign Language Recognition from Video", WACV 2020

Run:  python fetch_wlasl_clips.py
"""

import json
import os
import urllib.request
from collections import defaultdict

from word_sequences import VOCABULARY

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "word_clips")

HF_REPO    = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/main"
SAMPLES_JSON = f"{HF_REPO}/samples.json"
CACHE_JSON = os.path.join(BASE_DIR, "wlasl_samples.json")

# Our vocabulary names -> the gloss string WLASL actually uses.
GLOSS_ALIASES = {
    "thankyou": "thank you",
}

MIN_CLIPS_WARN = 10   # below this, a word is thin training data


def load_samples():
    """WLASL label index: which video file corresponds to which gloss."""
    if not os.path.exists(CACHE_JSON):
        print("Downloading WLASL label index (~9 MB)...")
        urllib.request.urlretrieve(SAMPLES_JSON, CACHE_JSON)
    with open(CACHE_JSON) as f:
        return json.load(f)["samples"]


def main():
    samples = load_samples()

    by_gloss = defaultdict(list)
    for s in samples:
        by_gloss[s["gloss"]["label"]].append(s["filepath"])

    print(f"WLASL index: {len(by_gloss)} glosses, {len(samples)} clips total\n")

    wanted = {w: GLOSS_ALIASES.get(w, w) for w in VOCABULARY}
    missing = [w for w, g in wanted.items() if g not in by_gloss]
    if missing:
        print(f"Not in WLASL (record these yourself): {', '.join(missing)}\n")

    total_new = total_have = 0
    thin = []

    for word, gloss in wanted.items():
        paths = by_gloss.get(gloss, [])
        if not paths:
            continue

        folder = os.path.join(CLIPS_DIR, word)
        os.makedirs(folder, exist_ok=True)

        new = have = 0
        for p in paths:
            dest = os.path.join(folder, f"wlasl_{os.path.basename(p)}")
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                have += 1
                continue
            try:
                urllib.request.urlretrieve(f"{HF_REPO}/{p}", dest)
                new += 1
            except Exception as e:
                # A single bad file shouldn't abort a long download.
                print(f"    ! failed {p}: {e}")
                if os.path.exists(dest):
                    os.remove(dest)

        got = new + have
        total_new += new
        total_have += have
        flag = ""
        if got < MIN_CLIPS_WARN:
            thin.append((word, got))
            flag = "   <- thin, consider recording extra"
        print(f"  {word:12s} {got:3d} clips  ({new} new){flag}")

    print(f"\nDownloaded {total_new} new clips ({total_have} already present).")
    print(f"Saved to {CLIPS_DIR}")

    if thin:
        print(f"\n{len(thin)} words have fewer than {MIN_CLIPS_WARN} clips. WLASL is broad "
              "but shallow - a few clips per sign.")
        print("Top up the thin ones with:  python record_word_clips.py")
        print("Your own clips also match your camera and signing style, which "
              "usually helps more at inference than extra dataset clips.")

    print("\nThen train with:  python train_word_model.py")


if __name__ == "__main__":
    main()
