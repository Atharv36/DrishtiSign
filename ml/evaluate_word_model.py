"""
evaluate_word_model.py
─────────────────────────────────────────────────────────────────
Honest evaluation for the word model, plus a vocabulary-size sweep.

Why this exists: with ~300 clips across 35 words, a single train/val/test
split puts only ~1 clip per word in the test set. One unlucky prediction
swings a word from 100% to 0%, so that number is extremely noisy and not
quotable. K-fold cross-validation trains k times on different splits and
reports mean +/- std - every clip gets used for testing exactly once, which
is the right method at this data size.

It also sweeps VOCABULARY SIZE. Fewer classes trained on the same data is
usually a large, free accuracy gain - this measures that tradeoff instead of
guessing at it, so the scoping decision is evidence-based.

Run:  python evaluate_word_model.py
"""

import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train_word_model import WordSignGRU, SeqDataset, CACHE_PATH, EXPORT_DIR

K_FOLDS    = 5
EPOCHS     = 60
BATCH_SIZE = 16
LR         = 1e-3

# Keep only words with at least this many clips. Each value is one experiment.
MIN_CLIP_THRESHOLDS = [1, 7, 10]


def load_cache():
    if not os.path.exists(CACHE_PATH):
        raise SystemExit(f"No cache at {CACHE_PATH} - run train_word_model.py first.")
    d = np.load(CACHE_PATH, allow_pickle=True)
    return d["X"], d["y"], list(d["labels"])


def filter_by_min_clips(X, y, labels, min_clips):
    """Keep only classes with >= min_clips examples, re-indexed contiguously."""
    counts = Counter(y.tolist())
    keep = sorted(c for c, n in counts.items() if n >= min_clips)
    remap = {old: new for new, old in enumerate(keep)}
    mask = np.array([c in remap for c in y])
    return (X[mask],
            np.array([remap[c] for c in y[mask]], dtype=np.int64),
            [labels[c] for c in keep])


def run_fold(Xtr, ytr, Xte, yte, n_classes, device):
    model = WordSignGRU(n_classes).to(device)
    counts = Counter(ytr.tolist())
    w = torch.tensor([len(ytr) / (n_classes * counts.get(c, 1)) for c in range(n_classes)],
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    dl = DataLoader(SeqDataset(Xtr, ytr, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, yb in DataLoader(SeqDataset(Xte, yte), batch_size=BATCH_SIZE):
            preds += model(xb.to(device)).argmax(1).cpu().tolist()
    preds = np.array(preds)
    return float((preds == yte).mean()), preds


def cross_validate(X, y, labels, device):
    """Stratified k-fold: every clip is tested exactly once."""
    n_classes = len(labels)
    rng = np.random.RandomState(42)

    # Assign folds within each class so rare words appear in every fold.
    folds = np.zeros(len(y), dtype=int)
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % K_FOLDS

    accs, all_true, all_pred = [], [], []
    for k in range(K_FOLDS):
        te, tr = folds == k, folds != k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        acc, preds = run_fold(X[tr], y[tr], X[te], y[te], n_classes, device)
        accs.append(acc)
        all_true += y[te].tolist()
        all_pred += preds.tolist()
        print(f"    fold {k+1}/{K_FOLDS}: {acc:.1%}")

    return np.array(accs), np.array(all_true), np.array(all_pred)


def main():
    X, y, labels = load_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded {len(X)} clips, {len(labels)} words. Device: {device}\n")

    results, best = {}, None

    for min_clips in MIN_CLIP_THRESHOLDS:
        Xf, yf, lf = filter_by_min_clips(X, y, labels, min_clips)
        if len(lf) < 2:
            continue
        chance = 1.0 / len(lf)
        print(f"── min {min_clips} clips/word → {len(lf)} words, {len(Xf)} clips "
              f"(chance {chance:.1%})")

        accs, true, pred = cross_validate(Xf, yf, lf, device)
        mean, std = accs.mean(), accs.std()
        print(f"    {K_FOLDS}-fold accuracy: {mean:.1%} +/- {std:.1%}  "
              f"({mean/chance:.1f}x chance)\n")

        results[min_clips] = {"words": len(lf), "clips": len(Xf),
                              "accuracy_mean": float(mean), "accuracy_std": float(std),
                              "chance": chance, "vocabulary": lf}
        if best is None or mean > results[best]["accuracy_mean"]:
            best = min_clips
            best_pair = (true, pred, lf)

    if best is not None:
        print("=" * 58)
        r = results[best]
        print(f"Best setup: min {best} clips/word -> {r['words']} words, "
              f"{r['accuracy_mean']:.1%} +/- {r['accuracy_std']:.1%}")

        true, pred, lf = best_pair
        conf = Counter((lf[t], lf[p]) for t, p in zip(true, pred) if t != p)
        if conf:
            print("\nTop confusions across all folds (true -> predicted):")
            for (t, p), n in conf.most_common(10):
                print(f"  {t:12s} -> {p:12s} x{n}")

        os.makedirs(EXPORT_DIR, exist_ok=True)
        out = os.path.join(EXPORT_DIR, "word_crossval.json")
        json.dump({"k_folds": K_FOLDS, "results": results, "best_threshold": best},
                  open(out, "w"), indent=2)
        print(f"\nSaved -> {out}")
        print("\nQuote the k-fold mean +/- std in your report - it's far more "
              "stable than a single split at this data size.")


if __name__ == "__main__":
    main()
