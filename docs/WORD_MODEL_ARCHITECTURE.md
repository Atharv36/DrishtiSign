# Word-Sign Model — Architecture Plan

Plan for adding **whole-word sign support** to DrishtiSign, so common words are
one sign instead of being fingerspelled letter-by-letter.

Status: **design only — not implemented.** Written to be picked up in a later
session.

---

## 1. Why this is a separate model from the letter model

The existing letter model is a single-frame MLP: 21 hand landmarks → one of
A–Z. That works because a fingerspelled letter **is** a static handshape.

A word sign is not. ASL words are *movement + location*: "Hello" is a flat hand
travelling outward from the temple. This is a different problem type, with a
different input shape:

| | Input | Architecture |
|---|---|---|
| **Letter model** (exists) | 1 frame → 88 features | MLP |
| **Word model** (planned) | 32 frames × 88 features | Bi-GRU |

One model cannot represent both. This is not a preference — it's the same wall
that made "Hello" impossible to show as a static pose, and why **J and Z had to
be dropped** from the alphabet (both are motion letters).

Secondary benefits of keeping them separate:

- **No label collision.** A frame captured mid-"Hello" is a flat hand —
  indistinguishable from "B" to a static classifier. Separate label spaces
  remove the conflict entirely.
- **Independent retraining.** The word model can be rebuilt without risking
  regression in the letter model that already works.
- **J and Z come back**, handled by the sequence model instead of excluded.

---

## 2. The cost asymmetry that should drive scoping

Text→Sign and Sign→Text need very different things. Scope them differently:

| | Needs | Cost | Target |
|---|---|---|---|
| **Text→Sign** (display a word) | One reference clip per word. No training. | Cheap | Large vocabulary (100+) |
| **Sign→Text** (recognise a word) | Trained temporal model + training data | Expensive | Small, reliable set (~20–35) |

**Out-of-vocabulary words fall back to fingerspelling** — and that is
linguistically authentic, not a shortcut. Real signers fingerspell names,
places and technical terms. Document it as a design decision.

---

## 3. Data

**Layout** (source-agnostic — own recordings or a public dataset):

```
ml/word_clips/
    hello/     clip1.mp4 clip2.mp4 ...
    please/    ...
    thankyou/  ...
```

**Sources — use both, they're complementary:**

1. **WLASL** via `fetch_wlasl_clips.py` — **verified working**. The Hugging Face
   mirror (`Voxel51/WLASL`) hosts the actual video files, so the link-rot
   problem in the original GitHub distribution doesn't apply. The script
   downloads **selectively** (only vocabulary words — tens of MB, not the full
   5.5 GB) and is resumable. Academic/non-commercial — cite Li et al., WACV 2020.
   - **34 of our 35 vocabulary words are present** (only `thankyou` needed an
     alias — WLASL spells it `"thank you"`).
   - ⚠️ **WLASL is broad but shallow: 4–15 clips per word**, averaging ~9. That
     is below the 10+/word rule of thumb for several words.
2. **Record your own** via `record_word_clips.py` — to top up thin words. Your
   own clips also match your camera and signing style, which typically helps
   more at inference than extra dataset clips, and they double as Text→Sign
   reference media.

**Stay in one language.** The alphabet is currently **ASL**, so word signs
should be ASL too. ISL alternatives (INCLUDE / CISLR / ISL-CSLTR) exist if the
whole project switches — but don't mix.

**Vocabulary — keep it small and frequent on purpose.** A compact reliable set
beats a large flaky one. Starting list (~35):

> hello, bye, please, thankyou, sorry, yes, no, help, want, need, good, bad,
> love, friend, family, eat, drink, water, home, work, school, learn,
> understand, what, where, who, how, more, stop, go, happy, sad, name, you, me

### 3a. Vocabulary trim: 35 → 20 (evidence-based, not guessed)

The 35-word model scored 71.4% ± 4.7% (5-fold). Two problems in that set were
diagnosed and fixed rather than guessed at:

**Two-handed signs lose information with `num_hands=1` extraction.** A
genuinely bimanual sign only has its dominant hand captured — exactly the
information that would disambiguate it is thrown away. This was verified
against OUR OWN clips, not assumed from memory of ASL: `audit_two_handed.py`
runs `num_hands=2` detection over each word's clips and measures how often two
hands are detected clearly separated (≥0.15 normalized distance) in the same
frame.

```bash
python audit_two_handed.py     # writes exported_model/two_hand_audit.json
```

Confirmed two-handed (≥30% of frames): `family, friend, go, happy, how,
learn, more, name, sad, want, what`. Also excluded as borderline (25–29%,
real ASL confirms these are two-handed — ​the low measured ratio is likely
clip framing/occlusion): `help, school, stop`.

Cross-referencing against the confusion pairs from every evaluation run
showed **most recurring confusions were exactly these two-handed words**
(`how↔help`, `want↔family`, `sad↔family/learn`, `go↔drink`, `what↔good/work`)
— strong evidence the confusion was information loss, not the model being
weak. Their clip folders stay on disk for when two-hand extraction is added
(§4, two-handed note).

**One genuine sign-similarity collision remained after that:** `good` ↔
`thankyou` recurred in every run and both are, in real ASL, a flat hand near
the chin moving forward/down — a well-documented beginner minimal pair, not a
data artifact. `good` was dropped, keeping `thankyou`.

**Result: 20 words, all confirmed one-handed, 85.1% ± 5.6% (5-fold, 17×
chance)** — up from 71.4% ± 4.7% on the full 35. `word_sequences.py` keeps the
excluded lists (`EXCLUDED_TWO_HANDED`, `EXCLUDED_SIMILAR`) so the reasoning
travels with the code.

**Reference check:** a similar public project
([Sign-Language-To-Text-Conversion](https://github.com/emnikhil/Sign-Language-To-Text-Conversion))
independently documents confusable *letter* clusters (D/R/U, T/K/D/I, S/M/N)
from the same kind of single-hand landmark approach — external confirmation
that landmark-based confusability is a real, expected phenomenon worth
auditing for, not specific to this project.

---

## 4. Preprocessing: clip → fixed-size tensor

```
video ──► MediaPipe per frame ──► feature_utils.extract_features (88-dim)
      ──► resample to 32 frames ──► (32, 88) float32
```

Two deliberate choices:

- **Reuse `feature_utils.extract_features`** (the letter model's 88-dim
  representation: normalized coords + joint angles + fingertip distances).
  The angles are rotation-invariant, which matters even more for motion.
  Reusing it also means one shared, already-tested feature definition.
- **Resample to a fixed 32 frames** rather than padding. Signers move at
  different speeds; time-normalising makes the model learn the *shape* of the
  motion, not how fast it was performed.

Cache the extracted array (`.npz`) so retraining doesn't re-decode every video.

### ⚠️ Two-handed signs — decided for v1, revisit later

Many ASL word signs use **both hands**, but the current extraction is
`num_hands=1`. **v1 ships option A** (below) and scopes the vocabulary to
one-handed signs only (§3a) rather than accepting degraded accuracy on
two-handed ones. `EXCLUDED_TWO_HANDED` in `word_sequences.py` lists 14 words
ready to reintroduce once option B lands - their clip folders are still on
disk in `word_clips/`.

- **A (shipped):** dominant hand only — 88 features/frame. Loses two-handed
  signs entirely, but what it does recognize, it recognizes reliably
  (§3a: dropping them raised accuracy 71.4% → 85.1%).
- **B (upgrade path):** both hands — 176 features/frame, zero-filled when the
  second hand is absent, with a consistent left/right ordering. Re-run
  `audit_two_handed.py` afterward to confirm accuracy on the reintroduced
  words before trusting them live.

Recommend starting with **A**, designing the feature function so B is a drop-in
change.

---

## 5. Model

Bidirectional GRU over the clip:

```
(32, 88) → LayerNorm → GRU(88→128, 2 layers, bidirectional, dropout 0.3)
         → [final fwd state ‖ final bwd state ‖ mean over time]   (512)
         → Linear(512→128) → ReLU → Dropout → Linear(128→C)
```

- **Bidirectional**: a sign's meaning depends on the whole movement — where it
  ends matters as much as where it starts.
- **Pooling combines final hidden states *and* the mean.** Mean-pooling alone
  is **direction-blind** — a movement outward and the same movement inward
  average to the same vector, and in sign language that difference *is* the
  meaning. The final forward/backward states encode where the motion ended up;
  the mean adds robustness against dead frames at the clip edges. This was
  caught by a synthetic test that specifically pits `move_out` against
  `move_in` — worth keeping as a regression check.
- **Class-weighted loss + label smoothing**, matching the letter trainer.
- **Augmentation**: landmark noise + small time-shift jitter. Same rationale as
  the letter model — real webcam input is noisier and less consistent than
  curated clips.

Export: `exported_model/word_model.pth`, `word_labels.json`, `word_config.json`
(seq_len, feature_dim, num_classes) — so the server can self-configure, the way
it already auto-detects the letter model's input size.

---

## 6. Inference & routing

Two models means deciding which one runs.

- **v1 — explicit UI mode.** The frontend already has separate modals.
  Fingerspelling mode → letter model; word mode → word model. Zero ambiguity,
  nothing to misfire in a demo. **Ship this.**
- **v2 — motion-gated routing.** Hand held still → letter model; hand moving →
  buffer the last ~32 frames and run the word model. The stability signal
  already exists (the hold-to-commit logic in Sign→Text computes it), so this
  is mostly wiring.

Server side: keep a **rolling frame buffer** per session (same per-`sid` pattern
as `TextSession`), run the word model on the buffer, and commit a word when
confidence stays above threshold for a few consecutive predictions — mirroring
the existing dwell logic so behaviour feels consistent.

**Keep the letter path untouched** so it cannot regress.

---

## 7. Text→Sign side (no model needed)

Serve a **reference clip per word** (short looping mp4/gif) from the frontend.
This needs no ML at all — the vocabulary here can be much larger than what the
recogniser supports. It also removes the server-side avatar rendering from the
playback path entirely.

---

## 8. Implementation order

1. **Recorder** (`record_word_clips.py`) — webcam capture into `word_clips/`.
   Unblocks everything with no download dependency, and the clips double as
   Text→Sign reference media.
2. **Extraction + cache** — clips → `(N, 32, 88)` arrays.
3. **Train** (`train_word_model.py`) — Bi-GRU, export artifacts.
4. **Server inference** — load word model beside the letter model, rolling
   buffer, new socket events.
5. **Frontend word mode** — mode toggle + display.
6. *(optional)* WLASL ingestion to widen the vocabulary.
7. *(optional)* Restore **J and Z** as motion signs via this model.

---

## 9. Success criteria

- ≥10 clips per word before training is meaningful.
- Report **held-out test accuracy + a confusion matrix** — needed for the
  report regardless, and it shows which signs collide.
- Live behaviour matters more than val accuracy: verify on webcam, since
  training clips are always cleaner than real input.

---

## 10. What's built so far

| File | Status |
|---|---|
| `ml/word_sequences.py` | **Built & tested** — shared config (SEQ_LEN, FEATURE_DIM, VOCABULARY), resampling, video→sequence |
| `ml/record_word_clips.py` | **Built** — webcam recorder with live hand-detection feedback (needs a camera to exercise) |
| `ml/train_word_model.py` | **Built & tested** — Bi-GRU trainer, class weighting, augmentation, early stopping, export |

Verified by smoke test (not just syntax):
- variable-length clips (5–200 frames) all resample to `(32, 88)`
- model forward pass `(B, 32, 88) → (B, C)`, ~500k params
- **full training loop learns**: 100% val accuracy on synthetic data, including
  separating `move_out` from `move_in` (the direction test the original
  mean-pooling design failed)

Not yet done: steps 4–6 below (server inference, routing, frontend word mode).

### Next step for whoever picks this up

```bash
cd ml && source venv/bin/activate
python record_word_clips.py     # record 10+ clips per word
python train_word_model.py      # trains and exports word_model.pth
```

Then wire inference into `server.py` alongside the letter model (§6).
