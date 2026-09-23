"""
isl_text_to_sign.py
─────────────────────────────────────────────────────────────────
Romanized text -> ordered ISL consonant fingerspelling queue.

This is the ISL counterpart to text_to_sign.py, and it is deliberately a
DIFFERENT kind of operation. text_to_sign.py does English -> ASL gloss:
it drops articles, strips auxiliaries and lemmatizes, because ASL gloss is a
grammar. Nothing here is a grammar.

What this does is TRANSLITERATION: it spells the input sound by sound using
consonant signs. Two honest limitations, both surfaced in the UI rather than
hidden:

  * It is fingerspelling, not translation. Word order and grammar are the
    input's, not ISL's.
  * The sign set is consonants only, so vowels are dropped entirely.
    "namaste" spells NA MA SA TA.

Matching is longest-first so multi-letter romanizations win over their
prefixes - "kha" must not become K + HA, and "chha" must not become CHA + HA.

The match table is DERIVED from the trained label set (isl_labels.json) rather
than hand-written, so trimming a class after evaluation can't leave a stale
mapping pointing at a sign the model no longer knows.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(BASE_DIR, "exported_model", "isl_labels.json")

VOWELS = set("aeiou")

_patterns = None   # [(pattern, label)] sorted longest-first
_labels = None


def _build_patterns(labels):
    """
    Map every romanized spelling we can recognize onto a label.

    Each label contributes two spellings:
      * its full romanization      - "kha" -> KHA
      * its bare consonant cluster - "kh"  -> KHA   (the inherent trailing
        vowel stripped, so "khan" still spells KHA + NA)

    Shorter labels are processed first so that when two labels reduce to the
    same cluster (DA and DAA both -> "d"), the base form wins. The doubled-vowel
    variants encode a vowel-length distinction that romanized input can't
    express anyway, so preferring the base form is the honest default.
    """
    patterns = {}
    for label in sorted(labels, key=len):
        spelling = label.lower()
        patterns.setdefault(spelling, label)
        cluster = spelling.rstrip("a")
        if cluster:
            patterns.setdefault(cluster, label)
    return sorted(patterns.items(), key=lambda kv: -len(kv[0]))


def _get_patterns():
    global _patterns, _labels
    if _patterns is None:
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(
                f"No ISL labels at {LABELS_PATH} - run train_isl_model.py first."
            )
        _labels = json.load(open(LABELS_PATH))
        _patterns = _build_patterns(_labels)
    return _patterns


def word_to_signs(word):
    """One romanized word -> list of consonant labels."""
    patterns = _get_patterns()
    text = word.lower()
    out, i = [], 0

    while i < len(text):
        if text[i] in VOWELS:        # vowels have no sign in this set
            i += 1
            continue

        for pattern, label in patterns:      # longest-first
            if text.startswith(pattern, i):
                out.append(label)
                i += len(pattern)
                break
        else:
            i += 1                   # unspellable character (digit, punctuation)

    return out


def translate(text):
    """
    Same return shape as text_to_sign.translate() so the frontend renders an
    ISL queue with no changes: {"simplified", "gloss", "queue"}.
    """
    gloss, queue = [], []
    for word in text.split():
        for label in word_to_signs(word):
            gloss.append(label)
            queue.append({"label": label, "kind": "letter", "word": word})
    return {"simplified": text, "gloss": gloss, "queue": queue}


if __name__ == "__main__":
    # Self-check on a fixed label set, so it runs before a model is trained.
    demo = ["KA", "KHA", "GA", "CHA", "CHHA", "JA", "TA", "THA", "DA", "DAA",
            "NA", "PA", "BA", "MA", "YA", "RA", "LA", "SA", "SHA", "HA", "KSHA"]
    _labels = demo
    _patterns = _build_patterns(demo)

    # Longest-match precedence: a multi-letter romanization must not be split.
    assert word_to_signs("kha") == ["KHA"], word_to_signs("kha")
    assert word_to_signs("chha") == ["CHHA"], word_to_signs("chha")
    assert word_to_signs("ksha") == ["KSHA"], word_to_signs("ksha")

    # Vowels carry no sign and must not block the consonant after them.
    assert word_to_signs("namaste") == ["NA", "MA", "SA", "TA"], word_to_signs("namaste")
    assert word_to_signs("aaa") == []

    # Bare clusters still resolve when no inherent vowel follows.
    assert word_to_signs("khan") == ["KHA", "NA"], word_to_signs("khan")

    # Collision tie-break: DA and DAA both reduce to "d", base form wins.
    assert word_to_signs("dil") == ["DA", "LA"], word_to_signs("dil")

    # Unspellable characters are skipped, not guessed at.
    assert word_to_signs("k9!a") == ["KA"], word_to_signs("k9!a")

    assert translate("ka kha")["gloss"] == ["KA", "KHA"]
    assert translate("ka")["queue"] == [{"label": "KA", "kind": "letter", "word": "ka"}]

    print("isl_text_to_sign self-check passed")
