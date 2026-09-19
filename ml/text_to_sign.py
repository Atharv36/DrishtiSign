"""
text_to_sign.py
─────────────────────────────────────────────────────────────────
English text -> ASL-style gloss -> ordered sign queue for the avatar.

Pipeline:
    text
      │  (optional) local LLM pre-pass via Ollama — paraphrases long/
      │  messy sentences into short, plain ones before gloss conversion.
      │  Off by default; the pipeline works fully without it.
      ▼
    simplified text
      │  rule-based gloss conversion (spaCy POS tagging):
      │    - drop articles (a/an/the)
      │    - drop auxiliary "to be" / helper verbs (am/is/are/was/do/have...)
      │    - lemmatize remaining words to base form (running -> run)
      ▼
    gloss tokens, e.g. ["I", "GO", "STORE"]
      │  sign lookup per token:
      │    - known word sign (Hello, Thankyou, ...) -> one whole-word sign
      │    - otherwise -> fingerspell letter by letter
      ▼
    sign queue: [{"label": "I", "kind": "letter", "word": "I"}, ...]

The queue is what server.py feeds to HandAvatar3D.load_gesture() in
sequence. Depends only on spaCy + (optionally) a local Ollama server —
no new ML model is trained here.
"""

import os

import spacy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Words whose meaning is purely grammatical in English but carries no
# separate ASL sign - dropped from the gloss. Kept as an explicit rule
# set (not just spaCy's stopword list) so the behaviour is easy to explain
# and tune: spaCy's default stopwords are broader than what we want to
# drop (e.g. it would also strip "not", which we need to keep).
DROP_POS = {"DET"}                       # articles: a, an, the
AUX_LEMMAS = {"be", "do", "have", "will", "shall", "would", "could"}
# Individual words dropped regardless of POS - the infinitive/directional
# "to" ("going TO the store", "want TO go") isn't signed on its own in
# basic ASL gloss.
DROP_WORDS = {"to"}

# The 8 word signs we have real avatar pose data for (see
# generate_word_poses.py), in their pose-file casing (Hello.json, etc).
# Matched case-insensitively against the gloss.
WORD_SIGN_LABELS = ["Bye", "Deaf", "Hello", "NotOk", "Pen", "Please", "Thankyou", "Yes"]
WORD_SIGNS_LOWER = {w.lower(): w for w in WORD_SIGN_LABELS}

# Multi-word English phrases that map to ONE of our word signs (the sign
# folders were named as single words, e.g. Thankyou/, but the phrase is two
# words in English). Matched as adjacent lemma pairs before POS filtering.
PHRASE_SIGNS = {
    ("thank", "you"): "Thankyou",
    ("not", "ok"): "NotOk",
    ("not", "okay"): "NotOk",
}

# Letters we have pose data for (A-Z). Fingerspelling skips anything else
# (digits, punctuation) rather than guessing.
FINGERSPELL_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

_nlp = None  # lazy-loaded spaCy pipeline


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── Step 1 (optional): local LLM pre-simplification ───────────
# Tested both locally pulled models: gemma3:270m is fast (~1-2s) but too
# weak to reliably follow the rewrite instruction (it just echoed the input
# back). qwen3:8b actually simplifies correctly, at the cost of ~15-20s on
# CPU - acceptable for an explicit, opt-in "Simplify with AI" action, not
# for something run silently on every translation.
def simplify_with_llm(text, model="qwen3:8b", timeout=60):
    """
    Ask a local Ollama model to rewrite a sentence as short, simple,
    literal English before gloss conversion. Returns the original text
    unchanged if Ollama isn't reachable or the model isn't pulled -
    this step is an enhancement, never a hard dependency.
    """
    import requests

    prompt = (
        "Rewrite the following sentence using short, simple, literal English. "
        "Keep all the original meaning and keep any greeting or courtesy phrases "
        "EXACTLY as written - do not paraphrase 'thank you' as 'thanks', do not "
        "shorten 'very much' to 'a lot', do not swap 'hello' for 'hi'. Only "
        "simplify parts that are genuinely complex. Do not add commentary. "
        "Reply with ONLY the rewritten sentence, nothing else.\n\n"
        f"Sentence: {text}"
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        simplified = resp.json().get("response", "").strip().strip('"')
        return simplified if simplified else text
    except Exception as e:
        print(f"[text_to_sign] LLM simplification skipped ({e}); using original text.")
        return text


# ── Step 2: rule-based gloss conversion ────────────────────────
def text_to_gloss(text):
    """
    English sentence -> list of gloss tokens (uppercase base words),
    e.g. "I am going to the store" -> ["I", "GO", "STORE"].
    """
    doc = _get_nlp()(text)
    # Keep POS alongside each lemma so a merged phrase (which has no
    # single POS of its own) can be tagged to skip the filtering pass below.
    words = [
        {"lemma": t.lemma_, "pos": t.pos_, "is_phrase": False}
        for t in doc if t.is_alpha
    ]

    # Merge known two-word phrases ("thank" + "you" -> "Thankyou") before
    # POS filtering, so neither half gets dropped or fingerspelled apart.
    merged = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            pair = (words[i]["lemma"].lower(), words[i + 1]["lemma"].lower())
            if pair in PHRASE_SIGNS:
                merged.append({"lemma": PHRASE_SIGNS[pair], "pos": None, "is_phrase": True})
                i += 2
                continue
        merged.append(words[i])
        i += 1

    gloss = []
    for w in merged:
        if w["is_phrase"]:
            gloss.append(w["lemma"].upper())
            continue
        if w["pos"] in DROP_POS:
            continue
        lemma_lower = w["lemma"].lower()
        if lemma_lower in DROP_WORDS:
            continue
        if lemma_lower in AUX_LEMMAS and w["pos"] in ("AUX", "VERB"):
            continue
        gloss.append(w["lemma"].upper())

    return gloss


# ── Step 3: gloss -> sign queue (word sign or fingerspell) ─────
def gloss_to_sign_queue(gloss_tokens):
    """
    Turn gloss tokens into an ordered list of avatar-playable sign units:
      {"label": <pose file name, no .json>, "kind": "word"|"letter", "word": <source token>}
    Tokens with no matching word sign and no fingerspellable letters are
    dropped (nothing to show, rather than guessing).
    """
    queue = []

    for token in gloss_tokens:
        lower = token.lower()

        if lower in WORD_SIGNS_LOWER:
            queue.append({"label": WORD_SIGNS_LOWER[lower], "kind": "word", "word": token})
            continue

        letters = [c for c in token if c in FINGERSPELL_ALPHABET]
        for letter in letters:
            queue.append({"label": letter, "kind": "letter", "word": token})

    return queue


# A short sentence has nothing for the LLM to usefully compress - it can only
# introduce drift (e.g. rewriting "thank you very much" as "thanks a lot",
# which breaks the literal phrase match in gloss_to_sign_queue). The LLM step
# exists for genuinely long/complex input, so skip it below this length
# regardless of the checkbox.
LLM_MIN_WORDS = 10


# ── Full pipeline ───────────────────────────────────────────────
def translate(text, use_llm=False):
    """
    Full text -> sign queue pipeline. Returns
      {"simplified": str, "gloss": [str], "queue": [sign unit dicts]}
    so the frontend can show what happened at each stage.
    """
    run_llm = use_llm and len(text.split()) >= LLM_MIN_WORDS
    simplified = simplify_with_llm(text) if run_llm else text
    gloss = text_to_gloss(simplified)
    queue = gloss_to_sign_queue(gloss)
    return {"simplified": simplified, "gloss": gloss, "queue": queue}
