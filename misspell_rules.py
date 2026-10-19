"""Heuristic nonword spelling edits; pronunciation labels are supplied externally."""

import json
import random
import re

VOWEL_GROUPS = [
    ["ee", "ea", "ie", "i"],
    ["oa", "ow", "o"],
    ["ow", "ou"],
    ["oo", "u", "ew"],
    ["ai", "ay", "ei"],
    ["igh", "ie", "y", "i"],
    ["er", "ar", "or", "ur"],
]
WEAK_VOWELS = "aeiou"
CONS = "bcdfghjklmnpqrstvwxz"


def load_wordlist(path):
    """Read UTF-8 text (one word/line) or JSONL objects with a word field."""
    words = set()
    jsonl = str(path).lower().endswith(".jsonl")
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or (not jsonl and line.startswith("#")):
                continue
            if jsonl:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError("invalid wordlist JSON at line {}".format(number)) from None
                word = item.get("word") if isinstance(item, dict) else None
            else:
                word = line
            if not isinstance(word, str) or not word.strip() or any(c.isspace() for c in word.strip()):
                raise ValueError("invalid wordlist entry at line {}".format(number))
            words.add(word.strip().lower())
    if not words:
        raise ValueError("wordlist is empty")
    return words


def rule_edits(w):
    """Enumerate historical single-step edits of a lowercase word, including duplicates."""
    out = []

    def sub(pat, rep):
        for m in re.finditer(pat, w):
            out.append(w[:m.start()] + m.expand(rep) + w[m.end():])

    sub(r"c(?=[eiy])", "s")
    sub(r"c", "k")
    sub(r"ck", "k")
    sub(r"ph", "f")
    sub(r"qu", "kw")
    sub(r"x", "ks")
    sub(r"g(?=[eiy])", "j")
    sub(r"tch", "ch")
    for grp in VOWEL_GROUPS:
        for p in grp:
            for m in re.finditer(p, w):
                for q in grp:
                    if q != p:
                        out.append(w[:m.start()] + q + w[m.end():])
    for m in re.finditer(r"[aeiou]", w):
        for v in WEAK_VOWELS:
            if v != m.group(0):
                out.append(w[:m.start()] + v + w[m.end():])
    sub(r"([" + CONS + r"])\1", r"\1")
    for m in re.finditer(r"([" + CONS + r"])", w):
        out.append(w[:m.start()] + m.group(1) + w[m.start():])
    sub(r"^kn", "n")
    sub(r"^gn", "n")
    sub(r"^wr", "r")
    sub(r"mb$", "m")
    sub(r"^wh", "w")
    sub(r"gh", "")
    sub(r"^ps", "s")
    sub(r"^pt", "t")
    sub(r"^h(?=[aeiou])", "")
    sub(r"([lrnv])e$", r"\1")
    if w.startswith("n"):
        out += ["kn" + w, "gn" + w]
    if w.startswith("r"):
        out.append("wr" + w)
    if w.startswith("w"):
        out.append("wh" + w)
    return out


def misspell_word(word, real_words, used, rng=None):
    """Return an uppercase nonword, or None after ten unsuccessful attempts.

    real_words must contain lowercase strings; used must contain uppercase strings.
    Edits preserve externally attached labels, not independently verified pronunciation.
    """
    rng = rng if rng is not None else random.Random()
    w0 = word.lower()
    if not re.fullmatch(r"[a-z'-]+", w0) or not re.search(r"[a-z]", w0):
        return None
    for _ in range(10):
        w = w0
        for _ in range(2 if rng.random() < 0.3 else 1):
            edits = [e for e in rule_edits(w) if e != w]
            if not edits:
                break
            w = rng.choice(edits)
        if (w != w0 and w not in real_words and w.upper() not in used
                and re.fullmatch(r"[a-z'-]+", w) and re.search(r"[a-z]", w)):
            return w.upper()
    return None
