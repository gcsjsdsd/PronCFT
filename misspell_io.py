"""Local JSONL and text helpers shared by the standalone commands."""

import json
import math
from pathlib import Path
import re

WORD_RE = re.compile(r"(?<![A-Za-z0-9_'-])[A-Za-z'-]+(?![A-Za-z0-9_'-])")
CAND_RE = re.compile(r"^([A-Za-z'-]+)\(([^()\r\n]+)\)$")


def normalize_word(word):
    if (not isinstance(word, str) or not re.fullmatch(r"[A-Za-z'-]+", word)
            or not re.search(r"[A-Za-z]", word)):
        raise ValueError("candidate words must contain ASCII letters, apostrophes or hyphens")
    return word.upper()


def text_words(text):
    return {m.group(0).upper() for m in WORD_RE.finditer(text) if re.search(r"[A-Za-z]", m.group(0))}


def replace_words(text, mapping):
    return WORD_RE.sub(lambda m: mapping.get(m.group(0).upper(), m.group(0)), text)


def contains_word(text, word):
    return word.upper() in text_words(text)


def parse_context(text):
    if not isinstance(text, str):
        raise ValueError("context_text must be a string")
    entries, phones_of = [], {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = CAND_RE.fullmatch(line.strip())
        if not match:
            if entries:
                raise ValueError("invalid candidate line after the first candidate")
            continue
        word = normalize_word(match.group(1))
        phones = match.group(2)
        if not phones.strip():
            raise ValueError("candidate phoneme label is empty")
        if word in phones_of and phones_of[word] != phones:
            raise ValueError("repeated candidate has inconsistent phoneme labels")
        entries.append((word, phones))
        phones_of[word] = phones
    if not entries:
        raise ValueError("context_text has no candidates")
    return entries


def render_context(entries, mapping):
    return "\n".join("{}({})".format(mapping.get(w, w), p) for w, p in entries)


def target_words(record):
    labels = record.get("oov_labels", [])
    if not isinstance(labels, list):
        raise ValueError("oov_labels must be a list")
    targets = []
    for item in labels:
        if not isinstance(item, dict) or "word" not in item:
            raise ValueError("each oov_labels entry requires a word field")
        word = normalize_word(item["word"])
        if word not in targets:
            targets.append(word)
    return targets


def check_probability(value):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probability must be between zero and one")


def read_jsonl(path, limit=0):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    count = 0
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError("invalid JSON at line {}".format(number)) from None
            if not isinstance(item, dict):
                raise ValueError("JSONL record at line {} must be an object".format(number))
            yield item
            count += 1
            if limit and count >= limit:
                break


def open_output(path, inputs):
    target = Path(path)
    if target.is_symlink():
        raise ValueError("output must not be a symbolic link")
    if target.resolve() in {Path(source).resolve() for source in inputs}:
        raise ValueError("output must differ from every input")
    return target.open("x", encoding="utf-8")


def write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_id(record):
    values = [record[name] for name in ("id", "uid") if record.get(name) is not None]
    if not values or any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in values):
        raise ValueError("each record requires a nonempty string or integer id")
    if any(not str(value).strip() for value in values) or len({str(value) for value in values}) != 1:
        raise ValueError("id and uid must be nonempty and agree when both are present")
    return str(values[0])
