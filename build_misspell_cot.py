#!/usr/bin/env python3
"""Generate standalone spelling variants of phoneme/match-annotated text."""

import argparse
import json
import random
import re

from misspell_io import (check_probability, normalize_word, open_output, parse_context,
                         read_jsonl, render_context, replace_words, target_words,
                         text_words, write_record)
from misspell_rules import load_wordlist, misspell_word

COT_RE = re.compile(r"^<phoneme>(.*?)</phoneme><match>(.*?)</match>(.*)$", re.S)
TOKEN_RE = re.compile(r"\[([^\[\]]+)\]")


def build_variants(record, real_words, rng, n_var_min=3, n_var_max=4, other_prob=0.5):
    check_probability(other_prob)
    if not 1 <= n_var_min <= n_var_max:
        raise ValueError("variant counts must satisfy 1 <= minimum <= maximum")
    text = record.get("text")
    match = COT_RE.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError("text must have phoneme, match and transcript sections")
    phonemes, match_text, transcript = match.groups()
    segments = phonemes.split(" | ")
    tokens = TOKEN_RE.findall(match_text)
    if TOKEN_RE.sub("", match_text).strip() or not all(segment.strip() for segment in segments):
        raise ValueError("invalid phoneme or match section")
    if len(segments) != len(tokens) or len(tokens) != len(transcript.split()):
        raise ValueError("phoneme, match and transcript word counts must align")
    entries = parse_context(record.get("context_text"))
    phones_of = dict(entries)
    candidates = list(dict.fromkeys(w for w, _ in entries))
    targets = target_words(record)
    if not targets or not set(targets).issubset(phones_of):
        raise ValueError("oov_labels must identify at least one existing candidate")
    if not set(targets).issubset(text_words(transcript)):
        raise ValueError("every target must occur in the transcript")
    normalized_tokens = []
    required = set(targets)
    for segment, token, transcript_word in zip(segments, tokens, transcript.split()):
        if token == "None":
            normalized_tokens.append(token)
            continue
        word = normalize_word(token)
        if word not in phones_of or segment.split() != phones_of[word].split():
            raise ValueError("match token and candidate phonemes disagree")
        if text_words(transcript_word) != {word}:
            raise ValueError("match token and transcript word disagree")
        normalized_tokens.append(word)
        required.add(word)
    hit_words = set(normalized_tokens) - {"None"}
    if not hit_words or not set(targets).issubset(hit_words):
        raise ValueError("every target must have a match hit")
    seen, variants = set(), []
    stats = {"variants": 0, "dropped_variants": 0, "target_changed": 0,
             "other_changed": 0, "other_kept": 0}
    for _ in range(rng.randint(n_var_min, n_var_max)):
        for _retry in range(3):
            used = set(candidates) | text_words(transcript)
            mapping, failed = {}, False
            for word in candidates:
                if word not in required and rng.random() >= other_prob:
                    continue
                fake = misspell_word(word, real_words, used, rng)
                if fake is None:
                    if word in required:
                        failed = True
                        break
                    continue
                mapping[word] = fake
                used.add(fake)
            signature = tuple(sorted(mapping.items()))
            if failed or signature in seen:
                continue
            seen.add(signature)
            ordered = list(entries)
            rng.shuffle(ordered)
            transcript_map = {w: f for w, f in mapping.items() if w in text_words(transcript)}
            updated_tokens = ["[{}]".format(mapping.get(w, w)) for w in normalized_tokens]
            output = {
                "text": "<phoneme>{}</phoneme><match>{}</match>{}".format(
                    phonemes, " ".join(updated_tokens), replace_words(transcript, transcript_map)),
                "context_text": render_context(ordered, mapping),
                "oov_labels": [{"word": mapping[w]} for w in targets],
                "misspell_map": ["{}->{}".format(w, f) for w, f in transcript_map.items()],
                "context_map": ["{}->{}".format(w, f) for w, f in mapping.items()],
                "variant": "misspell",
            }
            variants.append(output)
            stats["variants"] += 1
            stats["target_changed"] += len(required)
            stats["other_changed"] += len(set(mapping) - required)
            stats["other_kept"] += len(set(candidates) - required - set(mapping))
            break
        else:
            stats["dropped_variants"] += 1
    return variants, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--n-var-min", type=int, default=3)
    parser.add_argument("--n-var-max", type=int, default=4)
    parser.add_argument("--other-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    try:
        check_probability(args.other_prob)
        if not 1 <= args.n_var_min <= args.n_var_max or args.limit < 0:
            raise ValueError("invalid variant counts or limit")
        real_words = load_wordlist(args.vocab)
        totals = {"base_records": 0}
        with open_output(args.out, [args.inp, args.vocab]) as stream:
            for index, record in enumerate(read_jsonl(args.inp, args.limit), 1):
                try:
                    variants, stats = build_variants(record, real_words, random.Random(args.seed + index),
                                                      args.n_var_min, args.n_var_max, args.other_prob)
                except ValueError as error:
                    raise ValueError("record {}: {}".format(index, error)) from None
                totals["base_records"] += 1
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0) + value
                for number, variant in enumerate(variants, 1):
                    variant["id"] = "sample-{:06d}-v{}".format(index, number)
                    write_record(stream, variant)
        print(json.dumps(totals, sort_keys=True))
    except OSError:
        parser.error("local file operation failed; check inputs and ensure output does not exist")
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
