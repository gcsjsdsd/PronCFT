#!/usr/bin/env python3
# Copyright (c) 2026, Tencent. All rights reserved.
"""Apply nonword spelling edits to standalone text/context JSONL records."""

import argparse
import json
import random

from misspell_io import (check_probability, open_output, parse_context, read_jsonl,
                         render_context, replace_words, target_words, text_words,
                         write_record)
from misspell_rules import load_wordlist, misspell_word


def transform_record(record, real_words, rng, prob=0.5, context_only=False, targets_only=False):
    check_probability(prob)
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty plain transcript")
    if text.lstrip().startswith("<phoneme>") or text.lstrip().startswith("<match>"):
        raise ValueError("use the annotated-text command for phoneme/match input")
    entries = parse_context(record.get("context_text"))
    candidates = list(dict.fromkeys(w for w, _ in entries))
    transcript_words = text_words(text)
    targets = target_words(record)
    if not set(targets).issubset(transcript_words):
        raise ValueError("every target label must occur in the transcript")
    if targets_only and (not targets or not set(targets).issubset(candidates)):
        raise ValueError("target-only mode requires candidate labels present in the transcript")
    used = set(candidates) | transcript_words
    mapping = {}
    stats = {"candidate_words": len(candidates), "changed_words": 0,
             "generation_failures": 0, "dropped_records": 0}
    for word in candidates:
        if targets_only:
            if word not in targets:
                continue
        elif rng.random() >= prob:
            continue
        fake = misspell_word(word, real_words, used, rng)
        if fake is None:
            stats["generation_failures"] += 1
            if targets_only:
                stats["dropped_records"] = 1
                return None, stats
            continue
        mapping[word] = fake
        used.add(fake)
    transcript_map = {w: f for w, f in mapping.items() if w in transcript_words}
    output = {
        "text": text if context_only else replace_words(text, transcript_map),
        "context_text": render_context(entries, mapping),
        "misspell_map": ["{}->{}".format(w, f) for w, f in transcript_map.items()],
        "context_map": ["{}->{}".format(w, f) for w, f in mapping.items()],
    }
    if targets:
        output["oov_labels"] = [{"word": w if context_only else mapping.get(w, w)} for w in targets]
    stats["changed_words"] = len(mapping)
    return output, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-only", action="store_true", help="keep the reference transcript unchanged")
    parser.add_argument("--targets-only", action="store_true", help="replace only oov_labels targets; probability is ignored")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    try:
        check_probability(args.prob)
        if args.limit < 0:
            raise ValueError("limit must be nonnegative")
        real_words = load_wordlist(args.vocab)
        totals = {"input_records": 0, "output_records": 0}
        with open_output(args.out, [args.inp, args.vocab]) as stream:
            for index, record in enumerate(read_jsonl(args.inp, args.limit), 1):
                try:
                    output, stats = transform_record(record, real_words, random.Random(args.seed + index),
                                                     args.prob, args.context_only, args.targets_only)
                except ValueError as error:
                    raise ValueError("record {}: {}".format(index, error)) from None
                totals["input_records"] += 1
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0) + value
                if output is not None:
                    output["id"] = "sample-{:06d}".format(index)
                    write_record(stream, output)
                    totals["output_records"] += 1
        print(json.dumps(totals, sort_keys=True))
    except OSError:
        parser.error("local file operation failed; check inputs and ensure output does not exist")
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
