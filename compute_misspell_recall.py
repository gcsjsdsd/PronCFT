#!/usr/bin/env python3
# Copyright (c) 2026, Tencent. All rights reserved.
"""Count fake/original/both/neither/missing spelling outcomes, without causal claims."""

import argparse
from collections import Counter
import json
import re

from misspell_io import normalize_word, read_jsonl, record_id, text_words

ANNOTATED_RE = re.compile(r"^(?:<phoneme>.*?</phoneme>)?<match>.*?</match>(.*)$", re.S)


def load_map(path):
    mapping = {}
    for record in read_jsonl(path):
        key = record_id(record)
        if key in mapping:
            raise ValueError("duplicate source id")
        items = record.get("misspell_map", [])
        if not isinstance(items, list):
            raise ValueError("misspell_map must be a list")
        pairs, seen_orig, seen_fake = [], set(), set()
        for item in items:
            if not isinstance(item, str) or item.count("->") != 1:
                raise ValueError("misspell_map entries must have ORIG->FAKE format")
            orig, fake = [normalize_word(word) for word in item.split("->")]
            if orig == fake or orig in seen_orig or fake in seen_fake:
                raise ValueError("spelling mapping must contain distinct one-to-one pairs")
            pairs.append((orig, fake))
            seen_orig.add(orig)
            seen_fake.add(fake)
        if seen_orig & seen_fake:
            raise ValueError("original and fake spelling sets must not overlap")
        mapping[key] = pairs
    return mapping


def score_files(src, hyp):
    mapping = load_map(src)
    predictions = {}
    for record in read_jsonl(hyp):
        key = record_id(record)
        if key in predictions:
            raise ValueError("duplicate hypothesis id")
        hypothesis = record.get("hypothesis")
        if not isinstance(hypothesis, str):
            raise ValueError("hypothesis must be a string")
        hypothesis = hypothesis.strip()
        annotated = ANNOTATED_RE.fullmatch(hypothesis)
        if annotated:
            hypothesis = annotated.group(1)
        if re.search(r"</?(?:phoneme|match)(?:>|\\s|$)", hypothesis):
            raise ValueError("malformed annotated hypothesis")
        predictions[key] = text_words(hypothesis)
    counts = Counter({name: 0 for name in ("fake", "original", "both", "neither", "missing")})
    relevant_records = predicted_records = 0
    for key, pairs in mapping.items():
        if not pairs:
            continue
        relevant_records += 1
        if key in predictions:
            predicted_records += 1
        for orig, fake in pairs:
            if key not in predictions:
                counts["missing"] += 1
                continue
            words = predictions[key]
            fake_hit, orig_hit = fake in words, orig in words
            category = "both" if fake_hit and orig_hit else "fake" if fake_hit else "original" if orig_hit else "neither"
            counts[category] += 1
    total = sum(counts.values())
    return {
        "records_with_targets": relevant_records,
        "records_with_predictions": predicted_records,
        "prediction_coverage": predicted_records / relevant_records if relevant_records else 0.0,
        "unknown_hypothesis_records": len(set(predictions) - set(mapping)),
        "total_pairs": total,
        "counts": dict(counts),
        "rates": {key: value / total if total else 0.0 for key, value in counts.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--hyp", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(score_files(args.src, args.hyp), indent=2, sort_keys=True))
    except OSError:
        parser.error("local file operation failed; check that input files are readable")
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
