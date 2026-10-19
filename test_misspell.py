"""Standard-library tests using exclusively synthetic records."""

import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from build_misspell_cot import build_variants
from compute_misspell_recall import score_files
from make_misspell_phonectx import transform_record
from misspell_io import (check_probability, contains_word, open_output, parse_context,
                         read_jsonl, record_id, replace_words)
from misspell_rules import load_wordlist, misspell_word, rule_edits

BASE = Path(__file__).resolve().parent


def plain_record():
    return {
        "id": "private-id",
        "text": "THE PHONE IS ON THE TABLE.",
        "context_text": "PHONE(F  OW N)\nBOAT(B OW T)\nTABLE(T EY B AH L)\nPHONE(F  OW N)",
        "oov_labels": [{"word": "PHONE", "private_metadata": "not-exported"}],
        "private_metadata": "not-exported",
    }


def cot_record():
    return {
        "id": "private-id",
        "text": "<phoneme>DH AH | F OW N | IH Z | R EH D IY</phoneme>"
                "<match>[None] [PHONE] [None] [None]</match>THE PHONE IS READY.",
        "context_text": "PHONE(F OW N)\nBOAT(B OW T)\nTABLE(T EY B AH L)",
        "oov_labels": [{"word": "PHONE", "private_metadata": "not-exported"}],
        "private_metadata": "not-exported",
    }


def pairs(record, field="misspell_map"):
    return dict(item.split("->", 1) for item in record[field])


class RulesTests(unittest.TestCase):
    def test_specific_edits(self):
        self.assertIn("fone", rule_edits("phone"))
        self.assertIn("leter", rule_edits("letter"))
        self.assertIn("kat", rule_edits("cat"))
        self.assertIn("kent", rule_edits("cent"))
        self.assertIn("knname", rule_edits("name"))
        self.assertIn("wrrain", rule_edits("rain"))
        self.assertIn("whword", rule_edits("word"))
        self.assertFalse(any("\\1" in edit for edit in rule_edits("letter")))

    def test_reproducibility(self):
        first = misspell_word("PHONE", {"phone"}, {"PHONE"}, random.Random(42))
        self.assertEqual(first, misspell_word("PHONE", {"phone"}, {"PHONE"}, random.Random(42)))
        self.assertIsNotNone(first)
        self.assertNotEqual(first, "PHONE")

    def test_vocabulary_and_used_constraints(self):
        with patch("misspell_rules.rule_edits", return_value=["fone"]):
            self.assertIsNone(misspell_word("phone", {"fone"}, set(), random.Random(1)))
            self.assertIsNone(misspell_word("phone", set(), {"FONE"}, random.Random(1)))
            self.assertEqual("FONE", misspell_word("phone", set(), set(), random.Random(1)))

    def test_invalid_words(self):
        for word in ("", "123", "--", "A_B", "WORD2"):
            self.assertIsNone(misspell_word(word, set(), set(), random.Random(1)))

    def test_wordlist_formats(self):
        with tempfile.TemporaryDirectory(dir=str(BASE)) as directory:
            text = Path(directory) / "vocab.txt"
            text.write_text("# example\nPHONE\nphone\ntable\n", encoding="utf-8")
            self.assertEqual({"phone", "table"}, load_wordlist(text))
            data = Path(directory) / "vocab.jsonl"
            data.write_text('{"word":"PHONE","extra":"ignored"}\n', encoding="utf-8")
            self.assertEqual({"phone"}, load_wordlist(data))
            data.write_text('{"word":5}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_wordlist(data)
            text.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_wordlist(text)


class TransformTests(unittest.TestCase):
    def test_boundaries_and_one_pass_replacement(self):
        self.assertEqual("FAKE ONE XPHONE DON'T RE-WRITE 2PHONE",
                         replace_words("PHONE TABLE XPHONE DON'T RE-WRITE 2PHONE",
                                       {"PHONE": "FAKE", "TABLE": "ONE", "FAKE": "BAD"}))
        self.assertEqual("F1 F2", replace_words("'PHONE' RE-WRITE", {"'PHONE'": "F1", "RE-WRITE": "F2"}))
        self.assertFalse(contains_word("XPHONE", "PHONE"))

    def test_first_candidate_and_label_preservation(self):
        entries = parse_context("A description.\nPHONE(F  OW N)\nPHONE(F  OW N)")
        self.assertEqual([("PHONE", "F  OW N"), ("PHONE", "F  OW N")], entries)
        with self.assertRaises(ValueError):
            parse_context("PHONE(F OW N)\ninvalid candidate")
        with self.assertRaises(ValueError):
            parse_context("PHONE(F OW N)\nPHONE(F AH N)")

    def test_probability_and_metadata(self):
        record = plain_record()
        output, stats = transform_record(record, {"phone", "table", "boat"}, random.Random(10), prob=1)
        mapping = pairs(output, "context_map")
        original = parse_context(record["context_text"])
        updated = parse_context(output["context_text"])
        self.assertEqual(len(original), len(updated))
        for (word, phone), (fake, new_phone) in zip(original, updated):
            self.assertEqual(phone, new_phone)
            self.assertEqual(mapping.get(word, word), fake)
        for word, fake in pairs(output).items():
            self.assertTrue(contains_word(output["text"], fake))
            self.assertFalse(contains_word(output["text"], word))
        self.assertNotIn("BOAT", pairs(output))
        self.assertNotIn("id", output)
        self.assertNotIn("private_metadata", output)
        self.assertEqual("private-id", record["id"])
        self.assertGreater(stats["changed_words"], 0)

    def test_original_text_word_collision_is_forbidden(self):
        record = plain_record()
        record["text"] += " FONE"
        with patch("make_misspell_phonectx.misspell_word") as generate:
            generate.return_value = None
            transform_record(record, set(), random.Random(0), prob=1)
            self.assertIn("FONE", generate.call_args[0][2])
            self.assertIn("BOAT", generate.call_args[0][2])

    def test_context_only_targets(self):
        record = plain_record()
        output, stats = transform_record(record, {"phone"}, random.Random(10),
                                         context_only=True, targets_only=True)
        self.assertEqual(record["text"], output["text"])
        self.assertEqual({"PHONE"}, set(pairs(output)))
        self.assertEqual([{"word": "PHONE"}], output["oov_labels"])
        self.assertIn("BOAT(B OW T)", output["context_text"])
        self.assertEqual(0, stats["dropped_records"])

    def test_target_failure_drops_record(self):
        with patch("make_misspell_phonectx.misspell_word", return_value=None):
            output, stats = transform_record(plain_record(), set(), random.Random(10), targets_only=True)
            self.assertIsNone(output)
            self.assertEqual(1, stats["dropped_records"])

    def test_generation_failure_preserves_non_target(self):
        with patch("make_misspell_phonectx.misspell_word", return_value=None):
            record = plain_record()
            output, stats = transform_record(record, set(), random.Random(10), prob=1)
            self.assertEqual(record["text"], output["text"])
            self.assertEqual([], output["misspell_map"])
            self.assertEqual(3, stats["generation_failures"])

    def test_probability_zero_and_invalid(self):
        output, _ = transform_record(plain_record(), set(), random.Random(10), prob=0)
        self.assertEqual([], output["context_map"])
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                check_probability(value)


class AnnotatedTests(unittest.TestCase):
    def test_alignment_sync_and_sanitization(self):
        record = cot_record()
        outputs, stats = build_variants(record, {"phone", "table", "boat"}, random.Random(4), 3, 4)
        self.assertTrue(outputs)
        signatures = set()
        for output in outputs:
            mapping = pairs(output)
            fake = mapping["PHONE"]
            self.assertEqual(record["text"].split("</phoneme>")[0], output["text"].split("</phoneme>")[0])
            self.assertIn("[{}]".format(fake), output["text"])
            self.assertTrue(contains_word(output["text"].split("</match>")[1], fake))
            self.assertEqual([{"word": fake}], output["oov_labels"])
            self.assertIn("{}(F OW N)".format(fake), output["context_text"])
            self.assertNotIn("private_metadata", output)
            self.assertNotIn("id", output)
            signatures.add(tuple(output["context_map"]))
        self.assertEqual(len(outputs), len(signatures))
        self.assertEqual(len(outputs), stats["variants"])

    def test_label_mismatch_rejected(self):
        for replacement in ("[BOAT]", "[None]"):
            record = cot_record()
            record["text"] = record["text"].replace("[PHONE]", replacement)
            with self.assertRaises(ValueError):
                build_variants(record, set(), random.Random(4))
        record = cot_record()
        record["text"] = record["text"].replace("THE PHONE", "THE BOAT")
        with self.assertRaises(ValueError):
            build_variants(record, set(), random.Random(4))

    def test_present_but_unmatched_candidate_sync(self):
        record = cot_record()
        record["text"] = record["text"].replace("R EH D IY", "T EY B AH L").replace("READY.", "TABLE.")
        outputs, _ = build_variants(record, {"phone", "table", "boat"}, random.Random(4), 1, 1, other_prob=1)
        self.assertTrue(outputs)
        fake = pairs(outputs[0])["TABLE"]
        self.assertTrue(contains_word(outputs[0]["text"].split("</match>")[1], fake))
        self.assertIn("[None] [None]</match>", outputs[0]["text"])

    def test_forced_generation_failure(self):
        with patch("build_misspell_cot.misspell_word", return_value=None):
            outputs, stats = build_variants(cot_record(), set(), random.Random(4), 3, 3)
            self.assertEqual([], outputs)
            self.assertEqual(3, stats["dropped_variants"])


class ScoringAndIOTests(unittest.TestCase):
    def test_five_scoring_categories_and_missing_denominator(self):
        with tempfile.TemporaryDirectory(dir=str(BASE)) as directory:
            source = Path(directory) / "source.jsonl"
            hypothesis = Path(directory) / "hypothesis.jsonl"
            source.write_text("".join(json.dumps({"id": str(i), "misspell_map": ["PHONE->FONE"]}) + "\n"
                                      for i in range(5)), encoding="utf-8")
            texts = ["fone", "PHONE", "PHONE FONE", "XPHONE"]
            hypothesis.write_text("".join(json.dumps({"id": str(i), "hypothesis": text}) + "\n"
                                          for i, text in enumerate(texts)), encoding="utf-8")
            report = score_files(source, hypothesis)
            self.assertEqual({"fake": 1, "original": 1, "both": 1, "neither": 1, "missing": 1}, report["counts"])
            self.assertEqual(5, report["total_pairs"])
            self.assertEqual(0.8, report["prediction_coverage"])
            with hypothesis.open("a", encoding="utf-8") as stream:
                stream.write('{"id":"0","hypothesis":"FONE"}\n')
            with self.assertRaises(ValueError):
                score_files(source, hypothesis)

    def test_annotated_prediction_ignores_match_section(self):
        with tempfile.TemporaryDirectory(dir=str(BASE)) as directory:
            source = Path(directory) / "source.jsonl"
            hypothesis = Path(directory) / "hypothesis.jsonl"
            source.write_text('{"id":"1","misspell_map":["PHONE->FONE"]}\n', encoding="utf-8")
            hypothesis.write_text('{"id":"1","hypothesis":"<phoneme>F OW N</phoneme><match>[FONE]</match>PHONE"}\n',
                                  encoding="utf-8")
            self.assertEqual(1, score_files(source, hypothesis)["counts"]["original"])
            source.write_text('{"id":"1","misspell_map":["PHONE->FONE"]}\n' * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                score_files(source, hypothesis)

    def test_id_validation(self):
        self.assertEqual("1", record_id({"id": 1, "uid": "1"}))
        for record in ({}, {"id": ""}, {"id": True}, {"id": "a", "uid": "b"}):
            with self.assertRaises(ValueError):
                record_id(record)

    def test_output_protection_and_jsonl_limit(self):
        with tempfile.TemporaryDirectory(dir=str(BASE)) as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text('{}\n{}\n', encoding="utf-8")
            self.assertEqual([{}], list(read_jsonl(source, 1)))
            with self.assertRaises(ValueError):
                open_output(source, [source])
            with self.assertRaises(FileExistsError):
                open_output(source, [])
            output = Path(directory) / "output.jsonl"
            with open_output(output, [source]) as stream:
                stream.write('{}\n')
            self.assertEqual('{}\n', output.read_text(encoding="utf-8"))
            link = Path(directory) / "link.jsonl"
            link.symlink_to(output)
            with self.assertRaises(ValueError):
                open_output(link, [])


class CommandAndRegressionTests(unittest.TestCase):
    def test_command_line_pipeline(self):
        import build_misspell_cot
        import compute_misspell_recall
        import make_misspell_phonectx

        with tempfile.TemporaryDirectory(dir=str(BASE)) as directory:
            root = Path(directory)
            commands = [
                (make_misspell_phonectx, "plain.jsonl", "plain.jsonl", ["--prob", "1"]),
                (make_misspell_phonectx, "plain.jsonl", "context.jsonl", ["--targets-only", "--context-only"]),
                (build_misspell_cot, "cot.jsonl", "cot.jsonl", ["--n-var-min", "3", "--n-var-max", "3"]),
            ]
            for module, example, output, extras in commands:
                argv = [module.__name__, "--in", str(BASE / "examples" / example),
                        "--out", str(root / output), "--vocab", str(BASE / "examples" / "vocab.txt"),
                        "--seed", "42"] + extras
                with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    module.main()
                    json.loads(stdout.getvalue())
                records = list(read_jsonl(root / output))
                self.assertTrue(records)
                self.assertTrue(all(record["id"].startswith("sample-") for record in records))
            source = root / "plain.jsonl"
            records = list(read_jsonl(source))
            prediction = root / "predictions.jsonl"
            prediction.write_text("".join(json.dumps({"id": record["id"], "hypothesis": record["text"]}) + "\n"
                                          for record in records), encoding="utf-8")
            argv = ["score", "--src", str(source), "--hyp", str(prediction)]
            with patch("sys.argv", argv), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                compute_misspell_recall.main()
                report = json.loads(stdout.getvalue())
            self.assertGreater(report["total_pairs"], 0)
            self.assertEqual(1.0, report["rates"]["fake"])
            argv = ["generate", "--in", str(BASE / "examples" / "plain.jsonl"),
                    "--out", str(source), "--vocab", str(BASE / "examples" / "vocab.txt")]
            with patch("sys.argv", argv), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit):
                    make_misspell_phonectx.main()
                self.assertNotIn(str(root), stderr.getvalue())
            self.assertEqual(records, list(read_jsonl(source)))

    def test_annotated_prediction_whitespace_and_malformed(self):
        with tempfile.TemporaryDirectory(dir=str(BASE)) as directory:
            source = Path(directory) / "source.jsonl"
            prediction = Path(directory) / "predictions.jsonl"
            source.write_text('{"id":"1","misspell_map":["PHONE->FONE"]}\n', encoding="utf-8")
            hypothesis = " \n<phoneme>F OW N</phoneme><match>[FONE]</match>PHONE\n"
            prediction.write_text(json.dumps({"id": "1", "hypothesis": hypothesis}) + "\n", encoding="utf-8")
            self.assertEqual(1, score_files(source, prediction)["counts"]["original"])
            for hypothesis in (" <match>[FONE] PHONE", "<phoneme>F OW N</phoneme>PHONE", "<match>[FONE]</match><match>"):
                prediction.write_text(json.dumps({"id": "1", "hypothesis": hypothesis}) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    score_files(source, prediction)

    def test_plain_target_labels_are_synchronized(self):
        output, _ = transform_record(plain_record(), {"phone"}, random.Random(10), prob=1)
        self.assertEqual([{"word": pairs(output)["PHONE"]}], output["oov_labels"])


if __name__ == "__main__":
    unittest.main()
