# PronCFT

**Pronunciation Contextual Fine-Tuning for Contextual Biasing of Out-of-Vocabulary Words in LLM-Based ASR**

[Paper (arXiv:2610.25766)](https://arxiv.org/abs/2610.25766)

Yifan Xu, Kaixun Huang, Mengzhe Geng, Mingyu Cui, Shen Huang, Shidong Shang, Xunying Liu

The Chinese University of Hong Kong · Tenlai Team

This repository introduces PronCFT and releases tools for constructing misspelled test variants and scoring spelling recall.

## Overview

Large language model (LLM) based automatic speech recognition (ASR) can replace rare or out-of-vocabulary (OOV) words with familiar words that sound similar. PronCFT teaches the model to connect speech, candidate pronunciations, and output spellings through phoneme transcription and candidate matching. Phonemes are symbols representing speech sounds.

Spelling augmentation changes candidate spellings while retaining their stored pronunciations. A training curriculum then progressively removes intermediate outputs, producing PronCFT-C.

## Main Results and Training Setup

All three systems use **Qwen3-ASR-1.7B** and are fine-tuned on **GigaSpeech**. The audio encoder and adaptor are frozen; only the LLM is updated. The common fine-tuning setup uses one epoch, a global batch size of 384, a maximum sequence length of 2,048, and a learning rate from 1e-5 to 1e-6.

| System | How it is trained |
|---|---|
| **Giga-SFT** | Baseline supervised fine-tuning on speech transcription, without phoneme annotations or auxiliary output targets. |
| **PronCFT** | Adds supervision for utterance phonemes, candidate matching, and final transcription. All context candidates carry pronunciations. Structured auxiliary targets are included in 50% of training samples; 65% of those samples also use spelling augmentation with the stored phonemes unchanged. |
| **PronCFT-C** | Curriculum-trained version of PronCFT. Starting at 20k updates, five stages (L0–L4), each lasting 1.5k updates, progressively mask/remove phoneme outputs, simplify matching outputs, and reach transcription-only output. After L0, loss is computed only on transcription tokens; audio, context prompts, and spelling augmentation are kept fixed. |

The OOV evaluation set contains **10,533 sentences**. OOV targets are absent from CMUdict, a pronunciation dictionary, and do not overlap with the training vocabulary. Systems receive the same seeded candidate lists.

| System | OOV recall: transcription only, no phonemes | OOV recall: transcription only, with phonemes | OOV recall: structured output, with phonemes | General ASR average WER |
|---|---:|---:|---:|---:|
| Giga-SFT | 59.89% | — | — | 6.05% |
| PronCFT | 61.46% | 65.45% | 74.12% | 6.02% |
| PronCFT-C | 61.08% | 70.64% | **83.57%** | **6.01%** |

OOV recall measures recovery of the target words; higher is better. Word error rate (WER) measures transcription errors; lower is better. The recall columns preserve the evaluation labels in Table 2; average WER comes from Table 3 and is the macro-average across six test sets.

The best reported OOV recall improves from **59.89% to 83.57% (+23.68 percentage points)**. These are paper results; the columns distinguish decoding conditions rather than comparing every system under one output format.

## Released Data Tools

| File | Purpose |
|---|---|
| [make_misspell_phonectx.py](make_misspell_phonectx.py) | Generate spelling variants of plain transcripts, or perturb candidate spellings while keeping the reference transcript unchanged. |
| [build_misspell_cot.py](build_misspell_cot.py) | Generate variants of text with `<phoneme>/<match>` annotations and synchronize transcript and target labels. |
| [compute_misspell_recall.py](compute_misspell_recall.py) | Count replacement-only, original-only, both, neither, and missing-prediction outcomes. |
| [misspell_rules.py](misspell_rules.py) / [misspell_io.py](misspell_io.py) | Spelling edits, vocabulary filtering, text replacement, and JSONL input/output. |
| [examples/](examples/) / [test_misspell.py](test_misspell.py) | Synthetic examples and unit tests. |

The tools take prepared transcripts, candidates, phonemes, and target labels. They preserve supplied phoneme labels rather than verifying that a new spelling naturally has the same pronunciation.

This release contains partial data tools. Full test sets, model weights, and training/inference implementations are not included.

## Quick Start

Python 3.8+ is recommended. No third-party packages are required. Run these commands from the repository root; output files must not already exist.

```bash
# Change candidate spellings and synchronize transcripts and target labels.
python3 make_misspell_phonectx.py --in examples/plain.jsonl --out plain_misspell.jsonl --vocab examples/vocab.txt --prob 0.5 --seed 42

# Change only target candidate spellings; keep the reference transcript.
python3 make_misspell_phonectx.py --in examples/plain.jsonl --out context_misspell.jsonl --vocab examples/vocab.txt --targets-only --context-only --seed 42

# Run tests on synthetic records.
python3 -B -m unittest -v test_misspell
```

The example vocabulary is for demonstration. Use an appropriate exclusion vocabulary for actual experiments.

See the [tool guide](docs/data-tools.md) for formats, annotated variants, and scoring, and [evaluation and release scope](docs/evaluation.md) for how the released tools relate to the experiments. These detailed guides are currently in Chinese.

## Citation

```bibtex
@misc{xu2026proncft,
  title={PronCFT: Pronunciation Contextual Fine-Tuning for Contextual Biasing of Out-of-Vocabulary Words in LLM-Based ASR},
  author={Yifan Xu and Kaixun Huang and Mengzhe Geng and Mingyu Cui and Shen Huang and Shidong Shang and Xunying Liu},
  year={2026},
  eprint={2610.25766},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
```

A machine-readable citation is available in [CITATION.cff](CITATION.cff).

## License

Existing copyright notices are retained. A code license has not yet been provided. Third-party speech datasets and vocabularies remain subject to their respective licenses.
