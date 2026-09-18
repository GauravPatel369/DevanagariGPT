# Monolingual Transformer LMs — Hindi (Model H) & Nepali (Model L)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

**Individual Project** | Language Models and Agents (Monsoon 2026)
**Author**: Gaurav Patel
**Repository Branch**: `phase-3`

---

## Executive Overview

Two completely independent monolingual decoder-only Transformer language models, built from PyTorch
primitives and trained from scratch:

- **Model H**: Hindi (`hi`, higher-resource Devanagari)
- **Model L**: Nepali (`ne`, lower-resource Devanagari)

They share no data, tokenizer, vocabulary or weights. No pretrained model, no pretrained tokenizer,
no HuggingFace Transformer class, and no pre-built attention block is used anywhere. Attention,
positional encoding (RoPE) and causal masking are all implemented directly.

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| **Parameters** | **25,350,912** | **25,350,912** |
| Architecture | 7 layers, d_model 512, 8 heads, d_ff 1792, RoPE, pre-norm, tied embeddings | identical |
| Vocabulary | Unigram 10,000 (`hi`) | Unigram 10,000 (`ne`) — separate |
| Training tokens | 450,667,252 (1 epoch) | 453,282,358 (1 epoch) |
| Training time | 5.73 h, Kaggle Tesla P100 | 5.74 h, Kaggle Tesla P100 |
| **Test perplexity** | **24.39** | 29.87 |
| **Test bits-per-byte** | 0.4709 | **0.4600** |
| Test tokens scored | 51,399,168 | 56,785,408 |

**The two intrinsic metrics disagree, and that disagreement is the project's central finding.**
Perplexity favours Hindi; bits-per-byte favours Nepali. Because perplexity is measured per *token*
and the two tokenizers differ, only bits-per-byte is comparable. Section 8 of the Phase 2 report
tests this under a control that removes the corpus confound entirely — six models trained on
byte-identical text with only the tokenizer varying — and the reversal reproduces in both languages.

### Phase 1 corpus summary

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Clean total tokens | 554,935,719 | 566,889,900 |
| Manual scraped tokens | 113,763,877 (>= 100M) | 109,634,743 (>= 100M) |
| Train manual share | 20.43% (>= 20%) | 20.00% (>= 20%) |
| Tokenizer | Unigram 10,000 | Unigram 10,000 |
| Fertility | 1.34 tokens/word | 1.59 tokens/word |
| Compression | 3.79 chars/token | 4.03 chars/token |

---

## Deliverables & Google Drive Artifacts Index

Checkpoints are on Drive per the brief's instruction not to commit large binaries. **Every figure,
table and metric that is graded lives inside this repository**, under `report/`.

### Phase 3 — finetuned reasoning models

| Artifact | Local Repository Location | Google Drive |
|---|---|---|
| Model H — Hindi finetuned (final) | `hindi/reasoning/checkpoints/best.pt` | [Phase 3 Hindi Finetuned Checkpoints](https://drive.google.com/drive/folders/1Lr87SvS90iGCk9hJgFYK9_xQC6mwFls8) |
| Model L — Nepali finetuned (final) | `nepali/reasoning/checkpoints/best.pt` | [Phase 3 Nepali Finetuned Checkpoints](https://drive.google.com/drive/folders/1sZsSFSZOv9eo3TyLx0h3Rka4nC6hwsNy) |

`best.pt` is the selected model. The Hindi `best.pt` also stores optimizer state, so finetuning can resume from it; each language also has a resumable `last.pt`.

### Phase 2 — model checkpoints

Each `.pt` holds model weights, optimizer state, training step, model config, and RNG streams, so a
run resumes as a continuation rather than a restart.

| Artifact | Local Repository Location | Google Drive |
|---|---|---|
| **Model H — Hindi pretrained** (304 MB) | `hindi/checkpoints/last.pt` | [5_CHECKPOINTS/hindi](https://drive.google.com/drive/folders/1-iqL5RWhZAFUwB5kLRZImCbTz6z748hc) |
| **Model L — Nepali pretrained** (304 MB) | `nepali/checkpoints/last.pt` | [5_CHECKPOINTS/nepali](https://drive.google.com/drive/folders/1ZMgUWLVdShQk-UDRFPk16hQfx57gmiaZ) |
| **Ablation — Hindi, no positional encoding** (304 MB) | `hindi_norope/checkpoints/last.pt` | [5_CHECKPOINTS/ablation](https://drive.google.com/drive/folders/1FROyYzDpGYg9SHkzHnDLwLhxFsUkYeVE) |

### Phase 1 — data and tokenizers

| Artifact | Local Repository Location | Google Drive |
|---|---|---|
| **Hindi Raw Data Shards** | `hindi/data/raw/` | [0_RAW_DATA/hindi_raw_shards](https://drive.google.com/drive/folders/1DbIJLjDIwjB92CIbKfiOn0t--3RRv3Ej?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto) |
| **Nepali Raw Data Shards** | `nepali/data/raw/` | [0_RAW_DATA/nepali_raw_shards](https://drive.google.com/drive/folders/1b3e3ooC9BuYomN_5NDPwwPfF-SBN0BQk?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto) |
| **Hindi Cleaned Shards** | `hindi/data/dedup_combined/` | [1_CLEAN_DATA/hindi_clean_shards](https://drive.google.com/drive/folders/1pglxQA_HGYy4dzPxctyEOSosniFm37YI?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto) |
| **Nepali Cleaned Shards** | `nepali/data/dedup_combined/` | [1_CLEAN_DATA/nepali_clean_shards](https://drive.google.com/drive/folders/19RItUqwtElCVv_r8fbW69oyaMNvwuLVy?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto) |
| **Hindi Tokenizer** | `hindi/tokenizer/hi_unigram_10000.model` | [4_TOKENIZERS/hi_unigram_10000.model](https://drive.google.com/file/d/1NxHPLY6ftYCRQC0z0P39I_0KE8Ov3F2A/view?usp=drive_link) |
| **Nepali Tokenizer** | `nepali/tokenizer/ne_unigram_10000.model` | [4_TOKENIZERS/ne_unigram_10000.model](https://drive.google.com/file/d/1GvkdBjl-mH0tdjcjaiE-1nIxrYlLnyYT/view?usp=drive_link) |

### Reports (in-repo, graded)

| | |
|---|---|
| **Phase 3 report** | [`report/Phase 3/phase3.md`](report/Phase%203/phase3.md) |
| **Phase 2 report** | [`report/Phase 2/phase2.md`](report/Phase%202/phase2.md) — 11 sections, 11 figures |
| **Phase 1 report** | [`report/Phase 1/phase1.md`](report/Phase%201/phase1.md) |

---

## Phase 2 Results

### Intrinsic language modelling (complete test split, sequential non-overlapping windows)

| Metric | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Tokens scored | 51,399,168 | 56,785,408 |
| Cross-entropy (nats/token) | 3.1942 | 3.3968 |
| **Perplexity** | **24.39** | 29.87 |
| Bytes per token | 9.7857 | 10.6530 |
| **Bits per byte** | 0.4709 | **0.4600** |
| Compression vs raw text | 16.99x | 17.39x |

### Generation quality (best of 12 decoding settings)

| | Model H | Model L |
|---|---|---|
| Best BLEU (char-tokenized) | 23.87 | 24.81 |
| Best chrF++ | 21.80 | 21.00 |
| Sweet spot | T=1.0 + top-p 0.95 | T=1.0 |
| Greedy repetition rate (4-gram) | 0.756 | 0.733 |
| Human reference repetition rate | 0.009 | 0.002 |

Greedy decoding degenerates in **every one of the nine models trained** — repetition two orders of
magnitude above human text — and `no_repeat_ngram_size=4` drives it to ~0 every time.

### Experiments beyond the two required models

| Experiment | Finding | Report |
|---|---|---|
| **Vocabulary sweep** (6 models, V = 5k/8k/10k, both languages, parameters held at ~25.4M) | Perplexity and bits-per-byte rank the models in **opposite orders**, in both languages. Phase 1's V=10,000 choice is vindicated. | Section 8 |
| **Bonus ablation** — no positional encoding | +13.7% perplexity at an *identical* parameter count; layers 4-5 reorganise into short-range order detectors | Section 9 |
| **Second epoch** on Model H | 24.39 -> 23.02 perplexity; quantifies the one-epoch budget's cost | Section 10 |

---

## Phase 3 Results

Each pretrained model was finetuned to answer comparison questions in its own language ("A is taller than B, B is taller than C. Who is the shortest?"), on a generated dataset with separate names, numbers and sentence patterns for training and testing. Full details are in the Phase 3 report.

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Training questions | 20k (chosen on validation) | 10k (chosen on validation) |
| Final recipe | stop early on new-name validation | large name pool + replay |
| test_b choice accuracy, pretrained / final (chance 38.5%) | 39.5% / 37.8% | 38.8% / 41.2% |
| test_b written accuracy, pretrained / final | 15.5% / 33.0% | 4.0% / 10.2% (26.8% with endings) |
| Test perplexity, pretrained / final | 23.02 / 23.81 | 29.87 / 30.81 |

The main findings:

- The first dataset had shortcuts (the answer was always the first name in relation questions), and both models learned them. The generator was fixed, now checks itself for position shortcuts, and every experiment was rerun.
- On the clean data neither model learned the comparison: choice accuracy stays at chance for every data size and technique. The finetuned models mainly answer with the person named last.
- Plain finetuning makes models answer with memorised training names and costs up to 25% perplexity. Early stopping, a large name pool and replay reduce wrong names and keep perplexity within 3% of pretrained.
- Nepali copies names with their attached case endings ("योगेशको"), which lowers its strict written score; allowing the ending closes most of the gap to Hindi.

---

## Repository Structure

```
├── lmagpt/                      # Phase 2: the model and its training/eval stack
│   ├── model.py                 # Decoder-only Transformer from primitives; RoPE; 6 decoders
│   ├── data.py                  # Memmap token dataset, target shifting, batching
│   ├── train.py                 # AdamW, cosine schedule, resume-capable checkpointing
│   ├── evaluate.py              # Perplexity/BPB, BLEU/chrF++/ROUGE-L, diversity, sweet spot
│   ├── attention.py             # Entropy, mean distance, head classification, heatmaps
│   └── kaggle_bootstrap.py      # P100 (sm_60) torch compatibility shim
├── lmacorpus/                   # Phase 1: language-agnostic corpus pipeline
│   ├── acquire/                 # Crawling, sitemap parsing, HF streaming
│   ├── clean/                   # NFC normalisation, quality filters, 6-class Devanagari LID, dedup
│   ├── tokenizer/               # SentencePiece training and encoding
│   └── splits.py                # Group-aware deterministic splits
├── hindi/ , nepali/             # Self-contained per-language data, configs, tokenizers, checkpoints
│   └── reasoning/               # Phase 3: templates, question data, results, finetuned checkpoints
├── hindi_norope/                # Bonus ablation run
├── hindi_2ep/                   # Two-epoch continuation of Model H
├── kaggle/                      # Headless training kernels
├── scripts/                     # Operational scripts (see Reproduction below)
├── tests/                       # 63 unit tests
└── report/
    ├── Phase 1/                 # Corpus report, tables, 9 figures
    ├── Phase 2/                 # Model report, evaluation JSONs, logs, 11 figures
    └── Phase 3/                 # Reasoning report, attention results, figures
```

---

## Quickstart & Reproduction

### 1. Environment

```bash
git clone https://github.com/Language-Models-and-Agents-2026/individual-project-GauravPatel369.git
cd individual-project-GauravPatel369
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Unit tests (63)

```bash
python -m pytest tests/ -q
```

Includes the brief's required empirical causal-mask verification: changing token *t+1* leaves the
logits at position *t* bit-identical, post-softmax mass above the diagonal is exactly 0.0, and
position 0 attends only to itself.

### 3. Train

```bash
# locally
python -m lmagpt.train --lang hindi
python -m lmagpt.train --lang nepali

# headless on Kaggle (push / poll / fetch, resuming across sessions)
python scripts/kaggle_run.py train  --lang hindi --fresh
python scripts/kaggle_run.py status
python scripts/kaggle_run.py fetch  --lang hindi
```

Training stops on a wall-clock budget and exits with a loadable checkpoint, so an interrupted
session always resumes rather than restarting. Re-running the same command continues from
`last.pt`.

### 4. Evaluate

```bash
# intrinsic + generation, complete test split, all 12 decoding settings
python -m lmagpt.evaluate --lang both --split test --n-samples 128

# attention: entropy, mean distance, head classification, heatmaps
python -m lmagpt.attention --lang both --stats-tokens 256 --stats-passages 16
```

### 5. Figures

```bash
python scripts/plot_training_curves.py     # loss, perplexity, LR/throughput, gradient norm
python scripts/plot_vocab_sweep.py         # vocabulary sweep panels
python scripts/make_figures.py             # Phase 1 corpus figures
```

### 6. Interactive generation

```bash
python scripts/generate.py --lang hindi
```

Opens a REPL where decoding settings change mid-session (`:set T=0.8`, `:set top_p=0.95`), logging
every turn with the exact configuration that produced it.

### 7. Phase 1 data pipeline

```bash
python scripts/run_strict_pipeline_and_eval.py   # ingest -> clean -> dedup -> split -> tokenize
python scripts/test_tokenizer.py --lang both     # qualitative tokenizer evaluation
python -m lmacorpus.cli encode --lang hindi      # re-encode the uint16 memmaps
python scripts/smoke_test.py                     # end-to-end synthetic pipeline test
```

### 8. Phase 3 reasoning finetuning

```bash
# data (stops if the shortcut check fails)
python scripts/make_reasoning_data.py --lang hindi
python scripts/make_reasoning_data.py --lang nepali
python scripts/make_ablation_data.py --lang hindi
python scripts/make_ablation_data.py --lang nepali

# Kaggle: sweep, techniques, final repeats (Nepali adds --config-dir <second account>)
python scripts/kaggle_run.py sweep --lang hindi
python scripts/kaggle_run.py ablation --lang hindi --size 20000 --runs baseline ood_val ood_val_2ep name_pool replay name_pool_replay
python scripts/score_val_ood.py --lang hindi

# analysis
python scripts/shortcut_probe.py --lang hindi
python scripts/shortcut_audit.py --lang hindi
python scripts/score_with_endings.py --lang nepali
python scripts/attention_reasoning.py
python scripts/plot_ablation.py

# ask your own question (reads input.txt, writes output.txt)
python scripts/ask.py --lang hindi --checkpoint hindi/reasoning/checkpoints/best.pt
```

The complete command list is in Section 11 of the Phase 3 report.

---

## Key Technical Decisions

**RoPE over a learned positional table.** Costs zero parameters (a learned table would need 262,144),
encodes position *relatively* so the model never learns separately that positions 300 and 301 are
adjacent, and imposes no hard sequence-length ceiling. The parameters saved are what fund a seventh
layer inside the ~25M budget.

**d_ff at 3.5x d_model, not the conventional 4x.** At 4x, seven layers cost 27.19M — 8.8% over
target. Trimming to 3.5x lands at exactly 25,350,912. Depth was bought with FFN width.

**Tied embeddings.** Saves 5,120,000 parameters, a fifth of the budget. One consequence found while
testing: the tied output projection makes predicting the *current* token trivially easy, so a
dataloader that forgets to shift targets produces a beautiful loss curve and a model that learned
nothing. Pinned by a permanent test.

**Attention written out explicitly, not via `F.scaled_dot_product_attention`.** Partly because the
brief forbids pre-built attention, but also because fused kernels never materialise the
`(B, h, T, T)` weight matrix — and that matrix is exactly what the attention analysis measures.

**Bits-per-byte as the comparison metric.** Perplexity is per token and the two tokenizers differ,
so it is not comparable across models. This is not a technicality: the two metrics rank the models
in opposite orders, and the vocabulary sweep shows the reversal is caused by tokenization alone.

**Group-aware deterministic splits** (Phase 1). Document-level partitioning by Blake2b hash of
`(source, publication_month)`, 80/10/10, guaranteeing no leakage between splits — and the tokenizers
are trained strictly on `train.txt`.
