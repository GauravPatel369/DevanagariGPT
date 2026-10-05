# Phase 1: Code Implementation & Conceptual Architecture Guide

**Project**: Monolingual Transformer Language Models — Hindi (`hi`) & Nepali (`ne`)  
**Course**: Language Models and Agents (Monsoon 2026)  
**Author**: Gaurav Patel  
**Repository Branch**: `phase-1` / `phase-3`

---

## 1. Overview & Purpose of this Guide

This document provides a complete, dual-perspective reference for **Phase 1 (Data Engine, Quality Pipeline & Tokenization Infrastructure)**:
1. **"Where is implemented what"**: A precise codebase map showing the exact Python files, functions, classes, and scripts responsible for each stage.
2. **"What does that mean in that way"**: The conceptual intuition, mathematical definitions, linguistic principles, and engineering rationales behind every choice.

```
                                 END-TO-END PHASE 1 DATA PIPELINE
                                 
   [ Public Web & Archives ]             [ Manual Scraping (>= 20%) ]
      (Wikipedia, Sangraha,                  (News Portals, Literature,
       IndicCorp, CC-100)                     Government Archives)
              │                                        │
              ▼                                        ▼
    lmacorpus.acquire.hf_stream              lmacorpus.acquire.crawler
              │                                        │
              └───────────────────┬────────────────────┘
                                  │
                                  ▼
                         Stage 1 & 2: Raw Storage
                           (<lang>/data/raw/)
                                  │
                                  ▼
                     Stage 3: Unicode Normalization
                      lmacorpus.clean.normalize
                  (NFC, Nukta, Deva Digits, Invisible)
                                  │
                                  ▼
                     Stage 4: Quality & Heuristics
                       lmacorpus.clean.filters
               (Script ratio >= 0.75, Gopher heuristics)
                                  │
                                  ▼
                      Stage 5: Language Identification
                        lmacorpus.clean.langid
              (6-Class FastText/Logistic: hi, ne, mr, bh, mai, sa)
                                  │
                                  ▼
                        Stage 6: Deduplication
                         lmacorpus.clean.dedup
             (Exact hash -> MinHash LSH -> Cross-Corpus Purge)
                                  │
                                  ▼
                     Stage 7: Group-Aware Partitioning
                          lmacorpus.splits
                (blake2b hash of (source, month) -> train/val/test)
                                  │
                                  ▼
               Stage 8: Tokenizer Training & Evaluation
                     lmacorpus.tokenizer.train_spm
                    lmacorpus.tokenizer.vocab_sweep
            (Unigram 10k, Byte Fallback, Fertility & Compression)
                                  │
                                  ▼
                      Stage 9: Binary Tokenization
                   lmacorpus.tokenizer.encode_corpus
                   (uint16 memmap arrays: train.bin, val.bin)
```

---

## 2. Codebase Blueprint: Where is Implemented What?

The table below maps every Phase 1 functional requirement to its corresponding implementation in the repository:

| Pipeline Component | Module / File Path | Core Functions / Classes | CLI Command / Entry Point | Artifacts & Outputs Produced |
|---|---|---|---|---|
| **Manual Web Scraping** | [`lmacorpus/acquire/crawler.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/crawler.py) | `NewsScraper`, `fetch_sitemap()`, `crawl()` | `python -m lmacorpus acquire-manual --lang {hi,ne}` | `<lang>/data/raw/manual/*.jsonl.zst` |
| **URL Crawl Frontier** | [`lmacorpus/acquire/frontier.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/frontier.py) | `Frontier`, `seen_url()`, `enqueue()` | Internal to crawler | Crawl state database |
| **HTML Content Extraction** | [`lmacorpus/acquire/extract.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/extract.py) | `extract_article()`, Trafilatura wrapper | Internal to crawler | Clean body text extracted from raw HTML |
| **Public Dataset Streaming** | [`lmacorpus/acquire/hf_stream.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/hf_stream.py) | `stream_dataset()`, `iter_sangraha()`, `iter_indiccorp()` | `python -m lmacorpus acquire-public --lang {hi,ne}` | `<lang>/data/raw/downloaded/*.jsonl.zst` |
| **Stage 3: Normalization** | [`lmacorpus/clean/normalize.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/normalize.py) | `normalize()`, `strip_invisible()`, `filter_english_statements()` | `python -m lmacorpus clean --lang {hi,ne}` | Clean Unicode NFC, decomposed Nukta text |
| **Stage 4: Quality Filtering** | [`lmacorpus/clean/filters.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/filters.py) | `QualityFilter`, `FilterConfig`, `mean_word_len()`, `frac_lines_terminal()` | Called inside `lmacorpus.pipeline.run_clean()` | Rejection reports by reason & source |
| **Stage 4: Boilerplate Removal** | [`lmacorpus/clean/boilerplate.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/boilerplate.py) | `make_blocklist()`, `strip_boilerplate()` | Called inside `lmacorpus.pipeline.run_clean()` | Site navbars, copyright disclaimers removed |
| **Stage 5: Language ID** | [`lmacorpus/clean/langid.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/langid.py) | `LanguageIdentifier`, `train_classifier()`, `predict()` | `python -m lmacorpus train-langid` | `models/langid.pkl`, `langid_confusion.png` |
| **Stage 6: Deduplication** | [`lmacorpus/clean/dedup.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/dedup.py) | `signature()`, `dedup_exact()`, `dedup_minhash()`, `dedup_cross_corpus()` | `python -m lmacorpus dedup --lang {hi,ne}` | Deduplicated document shards |
| **Stage 7: Partitioning** | [`lmacorpus/splits.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/splits.py) | `group_key()`, `assign()`, `build_splits()` | `python -m lmacorpus make-splits --lang {hi,ne}` | `train.manifest`, `val.manifest`, `test.manifest` |
| **Corpus Accounting** | [`lmacorpus/stats.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/stats.py) | `compute_corpus_stats()`, `verify_compliance()` | `python -m lmacorpus stats --lang {hi,ne}` | `<lang>/data/tokens/meta.json` |
| **Tokenizer Training** | [`lmacorpus/tokenizer/train_spm.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/tokenizer/train_spm.py) | `train_spm()`, `write_training_sample()` | `python -m lmacorpus train-spm --lang {hi,ne}` | `<lang>/tokenizer/*.model`, `*.vocab` |
| **Hyperparameter Sweeps** | [`lmacorpus/tokenizer/vocab_sweep.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/tokenizer/vocab_sweep.py) | `sweep_vocab_sizes()`, `evaluate_tokenizer()` | `python -m lmacorpus vocab-sweep --lang {hi,ne}` | `report/Phase 1/{Hindi,Nepli}_report.json` |
| **Binary Tokenization** | [`lmacorpus/tokenizer/encode_corpus.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/tokenizer/encode_corpus.py) | `encode_split_to_bin()`, `ShardTokenizer` | `python -m lmacorpus encode --lang {hi,ne}` | `train.bin`, `val.bin`, `test.bin` (`uint16`) |
| **Phase 1 Figure Generation** | [`scripts/make_figures.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/make_figures.py) | `plot_manual_fractions()`, `plot_vocab_sweeps()`, `plot_zipf()` | `python scripts/make_figures.py` | `report/Phase 1/figures/*.png` |
| **Phase 1 Table Generation** | [`scripts/build_report_tables.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/build_report_tables.py) | `generate_split_tables()`, `generate_tokenizer_tables()` | `python scripts/build_report_tables.py` | `report/Phase 1/generated_tables.md` |

---

## 3. Deep-Dive: What Does That Mean in That Way?

This section explains the **conceptual meaning**, **linguistic rationale**, and **mathematical mechanics** behind every single stage and metric in Phase 1.

---

### Stage 1 & 2: Ingestion & The Manual Collection Constraint

#### 1. What does "Manual Collection ($\ge 20\%$)" mean?
- **The Requirement**: At least $20\%$ of the training corpus tokens must come from custom, directly scraped web sources rather than standard pre-packaged machine learning archives (like HuggingFace Common Crawl dumps).
- **Where it is implemented**:
  - Scrapers: [`lmacorpus/acquire/crawler.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/crawler.py)
  - Verification: [`lmacorpus/stats.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/stats.py#L40-L75)
- **Why this was done**: Pre-packaged datasets (e.g. CC-100, mC4) suffer from machine-translation artifacts, spam, and severe duplicate boilerplate. Writing custom scrapers targeting reputable regional publishers (Dainik Jagran, Amar Ujala, BBC Hindi, Kantipur, Setopati, Ratopati) ensures high-quality contemporary linguistic register.
- **The Numbers Verified in `meta.json`**:
  - **Hindi Train Manual Fraction**: **$20.43\%$** ($92,091,238$ manual tokens / $450,667,252$ total train tokens) $\rightarrow$ **PASS**.
  - **Nepali Train Manual Fraction**: **$20.00\%$** ($90,673,349$ manual tokens / $453,282,358$ total train tokens) $\rightarrow$ **PASS**.

#### 2. Why document-level metadata (`source`, `url`, `timestamp`)?
- Every document is stored using a strict schema ([`lmacorpus/schema.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/schema.py)) preserving:
  - `doc_id`: Deterministic 64-bit content hash.
  - `source`: Domain identifier (e.g. `kantipur`, `amarujala`).
  - `source_type`: Either `"manual"` (scraped by us) or `"downloaded"` (public archive).
  - `url` & `timestamp`: Required for group-aware temporal splitting (Stage 7).

---

### Stage 3: Normalization & Character Sanitization

**File**: [`lmacorpus/clean/normalize.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/normalize.py)

#### 1. What does Unicode NFC and Nukta Decomposition mean?
- In Devanagari, letters with a nukta dot (e.g., क़, ख़, ग़, ज़, ड़, ढ़, फ़) can be represented in two ways:
  1. **Single pre-composed code point**: e.g., क़ (`U+0958`).
  2. **Decomposed two-character sequence**: Base consonant क (`U+0915`) + Nukta sign ़ (`U+093C`).
- **Why NFC decomposes Nukta**: Under Unicode standard NFC (Normalization Form Canonical Composition), characters `U+0958–U+095F` sit on the Unicode *Composition Exclusion List*. Therefore, applying `unicodedata.normalize("NFC", text)` explicitly **decomposes** them into `Base + U+093C`.
- **Linguistic Meaning**: If not normalized, the tokenizer would treat `क़` and `क + ़` as two completely distinct vocabulary pieces, causing duplicate token IDs and splitting identical words.

#### 2. Why are Anusvara (ं) and Chandrabindu (ँ) NOT merged?
- Some naive NLP scripts fold Chandrabindu (ँ, `U+0901`) into Anusvara (ं, `U+0902`).
- **Linguistic Meaning**: In Hindi and Nepali, Anusvara represents a homorganic nasal consonant or vowel nasalization, while Chandrabindu denotes pure phonemic vowel nasalization. For example, **हंस** (*hans*, swan) vs. **हँस** (*hans*, laugh). Merging them destroys authentic phonemic and semantic contrast.
- **Decision**: Left strictly untouched. Halants (्) and Visarga (ः) are likewise preserved.

#### 3. What does Invisible & Control Character Stripping mean?
- Zero-Width Joiners (ZWJ, `\u200d`) and Zero-Width Non-Joiners (ZWNJ, `\u200c`) are used in Devanagari typesetting to control conjunct rendering (e.g. explicit halant vs half-consonant).
- However, web crawls contain huge volumes of stray invisible formatting artifacts (soft hyphens `\u00ad`, byte-order marks `\ufeff`, bidi overrides `\u200e`). Function `strip_invisible()` purges them and records counts per source as a data hygiene metric.

#### 4. Why map Devanagari digits (०-९) to ASCII digits (0-9)?
- `normalize()` applies `text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))`.
- **Conceptual Meaning**: Hindi and Nepali web text unpredictably mixes native Devanagari numerals (`४२`) and Latin numerals (`42`). Mapping them to standard ASCII digits reduces vocabulary fragmentation by saving 10 vocabulary slots and ensuring numeric facts are represented consistently.

---

### Stage 4: Quality & Heuristic Filtering

**File**: [`lmacorpus/clean/filters.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/filters.py)

#### 1. What does "Letter-Aware Script Filtering" mean?
- Standard python `str.isalpha()` fails on Devanagari non-spacing combining vowel signs (Matras like ा, ि, ी, ु, ू) and combining marks (Halant, Anusvara) because Unicode categorizes them as `Mn` (Nonspacing Mark) or `Mc` (Spacing Mark), not `L` (Letter).
- Naive alphabetic checks reject valid Devanagari words like *"किताब"* because `ि` (`U+093F`) is not categorized under `str.isalpha()`.
- **Implementation**: Function `script_ratio(text)` in [`normalize.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/normalize.py#L115-L125) explicitly matches `[\u0900-\u097F]`, ensuring all consonants, vowels, nuktas, halants, and matras count toward the Devanagari script ratio. Documents with Devanagari script ratio $< 0.75$ are rejected.

#### 2. What are Gopher-Style Quality Heuristics?
- `FilterConfig` enforces:
  - `min_chars = 200`: Drops short fragments, headlines without content, and stub pages.
  - `min_word_len = 1.5` & `max_word_len = 12.0`: Catches run-together text without whitespace (e.g. HTML without spacing) or single-character noise dumps.
  - `min_terminal_frac = 0.20`: At least 20% of lines must end in a sentence terminal (`।`, `॥`, `?`, `!`). Catches menu bars, product catalogs, and raw link dumps.
  - `max_dup_lines = 0.30` & `max_dup_ngrams = 0.20`: Drops documents where repetitive sentences or 5-gram phrases exceed thresholds (e.g. SEO link farms, site footers).

---

### Stage 5: Six-Class Devanagari Language Identification

**File**: [`lmacorpus/clean/langid.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/langid.py)

#### 1. Why 6 classes instead of a binary Hindi vs. Nepali classifier?
- Classes: **Hindi (`hi`)**, **Nepali (`ne`)**, **Marathi (`mr`)**, **Bhojpuri (`bh`)**, **Maithili (`mai`)**, and **Sanskrit (`sa`)**.
- **The Problem**: All six languages share the identical Devanagari Unicode block (`U+0900–U+097F`). If you train a binary classifier ($P(\text{hi}) + P(\text{ne}) = 1.0$), any Bhojpuri, Maithili, or Marathi document will be forcefully assigned to either Hindi or Nepali with high false confidence!
- **Conceptual Meaning**: By training on all six major Devanagari languages, the classifier accurately routes non-target languages away.

#### 2. Why character 3–5-gram TF-IDF + Logistic Regression?
- Consistent with the course mandate, **zero pretrained checkpoints** (such as external FastText or XLM-R) were used. The model was trained completely from scratch on Wikipedia paragraph samples.
- Character 3–5-grams capture subword inflectional morphemes that uniquely discriminate closely related languages:
  - **Nepali markers**: `छ`, `छन्`, `थियो`, `भन्दा`, `ले`, `लाई`, `गरेको`, `र`.
  - **Hindi markers**: `है`, `हैं`, `था`, `और`, `से`, `ने`, `को`, `किया`.
- **Threshold**: Only documents with predicted class probability $P(\text{target}) \ge 0.90$ are accepted. Borderline documents are quarantined in `data/quarantine/` for audit rather than silently deleted.

---

### Stage 6: Deduplication Engine

**File**: [`lmacorpus/clean/dedup.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/dedup.py)

#### 1. The Three Sequential Passes:
1. **Exact Deduplication**: Hashes the full document text via 64-bit blake2b (`doc_id`). Drops bit-identical copies at zero computational cost.
2. **Near-Duplicate Detection (MinHash + LSH)**: Catches syndicated news articles, reposts with slight editorial changes, or differing timestamps.
3. **Cross-Corpus Deduplication**: Indexes the entire Hindi corpus into an LSH index, then queries every Nepali document against it.

#### 2. Why Character 5-grams (Shingles) instead of Word n-grams?
- In morphologically rich languages like Nepali (which attaches bound clitics like `-लाई`, `-बाट`), two sentences can be semantically and lexically identical, but differ slightly in orthographic surface form due to inflectional variation.
- Word shingles treat inflected forms as totally distinct words, missing near-duplicates. Character 5-grams with stride 2 capture substring overlap across inflections, providing robust Jaccard similarity estimation.

#### 3. What does "MinHash with 128 Permutations" mean?
- MinHash estimates the Jaccard similarity between two shingle sets:
  $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
- Using 128 independent hash functions generates a compact 128-integer signature per document. Locality-Sensitive Hashing (LSH) bands these signatures into buckets, allowing near-duplicate detection in $O(N)$ time instead of an impossible $O(N^2)$ all-pairs comparison.
- **Threshold**: Jaccard similarity $\ge 0.80$.

#### 4. Why are Cross-Corpus Duplicates dropped from BOTH sides?
- If a document appears in both the Hindi and Nepali collections (common in crawled regional news covering national events), keeping it on either side leaves cross-lingual leakage in place. Purging it from **both** corpora guarantees absolute zero overlap between Model H and Model L.

---

### Stage 7: Group-Aware Partitioning

**File**: [`lmacorpus/splits.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/splits.py)

#### 1. Why document-level, never line-level?
- Line-level splitting places consecutive sentences from the same news story into both `train` and `test`. The model would then be evaluated on text whose immediate context it memorized during training, making held-out perplexity artificially low and meaningless.

#### 2. What does "Group-Aware Splitting" mean?
- **The Group Key**: `group_key(doc) = f"{doc.source}|{publication_month}"`.
- **The Problem**: Different news outlets cover the exact same breaking news event on the same day using nearly identical language. If documents are partitioned purely at random, stories about the same event scatter across both train and test.
- **The Solution**: Splitting on the hash of `(source, publication_month)` guarantees that all documents from a specific outlet published in a given month are assigned **together** to the same partition.
- **Deterministic Hashing**: The assignment uses `blake2b(f"lma|{group_key}".encode())` mapped to $[0, 1)$. It is a pure mathematical function of the metadata—no random seeds, 100% reproducible across re-runs.

---

### Stage 8: Tokenizer Engine & Hyperparameter Sweeps

**Files**: [`lmacorpus/tokenizer/train_spm.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/tokenizer/train_spm.py), [`vocab_sweep.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/tokenizer/vocab_sweep.py)

#### 1. What does "Fertility" (Tokens per Word) mean?
- **Definition & Formula**:
  $$\text{Fertility} = \frac{\text{Total Subword Tokens Generated}}{\text{Total Whitespace-Delimited Words}}$$
- **Linguistic Meaning**: Measures how many subword fragments an average word is broken into.
  - A fertility of $1.0$ would mean every single word is an intact token.
  - Higher fertility means words are fragmented into smaller pieces, increasing sequence length and taxing the model's 512-token context window.
- **Measured Values on Held-Out Validation (`val.txt`)**:
  - **Hindi (Model H)**: **$1.34$ tokens/word**.
  - **Nepali (Model L)**: **$1.59$ tokens/word** ($+18.6\%$ higher fragmentation).
- **Linguistic Cause**: Hindi has analytic morphology where case markers are free-standing words separated by spaces (\dn{राम ने}, \dn{घर से}). Nepali has agglutinative morphology where case clitics attach directly to the noun stem (\dn{रामले}, \dn{घरबाट}), producing longer orthographic words that demand more subword splits under a fixed vocabulary.

#### 2. What does "Compression" (Characters per Token) mean?
- **Definition & Formula**:
  $$\text{Compression} = \frac{\text{Total Raw Unicode Characters}}{\text{Total Subword Tokens Generated}}$$
- **Meaning**: How many characters are packed into each token on average. Higher compression means the tokenizer is capturing longer, more meaningful morphological units.
- **Measured Values**:
  - **Hindi**: **$3.79$ chars/token**.
  - **Nepali**: **$4.03$ chars/token**.

#### 3. What does UNK Rate & Byte Fallback mean?
- **Unknown Rate (UNK)**: Percentage of tokens in held-out text mapped to the generic `<unk>` token. `<unk>` represents permanent information loss because the original character cannot be recovered.
- **Byte Fallback (`byte_fallback=True`)**:
  - Rather than mapping unseen characters to `<unk>`, SentencePiece decomposes any out-of-vocabulary UTF-8 character into raw byte tokens: `<0x00>` through `<0xFF>`.
  - **Result**: **$\text{UNK Rate} = 0.0000\%$** (exact zero).
  - **ASCII byte tokens** (`<0x43>`, `<0x68>`): Represent Latin characters (like 'C', 'h' in `ChatGPT`).
  - **Devanagari byte tokens** (`<0xE0>`, `<0xA4>`, `<0x8B>`): Devanagari characters require 3 UTF-8 bytes. Rare Devanagari glyphs (like `ऋ` in Hindi) decompose into three consecutive byte tokens.
  - **Byte Fallback Rates**: Hindi validation text has **$1.2032\%$** byte fallback; Nepali has **$2.8347\%$**.

#### 4. Why Unigram strictly beats BPE?
- **BPE (Byte-Pair Encoding)**: Greedily merges the most frequent adjacent bigrams in the training corpus. Once two pieces are merged, the merge is permanent and deterministic. On morphologically rich Devanagari, BPE often locks in unnatural, arbitrary character clusters based on accidental frequency.
- **Unigram**: Starts with a massive candidate vocabulary and iteratively prunes subwords using the Expectation-Maximization (EM) algorithm based on overall corpus sequence likelihood. It allows multiple segmentation candidates and selects the segmentation that maximizes sentence probability, keeping clean morphological stems (e.g. \dn{पर्यावरण} + \dn{विद} + \dn{ों}).
- **Empirical Proof**: Across all swept sizes (5k, 8k, 10k) in both Hindi and Nepali, Unigram achieved lower fertility and higher compression than BPE.

#### 5. Why $V = 10,000$ instead of $32,000$? (The Parameter Allocation Budget)
- In a Transformer with tied input/output embeddings and model dimension $d_{\text{model}} = 512$:
  $$\text{Embedding Parameters} = V \times d_{\text{model}} = V \times 512$$
- At a fixed total parameter budget of $\sim 25$ million:
  - **$V = 32,000$**: Lookup table alone consumes $32,000 \times 512 = 16,384,000$ parameters (**$65.5\%$ of the total budget**)! Only $34.5\%$ would remain for Transformer attention and MLP layers.
  - **$V = 10,000$**: Lookup table consumes $10,000 \times 512 = 5,120,000$ parameters (**$20.2\%$ of the total budget**), leaving **$79.8\%$** ($20.23$M parameters) for the actual 7 Transformer blocks.
- Setting $V = 10,000$ captured the majority of compression gains while buying an entire additional Transformer layer ($7$ layers instead of $6$).

---

## 4. Summary Metric Dictionary

| Metric / Term | Formula / Expression | Ideal / Target | Model H (Hindi) | Model L (Nepali) | Conceptual Significance |
|---|---|:---:|:---:|:---:|---|
| **Manual Fraction** | $\frac{\text{Manual Scraped Tokens}}{\text{Total Split Tokens}}$ | $\ge 20.00\%$ | **20.43%** | **20.00%** | Regulatory project constraint; guarantees fresh web crawl quality. |
| **Token Fertility** | $\frac{\text{Total Subword Tokens}}{\text{Total Whitespace Words}}$ | Lower is better | **1.34 tok/word** | **1.59 tok/word** | Measures subword fragmentation; higher in Nepali due to bound clitics. |
| **Char Compression** | $\frac{\text{Total UTF-8 Characters}}{\text{Total Subword Tokens}}$ | Higher is better | **3.79 char/tok** | **4.03 char/tok** | Average length of text represented by each subword token. |
| **UNK Rate** | $\frac{\text{Count of <unk> Tokens}}{\text{Total Tokens}}$ | $0.0000\%$ | **0.0000%** | **0.0000%** | Zero information loss; enabled by SentencePiece byte fallback. |
| **Byte Fallback Rate** | $\frac{\text{Byte-tier Tokens}}{\text{Total Tokens}}$ | Low (< 3%) | **1.2032%** | **2.8347%** | Percentage of tokens that decomposed into raw byte tokens (`<0x..>`). |
| **Vocabulary Size ($V$)**| Number of unique tokens | Sized to budget | **10,000** | **10,000** | Balances subword compression against 25.4M parameter budget. |
| **Embedding Share** | $\frac{V \times d_{\text{model}}}{\text{Total Parameters}}$ | $\le 25\%$ | **20.20%** | **20.20%** | Leaves ~80% of model parameter budget for attention layers. |

---

## 5. Execution & Verification Commands

To reproduce or verify every artifact and stage of Phase 1 from the terminal:

```bash
# 1. Run corpus stats calculation and manual fraction check:
python -m lmacorpus stats --lang hi
python -m lmacorpus stats --lang ne

# 2. Run the 6-class Devanagari language ID classifier:
python -m lmacorpus train-langid

# 3. Execute SentencePiece tokenizer training:
python -m lmacorpus train-spm --lang hi --vocab-size 10000 --model-type unigram
python -m lmacorpus train-spm --lang ne --vocab-size 10000 --model-type unigram

# 4. Evaluate tokenizers across BPE/Unigram sweeps on held-out val:
python -m lmacorpus vocab-sweep --lang hi
python -m lmacorpus vocab-sweep --lang ne

# 5. Re-generate all Phase 1 markdown tables and figures:
python scripts/build_report_tables.py
python scripts/make_figures.py
```
