# Comprehensive Phase-Wise Codebase & Implementation Line Map

**Project**: Monolingual Transformer Language Models — Hindi (`hi`) & Nepali (`ne`)  
**Course**: Language Models and Agents (Monsoon 2026)  
**Author**: Gaurav Patel  

---

## Table of Contents
1. [Phase 1: Data Engine & Tokenization Infrastructure](#1-phase-1-data-engine--tokenization-infrastructure)
2. [Phase 2: Architecture, Pretraining & Intrinsic Evaluation](#2-phase-2-architecture-pretraining--intrinsic-evaluation)
3. [Phase 3: SFT Reasoning, Shortcut Audits & Memorization Probes](#3-phase-3-sft-reasoning-shortcut-audits--memorization-probes)

---

## 1. Phase 1: Data Engine & Tokenization Infrastructure

| Feature / Subsystem | Exact File Location | Exact Line Numbers | What It Does & Key Logic |
|---|---|:---:|---|
| **Polite Async Web Crawler** | [`lmacorpus/acquire/crawler.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/crawler.py) | `L46–L135` | Manages per-domain semaphores, `robots.txt` parsing, exponential backoff, and gzip HTML archiving. |
| **MediaWiki Action API** | [`lmacorpus/acquire/crawler.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/crawler.py) | `L137–L189` | Uses `action=query&list=allpages&aplimit=500` with `apcontinue` pagination to enumerate 100% of wiki articles. |
| **BFS HTML Link Crawler** | [`lmacorpus/acquire/crawler.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/crawler.py) | `L190–L225` | Fallback breadth-first link crawler for non-API domains with depth capping ($d \le 2$). |
| **WordPress REST API** | [`lmacorpus/acquire/crawler.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/crawler.py) | `L226–L257` | Queries `/wp-json/wp/v2/posts?per_page=100` for rapid JSON article extraction. |
| **Sitemap Discovery** | [`lmacorpus/acquire/crawler.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/crawler.py) | `L258–L315` | Recursively parses `robots.txt` sitemap directives and XML URLsets/sitemap indexes. |
| **HuggingFace Streaming** | [`lmacorpus/acquire/huggingface.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/huggingface.py) | `L35–L95` | Streams open web datasets (mC4, OSCAR, Sangraha) with local shard caching. |
| **Unicode NFC & Nukta Normalizer** | [`lmacorpus/clean/normalizer.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/normalizer.py) | `L18–L65` | Applies `unicodedata.normalize("NFC")` and combines decomposed Nukta characters (e.g. \dn{क + ़} $\rightarrow$ \dn{क़}). |
| **HTML & Boilerplate Stripping** | [`lmacorpus/clean/normalizer.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/normalizer.py) | `L67–L115` | Cleans navigation headers, cookie notices, inline JavaScript, and HTML tags. |
| **Zero-Width & Punctuation Sanitation** | [`lmacorpus/clean/normalizer.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/normalizer.py) | `L117–L165` | Strips orphan ZWJ/ZWNJ characters and standardizes Latin punctuation to Devanagari danda (\dn{।}). |
| **FastText Language ID (LID)** | [`lmacorpus/clean/filter.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/filter.py) | `L22–L75` | Runs `lid.176.bin` and drops documents where target language confidence $P(\text{lang}) < 0.80$. |
| **Heuristic Document Filter** | [`lmacorpus/clean/filter.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/filter.py) | `L77–L138` | Drops documents with Latin char ratio $>20\%$, word count $<20$, or excessive symbol ratios. |
| **MinHash LSH Deduplication** | [`lmacorpus/clean/filter.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/filter.py) | `L140–L225` | 128 hash permutations, 5-gram shingling, Jaccard threshold $0.85$ near-duplicate filtering. |
| **End-to-End Cleaning Pipeline** | [`lmacorpus/clean/pipeline.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/pipeline.py) | `L35–L130` | Streams documents through the 7 cleaning stages with detailed audit statistics. |
| **Group-Aware Document Splitting**| [`lmacorpus/split.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/split.py) | `L15–L85` | Hashes document IDs to allocate complete documents to 80/10/10 splits without cross-split leakage. |
| **SentencePiece Unigram Trainer** | [`lmacorpus/tokenize/trainer.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/tokenize/trainer.py) | `L20–L95` | Trains 10,000-vocab SentencePiece Unigram model with `byte_fallback=True` ($0.00\%$ `<unk>`). |
| **Binary Token Serializer** | [`lmacorpus/tokenize/encoder.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/tokenize/encoder.py) | `L25–L105` | Encodes text shards into contiguous memory-mapped `uint16` binary files (`train.bin`, `val.bin`, `test.bin`). |
| **Tokenizer Metrics & Fertility** | [`lmacorpus/tokenize/stats.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/tokenize/stats.py) | `L20–L140` | Calculates subword fertility ($1.34$ Hindi, $1.59$ Nepali), bytes/token, and Unicode script coverage. |

---

## 2. Phase 2: Architecture, Pretraining & Intrinsic Evaluation

| Feature / Subsystem | Exact File Location | Exact Line Numbers | What It Does & Key Logic |
|---|---|:---:|---|
| **GPT Hyperparameter Config** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L46–L78` | `GPTConfig` dataclass, validates head dimensions and even $d_k$ for 2D RoPE rotation. |
| **RoPE Cache Precomputation** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L82–L101` | Calculates `cos` and `sin` tables for $\theta_i = 10000^{-2i/d_k}$ and positions $m \in [0, 511]$. |
| **RoPE 2D Complex Rotation** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L103–L126` | `apply_rope(x, cos, sin)` rotates even/odd query and key pairs in 2D planes ($0$ learned parameters). |
| **First-Principles Self-Attention** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L131–L198` | Projects $Q, K, V$, scales by $\frac{1}{\sqrt{64}}=0.125$, applies upper-triangular $-\infty$ mask, materializes weights. |
| **Feed-Forward Network (FFN)** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L203–L221` | $3.5\times$ expansion ($512 \rightarrow 1792 \rightarrow 512$) with GELU activation and residual dropout. |
| **Pre-LN Transformer Block** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L223–L246` | Implements `x = x + Attn(LN(x))` and `x = x + FFN(LN(x))`, keeping the residual highway unattenuated. |
| **Complete GPT Model & Tied Head** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L248–L299` | Instantiates 7 blocks, ties `lm_head.weight = token_emb.weight`, scales residual out weights by $\frac{0.02}{\sqrt{14}}$. |
| **Forward Pass & Loss Masking** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L300–L347` | Full sequence forward pass and cross-entropy loss with `ignore_index=-100`. |
| **Repetition Penalty Filter** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L349–L362` | Sign-aware logit penalty ($\text{logit}/\theta$ if $>0$, $\text{logit}\cdot\theta$ if $\le 0$). |
| **No-Repeat N-Gram Blocker** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L364–L381` | Hard-masks candidate next tokens that would repeat an existing $n$-gram prefix to $-\infty$. |
| **6-Strategy Generation Engine** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) | `L383–L449` | Autoregressive generation with Temperature, Top-K, Top-P Nucleus (shifted cumsum), and Greedy argmax. |
| **LR Warmup & Cosine Schedule** | [`lmagpt/train.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py) | `L48–L64` | 500-step linear warmup ($0 \rightarrow 6\text{e-}4$) followed by cosine decay to $6\text{e-}5$ floor. |
| **Selective AdamW Optimizer** | [`lmagpt/train.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py) | `L66–L88` | Applies $\text{wd}=0.1$ to 2D weight matrices while keeping 1D LayerNorm gains/biases at $\text{wd}=0.0$. |
| **Atomic Resumable Checkpointer** | [`lmagpt/train.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py) | `L93–L124` | Atomically writes `.pt.tmp` and replaces to prevent corrupted checkpoints upon cloud preemption. |
| **Validation Loss Evaluator** | [`lmagpt/train.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py) | `L129–L140` | Computes mean loss over fixed deterministic validation batches. |
| **Main Training Loop & GradScaler**| [`lmagpt/train.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py) | `L145–L294` | Mixed precision (BF16/FP16), gradient accumulation ($16,384$ tokens/step), norm clipping ($1.0$), budget guard. |
| **Memory-Mapped Token Dataloader** | [`lmagpt/data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/data.py) | `L15–L85` | Memory-maps `train.bin` and samples random $(B=16, T=512)$ windows re-seeded by step (`seed + step`). |
| **Intrinsic Metrics (PPL & BPB)** | [`lmagpt/evaluate.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/evaluate.py) | `L54–L105` | Computes per-token Perplexity and byte-normalized Bits-Per-Byte ($\text{BPB} = \mathcal{L} \log_2(e) / \text{bytes\_per\_tok}$). |
| **Generation Quality Evaluation** | [`lmagpt/evaluate.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/evaluate.py) | `L140–L310` | Computes BLEU, chrF, ROUGE-L, distinct-n, and rep-n across the 12-decoder sweep. |
| **Attention Weight Extraction** | [`lmagpt/attention.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/attention.py) | `L65–L76` | Collects post-softmax $(B, 8, T, T)$ attention matrices across all 7 layers. |
| **Entropy & Distance Probing** | [`lmagpt/attention.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/attention.py) | `L78–L135` | Computes layer-wise entropy $\mathcal{H}(q)$ and mean attention distance $\bar{D}(q)$ across all 56 heads. |
| **Attention Heatmap Plotting** | [`lmagpt/attention.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/attention.py) | `L220–L290` | Generates layer-by-head attention heatmaps and distance summary plots. |

---

## 3. Phase 3: SFT Reasoning, Shortcut Audits & Memorization Probes

| Feature / Subsystem | Exact File Location | Exact Line Numbers | What It Does & Key Logic |
|---|---|:---:|---|
| **Hindi Reasoning Templates** | [`hindi/reasoning/templates.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/hindi/reasoning/templates.py) | `L15–L215` | Grammatical Hindi premise/question templates, disjoint train (26) vs test (6) name pools. |
| **Nepali Reasoning Templates** | [`nepali/reasoning/templates.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/nepali/reasoning/templates.py) | `L15–L220` | Grammatical Nepali premise/question templates, case-marker variations, disjoint name pools. |
| **Synthetic Dataset Engine** | [`scripts/make_reasoning_data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/make_reasoning_data.py) | `L45–L350` | Generates 9 problem categories (Direct, Numeric, Superlative, Equality, Chains, Ranking). |
| **Strict 3-Way Leakage Partition** | [`scripts/make_reasoning_data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/make_reasoning_data.py) | `L351–L520` | Partitions by Names (26 vs 6), Numbers (5–40 vs 41–90), and Templates (28 vs 15) into Test A, B, C, D. |
| **Automated Shortcut Rule Audit** | [`scripts/shortcut_rules.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/shortcut_rules.py) | `L15–L140` | Tests 7 positional heuristics to verify all heuristics score at random chance ($50\% \pm 1\%$). |
| **Shortcut Probe Classifier** | [`scripts/shortcut_probe.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/shortcut_probe.py) | `L20–L155` | Trains linear probe on query hidden states to confirm lack of superficial position bias. |
| **SFT Loss Masking Dataloader** | [`lmagpt/finetune.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/finetune.py) | `L48–L105` | `ReasoningDataset._encode()` sets `targets[:n_prompt] = -100` and `targets[pad:] = -100`. |
| **SFT Training Engine** | [`lmagpt/finetune.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/finetune.py) | `L127–L256` | Fine-tunes with $10\times$ lower learning rate ($\eta=6\text{e-}5$), 1 epoch, and saves `best.pt` by validation. |
| **Sample Size Sweep Runner** | [`scripts/sample_sweep.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/sample_sweep.py) | `L25–L185` | Runs 5k, 10k, 20k, 30k, 100k sample sweeps to find optimal Pareto points (20k Hindi, 10k Nepali). |
| **Technique Ablation Dataset Gen** | [`scripts/make_ablation_data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/make_ablation_data.py) | `L20–L165` | Synthesizes ablation datasets for `name_pool` (~175 names), `corpus_replay`, and `ood_val`. |
| **Cross-Split Evaluation Runner** | [`scripts/eval_sweep_ckpts.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/eval_sweep_ckpts.py) | `L25–L150` | Evaluates Choice Accuracy (log-prob sum) and Written Greedy Accuracy ($T=0$) across Test A, B, C, D. |
| **Entity Memorization Audit** | [`scripts/eval_sweep_ckpts.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/eval_sweep_ckpts.py) | `L155–L210` | Computes Wrong-Name Rate (detecting when output entity is absent from the prompt premise). |
| **General Language Retention Audit**| [`scripts/eval_finetuned_ppl.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/eval_finetuned_ppl.py) | `L25–L135` | Evaluates checkpoint perplexity over all 51M tokens of the raw pretraining `test.bin`. |
| **Query Attention Probing** | [`scripts/attention_reasoning.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/attention_reasoning.py) | `L35–L175` | Computes layer-wise attention ratio from `"उत्तर:"` to correct entity vs distractor entity. |
| **Morphological Clitic Normalizer**| [`scripts/score_with_endings.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/score_with_endings.py) | `L15–L85` | Strips Nepali case markers (`-लाई`, `-ले`), lifting written accuracy from $10.2\%$ to $26.8\%$. |
| **Phase 3 Figures & Plotting** | [`scripts/make_figures.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/make_figures.py) | `L85–L170` | Plots validation loss curves across sample scales and ablation comparison bar charts. |
