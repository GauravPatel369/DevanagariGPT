# Comprehensive Viva & Defense Preparation Guide (Phases 1–3)

**Project**: Monolingual Transformer Language Models — Hindi (`hi`) & Nepali (`ne`)  
**Course**: Language Models and Agents (Monsoon 2026)  
**Author**: Gaurav Patel  

---

## Table of Contents
1. [General & High-Level Project Questions](#1-general--high-level-project-questions)
2. [Phase 1: Data Acquisition, Cleaning & Tokenization](#2-phase-1-data-acquisition-cleaning--tokenization)
3. [Phase 2: Transformer Architecture & Mathematical Primitives](#3-phase-2-transformer-architecture--mathematical-primitives)
4. [Phase 2: Pretraining Dynamics & Intrinsic Metrics (PPL vs. BPB)](#4-phase-2-pretraining-dynamics--intrinsic-metrics-ppl-vs-bpb)
5. [Phase 2: Attention Probing & Decoding Strategies](#5-phase-2-attention-probing--decoding-strategies)
6. [Phase 3: SFT Reasoning, Loss Masking & Leakage Isolation](#6-phase-3-sft-reasoning-loss-masking--leakage-isolation)
7. [Phase 3: Entity Memorization, Mitigations & Query Attention Probing](#7-phase-3-entity-memorization-mitigations--query-attention-probing)
8. [Professor "Trap" Questions & Bulletproof Answers](#8-professor-trap-questions--bulletproof-answers)

---

## 1. General & High-Level Project Questions

### Q1: What was the primary objective of this project?
**Answer**:  
To design, train, and empirically evaluate two strictly monolingual 25.4M-parameter decoder-only Transformer language models from scratch (Model H for Hindi, Model L for Nepali) without using pre-packaged architectures. The project spans the entire lifecycle: web scraping/cleaning raw corpora, building morphological tokenizers, pretraining on $\sim 450$M tokens, evaluating intrinsic compression and attention mechanisms, and performing Supervised Fine-Tuning (SFT) to rigorously audit whether small language models learn multi-step relational reasoning or merely memorize entity names.

### Q2: Did Hindi and Nepali share any data, tokenizers, or weights?
**Answer**:  
**No.** Both models are completely isolated and strictly monolingual. Each has its own independent web-crawled dataset, its own custom 10,000-vocabulary SentencePiece Unigram tokenizer, and its own separately initialized and trained weights.

### Q3: What is the exact parameter count, and why does it match the budget?
**Answer**:  
Both models have exactly **25,350,912 trainable parameters** ($99.8\%$ of the 25.4M ceiling). Every single weight is accounted for:
* Token Embedding Table ($10,000 \times 512$): $5,120,000$
* Attention $Q, K, V$ Projections ($7 \times 3 \times (512 \times 512 + 512)$): $5,531,904$
* Attention Output Projections ($7 \times (512 \times 512 + 512)$): $1,843,968$
* FFN Expansion Projections ($7 \times (512 \times 1792 + 1792)$): $6,437,056$
* FFN Contraction Projections ($7 \times (1792 \times 512 + 512)$): $6,419,968$
* Sublayer LayerNorms ($7 \times 2 \times (512 + 512)$): $14,336$
* Final LayerNorm ($512 + 512$): $1,024$
* Output Head: $0$ (Tied to Token Embedding Matrix)

---

## 2. Phase 1: Data Acquisition, Cleaning & Tokenization

### Q4: What acquisition strategies were used to collect $\sim 500$M tokens?
**Answer**:  
A 4-tier acquisition hierarchy was implemented in [`lmacorpus/acquire/crawler.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/acquire/crawler.py):
1. **MediaWiki Action API (`action=query&list=allpages`)**: Directly enumerated 100% of wiki articles (Wikipedia, Wikibooks, Wikisource, Kavita Kosh) avoiding crawl-traps.
2. **WordPress REST API (`/wp-json/wp/v2/posts`)**: Fast JSON post extraction at 100 posts per request.
3. **XML Sitemaps (`sitemap.xml`)**: Discovered canonical deep URLs across news portals (e.g., Setopati, Ratopati, Amar Ujala).
4. **Asynchronous Politeness Crawler**: Crawled robots.txt-compliant seeds using `httpx` + `asyncio` with exponential backoff and domain-level concurrency limiting.

### Q5: What are the 7 stages of your data cleaning pipeline?
**Answer**:  
Implemented in [`lmacorpus/clean/normalizer.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/normalizer.py) and [`lmacorpus/clean/filter.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmacorpus/clean/filter.py):
1. **Unicode NFC & Nukta Normalization**: Standardizes canonical equivalence (e.g. \dn{क़} vs \dn{क + ़}).
2. **HTML & Boilerplate Stripping**: Removes script tags, cookie banners, navigation links, and markup.
3. **Punctuation & Zero-Width Sanitation**: Replaces English quotes/dashes with standard Devanagari danda (\dn{।}) and cleans orphan zero-width joiners (`ZWJ` / `ZWNJ`).
4. **FastText Language Identification (LID)**: Discards documents where target language confidence $P(\text{lang}) < 0.80$.
5. **Heuristic Document Filtering**: Rejects documents with $>20\%$ Latin characters, word counts $<20$, or excessive symbol ratios.
6. **MinHash LSH Deduplication**: Removes near-duplicate articles at Jaccard threshold $0.85$ using 128 hash permutations.
7. **Perplexity Filtering**: Eliminates corrupt web machine translations using an n-gram KenLM / base model threshold.

### Q6: Why did you use Group-Aware Document Splitting instead of random line splitting?
**Answer**:  
Random line/token splitting distributes contiguous lines of the same document across train, validation, and test splits, causing **data leakage** (the model sees the continuation of a training story in the test set). Group-aware splitting hashes the complete document identifier so entire documents are assigned strictly to one split (Train: $80\%$, Val: $10\%$, Test: $10\%$), guaranteeing zero test-set leakage.

### Q7: Why SentencePiece Unigram over BPE, and why Byte Fallback?
**Answer**:  
1. **Unigram vs. BPE**: Standard BPE merges frequent pairs greedily, which often chops agglutinative Devanagari suffixes inconsistently. SentencePiece Unigram starts with an oversized vocabulary and optimizes a probabilistic unigram language model to find the most linguistically coherent subwords via Viterbi EM.
2. **Byte Fallback**: Instead of mapping rare or out-of-vocabulary characters to `<unk>`, byte fallback encodes them into UTF-8 byte tokens (e.g. `<0xE0><0xA4>...`), guaranteeing **zero `<unk>` tokens ($0.00\%$)** and making the vocabulary 100% lossless.

### Q8: How was the 10,000 vocabulary budget allocated?
**Answer**:  
* **$V=10,000$** was chosen after an empirical vocabulary sweep ($V \in \{5\text{k}, 8\text{k}, 10\text{k}, 16\text{k}\}$).
* At $V=10\text{k}$, embedding parameters ($5.12$M) consume only $20.2\%$ of the total parameter budget, leaving $79.8\%$ for 7 Transformer layers.
* At $V=16\text{k}$, the embedding table eats $32.8\%$ of the budget, starving the network of expressive depth while Bits-per-Byte plateaued ($0.471 \rightarrow 0.470$).

---

## 3. Phase 2: Transformer Architecture & Mathematical Primitives

### Q9: Why was `F.scaled_dot_product_attention` avoided in `model.py`?
**Answer**:  
1. **Course Constraint**: Pre-packaged attention blocks were forbidden; every attention equation had to be written from atomic PyTorch primitives (`nn.Linear`, `nn.Dropout`, matrix multiplications).
2. **Attention Probing Requirement**: `F.scaled_dot_product_attention` uses FlashAttention CUDA kernels in SRAM that never materialize the $(B, H, T, T)$ attention weight matrix. To extract attention matrices for layer-wise entropy probing, attention distance analysis, and heatmap visualization, explicit tensor materialization is mandatory.

### Q10: Why do we divide by $\sqrt{d_k} = 8$ in Scaled Dot-Product Attention?
**Answer**:  
Given $q, k \in \mathbb{R}^{64}$ with mean 0 and variance 1:
$$\text{Var}(q \cdot k) = \sum_{i=1}^{64} \text{Var}(q_i k_i) = 64$$
Without scaling, the logits have $\sigma = 8$. Large logits force the softmax function to saturate into an extreme one-hot distribution. In saturation, the softmax derivative $\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$ approaches $0$, which **extinguishes the gradient** during backpropagation. Dividing by $\sqrt{d_k} = \sqrt{64} = 8$ scales the variance back to $1.0$, preserving steady gradient flow.

### Q11: Explain Rotary Position Embeddings (RoPE) and why it was chosen over learned embeddings.
**Answer**:  
RoPE rotates query and key vectors in 2D planes by position angles $\theta_i = 10000^{-2i/d_k}$:
$$\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$
**Key Advantages**:
1. **Zero Learned Parameters**: A learned table costs $512 \times 512 = 262,144$ parameters. RoPE is a mathematical buffer, freeing budget to fund a 7th layer.
2. **Relative Distance by Construction**: $\langle R_m q, R_n k \rangle = q^T R_{n-m} k$ depends strictly on relative offset $(m-n)$.
3. **Applied to $Q$ and $K$ ONLY**: Position determines *where to look* ($Q, K$), whereas value $V$ carries *what content to retrieve*.

### Q12: Why Pre-LayerNorm over Post-LayerNorm?
**Answer**:  
In Pre-LN (`x = x + SubLayer(LayerNorm(x))`), the residual stream acts as an unadulterated identity highway:
$$x_L = x_0 + \sum_{l=1}^L \text{SubLayer}_l(\text{LN}(x_{l-1}))$$
Gradients backpropagate directly from the loss to layer 0 through the identity matrix without passing through 7 attenuating LayerNorm Jacobians. This allowed our models to train with perfectly flat gradient norms ($0.65$--$0.70$) with zero instability.

### Q13: How did you "buy depth with width" ($d_{\text{ff}} = 1792$)?
**Answer**:  
Standard GPT uses $d_{\text{ff}} = 4 \times d_{\text{model}} = 2048$, which at 7 layers requires $27.2$M parameters (exceeding 25.4M). By shrinking FFN expansion to $3.5 \times d_{\text{model}} = 1792$, we saved $1.84$M parameters. This parameter saving funded an entire additional Transformer block (**7 layers** instead of 6), allowing the model to compose higher-order syntactic and semantic abstractions.

---

## 4. Phase 2: Pretraining Dynamics & Intrinsic Metrics (PPL vs. BPB)

### Q14: Explain the difference between Perplexity (PPL) and Bits-Per-Byte (BPB).
**Answer**:  
* **Perplexity**: $\text{PPL} = \exp(\mathcal{L}_{\text{token}})$, measuring branching uncertainty per *subword token*.
* **Bits-per-Byte**: $\text{BPB} = \frac{\mathcal{L}_{\text{token}} \cdot \log_2(e)}{\text{Bytes per Token}}$, measuring compression cost per raw *UTF-8 text byte*.

### Q15: Why is comparing cross-lingual Perplexity fundamentally invalid?
**Answer**:  
In our evaluation (>51M tokens scored):
* Model H (Hindi) achieved **PPL 24.39**, while Model L (Nepali) had **PPL 29.87** (Hindi looks $+22.5\%$ better).
* However, Nepali tokens carry more raw bytes ($10.65$ vs $9.79$ bytes/token) and Nepali has higher tokenizer fertility ($1.59$ vs $1.34$ tokens/word). Predicting a larger chunk of text is inherently harder per step, which artificially inflates Nepali's perplexity.
* When evaluated on **Bits-per-Byte (BPB)**—a language-agnostic unit of raw bytes—Nepali achieves **$0.4600$ BPB** vs. Hindi's **$0.4709$ BPB** ($+2.3\%$ better compression for Nepali).
* **Conclusion**: Perplexity cannot be compared across different tokenizers. Bits-per-Byte is the only mathematically sound metric for cross-tokenizer evaluation.

### Q16: Why was AdamW configured with $\beta_2 = 0.95$ and selective weight decay?
**Answer**:  
1. **$\beta_2 = 0.95$**: Autoregressive token gradients are dynamic. Lowering $\beta_2$ from default $0.999$ to $0.95$ shortens the second-moment moving average window, allowing faster adaptation to non-stationary token dynamics.
2. **Selective Weight Decay ($0.1$)**: Applied only to 2D weight matrices ($W_Q, W_K, W_V, W_{\text{out}}, W_{\text{up}}, W_{\text{down}}, W_{\text{emb}}$). Decaying 1D LayerNorm gains ($\gamma$) and biases ($\beta$) would pull them toward zero, fighting against the normalization layer.

---

## 5. Phase 2: Attention Probing & Decoding Strategies

### Q17: What were the findings from Attention Entropy and Distance probing?
**Answer**:  
Across all 56 attention heads (7 layers $\times$ 8 heads):
1. **Monotonic Entropy Decay**: Attention entropy falls steadily with depth (Hindi: $5.76 \rightarrow 3.52$ bits; Nepali: $6.03 \rightarrow 3.14$ bits). Early layers attend diffusely across the whole context; deep layers narrow into razor-sharp focal points.
2. **U-Shaped Attention Distance Curve**:
   * *Layer 0*: Broad intake ($45$--$48$ tokens).
   * *Layers 1–4*: Local syntax ($25$--$30$ tokens: postpositions, case markers, noun-matra bindings).
   * *Layer 6*: Long-range cross-clause syntactic resolution ($53$--$55$ tokens: subject-verb agreement).

### Q18: What are the 4 functional head specializations identified?
**Answer**:  
1. **Local / Positional** (18 Hindi, 19 Nepali): Sharp attention on immediately adjacent tokens.
2. **Diffuse Global** (18 Hindi, 19 Nepali): Broad context tracking over the entire 512 context.
3. **Diffuse Local** (10 Hindi, 9 Nepali): Capturing phrase and clause boundaries.
4. **Long-Range Selective** (10 Hindi, 9 Nepali): Connecting distant syntactic dependencies.

### Q19: How did you select the optimal decoding strategy?
**Answer**:  
Rather than maximizing BLEU or diversity blindly, we selected the strategy that **minimized the distance to human reference text statistics** (`rep-4` and `distinct-2`):
* Greedy decoding failed due to extreme repetition loops ($\text{rep-4} = 0.756$).
* **Winning Hindi Configuration**: $T=1.0 + \text{top-}p\ 0.95$ ($\text{rep-4} = 0.0106$ vs Human $0.0087$; $\text{distinct-2} = 0.8147$ vs Human $0.8206$).
* **Winning Nepali Configuration**: $T=1.0$ ($\text{rep-4} = 0.0031$ vs Human $0.0018$; $\text{distinct-2} = 0.9229$ vs Human $0.9181$).

---

## 6. Phase 3: SFT Reasoning, Loss Masking & Leakage Isolation

### Q20: Why is prompt loss masking (`ignore_index=-100`) mandatory in SFT?
**Answer**:  
In SFT, an example consists of $\sim 40$ prompt tokens (premise + question) and only $\sim 2$ answer tokens. If the prompt is unmasked, $95\%$ of the gradient updates reward the model for reproducing trivially predictable premise text. Masking the prompt with `-100` ensures that **100% of the backpropagated loss comes from predicting the correct answer tokens**.

### Q21: Why must the SFT learning rate be an order of magnitude lower ($6\text{e-}5$ vs. $6\text{e-}4$)?
**Answer**:  
Pretraining spent 450M tokens establishing the foundational language model. The SFT dataset has only $\sim 3.5$M tokens. Training on a small synthetic dataset at the full pretraining rate ($\eta = 6\text{e-}4$) causes **severe language forgetting**—the weights are aggressively overwritten by repetitive synthetic templates, causing pretraining test perplexity to explode from $23.02$ to $>72.57$. An LR of $6\text{e-}5$ allows gentle task adaptation while preserving language structure.

### Q22: Explain the 3-way partition across Test A, B, C, and D.
**Answer**:  
To prevent memorization, data was partitioned across three disjoint dimensions: **Entity Names** (26 train vs. 6 test), **Numeric Ranges** (5–40 train vs. 41–90 test), and **Templates** (28 train vs. 15 test).
* **Test A**: Unseen Names + Unseen Numbers + *Seen Templates* (Relational Transfer).
* **Test B**: Unseen Names + Unseen Numbers + *Unseen Templates* (**True Out-of-Distribution Benchmark**).
* **Test C**: Unseen Names + *Seen Numbers* + Unseen Templates (Disentangling number vs template transfer).
* **Test D**: *Seen Names* + *Seen Numbers* + *Seen Templates* (In-Distribution Control).

---

## 7. Phase 3: Entity Memorization, Mitigations & Query Attention Probing

### Q23: What is the empirical proof that small language models memorize entity names?
**Answer**:  
Comparing Plain SFT on **Test D (In-Distribution)** vs. **Test B (OOD)**:
* On Test D (which uses the 26 training names), Plain SFT achieves $45\%$ written accuracy with **$0\%$ wrong names**.
* On Test B (which uses 6 unseen names), Plain SFT's wrong-name rate surges to **$92\%$** (and written accuracy drops to $6.3\%$).
* **The Proof**: When presented with unseen names, the model outputs names from its training set (e.g. predicting *"विजय"* when the question only mentions *"अमित"* and *"रोहित"*). The model did not learn abstract logic ($A > B \land B > C \implies A > C$); it learned a surface prior to output memorized training names.

### Q24: How did you fix entity memorization?
**Answer**:  
1. **Hindi**: Early stopping on out-of-distribution validation accuracy (`val_ood`) dropped wrong-name rate from $92\%$ to **$15\%$**, boosting OOD written accuracy to **$33.0\%$**.
2. **Nepali**: Expanding the entity name pool to $\sim 175$ names + pretraining corpus replay dropped wrong-name rate from $90\%$ to **$38\%$**, lifting written accuracy to **$26.8\%$**.

### Q25: What did Query Attention Probing reveal about reasoning in Transformers?
**Answer**:  
By measuring the attention weights of the query token (`"उत्तर:"`) across all 7 layers:
* The attention preference ratio between the correct entity and the distractor entity remained flat at **$0.97$--$1.09$** (no selective preference for the logical answer).
* However, Layer 4 increased its total attention mass on **all entity names by $+40\%$**.
* **Insight**: The attention heads learned the *semantic type* of the expected answer (*"output a person's name"*), but lacked the multi-hop routing circuits required to determine *which* entity is logically correct.

### Q26: Why does Equality Detection succeed ($74\%$) while Multi-Hop Transitivity stays near chance ($34\%$)?
**Answer**:  
* **Equality Detection** (*"बराबर"*): Requires detecting identical string tokens (e.g., token "25" matching token "25"), which a single self-attention head can resolve via direct key-query token matching.
* **Multi-Hop Transitivity** ($A > B > C \implies A > C$): Requires composing attention across multiple disjoint entity-attribute bindings. In a 25.4M model without Chain-of-Thought scratchpads, this exceeds the representational capacity of 7 layers.

---

## 8. Professor "Trap" Questions & Bulletproof Answers

### Trap Q1: "Why didn't you train for 3 to 5 epochs in SFT to get better reasoning accuracy?"
**Bulletproof Answer**:  
"We ran training sample sweeps from 5k to 100k samples and tested multi-epoch runs. In SFT on synthetic data, training loss falls from $1.23$ to $0.40$ within 500 steps, but validation loss on unseen names begins climbing after step 600 ($1.37 \rightarrow 1.93$). Training for multiple epochs causes severe overfitting to training entity names and degrades general language perplexity ($23.02 \rightarrow 72.57$). In small Transformers, 1 epoch with early stopping on OOD validation is the optimal Pareto point."

### Trap Q2: "Why didn't you use Byte-Pair Encoding (BPE) since GPT-2 used BPE?"
**Bulletproof Answer**:  
"GPT-2 was designed for English, an isolating language with minimal inflection. Hindi and Nepali are morphologically rich and agglutinative Indo-Aryan languages with complex consonant conjuncts and case clitics. Greedy BPE merges frequently chop grammatical morphemes suboptimally. SentencePiece Unigram models tokenization as a probabilistic unigram distribution and uses Viterbi EM optimization, preserving coherent subwords. Combined with Byte Fallback, it guarantees $0.00\%$ `<unk>` tokens."

### Trap Q3: "Your choice accuracy on Test B is ~38%, which is around random chance. Does that mean the model learned nothing?"
**Bulletproof Answer**:  
"No, it proves a profound distinction between **format compliance** and **logical deduction**. The base pretrained model had $0\%$ written accuracy and could not follow the QA format. SFT taught the model to produce valid entity names in the correct QA format (lifting written accuracy from $0\%$ to $33.0\%$ with only $15\%$ wrong names), and taught the model string identity matching for equality detection ($74\%$). However, our strict 3-way leakage isolation (Test B) proved that multi-step relational reasoning ($A>B>C$) cannot be acquired by a 25.4M parameter Transformer via direct SFT without scratchpads."
