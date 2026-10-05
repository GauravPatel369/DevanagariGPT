# Phase 3: Supervised Finetuning, Reasoning & Memorization Audit Guide

**Project**: Monolingual Transformer Language Models — Hindi (`hi`) & Nepali (`ne`)  
**Course**: Language Models and Agents (Monsoon 2026)  
**Author**: Gaurav Patel  
**Repository Branch**: `phase-3` / `bonus`  
**Primary Source Modules**: [`lmagpt/finetune.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/finetune.py), [`scripts/make_reasoning_data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/make_reasoning_data.py), [`scripts/shortcut_rules.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/shortcut_rules.py), [`scripts/attention_reasoning.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/attention_reasoning.py), [`scripts/eval_finetuned_ppl.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/eval_finetuned_ppl.py)

---

## 1. Overview & Conceptual Architecture

This document provides the definitive implementation and theoretical reference for **Phase 3 (Synthetic Reasoning Dataset Generation, SFT Training, Shortcut & Memorization Auditing, Cross-Split Generalization, and Query Attention Probing)**:
1. **"Where is implemented what"**: A complete directory and code map linking every script, template generator, loss masking function, evaluation probe, and plotting module.
2. **"End-to-End Flow (`finetune.py`, `make_reasoning_data.py`)"**: A step-by-step trace of how comparative reasoning problems are synthesized, tokenized with loss masks, optimized without forgetting, and evaluated across in-distribution and out-of-distribution (OOD) test splits.
3. **"What does that mean in that way"**: The mathematical formulations, empirical proofs of entity memorization, linguistic analysis of morphological clitics, and attention mechanism probing.

```
══════════════════════════════════════════════════════════════════════════════════════════════════════
                               PHASE 3 SFT & REASONING AUDIT LIFECYCLE
══════════════════════════════════════════════════════════════════════════════════════════════════════

  [Grammatical Template Engine: hindi/nepali templates.py]
                     │
                     ▼
  [make_reasoning_data.py: 9 Problem Categories + Strict 3-Way Leakage Partitioning]
                     │
         ┌───────────┴─────────────────────────────────────────┐
         ▼                                                     ▼
  [Training Split (26 Names, 5-40 Values, 28 Templates)]  [Test Splits: A, B (OOD), C, D (Control)]
         │                                                     │
         ▼                                                     │
  [ReasoningDataset: Prompt Loss Masking (ignore_index=-100)]  │
         │                                                     │
         ▼                                                     │
  ┌────────────────────────────────────────────────────────┐   │
  │                  lmagpt/finetune.py                    │   │
  │                                                        │   │
  │  1. Load Pretrained Checkpoint (Step 55k / 27.5k)       │   │
  │  2. Low Learning Rate (6e-5, 10x below pretraining)    │   │
  │  3. Masked Cross-Entropy (Gradients on Answer ONLY)    │   │
  │  4. Mitigation Strategies:                             │   │
  │     ├── Early Stopping on OOD validation (val_ood)     │   │
  │     ├── Expanded Name Pool (~175 unique names)         │   │
  │     └── Pretraining Corpus Replay (prevents PPL drift) │   │
  │  5. Checkpoints: best.pt (by OOD metric), last.pt      │   │
  └───────────────────────────┬────────────────────────────┘   │
                              │                                │
                              ▼                                ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │                           Comprehensive Reasoning & Diagnostic Suite                       │
  │                                                                                            │
  │  1. Dual Metric Evaluation:                                                                │
  │     ├── Choice Accuracy (Log-probability sum over candidate choices)                       │
  │     └── Written Accuracy (Greedy autoregressive generation T=0)                            │
  │  2. Entity Memorization Audit (Wrong-Name Rate: Test D 0% vs. Test B 90% in Plain SFT)     │
  │  3. General Language Retention: Scored on full 51M pretraining test.bin                     │
  │  4. Attention Mechanism Query Probing (scripts/attention_reasoning.py):                     │
  │     └── Query token ("उत्तर:") attention ratio: Correct Name vs. Distractor across layers │
  │  5. Morphological Clitic Normalization (scripts/score_with_endings.py: -लाई, -ले)          │
  └────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Where is Implemented What: Codebase Directory Map

The table below catalogs every component, script, template generator, and evaluation artifact in Phase 3:

| Component / Subsystem | Source File Path | Key Classes & Functions | Execution CLI Command | Primary Outputs & Reports |
|---|---|---|---|---|
| **Hindi Reasoning Templates** | [`hindi/reasoning/templates.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/hindi/reasoning/templates.py) | `TEMPLATES`, `NAMES_TRAIN`, `NAMES_TEST`, `ATTRIBUTES` | Imported by data generator | Grammatical Hindi premises & questions |
| **Nepali Reasoning Templates** | [`nepali/reasoning/templates.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/nepali/reasoning/templates.py) | `TEMPLATES`, `NAMES_TRAIN`, `NAMES_TEST`, `ATTRIBUTES` | Imported by data generator | Grammatical Nepali premises & questions |
| **Synthetic Dataset Engine** | [`scripts/make_reasoning_data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/make_reasoning_data.py) | `generate_split()`, `generate_chains()`, 9 problem types | `python scripts/make_reasoning_data.py --lang {hi,ne}` | `train.jsonl`, `val.jsonl`, `test_{a,b,c,d}.jsonl` |
| **Position Shortcut Audit** | [`scripts/shortcut_rules.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/shortcut_rules.py), [`scripts/shortcut_audit.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/shortcut_audit.py) | `audit_split()`, 7 heuristic rules, `probe_shortcut()` | `python scripts/shortcut_audit.py` | `shortcut_audit.json`, `shortcut_probe.json` |
| **SFT Loss Masking Dataloader**| [`lmagpt/finetune.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/finetune.py#L48-L105) | `ReasoningDataset`, `_encode()`, `batches()` | Internal to finetuning | Masked input/target arrays with `-100` |
| **SFT Training Engine** | [`lmagpt/finetune.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/finetune.py#L127-L256) | `main()`, `validation_loss()`, cosine schedule | `python -m lmagpt.finetune --lang {hi,ne}` | `reasoning/checkpoints/best.pt`, `finetune_log.jsonl` |
| **Sample Size Sweep** | [`scripts/sample_sweep.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/sample_sweep.py) | `run_sweep()`, 5k, 10k, 20k, 30k, 100k runs | `python scripts/sample_sweep.py` | `sweep_results.json`, validation curves |
| **Technique Ablations** | [`scripts/make_ablation_data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/make_ablation_data.py) | `name_pool`, `corpus_replay`, `ood_val` | `python scripts/make_ablation_data.py` | `ablation/` dataset & checkpoint directories |
| **Cross-Split Evaluation** | [`scripts/eval_sweep_ckpts.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/eval_sweep_ckpts.py), [`scripts/ask.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/ask.py) | `evaluate_split()`, `choice_acc()`, `written_acc()`, wrong-name rate | `python scripts/eval_sweep_ckpts.py` | `eval_finetuned.json`, `examples_test_b.json` |
| **General Language Retention** | [`scripts/eval_finetuned_ppl.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/eval_finetuned_ppl.py) | `eval_ppl_on_test_bin()` | `python scripts/eval_finetuned_ppl.py` | Full pretraining test PPL post-SFT |
| **Query Attention Probing** | [`scripts/attention_reasoning.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/attention_reasoning.py) | `probe_query_attention()`, layer-wise ratio | `python scripts/attention_reasoning.py` | `attention_reasoning.json`, layer-wise ratios |
| **Morphological Normalization** | [`scripts/score_with_endings.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/score_with_endings.py) | `strip_nepali_clitics()`, case marker matcher | `python scripts/score_with_endings.py` | `written_with_endings.json` |
| **Phase 3 Figure Generation** | [`scripts/make_figures.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/make_figures.py), [`scripts/plot_ablation.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/plot_ablation.py) | `plot_validation_curves()`, `plot_ablation_summary()` | `python scripts/make_figures.py` | `validation_curves.png`, `ablation_summary.png` |

---

## 3. End-to-End System Flow: Step-by-Step Code Execution

### Step 1: Synthetic Reasoning Problem Synthesis (9 Problem Categories)
In [`scripts/make_reasoning_data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/make_reasoning_data.py), questions are synthesized across 9 problem categories using grammatical templates from `templates.py`:
1. **Direct Comparison** (12%): Two entities with explicit numeric values (*"अनिल 18 वर्ष, विजय 25 वर्ष। किसकी उम्र अधिक है?"*).
2. **Numeric Comparison** (12%): Price/quantity comparison with polarity inversion (*"सस्ता कौन है?"* vs *"महंगा कौन है?"*).
3. **Superlative of Three** (12%): Finding the extreme entity among three.
4. **Mixed / Equality** (12%): Values can be equal (*"बराबर"*).
5. **2-Step Chains (Place)** (12%): $A > B > C$, asking for extreme or intermediate entity.
6. **3-Step Chains (Place)** (12%): 4-entity chain ($A > B > C > D$), asking for extreme or second place.
7. **2-Step Chains (Pair Relation)** (10%): $A > B > C$, asking directly: *"A और C में कौन बड़ा है?"*.
8. **3-Step Chains (Pair Relation)** (8%): 4-entity chain asking about two non-adjacent entities ($A$ vs $C$, or $B$ vs $D$).
9. **Ranking** (10%): Identifying median or specific relative position (*"बीच में कौन है?"*).

---

### Step 2: Strict 3-Way Leakage Partitioning (Test A, B, C, D)
To guarantee that high benchmark scores cannot be achieved via memorization, the generator partitions data across **three disjoint dimensions**:

| Dimension | Training Split | Held-Out Test Splits |
|---|---|---|
| **Entity Names** | 26 disjoint names | 6 held-out names (never seen in training) |
| **Numeric Value Ranges** | 5 to 40 | 41 to 90 (disjoint number space) |
| **Sentence Templates** | 28 templates | 15 held-out templates |

#### The Four Evaluation Splits (4,000 generated each; 600 fixed eval sample):
* **Test A (Relational Transfer)**: Unseen Names + Unseen Numbers + *Seen Templates*.
* **Test B (True OOD Benchmark)**: Unseen Names + Unseen Numbers + *Unseen Templates*.
* **Test C (Disentangling Template Transfer)**: Unseen Names + *Seen Numbers* + Unseen Templates.
* **Test D (In-Distribution Control)**: *Seen Names* + *Seen Numbers* + *Seen Templates* in novel combinations.

---

### Step 3: Position Shortcut Audit & Rule Verification
Before any training, [`scripts/shortcut_rules.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/shortcut_rules.py) evaluates 7 positional heuristics to verify that answers cannot be guessed without reading the premise:
* *Rule 1*: "Always pick the first entity mentioned in the prompt."
* *Rule 2*: "Always pick the second entity."
* *Rule 3*: "Always pick the last entity mentioned in the question."
* *Rule 4–7*: "Chain-position heuristics (first in chain, middle in chain, last in chain)."
* **Audit Result**: All 7 heuristics score **$50.0\% \pm 1.0\%$ on binary relations** and **$33.3\% \pm 1.0\%$ on ternary comparisons**, exactly matching random chance.

---

### Step 4: SFT Loss Masking Mechanics
In [`lmagpt/finetune.py:L48-L105`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/finetune.py#L48-L105), `ReasoningDataset._encode()` constructs inputs and targets:
1. Formats each sample as: `prompt = "तथ्य: ... प्रश्न: ... उत्तर:"`, `target = " विजय"`.
2. Tokenizes both strings via SentencePiece: `ids = sp.encode(prompt) + sp.encode(target)`.
3. Sets up standard next-token prediction: $x = \text{ids}[:-1]$, $y = \text{ids}[1:]$.
4. **The Masking Step**:
   ```python
   n_prompt = len(prompt) - 1
   y = [-100] * n_prompt + y[n_prompt:]   # Mask prompt tokens
   pad = max_len - 1 - len(x)
   x = x + [pad_id] * pad
   y = y + [-100] * pad                  # Mask padding tokens
   ```
5. In `GPT.forward()`, `F.cross_entropy(logits, targets, ignore_index=-100)` skips all `-100` positions. Gradients flow **exclusively through answer tokens**.

---

### Step 5: Low Learning Rate Optimization & Early Stopping
In [`lmagpt/finetune.py:L127-L256`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/finetune.py#L127-L256):
1. Loads the pretrained checkpoint (`last.pt` / `best.pt`) from Phase 2.
2. Sets SFT learning rate to $\eta = \mathbf{6.0 \times 10^{-5}}$ ($10\times$ lower than pretraining's $6.0 \times 10^{-4}$).
3. Employs a 100-step linear warmup followed by cosine annealing down to $6.0 \times 10^{-6}$.
4. Trains for **1 epoch** (e.g. 625 steps for 20k samples). Training for $\ge 3$ epochs causes training loss to collapse while validation loss surges ($1.37 \rightarrow 1.93$), causing severe entity memorization.
5. Saves `best.pt` based on validation performance on unseen entity names (`val_ood`).

---

### Step 6: Four Mitigation Recipes
To combat entity memorization and general language degradation, four recipes were implemented and compared:
1. **`ood_val`**: Checkpoints selected exclusively by accuracy on unseen entity names rather than training loss.
2. **`name_pool`**: Expands the entity name pool from 26 names to $\sim 175$ diverse names in `templates.py`.
3. **`corpus_replay`**: Injects random 512-token chunks from the pretraining corpus (`train.bin`) into $20\%$ of training batches.
4. **`name_pool + replay`**: Combines the 175-name pool with pretraining corpus replay.

---

### Step 7: Dual Evaluation Framework
Evaluation in [`scripts/eval_sweep_ckpts.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/eval_sweep_ckpts.py) and [`scripts/ask.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/ask.py) uses two distinct scoring paradigms:

#### A. Choice Accuracy (Log-Probability Summation):
For candidate entity choices $\{C_1, C_2, \dots, C_k\}$:
$$\text{Score}(C_i) = \sum_{t=1}^{|C_i|} \log P(w_t^{(i)} \mid \text{Prompt}, w_{<t}^{(i)})$$
The model selects $\hat{C} = \arg\max_i \text{Score}(C_i)$. This metric works even on the raw pretrained base model before fine-tuning.

#### B. Written Accuracy & Wrong-Name Rate (Greedy Generation $T=0$):
The model autoregressively generates text following `"उत्तर:"`.
* **Written Accuracy**: The first emitted entity name matches the target answer.
* **Wrong-Name Rate**: The percentage of generated responses that output a name **not present anywhere in the question's premise**.

---

### Step 8: General Language Retention Scoring
To verify that SFT did not destroy the foundational language model, [`scripts/eval_finetuned_ppl.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/eval_finetuned_ppl.py) evaluates every checkpoint across all $51$M+ tokens of the raw pretraining test split (`test.bin`):
* Base Hindi Pretrained PPL: **23.02** $\rightarrow$ Final SFT Model H PPL: **23.81** (only $+3.4\%$ change).
* Base Nepali Pretrained PPL: **29.87** $\rightarrow$ Final SFT Model L PPL: **30.81** (only $+3.1\%$ change).

---

### Step 9: Query Attention Probing
In [`scripts/attention_reasoning.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/attention_reasoning.py):
1. Gathers post-softmax attention weights from the prompt's final query token (`"उत्तर:"`).
2. Computes the layer-wise **Preference Ratio**:
   $$\text{Ratio}(l) = \frac{\text{Mean Attention Mass on Correct Entity Tokens in Layer } l}{\text{Mean Attention Mass on Distractor Entity Tokens in Layer } l}$$
3. A ratio of $1.0$ represents equal, unselective attention.

---

### Step 10: Morphological Normalization (Nepali Clitic Strip)
In [`scripts/score_with_endings.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/score_with_endings.py):
Nepali is highly agglutinative. The model frequently generates the correct entity root combined with a valid case clitic (e.g. \dn{अनिललाई} [Anil-DAT] or \dn{विजयले} [Vijay-ERG] instead of bare \dn{अनिल}). Normalizing these clitics lifts Nepali written accuracy from **$10.2\%$** to **$26.8\%$**.

---

## 4. What Does That Mean in That Way? Mathematical & Conceptual Deep-Dive

---

### 4.1 Why SFT Loss Masking is Mandatory (`ignore_index=-100`)
- In standard next-token pretraining, loss is computed over every token.
- In SFT, each prompt consists of $\sim 40$ tokens of premise/question text and only $\sim 2$ tokens of target answer.
- **The Failure Mode of Unmasked SFT**: If the prompt is unmasked, $95\%$ of the gradient updates teach the model to verbatim copy and predict the premise words. The model wastes its small 25.4M capacity on trivially predictable surface text rather than routing information from premise to answer.
- **The Solution**: Masking with `-100` forces the optimizer to update weights strictly based on whether the final hidden state produces the correct answer token.

---

### 4.2 Why SFT Learning Rate Must Be $10\times$ Lower ($6.0 \times 10^{-5}$)
- Pretraining used $\eta = 6.0 \times 10^{-4}$ over $450$M tokens to learn grammar, vocabulary, and world knowledge.
- The SFT dataset contains only $\sim 3.5$M tokens.
- **The Mechanics of Severe Language Forgetting**: If fine-tuned at $\eta = 6.0 \times 10^{-4}$, the large gradient steps on repetitive synthetic templates overwrite the pretraining embedding and attention projections. The model quickly forgets general language (pretraining test perplexity explodes from $23.02$ to $>72.57$), outputting corrupted sentences.
- Sizing $\eta = 6.0 \times 10^{-5}$ restricts weight updates to gentle adaptations within the existing representation space.

---

### 4.3 The Memorization Proof: Test D vs. Test A, B, C
The cross-split benchmark provides conclusive empirical proof of entity memorization in small Transformers:

| Split | Description | Plain SFT Written Acc | Plain SFT Wrong-Name Rate | Final Mitigated Recipe | Final Wrong-Name Rate |
|---|---|:---:|:---:|:---:|:---:|
| **Test D** | Familiar training names | **45.0%** | **0%** | **43.5%** | **1%** |
| **Test A** | Unseen names, seen templates | 8.0% | **88%** | **38.0%** | **14%** |
| **Test B** | Unseen names, unseen templates (OOD) | 6.3% | **92%** | **33.0%** | **15%** |
| **Test C** | Unseen names, seen numbers | 7.0% | **89%** | **35.3%** | **17%** |

#### Theoretical Interpretation:
1. On **Test D** (which uses the 26 training names), Plain SFT achieves $45\%$ written accuracy with $0\%$ wrong names.
2. On **Test A, B, and C** (which use 6 unseen names), Plain SFT's wrong-name rate jumps to **$88\%$--$92\%$**. The model generates names from its 26 training names (e.g. predicting *"विजय"* when the question is strictly about *"अमित"* and *"रोहित"*).
3. **Conclusion**: Small language models do not learn abstract logical rules (such as $A > B \land B > C \implies A > C$). Instead, they learn a statistical prior to emit specific memorized high-frequency names from the training split.
4. **The Fix**: Early stopping on `val_ood` in Hindi and Name Pool + Replay in Nepali successfully suppressed wrong-name rates to **$14\%$--$15\%$**, allowing the model to generate names dynamically extracted from the prompt.

---

### 4.4 Why Choice Accuracy Remains Near Chance (~39%)
Across all training sizes (5k to 100k samples) and all ablation techniques:
- Binary relations chance baseline: $50.0\%$
- Ternary comparisons chance baseline: $33.3\%$
- Weighted overall random chance: $\mathbf{38.5\%}$
- Measured Model H Choice Accuracy on Test B: $\mathbf{37.8\% \pm 0.9\%}$
- Measured Model L Choice Accuracy on Test B: $\mathbf{41.2\% \pm 1.8\%}$

#### Linguistic & Computational Reason:
Choice accuracy measures whether $\log P(\text{Correct Choice}) > \log P(\text{Distractor})$. Because 25.4M parameter models lack deep multi-hop routing circuits, their log-probability distribution over candidate names remains uniformly flat. Fine-tuning teaches the model the *format* of answering questions (boosting written accuracy from $0\%$ to $33\%$), but does not endow the model with multi-step relational inference.

---

### 4.5 Attention Probing Truth: No Preference for Correct Answer
The attention ratio on the query token (`"उत्तर:"`) across all 7 layers:

| Transformer Layer | Hindi Pretrained | Hindi Final SFT | Nepali Pretrained | Nepali Final SFT |
|:---:|:---:|:---:|:---:|:---:|
| **Layer 0 (Input)** | 0.99 | 0.99 | 1.05 | 1.06 |
| **Layer 3** | 1.02 | 0.97 | 0.99 | 0.97 |
| **Layer 4** | 1.00 | 1.04 | 1.03 | 1.09 |
| **Layer 5** | 1.04 | 1.05 | 0.97 | 1.06 |
| **Layer 6 (Output)** | 1.04 | 1.04 | 1.00 | 0.99 |

* **The Finding**: The attention preference ratio remains locked between **$0.97$ and $1.09$** across all layers. The query token does not attend more strongly to the logically correct entity than to the distractor entity.
* **Salience Shift**: What changed during SFT was that Layer 4 increased its total attention mass on **all entity names by $\sim 40\%$**. The model learned the semantic category (*"I should output a name"*), but not the logical deduction (*"This specific name is the answer"*).

---

### 4.6 Category Difficulty Hierarchy
Breakdown of Test B accuracy by problem category:

| Problem Category | Chance Baseline | Final Hindi Acc | Final Nepali Acc | Computational Mechanism |
|---|:---:|:---:|:---:|---|
| **Mixed / Equality** (*"बराबर"*) | 33% | **44%** | **74%** | Identity matching between two identical numeric strings |
| **Numeric Comparison** | 50% | **61%** | **48%** | Direct single-step digit value comparison |
| **Direct Comparison** | 50% | **39%** | **47%** | Single-step comparison with lexical polarity |
| **2-Step Relation ($A>B>C$)** | 50% | **45%** | **59%** | Multi-hop transitivity |
| **3-Step Relation ($A>B>C>D$)** | 50% | **59%** | **50%** | Non-adjacent entity transitivity |
| **2-Step Chain (Place)** | 33% | **42%** | **36%** | Multi-hop extreme selection |
| **3-Step Chain (Place)** | 25% | **32%** | **34%** | 4-element sorting |
| **Ranking (Median)** | 33% | **36%** | **42%** | Total order sorting (Locked near chance) |

* **Why Equality Detection Succeeded ($74\%$)**: Equality detection requires string and token identity matching (e.g., detecting that token "25" matches token "25"), which self-attention heads can resolve via single-layer direct copy mechanisms.
* **Why Multi-Hop Transitivity Remained Near Chance**: Resolving $A > B > C \implies A > C$ requires composing two separate attention hops across intermediate tokens, which exceeds the circuit capacity of small 7-layer models without Chain-of-Thought scratchpads.

---

## 5. Execution & Verification Commands

To reproduce every Phase 3 dataset, training run, evaluation split, and attention probe from the terminal:

```bash
# 1. Generate synthetic reasoning datasets with strict 3-way partition:
python scripts/make_reasoning_data.py --lang hindi
python scripts/make_reasoning_data.py --lang nepali

# 2. Run automated position shortcut audit across all splits:
python scripts/shortcut_audit.py

# 3. Fine-tune Hindi and Nepali models with prompt loss masking:
python -m lmagpt.finetune --lang hindi --lr 6e-5 --epochs 1
python -m lmagpt.finetune --lang nepali --lr 6e-5 --epochs 1

# 4. Run sample size sweep (5k, 10k, 20k, 30k, 100k):
python scripts/sample_sweep.py

# 5. Evaluate final models across Test A, B, C, and D:
python scripts/eval_sweep_ckpts.py

# 6. Audit general pretraining language retention (PPL on 51M test.bin):
python scripts/eval_finetuned_ppl.py

# 7. Probe query token attention weights across layers:
python scripts/attention_reasoning.py

# 8. Score Nepali written accuracy with clitic/ending normalization:
python scripts/score_with_endings.py

# 9. Generate all Phase 3 figures (validation curves & ablation summary):
python scripts/make_figures.py
```
