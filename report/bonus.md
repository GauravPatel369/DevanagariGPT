# Bonus Deliverable: Experimental Ablations & Investigations

This document summarizes the bonus investigations completed across the project, their empirical results, theoretical mechanisms, and corresponding directory artifacts.

All code and experiments in this document are built directly on top of the `phase-3` branch on the `bonus` branch.

---

## Directory of Bonus Artifacts

The following table indexes every bonus file, its role, and its location in the repository:

| Component | Path / Directory | Description |
|---|---|---|
| **Model Configuration** | [`hindi/configs/model_norope.yaml`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/hindi/configs/model_norope.yaml) | Full 25.4M model config with `use_rope: false` |
| **Model Architecture** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L40-L90) | Implementation supporting RoPE toggle (`use_rope` flag) |
| **Attention Mechanism** | [`lmagpt/attention.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/attention.py) | RoPE rotation vs. identity key/query transformations |
| **Kaggle Kernel** | [`kaggle/train_hindi_norope/`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/kaggle/train_hindi_norope) | Self-contained kernel script for training the ablation |
| **Training Logs** | [`report/Phase 2/logs/hindi_norope_train_log.jsonl`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/report/Phase%202/logs/hindi_norope_train_log.jsonl) | Step-by-step training loss, learning rate, and throughput |
| **Evaluation Metrics** | [`report/Phase 2/ablation/evaluation_test.json`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/report/Phase%202/ablation/evaluation_test.json) | Complete 51.4M-token test evaluation (PPL, BPB, BLEU, chrF++) |
| **Attention Diagnostics** | [`report/Phase 2/ablation/attention_analysis.json`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/report/Phase%202/ablation/attention_analysis.json) | Layer-wise entropy, mean attention distance, and head specializations |
| **Ablation Figures** | [`report/Phase 2/ablation/figures/`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/report/Phase%202/ablation/figures) | Attention heatmaps and summary comparison charts |
| **Order Invariance Proof**| [`tests/test_model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/tests/test_model.py) | Unit tests verifying prefix permutation sensitivity by depth |
| **Extended Pretraining** | [`hindi/reports/two_epoch/`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/hindi/reports/two_epoch) | 2-Epoch (554.9M token) pretraining run evaluation and logs |
| **Vocab Size Sweeps** | [`report/Phase 2/vocab_sweep/`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/report/Phase%202/vocab_sweep) | Complete 5k, 8k, 10k, 16k tokenizer & model sweep results |

---

## 1. Primary Bonus: Positional Encoding Ablation (No-RoPE)

### 1.1 Motivation & Clean Separation
A critical question in Transformer architecture design is whether explicit positional encodings (such as RoPE) are necessary, or if the autoregressive causal mask alone provides sufficient positional signal.

Because Rotary Position Embedding (RoPE) is applied directly as a coordinate rotation on query and key vectors using precomputed sinusoidal buffers, **it contributes 0 learnable parameters**. 
* Baseline Model H (RoPE): **25,350,912 parameters**
* Ablated Model H (No Positional Encoding): **25,350,912 parameters**

Unlike learned positional embeddings (which would add or subtract $262,144$ parameters and confound capacity with positional signaling), this ablation provides a **strictly controlled scientific comparison**: identical parameters, identical token budget ($\sim 450$M tokens, 27,500 steps), identical optimizer/LR schedule, and identical random seed.

---

### 1.2 Quantitative Results

Scored over the complete 51,399,168-token held-out test split:

| Metric | Baseline (RoPE) | Ablated (No-RoPE) | Absolute $\Delta$ | Relative Change |
|---|:---:|:---:|:---:|:---:|
| **Parameters** | 25,350,912 | 25,350,912 | 0 | 0.0% |
| **Test Cross-Entropy** | 3.1942 nats | 3.3225 nats | +0.128 nats | +4.0% |
| **Test Perplexity (PPL)** | **24.39** | **27.73** | **+3.34** | **+13.7% (degraded)** |
| **Test Bits-Per-Byte (BPB)** | **0.4709** | **0.4898** | **+0.0189** | **+4.0% (degraded)** |
| **Compression Ratio** | 16.99x | 16.33x | -0.66x | Worse compression |
| **Best chrF++** | 21.80 | 21.89 | +0.09 | Indistinguishable |
| **Best BLEU** | 23.87 | 23.72 | -0.15 | Indistinguishable |
| **Decoding Sweet Spot** | $T=1.0, p=0.95$ | $T=1.0, p=0.95$ | — | Unchanged |

---

### 1.3 Key Theoretical Findings

1. **Explicit Position Accounts for $\sim 14\%$ of Language Modeling Capacity**:
   Removing positional encoding does **not** cause the model to collapse (PPL reaches 27.73 vs. ~10,000 for an untrained model). However, the model suffers a $+13.7\%$ perplexity penalty, establishing the precise quantitative value of explicit position representation.

2. **Why the Model is Not Order-Blind (The Causal Mask as a Ruler)**:
   In a causal decoder, token $t$ attends to exactly $t$ predecessor tokens. Thus, the attention mask itself encodes prefix length.
   * At **Layer 1**, the final position pools over preceding tokens as an unordered set (prefix permutations yield zero change in logits: $\Delta \approx 5.96 \times 10^{-8}$).
   * From **Layer 2 onwards**, positions attend over contextualized representations shaped by earlier causal masks, breaking permutation invariance ($\Delta \approx 1.3 \times 10^{-2}$).

3. **Internal Layer Re-Specialization**:
   Analyzing layer-wise attention distributions over 16 held-out 256-token passages reveals how the No-RoPE model compensates:

   | Layer | Entropy (RoPE) | Entropy (No-RoPE) | Mean Dist (RoPE) | Mean Dist (No-RoPE) |
   |:---:|:---:|:---:|:---:|:---:|
   | 0 | 5.76 bits | 6.17 bits | 44.5 tokens | **62.3 tokens (diffuse)** |
   | 1 | 4.94 bits | 5.09 bits | 29.9 tokens | **56.5 tokens (diffuse)** |
   | 2 | 4.38 bits | 5.96 bits | 29.8 tokens | **56.2 tokens (diffuse)** |
   | 3 | 4.20 bits | 4.82 bits | 30.3 tokens | 41.8 tokens |
   | **4** | 3.77 bits | **2.98 bits** | 25.1 tokens | **5.4 tokens (hyper-local)** |
   | **5** | 3.71 bits | **2.66 bits** | 29.9 tokens | **9.0 tokens (hyper-local)** |
   | 6 | 3.52 bits | 4.33 bits | 55.5 tokens | 59.4 tokens |

   * **Early Layers (0–2)** expand their receptive field to broad global contexts (mean distance expands to $56$–$62$ tokens).
   * **Middle Layers (4–5)** collapse into hyper-local order detectors (mean distance collapses to **5.4** and **9.0** tokens; entropy drops to 2.66 bits).
   * **Conclusion**: The model sacrifices two entire hidden layers (Layers 4 & 5) purely to reconstruct sequence order from the causal mask.

---

## 2. Additional Bonus Investigations

### 2.1 Multi-Epoch Pretraining (2 Full Epochs on Hindi)
* **Dataset**: Complete 554.9M tokens (2 epochs across corpus).
* **Result**: Test PPL improved from **24.39** (1 epoch) to **23.02** (2 epochs), confirming that sub-billion token models continue to gain non-trivial linguistic structure without overfitting.
* **Artifacts**: Located in [`hindi/reports/two_epoch/`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/hindi/reports/two_epoch).

### 2.2 Tokenizer Vocabulary Size Sweeps (5k, 8k, 10k, 16k)
* Swept across both Hindi and Nepali to determine the optimal fertility/compression trade-off.
* **Finding**: While 16k yields lower fertility (fewer tokens per word), 10,000 subwords maximizes Bits-Per-Byte efficiency and minimizes rare-token fragmentation.
* **Artifacts**: Located in [`report/Phase 2/vocab_sweep/`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/report/Phase%202/vocab_sweep).

---

## 3. How to Reproduce

### Run Model Evaluation
```bash
python -m lmagpt.evaluate --lang hindi --split test \
       --checkpoint hindi_norope/checkpoints/last.pt \
       --out "report/Phase 2/ablation"
```

### Run Attention Diagnostics
```bash
python -m lmagpt.attention --lang hindi \
       --checkpoint-dir hindi_norope/stage \
       --stats-tokens 256 --stats-passages 16 \
       --out "report/Phase 2/ablation"
```

### Run Prefix Permutation Unit Tests
```bash
pytest tests/test_model.py -k "test_permutation"
```
