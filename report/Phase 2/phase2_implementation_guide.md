# Phase 2: Complete Architecture, Pretraining & End-to-End System Guide

**Project**: Monolingual Transformer Language Models — Hindi (`hi`) & Nepali (`ne`)  
**Course**: Language Models and Agents (Monsoon 2026)  
**Author**: Gaurav Patel  
**Repository Branch**: `phase-2` / `phase-3`  
**Primary Source Modules**: [`lmagpt/train.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py), [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py), [`lmagpt/data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/data.py), [`lmagpt/evaluate.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/evaluate.py), [`lmagpt/attention.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/attention.py)

---

## 1. Overview & Conceptual Architecture

This guide delivers an exhaustive, dual-perspective technical blueprint of **Phase 2**:
1. **"Where is implemented what"**: A file-by-file, class-by-class, and function-by-function index mapping every theoretical construct directly into the code.
2. **"End-to-End Flow (`train.py` & `model.py`)"**: A step-by-step trace of how data, tensors, gradients, optimizer states, and checkpoints flow through the complete training and inference lifecycle.
3. **"What does that mean in that way"**: The mathematical derivations, first-principles implementations, and linguistic rationales behind every design choice.

```
══════════════════════════════════════════════════════════════════════════════════════════════════════
                               PHASE 2 END-TO-END SYSTEM LIFECYCLE
══════════════════════════════════════════════════════════════════════════════════════════════════════

  [Tokenized Binary Files: train.bin, val.bin]
                     │
                     ▼
  [TokenDataset Dataloader: Memory-mapped uint16] ──► Draws micro-batches: x, y ~ (B=16, T=512)
                     │
                     ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                    lmagpt/model.py : GPT                                   │
  │                                                                                            │
  │  Input Token IDs: x ~ (B, T) ∈ ℤ_{10000}                                                   │
  │       │                                                                                    │
  │       ▼                                                                                    │
  │  Token Embedding Table (10,000 × 512) + Dropout(p=0.1) ──────────► (B, T, 512)             │
  │       │                                                                                    │
  │       ▼                                                                                    │
  │  7 × Pre-LayerNorm Transformer Blocks (l = 0 .. 6):                                        │
  │    ┌──────────────────────────────────────────────────────────────────────────────────┐    │
  │    │ 1. Multi-Head Attention Sublayer:                                                │    │
  │    │    x_norm = LayerNorm(x)                                                         │    │
  │    │    Q, K, V = Linear_Q(x_norm), Linear_K(x_norm), Linear_V(x_norm)  [512 → 512]   │    │
  │    │    Split into 8 heads (d_k = 64) ───────────────────────────► (B, 8, T, 64)      │    │
  │    │    Apply RoPE to Q and K (2D complex rotation) ─────────────► q_rot, k_rot       │    │
  │    │    Scaled Dot-Product: Scores = (q_rot @ k_rot^T) / sqrt(64) ──► (B, 8, T, T)    │    │
  │    │    Causal Mask: Add -inf for future positions (j > i)                            │    │
  │    │    Softmax + Attn Dropout ──────────────────────────────────► Post-Softmax W     │    │
  │    │    Context = W @ V ─────────────────────────────────────────► (B, 8, T, 64)      │    │
  │    │    Concat heads + Out Projection (512 → 512) + Resid Dropout                     │    │
  │    │    Residual Connection: x = x + Attn_Output                                      │    │
  │    │                                                                                  │    │
  │    │ 2. Feed-Forward Network (FFN) Sublayer:                                          │    │
  │    │    x_norm = LayerNorm(x)                                                         │    │
  │    │    Linear Expansion (512 → 1792 = 3.5 × d_model)                                 │    │
  │    │    GELU Non-Linearity                                                            │    │
  │    │    Linear Contraction (1792 → 512) + Resid Dropout                               │    │
  │    │    Residual Connection: x = x + FFN_Output                                       │    │
  │    └──────────────────────────────────────────────────────────────────────────────────┘    │
  │       │                                                                                    │
  │       ▼                                                                                    │
  │  Final LayerNorm (512)                                                                     │
  │       │                                                                                    │
  │       ▼                                                                                    │
  │  Tied Output LM Head (re-uses Token Embedding Table W_emb^T) ────► Logits: (B, T, 10000)   │
  └───────────────────────────────────────┬────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                    lmagpt/train.py                                         │
  │                                                                                            │
  │  1. Shifted Cross-Entropy Loss: ℒ = CrossEntropy(Logits, Targets y)                         │
  │  2. Gradient Accumulation (accum=2 micro-batches ⟹ 16,384 tokens/step)                     │
  │  3. Mixed Precision: GradScaler (fp16) or native autocast (bf16)                           │
  │  4. Gradient Clipping: torch.nn.utils.clip_grad_norm_(params, 1.0)                          │
  │  5. Optimizer Step: AdamW (lr=6e-4, beta=(0.9, 0.95), weight_decay=0.1 on 2D weights only)│
  │  6. Cosine LR Schedule with 500-step linear warmup (decays to 6e-5 floor)                  │
  │  7. Atomic Checkpointing: saves model, optimizer, scheduler, RNGs to last.pt / best.pt     │
  │  8. Wall-Clock Budget Guard (--max-hours): exits cleanly before cloud timeout               │
  └────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Where is Implemented What: Codebase Directory Map

The table below catalogs every component, function, and file in Phase 2:

| Component / Subsystem | Source File Path | Key Classes & Functions | Execution CLI Command | Primary Outputs & Artifacts |
|---|---|---|---|---|
| **Model Hyperparameters** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L46-L78) | `GPTConfig`, `__post_init__()`, `to_dict()` | Imported by train & eval | `model.yaml`, `model_norope.yaml` |
| **RoPE Mathematical Rotation** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L82-L126) | `build_rope_cache()`, `apply_rope()` | `pytest tests/test_model.py` | Rotary cos/sin buffers |
| **Causal Self-Attention** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L131-L198) | `CausalSelfAttention`, `forward()` | Internal to `Block` | Scaled dot-product, post-softmax weights |
| **Feed-Forward Network (FFN)** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L203-L221) | `FeedForward`, `forward()` | Internal to `Block` | $3.5\times$ intermediate activations |
| **Transformer Block (Pre-LN)** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L223-L244) | `Block`, Pre-LN residual dataflow | Internal to `GPT` | Identity residual stream |
| **Complete GPT Model** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L248-L347) | `GPT`, `forward()`, `_init_weights()`, tied head | Base class for training | 25,350,912-parameter network |
| **Autoregressive Generation Engine** | [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L349-L449) | `generate()`, `_apply_repetition_penalty()`, `_block_repeat_ngrams()` | `python scripts/generate.py` | `generation_examples.json`, `generation_session.jsonl` |
| **Dataset Dataloader** | [`lmagpt/data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/data.py) | `TokenDataset`, `batch()`, `deterministic_batches()` | Internal to train & eval | Memory-mapped `(B, T)` token batches |
| **Pretraining Engine** | [`lmagpt/train.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py) | `main()`, `lr_at()`, `build_optimizer()`, `save_checkpoint()`, `load_checkpoint()` | `python -m lmagpt.train --lang {hi,ne}` | `checkpoints/{last,best}.pt`, `train_log.jsonl` |
| **Hardware & Environment Bootstrap**| [`lmagpt/kaggle_bootstrap.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/kaggle_bootstrap.py) | `bootstrap_env()`, sm_60 detection | Auto-run on Kaggle P100 | PyTorch 2.5.1+cu121 native binaries |
| **Intrinsic Evaluation (PPL & BPB)**| [`lmagpt/evaluate.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/evaluate.py#L54-L100) | `intrinsic_metrics()`, `add_bits_per_byte()` | `python -m lmagpt.evaluate --lang {hi,ne}` | `report/Phase 2/evaluation_test.json`, `evaluation_val.json` |
| **Generation Quality Evaluation** | [`lmagpt/evaluate.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/evaluate.py#L140-L310) | `compute_bleu()`, `compute_chrf()`, `compute_rouge()`, `distinct_n()`, `rep_n()` | `python -m lmagpt.evaluate --lang {hi,ne}` | 12-decoder sweep benchmarks |
| **Attention Probing & Diagnostics** | [`lmagpt/attention.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/attention.py) | `collect_attention()`, `attention_entropy()`, `mean_attention_distance()` | `python -m lmagpt.attention --lang {hi,ne}` | `attention_analysis.json`, head categorization |
| **Attention Heatmap Plotting** | [`lmagpt/attention.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/attention.py#L220-L290) | `plot_heatmaps()`, `plot_summary()` | `python -m lmagpt.attention --lang {hi,ne}` | `hindi_attention_heatmaps.png`, `nepali_attention_heatmaps.png` |
| **Training Dynamics Plotting** | [`scripts/plot_training_curves.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/plot_training_curves.py) | `plot_loss()`, `plot_ppl()`, `plot_lr_throughput()`, `plot_grad_norm()` | `python scripts/plot_training_curves.py` | `training_loss.png`, `validation_perplexity.png`, `gradient_norm.png` |
| **Vocab Sweep Training & Plots** | [`scripts/train_and_eval_multivocab.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/train_and_eval_multivocab.py), [`scripts/plot_vocab_sweep.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/scripts/plot_vocab_sweep.py) | `train_sweep()`, `plot_sweep_bpb()` | `python scripts/plot_vocab_sweep.py` | `vocab_sweep.png`, `vocab_sweep_bpb.png` |
| **Unit Test Suite (63 Tests)** | [`tests/test_model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/tests/test_model.py) | `test_causal_mask()`, `test_rope_is_relative()`, `test_param_count()` | `pytest tests/` | 63 Passing Test Assertions |

---

## 3. End-to-End System Flow: Step-by-Step Code Execution

Here is the exact runtime trace through [`lmagpt/train.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py) and [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py).

### Step 1: Configuration Loading & Verification
1. **Entry Point**: `python -m lmagpt.train --lang hindi` calls `main()` in [`lmagpt/train.py:L145`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py#L145).
2. **YAML Parsing**: Reads `hindi/configs/model.yaml` containing two sections:
   - `model`: `vocab_size: 10000`, `context: 512`, `d_model: 512`, `n_layer: 7`, `n_head: 8`, `d_ff: 1792`, `dropout: 0.1`, `tie_embeddings: true`, `rope_theta: 10000.0`, `use_rope: true`.
   - `train`: `max_steps: 27500`, `learning_rate: 0.0006`, `min_lr: 0.00006`, `warmup_steps: 500`, `micro_batch: 16`, `grad_accum: 2`, `weight_decay: 0.1`, `beta1: 0.9`, `beta2: 0.95`, `grad_clip: 1.0`, `seed: 1337`.
3. **Data Class Instantiation**: `model_cfg = GPTConfig(**mcfg)` validates that `d_model % n_head == 0` ($512/8 = 64$) and `head_dim % 2 == 0$ ($64\%2 = 0$).

### Step 2: Precision & Hardware Bootstrap
1. **CUDA & Device Detection**: Lines 169–179 in `train.py` detect available hardware:
   - On NVIDIA Ampere+ (compute capability $\ge 8.0$): selects `torch.bfloat16` with native autocast and disables `GradScaler`.
   - On NVIDIA Pascal / Turing (e.g. Tesla P100 `sm_60`, RTX 2060): selects `torch.float16` and enables `torch.amp.GradScaler`.
   - Enables TF32 for fast Tensor Core matrix multiplications: `torch.backends.cuda.matmul.allow_tf32 = True`.
2. **Seed Setting**: `torch.manual_seed(1337)`.

### Step 3: Model Instantiation & Weight Initialization
In [`lmagpt/model.py:L248-L299`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L248-L299), `GPT(model_cfg)` executes:
1. **Module Allocation**:
   - `token_emb`: `nn.Embedding(10000, 512)`
   - `blocks`: `nn.ModuleList([Block(cfg) for _ in range(7)])`
   - `norm_f`: `nn.LayerNorm(512)`
   - `lm_head`: `nn.Linear(512, 10000, bias=False)`
2. **Weight Tying**:
   - `self.lm_head.weight = self.token_emb.weight` (points to the exact same tensor in memory).
3. **RoPE Cache Buffer**:
   - `build_rope_cache(512, 64)` builds `cos` and `sin` tables of shape `(512, 32)` and registers them as non-persistent buffers (`rope_cos`, `rope_sin`).
4. **Custom Weight Initialization**:
   - Linear and Embedding weights: $\mathcal{N}(0, 0.02)$, biases initialized to $0$.
   - **Residual Output Scaling**: Weights in `out_proj` and `down` are scaled down by $\frac{1}{\sqrt{2 \cdot n_{\text{layer}}}} = \frac{1}{\sqrt{14}} \approx 0.267$ to prevent residual stream variance explosion across 7 layers:
     $$\sigma_{\text{resid}} = \frac{0.02}{\sqrt{2 \times 7}} \approx 0.005345$$
5. **Exact Parameter Verification**: `model.num_parameters()` returns **25,350,912**.

### Step 4: Selective Optimizer Construction
In [`lmagpt/train.py:L66-L88`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py#L66-L88), `build_optimizer(model, tcfg)` splits parameters into two groups:
1. **Decay Group (`dim >= 2`)**: $W_Q, W_K, W_V, W_{\text{out}}, W_{\text{up}}, W_{\text{down}}, W_{\text{emb}}$ receive `weight_decay = 0.1`.
2. **No-Decay Group (`dim < 2`)**: LayerNorm scale ($\gamma$) and shift ($\beta$), plus all 1D bias vectors, receive `weight_decay = 0.0`.
3. Returns `torch.optim.AdamW(groups, lr=6e-4, betas=(0.9, 0.95), eps=1e-8)`.

### Step 5: Dataset & Dataloader Memory-Mapping
In [`lmagpt/data.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/data.py):
1. Opens `train.bin` (uint16 binary token stream) via `np.memmap(..., mode="r", dtype=np.uint16)`.
2. Total train tokens: $450,601,984$ (Hindi) and $450,864,128$ (Nepali).
3. **Deterministic vs. Random Batching**:
   - During evaluation: `deterministic_batches()` yields fixed, non-overlapping sequential windows.
   - During training: `train_ds.batch(micro, gen)` samples random starting offsets $i \sim \mathcal{U}(0, N - T - 1)$ where `gen` is re-seeded at each step: `gen = np.random.default_rng(seed + step)`.
   - Returns input tensor $x = \text{tokens}[i : i+T]$ and target tensor $y = \text{tokens}[i+1 : i+T+1]$ of shape `(16, 512)`.

### Step 6: Detailed Forward Pass Tensor Dataflow
Tracing a single forward call `model(x, targets=y)` in [`lmagpt/model.py:L300-L347`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L300-L347):

```
1. Input: idx ~ (B=16, T=512) int64
2. Embedding: x = emb_dropout(token_emb(idx)) ──────────────► (16, 512, 512)
3. RoPE Cache Slice: cos = rope_cos[:512], sin = rope_sin[:512] ─► (512, 32)
4. Loop through 7 Transformer Blocks:
   ┌── For Block l = 0 .. 6:
   │   a. Pre-LayerNorm 1: x_norm1 = ln1(x) ────────────────► (16, 512, 512)
   │   b. Q, K, V Projections:
   │      q = q_proj(x_norm1).view(16, 512, 8, 64).transpose(1, 2) ─► (16, 8, 512, 64)
   │      k = k_proj(x_norm1).view(16, 512, 8, 64).transpose(1, 2) ─► (16, 8, 512, 64)
   │      v = v_proj(x_norm1).view(16, 512, 8, 64).transpose(1, 2) ─► (16, 8, 512, 64)
   │   c. RoPE Application (Q and K only):
   │      q_rot = apply_rope(q, cos, sin) ───────────────────► (16, 8, 512, 64)
   │      k_rot = apply_rope(k, cos, sin) ───────────────────► (16, 8, 512, 64)
   │   d. Scaled Dot-Product Attention:
   │      scores = (q_rot @ k_rot^T) / sqrt(64) ─────────────► (16, 8, 512, 512)
   │      scores = scores.masked_fill(causal_mask == 1, -inf)
   │      weights = attn_dropout(softmax(scores, dim=-1)) ───► (16, 8, 512, 512)
   │      attn_out = weights @ v ────────────────────────────► (16, 8, 512, 64)
   │   e. Output Projection:
   │      attn_out = attn_out.transpose(1, 2).contiguous().view(16, 512, 512)
   │      attn_out = resid_dropout(out_proj(attn_out)) ──────► (16, 512, 512)
   │   f. Residual Connection 1:
   │      x = x + attn_out ──────────────────────────────────► (16, 512, 512)
   │   g. Pre-LayerNorm 2: x_norm2 = ln2(x) ────────────────► (16, 512, 512)
   │   h. Feed-Forward Network:
   │      ffn_out = up(x_norm2) ─────────────────────────────► (16, 512, 1792)
   │      ffn_out = gelu(ffn_out) ───────────────────────────► (16, 512, 1792)
   │      ffn_out = resid_dropout(down(ffn_out)) ────────────► (16, 512, 512)
   │   i. Residual Connection 2:
   │      x = x + ffn_out ───────────────────────────────────► (16, 512, 512)
   └── End Block Loop
5. Final LayerNorm: x = norm_f(x) ───────────────────────────► (16, 512, 512)
6. Output Head Projection: logits = lm_head(x) ──────────────► (16, 512, 10000)
7. Cross-Entropy Loss:
   loss = F.cross_entropy(logits.view(-1, 10000), targets.reshape(-1), ignore_index=-100)
```

### Step 7: Backward Pass, Gradient Accumulation & Optimizer Step
In [`lmagpt/train.py:L226-L253`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py#L226-L253):
1. **Zero Gradients**: `optimizer.zero_grad(set_to_none=True)` (sets `.grad = None` to save VRAM memory writes).
2. **Accumulation Loop (`accum = 2`)**:
   - Iteration 1: Forward pass on micro-batch 1 ($B=16$), compute $\mathcal{L}_1 / 2$, call `backward()`.
   - Iteration 2: Forward pass on micro-batch 2 ($B=16$), compute $\mathcal{L}_2 / 2$, call `backward()`.
   - Effective batch size: $16 \times 2 \times 512 = \mathbf{16,384\text{ tokens/step}}$.
3. **Gradient Unscaling & Clipping**:
   - `scaler.unscale_(optimizer)`
   - `grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
4. **Optimizer Update**:
   - `scaler.step(optimizer)` updates weights using AdamW equations.
   - `scaler.update()` adjusts FP16 dynamic loss scale factor.
5. **Step Counter**: `step += 1`.

### Step 8: Learning Rate Schedule Updates
In [`lmagpt/train.py:L48-L64`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py#L48-L64), `lr_at(step, tcfg)` updates `optimizer.param_groups[0]["lr"]`:
- **Warmup Stage ($0 \le \text{step} < 500$)**:
  $$\eta_{\text{step}} = 6.0 \times 10^{-4} \times \frac{\text{step} + 1}{500}$$
- **Cosine Decay Stage ($500 \le \text{step} \le 27500$)**:
  $$\text{progress} = \frac{\text{step} - 500}{27500 - 500}, \quad \eta_{\text{step}} = 6.0 \times 10^{-5} + \frac{1}{2}(6.0 \times 10^{-4} - 6.0 \times 10^{-5})(1 + \cos(\pi \cdot \text{progress}))$$

### Step 9: Atomic Resumable Checkpointing
In [`lmagpt/train.py:L93-L124`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/train.py#L93-L124):
1. **Checkpoint Payload**: Packages `model.state_dict()`, `optimizer.state_dict()`, `step`, `tcfg`, `model_cfg.to_dict()`, `best_val`, `numpy_rng`, and `torch_rng`.
2. **Atomic Write**: Saves to `last.pt.tmp` then calls `os.replace("last.pt.tmp", "last.pt")`. This guarantees that if a cloud session is killed mid-write, the existing checkpoint remains uncorrupted.
3. **Best Checkpoint**: If validation loss achieves a new minimum, saves a separate copy to `best.pt`.
4. **Wall-Clock Guard (`--max-hours`)**: If elapsed time exceeds deadline, cleanly saves `last.pt` and exits with code 0 before platform SIGKILL.

### Step 10: Autoregressive Generation & Decoding Loop
In [`lmagpt/model.py:L383-L449`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L383-L449), `model.generate(prompt_ids, max_new_tokens=100, ...)` runs autoregressively:
1. Crop prompt to the last 512 tokens: `window = idx[:, -512:]`.
2. Forward pass through model to get logits: `logits = self(window)[:, -1, :]` (last sequence position only).
3. **Repetition Penalty**: If $\theta > 1.0$, for all tokens already generated:
   $$\text{logit} = \begin{cases} \text{logit} / \theta & \text{if } \text{logit} > 0 \\ \text{logit} \cdot \theta & \text{if } \text{logit} \le 0 \end{cases}$$
4. **No-Repeat N-gram Hard Blocking**: If $n$-gram prefix matches a previous occurrence, mask candidate next-token to $-\infty$.
5. **Temperature Scaling**: `logits = logits / max(temperature, 1e-6)`.
6. **Top-K Truncation**: Keep only top $K$ values, mask remaining to $-\infty$.
7. **Top-P Nucleus Truncation**:
   - Sort logits descending, compute cumulative softmax probabilities.
   - Shift cumulative sum right by subtracting individual softmax to guarantee at least one token survives.
   - Mask elements where cumulative sum exceeds $p$ to $-\infty$.
8. **Sampling**: Sample next token via `torch.multinomial(softmax(logits), num_samples=1)` or `logits.argmax()` if greedy.
9. **Append & Repeat**: Concatenate new token ID to sequence: `idx = torch.cat([idx, nxt], dim=1)`. Stop if `eos_id` is emitted.

---

## 4. What Does That Mean in That Way? Mathematical & Conceptual Deep-Dive

This section explains the exact theory, mathematics, and empirical reasoning behind every architectural decision.

---

### 4.1 PyTorch Primitives Constraint & Materialized Attention
- **The Constraint**: Black-box Transformer modules (`nn.Transformer`, HuggingFace `AutoModel`, or pre-built attention blocks) are strictly forbidden.
- **Implementation**: [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) uses only `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, and `nn.Dropout`.
- **Why Avoid `F.scaled_dot_product_attention`**:
  1. It is a pre-packaged attention wrapper.
  2. Its fused FlashAttention CUDA kernels compute attention in SRAM without materializing the $(B, H, T, T)$ attention weight matrix. To perform Phase 2 attention entropy probing, attention distance analysis, and heatmap visualization, the explicit $(B, H, T, T)$ tensor **must be materialized**.

---

### 4.2 Scaled Dot-Product: Why Divide by $\sqrt{d_k} = 8$?
- Let $q, k \in \mathbb{R}^{d_k}$ with $d_k = 64$.
- The dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ sums 64 independent random variables.
- Assuming zero mean and unit variance ($\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$, $\text{Var}(q_i) = \text{Var}(k_i) = 1$):
  $$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k = 64$$
- **The Problem**: Without scaling, logits would have standard deviation $\sigma = 8$. For large inputs, the softmax function $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$ collapses into an extreme one-hot distribution. Its Jacobian $\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$ vanishes toward $0$, completely extinguishing the backpropagated gradient.
- **The Solution**: Dividing by $\sqrt{d_k} = \sqrt{64} = 8$ normalizes the variance back to $1.0$, keeping gradients healthy throughout training.

---

### 4.3 Causal Masking Mechanics & Proof of Future-Blindness
- **Definition**: $M_{\text{causal}}(i, j) = -\infty$ for $j > i$, and $0$ for $j \le i$.
- **Why $-\infty$ before softmax**:
  $$p(j \mid i) = \frac{\exp\left( \frac{q_i \cdot k_j}{\sqrt{d_k}} + M_{ij} \right)}{\sum_{k=1}^T \exp\left( \frac{q_i \cdot k_k}{\sqrt{d_k}} + M_{ik} \right)}$$
  For any future token $j > i$, $\exp(-\infty) = 0.0000$. The token receives zero attention mass, ensuring the model never sees future context.
- **Unit Test Proofs in `tests/test_model.py`**:
  1. *Identical Prefix Test*: Sequences with identical prefixes up to token $t$ and completely different tokens after $t$ yield **bit-identical logits** at all positions $\le t$.
  2. *Strict Diagonal Test*: $W_{\text{attn}}[b, h, i, j] == 0.0$ for all $j > i$.
  3. *Position 0 Self-Attention*: At position $t=0$, $100.0\%$ of attention mass is on position 0.

---

### 4.4 Rotary Position Embeddings (RoPE) Formulation
- **File**: [`lmagpt/model.py:L82-L126`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py#L82-L126)
- **Concept**: Instead of adding an absolute position vector to the token embedding ($x + p$), RoPE rotates the query and key vectors in 2D planes by position-dependent angles.
- **Mathematical Formulation**: For a 2D vector pair $(x_{2i}, x_{2i+1})$ at sequence position $m$:
  $$\begin{pmatrix} x_{2i}' \\ x_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}, \quad \theta_i = 10000^{-2i/d_k}$$
- **Why RoPE is Superior to Learned Embeddings**:
  1. **Zero Trainable Parameters**: A learned table ($512 \times 512 = 262,144$ weights) wastes budget. RoPE is a precomputed trigonometric buffer, freeing parameter budget for an entire 7th Transformer block.
  2. **Relative Distance by Construction**: The inner product between query at position $m$ and key at position $n$ is:
     $$\langle R_m q, R_n k \rangle = q^T R_m^T R_n k = q^T R_{n-m} k$$
     Attention scores depend purely on the relative distance $(m - n)$, allowing natural generalizability across positions.
  3. **Applied Strictly to $Q$ and $K$ (Never $V$)**: $Q$ and $K$ determine *where to look* (position-dependent routing), while $V$ contains *what information to extract* (content-dependent value).

---

### 4.5 Dimensioning: Buying Depth with Width ($d_{\text{ff}} = 1792$)
- Standard GPT configurations use $d_{\text{ff}} = 4 \times d_{\text{model}} = 2048$.
- At $4\times$, a 7-layer model requires $27,188,736$ parameters, exceeding the 25.4M budget ceiling.
- **The Design Trade-Off**: Reducing FFN expansion to $3.5 \times d_{\text{model}} = 1792$ saved $1,837,824$ parameters. This enabled allocating a **7th Transformer block** while keeping total parameters at **25,350,912** ($99.8\%$ of the 25.4M limit).
- **Theoretical Value**: In natural language, deeper networks compose hierarchical linguistic abstractions (phonemes $\rightarrow$ morphemes $\rightarrow$ syntax $\rightarrow$ long-range discourse) far more effectively than wider single-layer MLPs.

#### Complete Parameter Accounting Table:
| Module Component | Tensor Dimensions & Mathematical Formula | Parameters | Share |
|---|---|:---:|:---:|
| **Token Embedding Table** | $V \times d_{\text{model}} = 10,000 \times 512$ | 5,120,000 | 20.20% |
| **Attention $Q, K, V$ Projections** | $7 \times 3 \times (512 \times 512 + 512)$ | 5,531,904 | 21.82% |
| **Attention Output Projections** | $7 \times (512 \times 512 + 512)$ | 1,843,968 | 7.27% |
| **FFN Expansion Projections** | $7 \times (512 \times 1792 + 1792)$ | 6,437,056 | 25.39% |
| **FFN Contraction Projections** | $7 \times (1792 \times 512 + 512)$ | 6,419,968 | 25.32% |
| **Block LayerNorms (2 per block)** | $7 \times 2 \times (512 + 512)$ | 14,336 | 0.06% |
| **Final LayerNorm** | $512 + 512$ | 1,024 | <0.01% |
| **Output Head** | Reuses Token Embedding Table | 0 (Tied) | 0.00% |
| **Total Trainable Parameters** | | **25,350,912** | **100.00%** |

---

### 4.6 Pre-LayerNorm vs. Post-LayerNorm Dynamics
- In **Pre-LN** (`x = x + SubLayer(LayerNorm(x))`), the residual stream acts as an unadulterated identity highway:
  $$x_L = x_0 + \sum_{l=1}^L \text{SubLayer}_l(\text{LN}(x_{l-1}))$$
- During backpropagation:
  $$\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} \left( I + \sum_{l=1}^L \frac{\partial \text{SubLayer}_l}{\partial x_0} \right)$$
  The identity matrix $I$ ensures gradients propagate directly from output to embedding layer without passing through $L$ consecutive LayerNorm Jacobians.
- **Empirical Confirmation**: Gradient norms remained stable in the $[0.65, 0.70]$ range throughout all 27,500 training steps, with zero exploding or vanishing gradients.

---

### 4.7 Tied Embeddings & The Current-Token Baseline
- The output language modeling projection reuses the input embedding matrix ($W_{\text{lm\_head}} = W_{\text{token\_emb}}^T$), saving $5.12$M parameters ($20.2\%$ of total budget).
- **The Current-Token Baseline Phenomenon**: Because the residual stream carries token $w_t$'s embedding forward and projects it directly against $W_{\text{token\_emb}}^T$, predicting the *current* token has an elevated baseline logit before training.
- An untrained model achieves loss $3.59$ when predicting current token vs $\ln(10,000) = 9.21$ for random next-token guessing. This confirms targets must be properly shifted ($x = \text{tokens}[:-1], y = \text{tokens}[1:]$).

---

### 4.8 Intrinsic Metric Divergence: Perplexity vs. Bits-per-Byte
- **Perplexity (PPL)**: $\text{PPL} = \exp(\mathcal{L}_{\text{token}})$, measuring branching uncertainty per subword token.
- **Bits-per-Byte (BPB)**: $\text{BPB} = \frac{\mathcal{L}_{\text{token}} \cdot \log_2(e)}{\text{Bytes per Token}}$, measuring compression efficiency per raw UTF-8 byte.

#### Empirical Test Evaluation Comparison (>51M Tokens Scored):
| Metric | Model H (Hindi) | Model L (Nepali) | Apparent Winner | True Technical Explanation |
|---|:---:|:---:|:---:|---|
| **Test Tokens Scored** | 51,399,168 | 56,785,408 | — | Full sequential test splits evaluated |
| **Test Perplexity (PPL)** | **24.39** | 29.87 | **Hindi (+22.5%)** | Distorted by tokenizer fertility |
| **Bytes per Token** | 9.7857 | 10.6530 | — | Nepali tokens carry 8.9% more UTF-8 text |
| **Test Bits-per-Byte (BPB)**| 0.4709 | **0.4600** | **Nepali (+2.3%)** | True byte-level compression score |

- **Why Cross-Lingual PPL is Invalid**: Nepali's tokenizer has higher fertility ($1.59$ vs $1.34$ tok/word) and each token packs more raw bytes ($10.65$ vs $9.79$ bytes/tok). Predicting a larger chunk of text is harder per token, artificially inflating Nepali's perplexity.
- **Why BPB Crowns Nepali**: When normalized by raw UTF-8 bytes—a shared, language-agnostic unit—Nepali compresses text better ($0.4600$ vs $0.4709$ BPB).

---

### 4.9 Attention Mechanism Probing Findings
By explicitly materializing the attention tensor $(B, H, T, T)$ across held-out test passages:
1. **Entropy Monotonically Sharpens with Depth**:
   - Hindi entropy drops from **$5.76$ bits** (Layer 0) down to **$3.52$ bits** (Layer 6).
   - Nepali entropy drops from **$6.03$ bits** (Layer 0) down to **$3.14$ bits** (Layer 6).
   - *Linguistic Insight*: Early layers attend broadly to absorb global context; deeper layers focus sharply on specific syntactic targets.
2. **U-Shaped Attention Distance Curve**:
   - Layer 0: Broad ingestion ($44.5$--$47.5$ tokens).
   - Layers 1–4: Local syntax ($25.1$--$33.1$ tokens: case markers, matras, postpositions).
   - Layer 6: Long-range syntactic resolution ($53.2$--$55.5$ tokens: subject-verb agreement across clauses).
3. **Head Specialization Taxonomy (56 Total Heads)**:
   - **Local/Positional** (18 Hindi, 19 Nepali): Sharp attention on adjacent tokens.
   - **Diffuse Global** (18 Hindi, 19 Nepali): Broad context tracking.
   - **Diffuse Local** (10 Hindi, 9 Nepali): Local phrase boundary detection.
   - **Long-Range Selective** (10 Hindi, 9 Nepali): Distant cross-clause syntactic binding.

---

### 4.10 Six Decoders & Generation Quality Sweet Spot
In [`lmagpt/model.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/model.py) and [`lmagpt/evaluate.py`](file:///c:/LMA/Mini_Project/individual-project-GauravPatel369/lmagpt/evaluate.py):
1. **Greedy**: Fast but collapses into degenerate repetition loops ($\text{rep-4} = 0.756$).
2. **Temperature ($T=0.7, 1.0, 1.2$)**: Smooths probability distribution.
3. **Top-K ($K=40, 50$)**: Constant-size truncation; over-truncates flat distributions.
4. **Top-P / Nucleus ($p=0.90, 0.95$)**: Dynamic truncation based on cumulative probability mass.
5. **Repetition Penalty ($\theta=1.2$)**: Sign-aware logit dampening preventing loops.
6. **No-Repeat N-gram ($n=3, 4$)**: Hard masking preventing recurring sequences.

#### Human Reference Calibration:
The optimal generation configuration was selected by minimizing the distance to human reference corpus statistics (rep-4 and distinct-2):

| Evaluation Metric | Hindi Human Ref | Hindi Winner: $T=1.0 + \text{top-}p\ 0.95$ | Nepali Human Ref | Nepali Winner: $T=1.0$ |
|---|:---:|:---:|:---:|:---:|
| **rep-4** | 0.0087 | **0.0106** | 0.0018 | **0.0031** |
| **distinct-2** | 0.8206 | **0.8147** | 0.9181 | **0.9229** |

---

## 5. Execution & Verification Commands

To train, evaluate, and verify every Phase 2 component from the terminal:

```bash
# 1. Run full unit test suite (63 passing tests verifying architecture):
pytest tests/

# 2. Train Hindi and Nepali models from scratch:
python -m lmagpt.train --lang hindi --config hindi/configs/model.yaml
python -m lmagpt.train --lang nepali --config nepali/configs/model.yaml

# 3. Resume training from an existing checkpoint:
python -m lmagpt.train --lang hindi --resume auto

# 4. Evaluate intrinsic metrics (PPL and BPB) over complete test splits:
python -m lmagpt.evaluate --lang hindi --split test
python -m lmagpt.evaluate --lang nepali --split test

# 5. Run 12-decoder sweep and qualitative generation benchmarks:
python -m lmagpt.evaluate --lang both --out "report/Phase 2/eval"

# 6. Extract attention weights, compute entropy, distance, and plot heatmaps:
python -m lmagpt.attention --lang both

# 7. Generate training loss, perplexity, and gradient norm curves:
python scripts/plot_training_curves.py
```
