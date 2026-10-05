"""Decoder-only Transformer language model, built from PyTorch primitives.

Only ``nn.Linear``, ``nn.Embedding``, ``nn.LayerNorm`` and ``nn.Dropout`` are
used.  No ``nn.Transformer*`` module, no pre-built attention: the
``QK^T / sqrt(d_k)`` -> mask -> softmax -> ``@V`` chain is written out
explicitly below.  ``F.scaled_dot_product_attention`` is deliberately avoided
too -- besides being a pre-built attention primitive, its fused kernels never
materialise the attention matrix, and Phase-2 requires those weights for the
heatmap / entropy / distance analysis.

Architecture (per ``<lang>/configs/model.yaml``):

    d_model 512 | n_layer 7 | n_head 8 | d_ff 1792 | context 512 | vocab 10,000
    -> 25,350,912 trainable parameters with tied embeddings

Two decisions worth stating up front, because the brief asks for both:

* **Positional encoding is RoPE**, not a learned table.  RoPE adds *no*
  parameters, which is what makes a seventh layer affordable inside the ~25M
  budget.  It also encodes position *relatively*: the attention score between a
  query at position m and a key at position n depends only on (m - n) after
  rotation, so the model never has to learn that "position 300" and
  "position 301" are adjacent.  Unlike a learned table -- which has exactly
  ``context`` rows and raises an IndexError beyond them -- RoPE has no hard
  ceiling; the limit is the precomputed cos/sin cache, which can be extended,
  after which quality degrades because those rotation angles were never trained.

* **Pre-norm**, not post-norm.  LayerNorm is applied to the *input* of each
  sublayer, leaving the residual stream an unnormalised identity path from
  embeddings to the final norm.  Gradients therefore reach layer 0 without
  passing through seven LayerNorm Jacobians, which is why pre-norm trains at
  depth without a long warmup.  Post-norm at 7 layers is trainable but needs
  more careful warmup and is likelier to diverge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Hyperparameters for one model.  Serialised into every checkpoint."""

    vocab_size: int = 10_000
    context: int = 512
    d_model: int = 512
    n_layer: int = 7
    n_head: int = 8
    d_ff: int = 1792
    dropout: float = 0.1
    tie_embeddings: bool = True
    rope_theta: float = 10_000.0
    use_rope: bool = True     # False = the no-positional-information ablation

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must divide evenly into n_head ({self.n_head})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even: RoPE rotates dimensions in pairs"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- RoPE


def build_rope_cache(context: int, head_dim: int, theta: float = 10_000.0,
                     device=None, dtype=torch.float32):
    """Precompute the cos/sin tables RoPE rotates by.

    Dimension pair ``i`` rotates at frequency ``theta ** (-2i / head_dim)``, so
    early pairs turn quickly (they resolve nearby positions) and late pairs turn
    slowly (they carry long-range position).  Position ``m`` rotates pair ``i``
    by angle ``m * freq_i``.

    Returns:
        ``(cos, sin)``, each of shape ``(context, head_dim // 2)``.  These are
        buffers, not parameters -- RoPE contributes nothing to the parameter
        count, which is precisely why the seventh layer fits in the budget.
    """
    half = head_dim // 2
    freqs = theta ** (-torch.arange(0, half, device=device, dtype=torch.float32) / half)
    positions = torch.arange(context, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)            # (context, half)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the head dimension of ``x`` by its position angle.

    Args:
        x: ``(B, n_head, T, head_dim)`` -- queries or keys.  Never values: RoPE
            biases *where* attention looks, and V carries *what* is retrieved.
        cos, sin: ``(T, head_dim // 2)`` slices of the cache.

    Each adjacent pair ``(x_even, x_odd)`` is treated as a 2-D vector and turned
    by the position's angle, the standard 2-D rotation:

        x_even' = x_even * cos - x_odd  * sin
        x_odd'  = x_odd  * cos + x_even * sin

    The relative property follows from rotation composition: rotating a query by
    ``m`` and a key by ``n`` makes their dot product a function of ``m - n``.
    """
    x_even, x_odd = x[..., 0::2], x[..., 1::2]        # each (B, h, T, half)
    cos = cos[None, None, :, :]                       # broadcast over B and heads
    sin = sin[None, None, :, :]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_odd * cos + x_even * sin
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


# ---------------------------------------------------------------- attention


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, written from first principles."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.use_rope = cfg.use_rope
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Upper-triangular mask, strictly above the diagonal. Registered as a
        # buffer so it moves with .to(device) and is saved with the module, but
        # is not a parameter.
        mask = torch.triu(torch.ones(cfg.context, cfg.context, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, cos, sin, need_weights: bool = False):
        """
        Args:
            x: ``(B, T, d_model)``.
            cos, sin: RoPE cache sliced to ``T``.
            need_weights: return the post-softmax attention matrix for analysis.
                Off during training -- keeping ``(B, h, T, T)`` alive costs about
                134 MB per layer at B=32, T=512 in bf16.

        Returns:
            ``(output, weights_or_None)`` where output is ``(B, T, d_model)``.
        """
        B, T, C = x.shape

        # (B, T, C) -> (B, T, h, head_dim) -> (B, h, T, head_dim).
        # The transpose puts the head axis before time so each head's (T, head_dim)
        # slice is contiguous for the matmuls below.
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Position enters here, on Q and K only. With use_rope=False nothing
        # in the whole forward pass depends on order, so the model sees an
        # unordered bag of tokens -- that is the bonus ablation.
        if self.use_rope:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Scaled dot-product. The 1/sqrt(head_dim) matters: q.k is a sum of
        # head_dim products, so its variance grows linearly with head_dim. Left
        # unscaled, logits at head_dim=64 would be ~8x larger, softmax would
        # saturate into a near one-hot distribution, and the gradient through it
        # would vanish. Dividing by sqrt(head_dim) holds the variance at ~1.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,h,T,T)

        # Causal masking: -inf before the softmax becomes exactly zero
        # probability after it, so position t cannot attend to t+1..T-1.
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(weights) @ v                          # (B,h,T,head_dim)

        # Concatenate heads back into the residual width, then project.
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(attn))
        return out, (weights if need_weights else None)


# ----------------------------------------------------------------- feed-forward


class FeedForward(nn.Module):
    """Position-wise FFN: expand to ``d_ff``, apply GELU, project back.

    Applied independently at every position -- attention is the only operation
    that moves information across time.  ``d_ff`` is 3.5x ``d_model`` rather than
    the conventional 4x: both matrices are ``d_model x d_ff``, so the FFN is two
    thirds of each block's parameters, and trimming the ratio is what buys the
    seventh layer inside the ~25M budget.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    """One pre-norm Transformer block.
    
    ``x = x + attn(norm(x))`` then ``x = x + ffn(norm(x))``.  Normalising the
    sublayer *input* leaves the residual stream itself untouched, so it forms an
    identity path from the embeddings to the final norm.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin, need_weights: bool = False):
        attn_out, weights = self.attn(self.norm1(x), cos, sin, need_weights)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, weights


# --------------------------------------------------------------------- model


class GPT(nn.Module):
    """Decoder-only Transformer trained with a causal language-modelling objective."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # One matrix serves both roles: it maps ids -> vectors on the way in
            # and vectors -> logits on the way out. Saves vocab*d_model = 5.12M
            # parameters, a fifth of the entire budget, and ties the two
            # representations of a token so they cannot drift apart.
            self.lm_head.weight = self.token_emb.weight

        cos, sin = build_rope_cache(cfg.context, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale the residual-path output projections by 1/sqrt(2*n_layer): each
        # block writes twice into the residual stream, so without this the
        # stream's variance grows with depth.
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Trainable parameter count.  Tied weights are counted once."""
        seen, total = set(), 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def forward(self, idx, targets=None, need_weights: bool = False):
        """
        Args:
            idx: ``(B, T)`` int64 token ids.
            targets: ``(B, T)`` next-token ids, or None for inference.
            need_weights: also return per-layer attention matrices.

        Returns:
            ``(logits, loss, attentions)``. ``logits`` is ``(B, T, vocab)``;
            ``loss`` is None when targets is None; ``attentions`` is a list of
            ``(B, h, T, T)`` tensors or None.
        """
        B, T = idx.shape
        if T > self.cfg.context:
            raise ValueError(
                f"sequence length {T} exceeds context {self.cfg.context}; the RoPE "
                f"cache holds {self.cfg.context} positions"
            )

        x = self.emb_dropout(self.token_emb(idx))
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]

        attentions = [] if need_weights else None
        for block in self.blocks:
            x, w = block(x, cos, sin, need_weights)
            if need_weights:
                attentions.append(w)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Cross-entropy between the logits at position t and the token at
            # t+1. The shift is already applied by the dataloader, so targets is
            # simply idx rolled left by one.
            #
            # ignore_index=-100 lets the caller mask positions out of the loss.
            # Pretraining never does; supervised finetuning (Phase 3) masks the
            # prompt so the gradient comes only from the answer tokens, since
            # the facts and question are given and reproducing them teaches
            # nothing. Positions set to -100 contribute no loss and no gradient.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, attentions

    @staticmethod
    def _apply_repetition_penalty(logits, generated, penalty: float):
        """Down-weight tokens the sequence has already produced (CTRL, Keskar 2019).

        Positive logits are divided by ``penalty`` and negative ones multiplied,
        so the operation always moves a score *towards* -inf regardless of sign.
        Dividing throughout would make a negative logit larger and reward the
        repetition it is meant to discourage.
        """
        for b in range(logits.size(0)):
            seen = torch.unique(generated[b])
            vals = logits[b, seen]
            logits[b, seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
        return logits

    @staticmethod
    def _block_repeat_ngrams(logits, generated, n: int):
        """Forbid any token that would complete an n-gram already in the text.

        A hard constraint rather than a soft penalty: it makes the exact
        degenerate loop that greedy decoding falls into impossible, while
        leaving every other choice untouched.
        """
        for b in range(logits.size(0)):
            seq = generated[b].tolist()
            if len(seq) < n:
                continue
            prefix = tuple(seq[-(n - 1):])
            banned = {seq[i + n - 1] for i in range(len(seq) - n + 1)
                      if tuple(seq[i:i + n - 1]) == prefix}
            if banned:
                logits[b, list(banned)] = float("-inf")
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None,
                 repetition_penalty: float | None = None,
                 no_repeat_ngram_size: int | None = None,
                 greedy: bool = False, eos_id: int | None = None):
        """Autoregressive decoding with the common truncation strategies.

        Args:
            idx: ``(B, T0)`` prompt ids.
            temperature: divides the logits. Below 1 sharpens the distribution,
                above 1 flattens it. Ignored when ``greedy``.
            top_k: keep only the k highest-probability tokens.  Fixed-size
                truncation: the same k applies whether the model is confident or
                not, which over-truncates flat distributions and under-truncates
                peaked ones.
            top_p: nucleus sampling -- keep the smallest set of tokens whose
                probability sums to ``p``.  Adaptive: a confident step keeps a
                couple of tokens, an uncertain one keeps many, which is why it
                usually beats top-k.
            repetition_penalty: > 1 discourages tokens already produced.
            no_repeat_ngram_size: hard-block any n-gram repeat.
            greedy: take the argmax instead of sampling.
            eos_id: stop early once every sequence has emitted it.

        The filters compose in the order penalty -> n-gram block -> temperature
        -> top-k -> top-p, which is the conventional ordering: penalties act on
        raw scores, truncation acts on the tempered distribution.
        """
        self.eval()
        prompt_len = idx.size(1)
        for _ in range(max_new_tokens):
            # Crop to the context window: RoPE has no cached angle beyond it.
            window = idx[:, -self.cfg.context:]
            logits, _, _ = self(window)
            logits = logits[:, -1, :].float()               # last position only

            if repetition_penalty and repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, idx, repetition_penalty)
            if no_repeat_ngram_size:
                logits = self._block_repeat_ngrams(logits, idx, no_repeat_ngram_size)

            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)

                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))

                if top_p is not None:
                    ordered, order = torch.sort(logits, descending=True, dim=-1)
                    cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
                    # Keep everything up to and including the token that crosses
                    # p: shifting right guarantees at least one token survives
                    # even when the top token alone already exceeds p.
                    drop = cumulative - torch.softmax(ordered, dim=-1) > top_p
                    ordered = ordered.masked_fill(drop, float("-inf"))
                    logits = torch.empty_like(logits).scatter_(-1, order, ordered)

                nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)

            idx = torch.cat((idx, nxt), dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
