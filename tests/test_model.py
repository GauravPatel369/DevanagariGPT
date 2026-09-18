"""Tests for the Phase-2 Transformer.

The causal-masking test is required explicitly by the brief: "verify empirically
that the model cannot see the future -- for example, show that changing token
t+1 does not change the logits at position t".  A broken mask does not raise;
it silently leaks the answer into the input and produces a beautiful loss curve
and a worthless model, so it is worth asserting rather than eyeballing.
"""

import math

import pytest
import torch

from lmagpt.model import GPT, GPTConfig, apply_rope, build_rope_cache


def tiny_cfg(**kw):
    """Small config so tests run in milliseconds on CPU."""
    base = dict(vocab_size=64, context=16, d_model=32, n_layer=2, n_head=4,
                d_ff=64, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


# ------------------------------------------------------------------ config


def test_head_dim_must_be_even_for_rope():
    """RoPE rotates dimension pairs, so an odd head_dim cannot work."""
    with pytest.raises(ValueError, match="even"):
        GPTConfig(d_model=30, n_head=10, context=8, vocab_size=8, d_ff=16)


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError, match="divide"):
        GPTConfig(d_model=32, n_head=5, context=8, vocab_size=8, d_ff=16)


def test_shipped_config_hits_the_parameter_budget():
    """The report claims 25,350,912 trainable parameters; hold the code to it."""
    model = GPT(GPTConfig())
    assert model.num_parameters() == 25_350_912


def test_tied_embeddings_share_storage():
    """Tying must be identity, not a copy, or the two drift apart in training."""
    model = GPT(tiny_cfg(tie_embeddings=True))
    assert model.lm_head.weight is model.token_emb.weight


def test_untied_embeddings_cost_a_vocab_matrix():
    cfg = tiny_cfg(tie_embeddings=False)
    tied = GPT(tiny_cfg(tie_embeddings=True)).num_parameters()
    untied = GPT(cfg).num_parameters()
    assert untied - tied == cfg.vocab_size * cfg.d_model


# ----------------------------------------------------------- causal masking


def test_future_tokens_cannot_change_past_logits():
    """The requirement, stated directly: perturb t+1, logits at <=t must not move.

    Two sequences identical up to position t and different afterwards must
    produce bit-identical logits at every position <= t.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = GPT(cfg).eval()

    a = torch.randint(0, cfg.vocab_size, (1, cfg.context))
    b = a.clone()
    t = cfg.context // 2
    # Change everything strictly after position t.
    b[0, t + 1:] = (b[0, t + 1:] + 1) % cfg.vocab_size
    assert not torch.equal(a, b)

    with torch.no_grad():
        la, _, _ = model(a)
        lb, _, _ = model(b)

    # Positions 0..t must be untouched.
    assert torch.equal(la[:, : t + 1, :], lb[:, : t + 1, :]), "future leaked into the past"
    # Sanity: the changed region really did change, so the test can fail.
    assert not torch.equal(la[:, t + 1:, :], lb[:, t + 1:, :])


def test_attention_weights_are_lower_triangular():
    """Post-softmax mass above the diagonal must be exactly zero."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.context))
    with torch.no_grad():
        _, _, attns = model(idx, need_weights=True)

    assert len(attns) == cfg.n_layer
    for w in attns:
        assert w.shape == (2, cfg.n_head, cfg.context, cfg.context)
        upper = torch.triu(torch.ones(cfg.context, cfg.context, dtype=torch.bool), diagonal=1)
        assert w[..., upper].abs().max().item() == 0.0
        # Each row is a distribution over allowed positions.
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_first_position_attends_only_to_itself():
    """With nothing to look back on, row 0 must be all mass on itself."""
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = GPT(cfg).eval()
    with torch.no_grad():
        _, _, attns = model(torch.randint(0, cfg.vocab_size, (1, cfg.context)), need_weights=True)
    for w in attns:
        assert pytest.approx(1.0, abs=1e-5) == w[0, :, 0, 0].min().item()


# -------------------------------------------------------------------- RoPE


def test_rope_adds_no_parameters():
    """RoPE is buffers, not parameters -- that is what funds the 7th layer."""
    model = GPT(tiny_cfg())
    names = [n for n, _ in model.named_parameters()]
    assert not any("rope" in n for n in names)


def test_rope_preserves_vector_norm():
    """Rotation is orthogonal, so it cannot change magnitudes."""
    cfg = tiny_cfg()
    cos, sin = build_rope_cache(cfg.context, cfg.head_dim)
    x = torch.randn(2, cfg.n_head, cfg.context, cfg.head_dim)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_is_relative_not_absolute():
    """The core RoPE property: q.k depends on (m - n), not on m and n alone.

    Two query/key pairs the same distance apart must give the same dot product,
    even at different absolute positions.
    """
    head_dim = 16
    cos, sin = build_rope_cache(32, head_dim)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def score(m, n):
        qm = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        kn = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum().item()

    # distance 3, at two different absolute offsets
    assert score(5, 2) == pytest.approx(score(20, 17), abs=1e-4)
    # a different distance must give a different score
    assert score(5, 2) != pytest.approx(score(10, 2), abs=1e-4)


def test_position_zero_is_an_identity_rotation():
    cfg = tiny_cfg()
    cos, sin = build_rope_cache(cfg.context, cfg.head_dim)
    x = torch.randn(1, 1, 1, cfg.head_dim)
    assert torch.allclose(apply_rope(x, cos[:1], sin[:1]), x, atol=1e-6)


# ------------------------------------------------------------ shapes / loss


def test_forward_shapes_and_loss():
    cfg = tiny_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, cfg.context))
    logits, loss, attns = model(idx, targets=idx)
    assert logits.shape == (3, cfg.context, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0
    assert attns is None  # not requested


def test_untrained_loss_is_near_uniform_entropy():
    """At init the model should be no better than guessing: loss ~ ln(V).

    Targets must be *independent* of the input here.  Reusing the input as the
    target measures something else entirely -- see the next test.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (8, cfg.context))
    targets = torch.randint(0, cfg.vocab_size, (8, cfg.context))
    _, loss, _ = model(idx, targets=targets)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.2


def test_predicting_the_current_token_is_trivially_easy():
    """Guards the reason the dataloader must shift targets by one.

    With tied embeddings the residual stream carries each token's own embedding
    forward, and the output projection is that same matrix -- so the logit for
    the *current* token is high before any training at all.  An untrained model
    scores well below ln(V) on ``targets == inputs``, which is why an unshifted
    dataloader produces a gorgeous loss curve and a useless model.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (8, cfg.context))
    _, copy_loss, _ = model(idx, targets=idx)
    _, honest_loss, _ = model(idx, targets=torch.randint(0, cfg.vocab_size, (8, cfg.context)))
    assert copy_loss.item() < honest_loss.item() - 0.3


def test_sequence_longer_than_context_is_rejected():
    cfg = tiny_cfg()
    model = GPT(cfg)
    too_long = torch.randint(0, cfg.vocab_size, (1, cfg.context + 1))
    with pytest.raises(ValueError, match="exceeds context"):
        model(too_long)


def test_generate_extends_the_sequence():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (2, 4))
    out = model.generate(prompt, max_new_tokens=5, greedy=True)
    assert out.shape == (2, 9)
    assert torch.equal(out[:, :4], prompt)  # prompt preserved


def test_greedy_generation_is_deterministic():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = GPT(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    a = model.generate(prompt, 6, greedy=True)
    b = model.generate(prompt, 6, greedy=True)
    assert torch.equal(a, b)


# ------------------------------------------------ no-positional ablation


def test_ablation_is_parameter_matched():
    """Removing RoPE must not change the parameter count.

    RoPE is buffers, not weights, so the ablation is exactly parameter-matched.
    A learned positional table would cost 262,144 parameters and any measured
    difference would be partly a capacity difference rather than a positional one.
    """
    assert GPT(GPTConfig()).num_parameters() == GPT(GPTConfig(use_rope=False)).num_parameters()


def test_single_layer_without_rope_is_order_invariant():
    """With one layer and no RoPE, the last position sees an unordered bag.

    Layer 1's final query attends over the prefix as a set, so permuting the
    prefix cannot change its output -- the definition of having no positional
    information.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg(n_layer=1, use_rope=False)
    model = GPT(cfg).eval()
    a = torch.tensor([[5, 9, 13, 21, 4, 8, 2, 7]])
    b = torch.tensor([[4, 21, 13, 9, 5, 8, 2, 7]])  # prefix permuted, suffix fixed
    with torch.no_grad():
        la, _, _ = model(a)
        lb, _, _ = model(b)
    assert torch.allclose(la[0, -1], lb[0, -1], atol=1e-5)


def test_depth_reintroduces_order_sensitivity_without_rope():
    """Stacking layers leaks position even with no positional encoding at all.

    The causal mask is itself positional information: token t knows exactly t
    tokens precede it. From layer 2 onward, each position attends over
    representations built from *its own* prefix, and permuting the sequence
    changes those prefixes. This is why the ablated model does not collapse to
    a bag-of-words model, and it is the substance of the bonus write-up.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg(n_layer=4, use_rope=False)
    model = GPT(cfg).eval()
    a = torch.tensor([[5, 9, 13, 21, 4, 8, 2, 7]])
    b = torch.tensor([[4, 21, 13, 9, 5, 8, 2, 7]])
    with torch.no_grad():
        la, _, _ = model(a)
        lb, _, _ = model(b)
    assert not torch.allclose(la[0, -1], lb[0, -1], atol=1e-4)


def test_rope_flag_actually_changes_the_forward_pass():
    """Guards against the flag being wired up but ignored."""
    torch.manual_seed(0)
    with_rope = GPT(tiny_cfg(use_rope=True)).eval()
    torch.manual_seed(0)
    without = GPT(tiny_cfg(use_rope=False)).eval()
    idx = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        a, _, _ = with_rope(idx)
        b, _, _ = without(idx)
    assert not torch.allclose(a, b, atol=1e-4)


# ---------------------------------------------------- decoding strategies


def _top_p_kept(probs, top_p):
    """Which tokens survive nucleus filtering, using the model's own logic."""
    logits = probs.log()
    ordered, order = torch.sort(logits, descending=True, dim=-1)
    p = torch.softmax(ordered, dim=-1)
    drop = p.cumsum(dim=-1) - p > top_p
    ordered = ordered.masked_fill(drop, float("-inf"))
    restored = torch.empty_like(logits).scatter_(-1, order, ordered)
    return ~torch.isinf(restored)


def test_top_p_keeps_smallest_set_reaching_p():
    """Nucleus sampling keeps tokens up to and including the one crossing p."""
    probs = torch.tensor([[0.50, 0.25, 0.125, 0.0625, 0.0625]])
    # cumulative: .50 .75 .875 .9375 1.0 -- reaching 0.9 requires the 4th token
    assert int(_top_p_kept(probs, 0.9).sum()) == 4
    assert int(_top_p_kept(probs, 0.95).sum()) == 5


def test_top_p_never_masks_everything():
    """Even at a p below the top token's mass, one token must survive."""
    probs = torch.tensor([[0.50, 0.25, 0.125, 0.0625, 0.0625]])
    for tp in (0.001, 0.01, 0.1):
        assert int(_top_p_kept(probs, tp).sum()) >= 1


def test_no_repeat_ngram_eliminates_greedy_loops():
    """The hard constraint makes the degenerate greedy loop impossible."""
    torch.manual_seed(0)
    cfg = tiny_cfg(vocab_size=50, context=64)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, 50, (1, 6))

    def repeat_rate(seq, n=4):
        grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
        return 1 - len(set(grams)) / max(len(grams), 1)

    torch.manual_seed(1)
    plain = model.generate(prompt, 50, greedy=True)[0, 6:].tolist()
    torch.manual_seed(1)
    blocked = model.generate(prompt, 50, greedy=True, no_repeat_ngram_size=4)[0, 6:].tolist()

    assert repeat_rate(plain) > 0.3, "greedy should degenerate, else the test proves nothing"
    assert repeat_rate(blocked) == 0.0


def test_repetition_penalty_moves_seen_logits_toward_minus_inf():
    """Penalty must lower a seen token's score whatever its sign.

    Dividing throughout would make a negative logit *larger* and reward the
    repetition it is meant to suppress, so signs are handled separately.
    """
    logits = torch.tensor([[2.0, -2.0, 0.5]])
    generated = torch.tensor([[0, 1]])          # tokens 0 and 1 already produced
    out = GPT._apply_repetition_penalty(logits.clone(), generated, 2.0)
    assert out[0, 0] < logits[0, 0]             # positive: divided
    assert out[0, 1] < logits[0, 1]             # negative: multiplied
    assert out[0, 2] == logits[0, 2]            # unseen: untouched


def test_every_decoder_returns_valid_tokens():
    """Each strategy must stay inside the vocabulary and extend the sequence."""
    torch.manual_seed(0)
    cfg = tiny_cfg(vocab_size=50, context=32)
    model = GPT(cfg).eval()
    prompt = torch.randint(0, 50, (2, 6))
    settings = [
        dict(greedy=True),
        dict(temperature=1.0),
        dict(temperature=1.0, top_k=10),
        dict(temperature=1.0, top_p=0.9),
        dict(temperature=0.8, top_p=0.9),
        dict(temperature=1.0, repetition_penalty=1.2),
        dict(temperature=0.8, top_p=0.9, no_repeat_ngram_size=4),
    ]
    for kw in settings:
        out = model.generate(prompt, 15, **kw)
        assert out.shape == (2, 21)
        assert out.min() >= 0 and out.max() < cfg.vocab_size
