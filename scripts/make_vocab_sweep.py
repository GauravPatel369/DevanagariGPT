"""Prepare the vocabulary-size sweep: encode corpora and write configs.

Question the sweep answers: **given a fixed ~25M parameter budget, how should it
be split between vocabulary and network width?**  Phase 1 chose V=10,000 on a
theoretical argument about embedding cost; this tests it empirically.

The design point that makes the experiment meaningful is holding the *parameter
count* constant rather than the architecture.  With a fixed architecture, a
larger vocabulary simply means a larger model:

    V= 5,000 -> 22.79M      V=10,000 -> 25.35M
    V= 8,000 -> 24.33M      V=16,000 -> 28.42M   (+13.7% over budget)

A 16k model would then beat a 5k one partly by being 25% bigger, and the result
would say nothing about tokenizers.  Instead ``d_ff`` absorbs the difference, so
every configuration lands within +/-1% of 25.35M and vocabulary trades directly
against FFN width.

Perplexity is meaningless across these runs -- each tokenizer produces a
different number of tokens for the same text -- so the comparison must be made
on **bits per byte**, exactly as for Hindi vs Nepali in Phase 2 section 4.3.

Usage:
    python scripts/make_vocab_sweep.py --lang hindi          # encode + configs
    python scripts/make_vocab_sweep.py --lang hindi --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VOCABS = (5000, 8000, 10000, 16000)
TARGET = 25_350_912          # the Phase-2 parameter count, held constant
D_MODEL, N_LAYER, N_HEAD, CONTEXT = 512, 7, 8, 512


def params(vocab: int, d_ff: int) -> int:
    """Total trainable parameters with tied embeddings and RoPE."""
    emb = vocab * D_MODEL
    attn = 4 * D_MODEL * D_MODEL + 4 * D_MODEL
    ln = 4 * D_MODEL
    ffn = 2 * D_MODEL * d_ff + d_ff + D_MODEL
    return emb + N_LAYER * (attn + ln + ffn) + 2 * D_MODEL


def best_d_ff(vocab: int) -> tuple[int, int]:
    """The d_ff bringing this vocabulary closest to the target budget.

    Kept a multiple of 64 so the matrices stay GPU-friendly.
    """
    best = None
    for d_ff in range(512, 4096, 64):
        total = params(vocab, d_ff)
        if best is None or abs(total - TARGET) < abs(best[1] - TARGET):
            best = (d_ff, total)
    return best


def write_config(lang: str, vocab: int, d_ff: int, total: int, dry: bool) -> Path:
    """One config per vocabulary, identical apart from vocab_size and d_ff."""
    base = yaml.safe_load((ROOT / lang / "configs" / "model.yaml").read_text(encoding="utf-8"))
    base["model"]["vocab_size"] = vocab
    base["model"]["d_ff"] = d_ff
    # Shorter runs: the sweep compares tokenizers, and the curves separate long
    # before convergence. A full epoch each would cost ~46 h of Kaggle quota.
    # One full epoch, derived from this vocabulary's own token count -- a smaller
    # vocabulary emits more tokens for the same text and so needs more steps to
    # cover it once. Every run in the sweep must see the corpus the same number
    # of times, or the comparison measures training budget instead of vocabulary.
    tokens_per_step = (base["train"]["micro_batch"] * base["train"]["grad_accum"]
                       * base["model"]["context"])
    meta = json.loads((ROOT / lang / "data" / f"tokens_v{vocab}" / "meta.json")
                      .read_text(encoding="utf-8"))
    base["train"]["max_steps"] = meta["splits"]["train"]["tokens"] // tokens_per_step
    base["train"]["warmup_steps"] = round(base["train"]["max_steps"] * 0.018 / 10) * 10
    base["train"]["eval_every"] = 500

    out = ROOT / lang / "configs" / f"model_v{vocab}.yaml"
    header = (
        f"# Vocabulary sweep: V={vocab:,}, d_ff={d_ff} ({d_ff / D_MODEL:.1f}x d_model).\n"
        f"#\n"
        f"# Total parameters {total:,} -- within {abs(total - TARGET) / TARGET:+.2%} of the\n"
        f"# Phase-2 budget, so vocabulary trades against FFN width rather than against\n"
        f"# model size. Compare these runs on BITS PER BYTE, never perplexity: each\n"
        f"# tokenizer emits a different number of tokens for the same text.\n"
    )
    if not dry:
        out.write_text(header + yaml.safe_dump(base, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")
    return out


def encode(lang: str, vocab: int, dry: bool) -> Path:
    """Re-encode the corpus with one tokenizer into its own token directory."""
    from lmagpt.data import resolve_data_dir
    from lmacorpus.cli import paths
    from lmacorpus.tokenizer.encode_corpus import encode_all

    code = "hi" if lang == "hindi" else "ne"
    model = ROOT / lang / "tokenizer" / f"{code}_unigram_{vocab}.model"
    if not model.exists():
        raise SystemExit(f"{model} missing -- it should exist from the Phase-1 sweep")

    p = paths(lang)
    src = p["final"] if p["final"].exists() else p["dedup"]
    out = ROOT / lang / "data" / f"tokens_v{vocab}"
    if out.joinpath("meta.json").exists():
        print(f"  V={vocab:>6,}: already encoded at {out}")
        return out
    if dry:
        print(f"  V={vocab:>6,}: would encode -> {out}")
        return out

    print(f"  V={vocab:>6,}: encoding -> {out} ...", flush=True)
    encode_all(src, p["splits"], model, out, vocab)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", default="hindi", choices=["hindi", "nepali"])
    ap.add_argument("--vocabs", type=int, nargs="+", default=list(VOCABS),
                    choices=list(VOCABS),
                    help="subset to prepare; each encoding costs ~1.1 GB and ~20 min")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Vocabulary sweep for {a.lang}, budget held at {TARGET:,} parameters\n")
    print(f"  {'V':>7} {'d_ff':>6} {'ratio':>6} {'params':>12} {'vs budget':>10}")
    plan = []
    for vocab in sorted(a.vocabs):
        d_ff, total = best_d_ff(vocab)
        print(f"  {vocab:>7,} {d_ff:>6} {d_ff / D_MODEL:>5.1f}x {total:>12,} "
              f"{(total - TARGET) / TARGET:>+9.2%}")
        plan.append((vocab, d_ff, total))

    # Encode BEFORE writing configs: max_steps is derived from the token count
    # in each encoding's meta.json, so the data must exist first.
    print("\nencoding:")
    for vocab, _, _ in plan:
        encode(a.lang, vocab, a.dry_run)

    print("\nconfigs:")
    for vocab, d_ff, total in plan:
        if a.dry_run:
            print(f"  would write model_v{vocab}.yaml")
            continue
        print(f"  {write_config(a.lang, vocab, d_ff, total, a.dry_run).name}")

    print("\nnext: launch training for each vocabulary, then compare on bits-per-byte")


if __name__ == "__main__":
    main()
