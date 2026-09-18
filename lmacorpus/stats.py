"""Corpus statistics.

Computation only.  Every function here returns a pandas DataFrame or a plain
dict; nothing in this module imports matplotlib.  Plotting lives exclusively in
``lmacorpus/viz/plots.py``.  That separation is an explicit project guideline,
and enforcing it architecturally is cheaper than remembering it.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from .io_utils import read_dir


def source_inventory(shard_dir) -> pd.DataFrame:
    """Per-source document, character and byte counts at one pipeline stage."""
    agg: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"documents": 0, "chars": 0, "bytes": 0}
    )
    for d in read_dir(shard_dir):
        a = agg[(d.source, d.source_type)]
        a["documents"] += 1
        a["chars"] += d.flags.get("n_chars", len(d.text))
        a["bytes"] += d.flags.get("n_bytes", len(d.text.encode("utf-8")))
    rows = [
        {"source": s, "source_type": t, **v} for (s, t), v in sorted(agg.items())
    ]
    return pd.DataFrame(rows)


def funnel(stage_counts: dict[str, int]) -> pd.DataFrame:
    """Documents surviving each pipeline stage, with retention rates."""
    rows, prev = [], None
    for stage, n in stage_counts.items():
        rows.append(
            {
                "stage": stage,
                "documents": n,
                "retained_vs_previous": None if prev is None else round(n / prev, 4),
            }
        )
        prev = n
    return pd.DataFrame(rows)


def length_histogram(shard_dir, bins=(0, 200, 500, 1000, 2000, 5000, 10000, 50000)) -> pd.DataFrame:
    """Document length distribution in characters, bucketed."""
    counts: Counter = Counter()
    for d in read_dir(shard_dir):
        n = d.flags.get("n_chars", len(d.text))
        for lo, hi in zip(bins, list(bins[1:]) + [float("inf")]):
            if lo <= n < hi:
                counts[f"{lo}-{hi}"] += 1
                break
    return pd.DataFrame(
        sorted(({"bucket": k, "documents": v} for k, v in counts.items()),
               key=lambda r: int(r["bucket"].split("-")[0]))
    )


def token_frequency(model_path, shard_dir, manifest, max_docs: int = 20_000) -> pd.DataFrame:
    """Token frequency table on held-out text, for the Zipf plot and top-k table."""
    import sentencepiece as spm

    from .splits import load_manifest

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    ids_wanted = load_manifest(manifest)
    freq: Counter = Counter()
    n = 0
    for d in read_dir(shard_dir):
        if d.doc_id not in ids_wanted:
            continue
        freq.update(sp.encode(d.text, out_type=int))
        n += 1
        if n >= max_docs:
            break
    total = sum(freq.values())
    rows = [
        {
            "rank": i + 1,
            "token_id": tid,
            "piece": sp.id_to_piece(tid),
            "count": c,
            "share": c / total,
        }
        for i, (tid, c) in enumerate(freq.most_common())
    ]
    return pd.DataFrame(rows)


def tokenization_examples(model_path, sentences: list[str]) -> pd.DataFrame:
    """Side-by-side surface form / pieces / token count for report examples."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    rows = []
    for s in sentences:
        pieces = sp.encode(s, out_type=str)
        rows.append(
            {
                "sentence": s,
                "n_chars": len(s),
                "n_tokens": len(pieces),
                "chars_per_token": round(len(s) / max(len(pieces), 1), 2),
                "pieces": " | ".join(pieces),
            }
        )
    return pd.DataFrame(rows)


def manual_fraction_by_split(meta: dict) -> pd.DataFrame:
    """Manual vs downloaded token split per split -- the headline Phase-1 number."""
    rows = []
    for split, s in meta["splits"].items():
        by = s["tokens_by_source_type"]
        rows.append(
            {
                "split": split,
                "tokens": s["tokens"],
                "manual_tokens": by.get("manual", 0),
                "downloaded_tokens": by.get("downloaded", 0),
                "manual_fraction": s["manual_fraction"],
                "meets_20pct": s["manual_fraction"] >= 0.20,
            }
        )
    return pd.DataFrame(rows)
