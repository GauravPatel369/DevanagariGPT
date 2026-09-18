"""Stage 8b: vocabulary size selection.

The trap: fertility decreases monotonically with vocabulary size, so selecting
on "lowest fertility" always picks the largest candidate.  The real trade-off is
against the Phase-2 parameter budget of ~25M trainable parameters.

With tied input/output embeddings at d_model = 512:

    V = 8k   ->  4.2M params in the embedding table
    V = 16k  ->  8.4M
    V = 24k  -> 12.6M
    V = 32k  -> 16.8M   (two thirds of the entire budget)

So this module reports fertility **and** embedding cost together, plus the
resulting non-embedding parameter budget, and the choice is made at the knee.
That single plot links Phase 1 to Phase 2 and is the strongest justification
available for the vocabulary decision.
"""

from __future__ import annotations

from pathlib import Path

from ..io_utils import read_dir
from ..splits import load_manifest


def held_out_text(shard_dir, manifest, max_docs: int = 5000) -> list[str]:
    """Collect validation documents for tokenizer evaluation."""
    ids = load_manifest(manifest)
    out = []
    for doc in read_dir(shard_dir):
        if doc.doc_id in ids:
            out.append(doc.text)
            if len(out) >= max_docs:
                break
    return out


def evaluate_model(model_path, texts: list[str]) -> dict:
    """Measure tokenizer quality on held-out text.

    Returns:
        fertility (tokens per whitespace word), chars/token, bytes/token,
        byte-fallback rate, single-character vocabulary fraction, and the
        coverage-95 token count (how many distinct tokens cover 95% of usage).
    """
    from collections import Counter

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.get_piece_size()
    byte_ids = {i for i in range(vocab_size) if sp.id_to_piece(i).startswith("<0x")}

    n_tokens = n_words = n_chars = n_bytes = n_byte_tokens = n_unk = 0
    freq: Counter = Counter()
    for t in texts:
        ids = sp.encode(t, out_type=int)
        freq.update(ids)
        n_tokens += len(ids)
        n_words += len(t.split())
        n_chars += len(t)
        n_bytes += len(t.encode("utf-8"))
        n_byte_tokens += sum(1 for i in ids if i in byte_ids)
        n_unk += sum(1 for i in ids if i == sp.unk_id())

    single_char = sum(
        1
        for i in range(vocab_size)
        if len(sp.id_to_piece(i).replace("\u2581", "")) == 1
    )
    # Distinct tokens needed to cover 95% of token occurrences.
    cum, cover95 = 0, 0
    for _, c in freq.most_common():
        cum += c
        cover95 += 1
        if cum >= 0.95 * n_tokens:
            break

    return {
        "vocab_size": vocab_size,
        "fertility_tokens_per_word": round(n_tokens / max(n_words, 1), 4),
        "chars_per_token": round(n_chars / max(n_tokens, 1), 4),
        "bytes_per_token": round(n_bytes / max(n_tokens, 1), 4),
        "byte_fallback_rate": round(n_byte_tokens / max(n_tokens, 1), 6),
        "unk_rate": round(n_unk / max(n_tokens, 1), 8),
        "single_char_vocab_frac": round(single_char / vocab_size, 4),
        "tokens_covering_95pct": cover95,
        "distinct_tokens_used": len(freq),
    }


def embedding_params(vocab_size: int, d_model: int = 512, tied: bool = True) -> int:
    """Parameters consumed by the embedding table(s) at a given vocabulary size."""
    return vocab_size * d_model * (1 if tied else 2)


def sweep_table(results: list[dict], d_model: int = 512, budget: int = 25_000_000) -> list[dict]:
    """Join tokenizer metrics with their Phase-2 parameter cost."""
    rows = []
    for r in results:
        emb = embedding_params(r["vocab_size"], d_model)
        rows.append(
            {
                **r,
                "d_model": d_model,
                "embedding_params": emb,
                "embedding_frac_of_budget": round(emb / budget, 4),
                "non_embedding_budget": budget - emb,
            }
        )
    return sorted(rows, key=lambda r: r["vocab_size"])
