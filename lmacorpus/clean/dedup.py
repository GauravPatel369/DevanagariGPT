"""Stage 6: deduplication.

Three passes, in this order:

1. **Exact** -- on ``doc_id`` (content hash).  Free.
2. **Near-duplicate within a language** -- MinHash + LSH over *character*
   5-gram shingles.  Character shingles beat word shingles for morphologically
   rich languages, where the same sentence appears with different inflections.
3. **Cross-corpus** -- index Hindi, query Nepali.  A hit means the identical
   document sits in both corpora, which is exactly what the project forbids.
   The document is dropped from **both** sides: removing it from one only would
   leave the contamination in place.

Processing order is deterministic (sorted by ``doc_id``) so the surviving member
of each duplicate cluster is reproducible across re-runs.

CC-100, mC4, CulturaX, OSCAR and HPLT are all Common Crawl derivatives, so
40-70% shrinkage relative to naive summed source counts is expected.  Only
post-dedup counts are reported.
"""

from __future__ import annotations

from pathlib import Path

from ..io_utils import ShardWriter, read_dir, shard_paths, read_shard

NUM_PERM = 128
SHINGLE = 5
SHINGLE_STRIDE = 2


def signature(text: str, num_perm: int = NUM_PERM):
    """MinHash signature over character n-gram shingles of one document.

    ``update_batch`` hashes the whole shingle set in one vectorised numpy call.
    The per-shingle ``update`` loop it replaces was the single hottest line in
    stage 6: a 5k-character document produces ~2,500 shingles, and at corpus
    scale that is billions of Python-level calls.
    """
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm)
    shingles = {
        text[i : i + SHINGLE].encode("utf-8")
        for i in range(0, max(1, len(text) - SHINGLE + 1), SHINGLE_STRIDE)
    }
    if shingles:
        m.update_batch(list(shingles))
    return m


def dedup_exact(docs):
    """Yield documents with unseen content hashes, counting drops."""
    seen: set[str] = set()
    for d in docs:
        if d.doc_id in seen:
            continue
        seen.add(d.doc_id)
        yield d


def dedup_near(in_dir, out_dir, threshold: float = 0.8) -> dict:
    """Exact + near-duplicate dedup within one language.

    Args:
        in_dir: Directory of shards after language ID.
        out_dir: Destination directory of shards.
        threshold: Jaccard similarity above which two documents are duplicates.

    Returns:
        Counts for the dedup funnel table in the report.
    """
    from datasketch import MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    stats = {"seen": 0, "exact_dupes": 0, "near_dupes": 0, "kept": 0}
    seen: set[str] = set()

    # Deterministic order: sorted shards, then sorted doc_id inside each shard.
    with ShardWriter(out_dir) as w:
        for path in shard_paths(in_dir):
            docs = sorted(read_shard(path), key=lambda d: d.doc_id)
            for d in docs:
                stats["seen"] += 1
                if d.doc_id in seen:
                    stats["exact_dupes"] += 1
                    continue
                seen.add(d.doc_id)
                sig = signature(d.text)
                if lsh.query(sig):
                    stats["near_dupes"] += 1
                    continue
                lsh.insert(d.doc_id, sig)
                stats["kept"] += 1
                w.add(d)
    return stats


def cross_corpus_collisions(dir_a, dir_b, threshold: float = 0.8) -> dict:
    """Find documents present in both language corpora.

    Args:
        dir_a: Shard directory for language A (indexed).
        dir_b: Shard directory for language B (queried).

    Returns:
        Dict with the colliding doc_ids on each side and per-source counts.
        Callers must drop the collisions from **both** corpora.
    """
    from collections import Counter

    from datasketch import MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    a_index: dict[str, str] = {}
    for d in read_dir(dir_a):
        lsh.insert(d.doc_id, signature(d.text))
        a_index[d.doc_id] = d.source

    hits_a: set[str] = set()
    hits_b: set[str] = set()
    pairs: Counter = Counter()
    for d in read_dir(dir_b):
        matched = lsh.query(signature(d.text))
        if matched:
            hits_b.add(d.doc_id)
            for m in matched:
                hits_a.add(m)
                pairs[f"{a_index.get(m, '?')} <-> {d.source}"] += 1
    return {
        "collisions_a": sorted(hits_a),
        "collisions_b": sorted(hits_b),
        "n_collisions_a": len(hits_a),
        "n_collisions_b": len(hits_b),
        "source_pairs": dict(pairs),
    }


def drop_ids(in_dir, out_dir, bad_ids: set[str]) -> int:
    """Rewrite a shard directory without the given doc_ids.  Returns docs kept."""
    kept = 0
    with ShardWriter(out_dir) as w:
        for d in read_dir(in_dir):
            if d.doc_id in bad_ids:
                continue
            w.add(d)
            kept += 1
    return kept
