"""Stage 2: downloaded collection from Hugging Face.

Always ``streaming=True``.  Sangraha and CulturaX would fill Colab's disk long
before they finished downloading; streaming lets us normalize, filter and shard
on the fly and keep only what survives.

Note that CC-100, mC4, CulturaX, OSCAR and HPLT are all Common Crawl derivatives
and overlap heavily.  Summing their advertised sizes is meaningless; only
post-dedup counts are reported.

The project permits public corpora (Hugging Face is named explicitly).  What is
forbidden is pretrained *models* and pretrained *tokenizers* -- not datasets.
"""

from __future__ import annotations

from ..io_utils import ShardWriter
from ..schema import DOWNLOADED, Document


def stream_source(
    repo: str,
    config: str | None,
    split: str,
    lang: str,
    source: str,
    out_dir,
    text_key: str = "text",
    url_key: str | None = "url",
    max_docs: int | None = None,
    min_chars: int = 200,
) -> dict:
    """Stream one HF dataset into Document shards.

    Args:
        repo: HF dataset repo id, e.g. "ai4bharat/sangraha".
        config: Dataset config / language subset, e.g. "hi".
        split: Usually "train".
        lang: Provisional language tag; the real assignment happens at stage 5.
        source: Name recorded on each document for provenance tables.
        out_dir: Shard output directory.
        text_key: Column holding the document text.
        url_key: Optional column holding a source URL.
        max_docs: Cap for smoke tests.
        min_chars: Skip trivially short records before they reach disk.

    Returns:
        Ingestion counters for the source-inventory table.
    """
    import os

    os.environ["PYTHONUTF8"] = "1"
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") 
    ds = load_dataset(repo, config, split=split, streaming=True, token=token)
    stats = {"source": source, "seen": 0, "written": 0, "chars": 0}
    with ShardWriter(out_dir) as w:
        for row in ds:
            stats["seen"] += 1
            text = row.get(text_key) or ""
            if len(text) < min_chars:
                continue
            doc = Document(
                text=text,
                lang=lang,
                source=source,
                source_type=DOWNLOADED,
                url=row.get(url_key) if url_key else None,
            ).finalize()
            w.add(doc)
            stats["written"] += 1
            stats["chars"] += doc.flags["n_chars"]
            if max_docs and stats["written"] >= max_docs:
                break
            if stats["written"] % 100_000 == 0:
                print(f"[hf:{source}] written={stats['written']:,}", flush=True)
    return stats
