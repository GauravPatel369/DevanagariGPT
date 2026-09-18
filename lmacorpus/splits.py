"""Stage 7: train / validation / test splits.

Two decisions that materially affect Phase-2 numbers:

* **Document level, never line level.**  Line-level splitting puts sentences
  from the same article on both sides and makes held-out perplexity meaningless.
* **Group aware.**  Even at document level, several outlets cover the same event
  in near-identical language on the same day.  Splitting on
  ``(source, publication month)`` keeps related articles together so
  near-duplicate coverage cannot straddle the train/test boundary.

Assignment is a hash of the group key, not an RNG draw: the split is a pure
function of the data and re-runs reproduce it exactly.  The split is stored as
manifests of doc_ids, not by copying text into three directories -- manifests
are small, diffable, and committable.
"""

from __future__ import annotations

import re
from collections import Counter
from hashlib import blake2b
from pathlib import Path

from .io_utils import read_dir

DATE_RE = re.compile(r"/(20\d{2})[/-](\d{1,2})[/-]")


def group_key(doc) -> str:
    """Stable grouping key for one document.

    Uses (source, publication month) when a date can be recovered from the URL,
    which is the common case for crawled news.  Falls back to a coarse hash
    bucket of the content so downloaded corpora without metadata are still
    grouped rather than split at random.
    """
    m = DATE_RE.search(doc.url or "")
    if m:
        return f"{doc.source}|{m.group(1)}-{int(m.group(2)):02d}"
    return f"{doc.source}|bucket-{doc.doc_id[:2]}"


def assign(doc, val_frac: float = 0.01, test_frac: float = 0.01, salt: str = "lma") -> str:
    """Deterministically map a document to 'train', 'val' or 'test'."""
    h = blake2b(f"{salt}|{group_key(doc)}".encode("utf-8"), digest_size=8).hexdigest()
    u = int(h, 16) / 2**64
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def build_splits(
    in_dir, out_dir, val_frac: float = 0.01, test_frac: float = 0.01
) -> dict:
    """Write train/val/test doc_id manifests and return split statistics.

    Statistics are reported **per split and per source_type**, not only
    globally: if the manual crawl is all news and the downloaded data is all
    encyclopedic, an unlucky split leaves test with a different source mixture
    from train.  That must be visible.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    handles = {s: open(out / f"{s}.txt", "w", encoding="utf-8") for s in ("train", "val", "test")}
    stats = {
        s: {
            "docs": 0,
            "chars": 0,
            "bytes": 0,
            "by_source_type": Counter(),
            "by_source": Counter(),
        }
        for s in handles
    }
    try:
        for d in read_dir(in_dir):
            s = assign(d, val_frac, test_frac)
            handles[s].write(d.doc_id + "\n")
            st = stats[s]
            st["docs"] += 1
            st["chars"] += d.flags.get("n_chars", len(d.text))
            st["bytes"] += d.flags.get("n_bytes", len(d.text.encode("utf-8")))
            st["by_source_type"][d.source_type] += 1
            st["by_source"][d.source] += 1
    finally:
        for h in handles.values():
            h.close()
    out_stats = {
        s: {
            **{k: v for k, v in st.items() if not isinstance(v, Counter)},
            "by_source_type": dict(st["by_source_type"]),
            "by_source": dict(st["by_source"]),
        }
        for s, st in stats.items()
    }
    # Group-aware splitting is coarse: with few groups, a requested 1% slice can
    # round to zero.  Fail loudly rather than silently shipping an empty test set.
    empty = [s for s, v in out_stats.items() if v["docs"] == 0]
    if empty:
        raise ValueError(
            f"splits {empty} are empty. The corpus has too few (source, month) "
            f"groups for the requested fractions. Add sources, recover publication "
            f"dates, or raise val_frac/test_frac."
        )
    return out_stats


def load_manifest(path) -> set[str]:
    """Read a split manifest into a set of doc_ids."""
    with open(path, "r", encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}
