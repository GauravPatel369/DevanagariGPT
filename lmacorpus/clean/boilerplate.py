"""Cross-document boilerplate removal (part of stage 4).

Per-document filters cannot see site furniture: nav menus, "कॉपीराइट ©",
"यो खबर पनि पढ्नुहोस्", share prompts.  Those lines are individually short and
plausible, but they appear in a large fraction of a domain's pages.

Two passes over the shards for one source:
  1. count line frequency,
  2. drop lines whose document frequency exceeds ``max_doc_freq``.

Learning the blocklist from data avoids hand-written per-site rules and
generalises to every domain in the crawl.
"""

from __future__ import annotations

from collections import Counter

MIN_LINE_CHARS = 3


def build_line_counts(docs, min_line_chars: int = MIN_LINE_CHARS) -> tuple[Counter, int]:
    """Count in how many documents each distinct line appears.

    Args:
        docs: Iterable of Document.
        min_line_chars: Ignore very short lines (single punctuation, numbers).

    Returns:
        (document-frequency counter over lines, number of documents seen).
    """
    df: Counter = Counter()
    n = 0
    for d in docs:
        n += 1
        seen = {
            l.strip()
            for l in d.text.split("\n")
            if len(l.strip()) >= min_line_chars
        }
        df.update(seen)
    return df, n


def make_blocklist(df: Counter, n_docs: int, max_doc_freq: float = 0.01) -> set[str]:
    """Lines appearing in more than ``max_doc_freq`` of a source's documents."""
    if n_docs == 0:
        return set()
    cutoff = max(2, int(max_doc_freq * n_docs))
    return {line for line, c in df.items() if c >= cutoff}


def strip_boilerplate(text: str, blocklist: set[str]) -> str:
    """Remove blocklisted lines from a document."""
    kept = [l for l in text.split("\n") if l.strip() not in blocklist]
    return "\n".join(kept).strip()
