"""Stage 4: document quality filters (Gopher-style heuristics).

Every filter returns True when the document should be *rejected*.  Rejections
are counted per reason and per source; that breakdown is a graded table in the
Phase-1 report ("we dropped 34% of cc100-ne, here is why").

Thresholds live in ``<lang>/configs/clean.yaml`` so they can be tuned without
touching code, and so the exact values used are recorded with the run.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .normalize import script_ratio

TERMINALS = "।॥?!"
SYMBOLS = set("#…•|©®™")
WORD_RE = re.compile(r"\S+")


def mean_word_len(text: str) -> float:
    """Mean length in characters of whitespace-delimited tokens."""
    words = WORD_RE.findall(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def frac_lines_terminal(text: str) -> float:
    """Fraction of non-empty lines ending in a sentence terminal (danda etc.)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    return sum(1 for l in lines if l[-1] in TERMINALS) / len(lines)


def symbol_word_ratio(text: str) -> float:
    """Ratio of junk symbols to words; high values indicate nav/boilerplate dumps."""
    words = WORD_RE.findall(text)
    if not words:
        return 1.0
    return sum(1 for c in text if c in SYMBOLS) / len(words)


def frac_dup_lines(text: str) -> float:
    """Fraction of lines that are exact repeats within the same document."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return 0.0
    counts = Counter(lines)
    return sum(c - 1 for c in counts.values()) / len(lines)


def frac_dup_ngrams(text: str, n: int = 5) -> float:
    """Fraction of word n-grams that repeat within the document."""
    words = WORD_RE.findall(text)
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    return sum(c - 1 for c in counts.values()) / len(grams)


@dataclass
class FilterConfig:
    """Thresholds for the quality filters (loaded from clean.yaml)."""

    min_chars: int = 200
    min_script_ratio: float = 0.75
    min_word_len: float = 1.5
    max_word_len: float = 12.0
    min_terminal_frac: float = 0.20
    max_symbol_ratio: float = 0.10
    max_dup_lines: float = 0.30
    max_dup_ngrams: float = 0.20
    dup_ngram_n: int = 5


IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf", ".ico", ".bmp", ".tif", ".tiff"
)
BINARY_HEADERS = ("JFIF", "Exif", "Photoshop", "XMP", "%PDF", "\x00\x00")

def is_image_or_binary(doc) -> bool:
    url = getattr(doc, "url", "") or ""
    url_clean = url.lower().split("?")[0]
    if any(url_clean.endswith(ext) for ext in IMAGE_EXTENSIONS):
        return True
    text_snippet = doc.text[:300]
    if any(header in text_snippet for header in BINARY_HEADERS):
        return True
    return False


class QualityFilter:
    """Applies all quality filters and tallies rejections by reason and source."""

    def __init__(self, cfg: FilterConfig):
        self.cfg = cfg
        self.rejected: dict[str, Counter] = defaultdict(Counter)
        self.kept: Counter = Counter()

    def reason(self, text: str) -> str | None:
        """Return the name of the first failing filter, or None if the doc passes."""
        c = self.cfg
        if len(text) < c.min_chars:
            return "too_short"
        sr = script_ratio(text)
        if sr < c.min_script_ratio:
            return "low_script_ratio"
        wl = mean_word_len(text)
        if not (c.min_word_len <= wl <= c.max_word_len):
            return "bad_word_length"
        if c.min_terminal_frac > 0 and frac_lines_terminal(text) < c.min_terminal_frac:
            return "no_sentence_terminals"
        if symbol_word_ratio(text) > c.max_symbol_ratio:
            return "symbol_heavy"
        if frac_dup_lines(text) > c.max_dup_lines:
            return "duplicate_lines"
        if frac_dup_ngrams(text, c.dup_ngram_n) > c.max_dup_ngrams:
            return "duplicate_ngrams"
        return None

    def keep(self, doc) -> bool:
        """Test one document, updating counters.  True means keep."""
        if is_image_or_binary(doc):
            self.rejected[doc.source]["image_or_binary_file"] += 1
            return False
        r = self.reason(doc.text)
        if r is None:
            doc.flags["script_ratio"] = round(script_ratio(doc.text), 4)
            self.kept[doc.source] += 1
            return True
        self.rejected[doc.source][r] += 1
        return False

    def report(self) -> dict:
        """Rejection breakdown by source and reason, plus survival rates."""
        out = {}
        for source in set(self.kept) | set(self.rejected):
            rej = dict(self.rejected.get(source, {}))
            kept = self.kept.get(source, 0)
            total = kept + sum(rej.values())
            out[source] = {
                "kept": kept,
                "total": total,
                "survival_rate": round(kept / total, 4) if total else 0.0,
                "rejected": rej,
            }
        return out
