"""Stage 3: Unicode and Devanagari normalization.

Order of operations matters.  Documented decisions, all of which are reported in
``report/phase1.md``:

* **NFC first.**  NFC *decomposes* the nukta letters (U+0958..U+095F, e.g. क़)
  because they sit on the Unicode composition exclusion list.  The two-codepoint
  form क + U+093C is the one consistent representation we keep; it changes
  character counts relative to naive expectations, so we state it explicitly.
* **ZWJ / ZWNJ are stripped.**  In plain Devanagari body text they are almost
  always crawl noise.  We count removals per source as a dirtiness signal.
* **Anusvara ं is NOT folded into chandrabindu ँ.**  They are contrastive in both
  Hindi and Nepali.  Visarga and halant are likewise left alone.  Over-
  normalizing would destroy real linguistic distinctions.
* **Identical policy for both languages.**  Any asymmetry between Hindi and
  Nepali would contaminate the Phase-3 resource-tier comparison.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width and bidi marks, soft hyphen, BOM.
INVISIBLE = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad"), None
)

# C0/C1 controls except newline and tab.
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_DEVA_OR_SPACE = r"[\u0900-\u097F\s]"


def strip_invisible(text: str) -> tuple[str, int]:
    """Remove zero-width/bidi/soft-hyphen characters.

    Returns:
        The cleaned text and the number of characters removed (reported per
        source as a corpus-quality signal).
    """
    out = text.translate(INVISIBLE)
    return out, len(text) - len(out)


# Image links, image markdown tags, HTML img tags, and image filenames (.jpg, .jpeg, .png, .gif, .webp, .svg, .pdf).
IMAGE_PATTERNS = re.compile(
    r"(?:!\[.*?\]\(.*?\)|<img[^>]*>|https?://\S+\.(?:jpg|jpeg|png|gif|webp|svg|pdf)|\b\S+\.(?:jpg|jpeg|png|gif|webp|svg|pdf)\b)",
    re.IGNORECASE,
)

# Strips URLs, email addresses, and all English/Latin words to guarantee ZERO English in the final dataset.
LATIN_AND_URL_RE = re.compile(r"https?://\S+|www\.\S+|\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b|\b[a-zA-Z]+\b")
LATIN_RE = re.compile(r"[a-zA-Z]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[।॥?!\n])")

def filter_english_statements(text: str) -> str:
    """Statement-wise filter: Keeps pure Devanagari statements, drops statements containing English words."""
    if not LATIN_RE.search(text):
        return text

    paragraphs = text.split("\n")
    clean_paragraphs = []
    for p in paragraphs:
        if not p.strip():
            clean_paragraphs.append("")
            continue
        sentences = re.split(r"(?<=[।॥?!])\s*", p)
        kept_sents = [s.strip() for s in sentences if s.strip() and not LATIN_RE.search(s)]
        if kept_sents:
            clean_paragraphs.append(" ".join(kept_sents))
        else:
            clean_paragraphs.append("")
    return "\n".join(clean_paragraphs)

def normalize(text: str, map_digits: bool = True) -> str:
    """Normalize one document's text.

    Args:
        text: Raw extracted text.
        map_digits: Map Devanagari digits ०-९ to ASCII 0-9.  Reduces vocabulary
            pressure; must be set identically for both languages.

    Returns:
        Normalized text, stripped of leading/trailing whitespace.
    """
    text = unicodedata.normalize("NFC", text)
    text = IMAGE_PATTERNS.sub(" ", text)
    text = filter_english_statements(text)
    text = LATIN_AND_URL_RE.sub(" ", text)
    text, _ = strip_invisible(text)
    text = CTRL.sub(" ", text)
    if map_digits:
        text = text.translate(DEVA_DIGITS)
    # Danda repair: crawls frequently emit ASCII pipes for danda/double danda.
    text = re.sub(r"\|\|", "॥", text)
    text = re.sub(rf"(?<={_DEVA_OR_SPACE})\|", "।", text)
    text = re.sub(r"।{2,}", "।", text)
    text = re.sub(r"॥{2,}", "॥", text)
    # Whitespace collapse (NBSP included), paragraph breaks preserved.
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_letter(ch: str) -> bool:
    """Letter test that is fair to Indic scripts.

    ``str.isalpha()`` is False for Devanagari matras and the anusvara (Unicode
    categories Mn/Mc), so an isalpha-only denominator counts every Latin letter
    but only the consonants of each Devanagari syllable.  On mixed text that
    biases the script ratio downwards by roughly a factor of two and silently
    discards good documents at the filter stage.  Combining marks are therefore
    counted as letters of whatever script they attach to.
    """
    return ch.isalpha() or unicodedata.category(ch) in ("Mn", "Mc")


def script_ratio(text: str) -> float:
    """Fraction of *letters* that are Devanagari.

    Computed over letters only.  Including punctuation and ASCII digits in the
    denominator drags the ratio down on perfectly good documents and throws them
    away at the filter stage.
    """
    letters = [c for c in text if _is_letter(c)]
    if not letters:
        return 0.0
    return sum(1 for c in letters if DEVANAGARI.match(c)) / len(letters)
