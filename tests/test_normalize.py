"""Tests for Devanagari normalization.

Normalization bugs are silent: they do not raise, they just quietly poison every
downstream stage and show up as an unexplained perplexity gap in Phase 2. These
cases were written before the normalizer itself.
"""

import unicodedata

from lmacorpus.clean.normalize import normalize, script_ratio, strip_invisible


def test_nfc_decomposes_nukta():
    """NFC yields the two-codepoint form for composition-excluded nukta letters."""
    precomposed = "\u0958"  # क़
    out = normalize(precomposed)
    assert out == "\u0915\u093c"
    assert len(out) == 2


def test_nukta_forms_unify():
    """Both spellings of क़ converge on one representation."""
    assert normalize("\u0958") == normalize("\u0915\u093c")


def test_zwnj_and_zwj_removed():
    """Zero-width joiners are crawl noise in plain Devanagari body text."""
    text = "क\u200dर्म और क\u200cर्म"
    assert "\u200d" not in normalize(text)
    assert "\u200c" not in normalize(text)


def test_strip_invisible_counts():
    """Removal count is reported per source as a corpus-dirtiness signal."""
    out, n = strip_invisible("अ\u200bब\ufeffस")
    assert out == "अबस"
    assert n == 2


def test_anusvara_and_chandrabindu_preserved():
    """ं and ँ are contrastive; folding them would destroy real distinctions."""
    a, b = normalize("हंस"), normalize("हँस")
    assert a != b
    assert "\u0902" in a and "\u0901" in b


def test_visarga_and_halant_preserved():
    """Visarga and halant carry meaning and must survive normalization."""
    assert "\u0903" in normalize("दुःख")
    assert "\u094d" in normalize("क्रम")


def test_devanagari_digits_mapped_when_enabled():
    """Digit mapping is opt-in and must be identical across both languages."""
    assert normalize("सन् २०२४ मा", map_digits=True) == "सन् 2024 मा"
    assert "२" in normalize("सन् २०२४ मा", map_digits=False)


def test_ascii_pipe_becomes_danda():
    """Crawls frequently emit ASCII pipes where a danda belongs."""
    assert normalize("यह एक वाक्य है |").endswith("।")
    assert "॥" in normalize("श्लोक ||")


def test_repeated_danda_collapsed():
    assert normalize("वाक्य।।।") == "वाक्य।"


def test_whitespace_collapsed_paragraphs_kept():
    out = normalize("पहला   वाक्य।\n\n\n\nदूसरा वाक्य।")
    assert "   " not in out
    assert out.count("\n") == 2  # exactly one blank line survives


def test_control_characters_removed():
    assert "\x07" not in normalize("अ\x07ब स है।")


def test_script_ratio_ignores_punctuation_and_digits():
    """Denominator is letters only; otherwise good documents get filtered out."""
    text = "यह एक वाक्य है। 12345 !!! ---"
    assert script_ratio(text) == 1.0


def test_script_ratio_mixed():
    """Matras must count as Devanagari letters, or the ratio halves unfairly."""
    assert script_ratio("हिंदी hindi") == 0.5


def test_script_ratio_empty():
    assert script_ratio("12345 !!!") == 0.0


def test_normalize_is_idempotent():
    """Running the pipeline twice must not change the text again."""
    raw = "क़िस्मत\u200c  ।। सन् २०२४ |"
    once = normalize(raw)
    assert normalize(once) == once


def test_output_is_nfc():
    out = normalize("हिन्दी भाषा।")
    assert out == unicodedata.normalize("NFC", out)
