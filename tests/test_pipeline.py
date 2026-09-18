"""Tests for filters, splits, schema invariants and shard I/O."""

import tempfile
from pathlib import Path

import pytest

from lmacorpus.clean.boilerplate import build_line_counts, make_blocklist, strip_boilerplate
from lmacorpus.clean.filters import (
    FilterConfig,
    QualityFilter,
    frac_dup_lines,
    frac_lines_terminal,
    mean_word_len,
)
from lmacorpus.io_utils import ShardWriter, read_dir
from lmacorpus.schema import Document
from lmacorpus.splits import assign, build_splits, group_key

# A realistic document. Sentences must be genuinely varied: templated text
# repeats 5-grams and is correctly rejected by the duplicate-ngram filter, which
# is the desired behaviour on real boilerplate pages.
GOOD = " ".join(
    [
        "भारत दक्षिण एशिया में स्थित एक विशाल देश है।",
        "यहाँ अनेक भाषाएँ बोली जाती हैं और संस्कृतियाँ फलती फूलती हैं।",
        "हिमालय की पर्वत श्रृंखला उत्तरी सीमा पर फैली हुई है।",
        "गंगा नदी कई राज्यों से होकर बंगाल की खाड़ी में गिरती है।",
        "कृषि आज भी बड़ी आबादी की आजीविका का मुख्य आधार बनी हुई है।",
        "पिछले कुछ दशकों में सूचना प्रौद्योगिकी क्षेत्र तेजी से बढ़ा है।",
        "शिक्षा के प्रसार से साक्षरता दर में उल्लेखनीय सुधार हुआ है।",
        "देश की अर्थव्यवस्था सेवा क्षेत्र पर बहुत निर्भर करती है।",
        "मानसून की वर्षा खेती के लिए निर्णायक भूमिका निभाती है।",
        "शहरीकरण के कारण महानगरों पर दबाव लगातार बढ़ रहा है।",
        "सार्वजनिक परिवहन व्यवस्था को सुधारने की आवश्यकता महसूस की जा रही है।",
        "साहित्य और संगीत की परंपरा सदियों पुरानी मानी जाती है।",
    ]
)


def doc(text, **kw):
    kw.setdefault("lang", "hi")
    kw.setdefault("source", "test")
    kw.setdefault("source_type", "downloaded")
    return Document(text=text, **kw).finalize()


# ------------------------------------------------------------------ schema


def test_doc_id_is_content_addressed():
    assert doc(GOOD).doc_id == doc(GOOD).doc_id
    assert doc(GOOD).doc_id != doc(GOOD + "x").doc_id


def test_bad_source_type_rejected():
    with pytest.raises(ValueError):
        Document(text="x", lang="hi", source="s", source_type="scraped").finalize()


def test_byte_count_recorded():
    """Phase-2 bits-per-byte needs the UTF-8 byte count of the raw text."""
    d = doc("नमस्ते")
    assert d.flags["n_bytes"] == len("नमस्ते".encode("utf-8"))
    assert d.flags["n_bytes"] > d.flags["n_chars"]


# ------------------------------------------------------------------ filters


def test_good_document_passes():
    qf = QualityFilter(FilterConfig())
    assert qf.reason(GOOD) is None


def test_short_document_rejected():
    assert QualityFilter(FilterConfig()).reason("छोटा।") == "too_short"


def test_latin_document_rejected_on_script_ratio():
    text = "This is an entirely English document repeated many times. " * 10
    assert QualityFilter(FilterConfig()).reason(text) == "low_script_ratio"


def test_nav_dump_rejected_for_no_terminals():
    text = "\n".join(["मुखपृष्ठ", "समाचार", "खेल", "मनोरंजन", "व्यापार"] * 20)
    assert QualityFilter(FilterConfig()).reason(text) in {
        "no_sentence_terminals",
        "duplicate_lines",
        "bad_word_length",
    }


def test_repeated_lines_rejected():
    text = ("यह एक दोहराई गई पंक्ति है।\n" * 50)
    assert frac_dup_lines(text) > 0.9
    assert QualityFilter(FilterConfig()).reason(text) == "duplicate_lines"


def test_filter_report_counts_reasons():
    qf = QualityFilter(FilterConfig())
    qf.keep(doc(GOOD))
    qf.keep(doc("छोटा।"))
    r = qf.report()["test"]
    assert r["kept"] == 1 and r["rejected"]["too_short"] == 1
    assert r["survival_rate"] == 0.5


def test_helpers():
    assert mean_word_len("अब सब") == pytest.approx(2.0)
    assert frac_lines_terminal("वाक्य।\nअधूरा") == pytest.approx(0.5)


# -------------------------------------------------------------- boilerplate


def test_boilerplate_blocklist_learned_from_frequency():
    docs = [doc(f"कॉपीराइट ©\nअसली सामग्री {i}।") for i in range(200)]
    df, n = build_line_counts(docs)
    block = make_blocklist(df, n, max_doc_freq=0.01)
    assert "कॉपीराइट ©" in block
    assert "असली सामग्री 5।" not in block
    assert strip_boilerplate("कॉपीराइट ©\nरखो।", block) == "रखो।"


# ------------------------------------------------------------------ splits


def test_split_assignment_deterministic():
    d = doc(GOOD, url="https://x.com/2023/05/story")
    assert assign(d) == assign(d)


def test_group_key_uses_url_date():
    d = doc(GOOD, source="jagran", url="https://jagran.com/2023/5/news-item")
    assert group_key(d) == "jagran|2023-05"


def test_same_group_lands_in_same_split():
    """Group-aware splitting keeps same-event coverage off both sides."""
    a = doc(GOOD + "a", source="j", url="https://j.com/2023/05/one")
    b = doc(GOOD + "b", source="j", url="https://j.com/2023/05/two")
    assert assign(a) == assign(b)


def test_build_splits_partitions_everything():
    with tempfile.TemporaryDirectory() as td:
        shards, out = Path(td) / "sh", Path(td) / "sp"
        with ShardWriter(shards, shard_docs=10) as w:
            for i in range(300):
                w.add(doc(GOOD + str(i), source=f"s{i % 7}"))
        stats = build_splits(shards, out, val_frac=0.1, test_frac=0.1)
        assert sum(s["docs"] for s in stats.values()) == 300
        ids = [set(open(out / f"{s}.txt").read().split()) for s in ("train", "val", "test")]
        assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_split_stats_track_source_type():
    """The >=20% manual rule needs source_type to survive to the split stage."""
    with tempfile.TemporaryDirectory() as td:
        shards, out = Path(td) / "sh", Path(td) / "sp"
        with ShardWriter(shards) as w:
            for i in range(100):
                w.add(doc(GOOD + str(i), source_type="manual" if i % 2 else "downloaded"))
        stats = build_splits(shards, out)
        total = sum(sum(s["by_source_type"].values()) for s in stats.values())
        assert total == 100


# ------------------------------------------------------------------ shard IO


def test_shard_roundtrip_preserves_devanagari_and_flags():
    with tempfile.TemporaryDirectory() as td:
        with ShardWriter(td, shard_docs=5) as w:
            for i in range(12):
                w.add(doc(GOOD + str(i), source_type="manual"))
        back = list(read_dir(td))
        assert len(back) == 12
        assert all(d.source_type == "manual" for d in back)
        assert "यह" in back[0].text
