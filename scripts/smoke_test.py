"""End-to-end pipeline smoke test on synthetic data.

Run this on day two, before the crawl has produced anything meaningful:

    python scripts/smoke_test.py

It exercises stages 3-8 (normalize, filter, dedup, split, tokenizer, encode,
stats, plots) on a few thousand generated documents in a temporary directory.
Schema mismatches, lost ``source_type`` tags and encoding bugs surface here in
about a minute instead of on day six of the real run.

Language ID (stage 5) is skipped: it needs the trained classifier, which needs
Wikipedia downloads.  Everything else runs for real.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmacorpus import stats as S  # noqa: E402
from lmacorpus.clean.dedup import cross_corpus_collisions, dedup_near, drop_ids  # noqa: E402
from lmacorpus.io_utils import ShardWriter, dump_json  # noqa: E402
from lmacorpus.pipeline import run_clean  # noqa: E402
from lmacorpus.schema import Document  # noqa: E402
from lmacorpus.splits import build_splits  # noqa: E402
from lmacorpus.tokenizer.encode_corpus import encode_all  # noqa: E402
from lmacorpus.tokenizer.train_spm import train_spm, write_training_sample  # noqa: E402
from lmacorpus.tokenizer.vocab_sweep import evaluate_model, held_out_text, sweep_table  # noqa: E402
from lmacorpus.viz import plots as P  # noqa: E402

HI_WORDS = "भारत देश नदी पर्वत शिक्षा किसान बाजार सरकार विद्यालय संगीत यात्रा मौसम प्रौद्योगिकी संस्कृति".split()
NE_WORDS = "नेपाल हिमाल गाउँ शिक्षा किसान बजार सरकार विद्यालय संगीत यात्रा मौसम प्रविधि संस्कृति काठमाडौं".split()
CFG = {
    "map_devanagari_digits": True,
    "boilerplate_max_doc_freq": 0.01,
    "filters": {"min_chars": 200, "min_script_ratio": 0.75},
}


def make_docs(words, lang, n, rng, boiler):
    """Generate n synthetic documents, ~5% of them exact duplicates."""
    out = []
    for i in range(n):
        # Unique filler keeps documents distinguishable: without it every
        # synthetic doc is a near-duplicate of every other and stage 6 collapses
        # the whole corpus (which is itself a useful thing to have observed).
        uniq = f"संख्या {i} खण्ड {rng.randrange(10**9)}"
        sents = [
            uniq + " " + " ".join(rng.sample(words, 8)) + "।"
            for _ in range(rng.randint(8, 20))
        ]
        text = boiler + "\n" + "\n".join(sents)
        out.append(
            Document(
                text=text,
                lang=lang,
                source=f"{lang}_src{i % 3}",
                source_type="manual" if i % 4 == 0 else "downloaded",
                # Year/month drawn independently of i so the synthetic corpus has
                # enough (source, month) groups for a 5% split to be non-empty.
                url=f"https://ex.com/{rng.randrange(2015, 2025)}/"
                    f"{rng.randrange(1, 13)}/a{i}",
            ).finalize()
        )
        if i % 20 == 0 and out:
            out.append(out[-1])  # exact duplicate for the dedup counter
    return out


def run_language(base: Path, lang: str, words, rng):
    """Run stages 3,4,6,7,8 for one synthetic language."""
    d = base / lang
    raw = d / "raw"
    boiler = "कॉपीराइट © सर्वाधिकार सुरक्षित" if lang == "hi" else "प्रतिलिपि अधिकार © सर्वाधिकार सुरक्षित"
    with ShardWriter(raw, shard_docs=500) as w:
        for doc in make_docs(words, lang, 2000, rng, boiler):
            w.add(doc)

    # workers=1: the synthetic corpus is a few thousand documents, where Windows
    # process-spawn overhead costs more than the parallelism saves.
    clean_stats = run_clean(raw, d / "clean", d / "reports/clean.json", CFG, workers=1)
    assert clean_stats["kept"] > 0, "everything was filtered out"
    assert any(clean_stats["boilerplate_lines_blocked"].values()), "boilerplate not caught"

    dd = dedup_near(d / "clean", d / "dedup", threshold=0.8)
    assert dd["exact_dupes"] > 0, "exact duplicates were not detected"
    return d, clean_stats, dd


def main():
    tmp = Path(tempfile.mkdtemp(prefix="lma-smoke-"))
    rng = random.Random(0)
    print(f"[smoke] workdir {tmp}")
    report = {}

    hi_dir, hi_clean, hi_dd = run_language(tmp, "hi", HI_WORDS, rng)
    ne_dir, ne_clean, ne_dd = run_language(tmp, "ne", NE_WORDS, rng)
    report["clean_hi"], report["dedup_hi"] = hi_clean["kept"], hi_dd
    report["clean_ne"], report["dedup_ne"] = ne_clean["kept"], ne_dd

    # Stage 6c: cross-corpus collisions, dropped from BOTH sides.
    cross = cross_corpus_collisions(hi_dir / "dedup", ne_dir / "dedup")
    drop_ids(hi_dir / "dedup", hi_dir / "final", set(cross["collisions_a"]))
    drop_ids(ne_dir / "dedup", ne_dir / "final", set(cross["collisions_b"]))
    report["cross_collisions"] = cross["n_collisions_a"]

    for d, lang in ((hi_dir, "hi"), (ne_dir, "ne")):
        sp_stats = build_splits(d / "final", d / "splits", val_frac=0.05, test_frac=0.05)
        assert all(v["docs"] > 0 for v in sp_stats.values()), "empty split"

        sample = d / "tok/spm_input.txt"
        n = write_training_sample(d / "final", d / "splits/train.txt", sample)
        assert n > 0
        texts = held_out_text(d / "final", d / "splits/val.txt", max_docs=200)

        rows = []
        for V in (320, 335):  # tiny: the synthetic corpus supports only a few hundred pieces
            model = train_spm(sample, d / f"tok/{lang}_{V}", V, model_type="unigram")
            rows.append(evaluate_model(model, texts))
        table = sweep_table(rows, d_model=512)
        dump_json(d / "reports/vocab_sweep.json", table)

        final_model = train_spm(sample, d / f"tok/{lang}_spm", 335)
        meta = encode_all(d / "final", d / "splits", final_model, d / "tokens", 335)
        assert meta["splits"]["train"]["tokens"] > 0
        assert "manual" in meta["splits"]["train"]["tokens_by_source_type"], (
            "source_type did not survive to the encode stage"
        )
        report[f"meta_{lang}"] = {
            "train_tokens": meta["splits"]["train"]["tokens"],
            "manual_fraction": meta["splits"]["train"]["manual_fraction"],
        }

        # Stage 8: tables + every figure type.
        fig = tmp / "figures"
        inv = S.source_inventory(d / "final")
        P.plot_source_inventory(inv, fig / f"{lang}_inventory.png", lang)
        P.plot_length_histogram(S.length_histogram(d / "final"), fig / f"{lang}_len.png", lang)
        P.plot_manual_fraction(S.manual_fraction_by_split(meta), fig / f"{lang}_manual.png", lang)
        P.plot_vocab_sweep(table, fig / f"{lang}_sweep.png", lang)
        tf = S.token_frequency(final_model, d / "final", d / "splits/val.txt")
        P.plot_zipf(tf, fig / f"{lang}_zipf.png", lang)
        P.plot_filter_rejections(
            (hi_clean if lang == "hi" else ne_clean)["filters"], fig / f"{lang}_rej.png", lang
        )
        P.plot_confusion_matrix([[9, 1], [2, 8]], ["hi", "ne"], fig / f"{lang}_cm.png")

    n_figs = len(list((tmp / "figures").glob("*.png")))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[smoke] {n_figs} figures rendered")
    assert n_figs == 14
    shutil.rmtree(tmp)
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
