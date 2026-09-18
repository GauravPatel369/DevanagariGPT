"""Render every Phase-1 figure into ``report/Phase 1/figures/``.

The brief requires that each plot carry a title, x-label, y-label and a legend
where applicable, and that figures live inside ``report/`` on the graded branch
to be considered at all.  Every figure here is produced by
``lmacorpus.viz.plots``, which sets all four unconditionally, so the requirement
is satisfied structurally rather than by remembering it per plot.

Computation stays out of this file: the plotting module takes DataFrames and
dicts, and the numbers come from artifacts the pipeline already wrote
(``meta.json``, the tokenizer evaluation JSONs, the language-ID report).  That
keeps the visualisation / computation split the brief asks for.

Figures produced, per language unless noted:

    <lang>_source_inventory.png   corpus composition by source and collection type
    <lang>_manual_fraction.png    manual vs downloaded tokens per split, vs the 20% rule
    <lang>_vocab_sweep.png        fertility against embedding parameter cost
    <lang>_zipf.png               token frequency vs rank on held-out text
    langid_confusion.png          six-class Devanagari classifier (both languages)

Usage:
    python scripts/make_figures.py                 # all figures, both languages
    python scripts/make_figures.py --lang hindi
    python scripts/make_figures.py --skip-zipf     # skip the one slow figure
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmacorpus import stats as S  # noqa: E402
from lmacorpus.cli import paths  # noqa: E402
from lmacorpus.tokenizer.vocab_sweep import embedding_params  # noqa: E402
from lmacorpus.viz import plots as P  # noqa: E402

FIG_DIR = ROOT / "report" / "Phase 1" / "figures"
REPORT_DIR = ROOT / "report" / "Phase 1"
EVAL_JSON = {"hindi": "Hindi_report.json", "nepali": "Nepli_report.json"}
DOWNLOADED_SOURCES = {"sangraha", "sangraha_unverified", "indiccorp_v2", "cc100"}


def load_meta(lang: str) -> dict:
    path = ROOT / lang / "data" / "tokens" / "meta.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run the encode stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_frame(meta: dict) -> pd.DataFrame:
    """Per-source totals shaped for :func:`plots.plot_source_inventory`.

    Built from ``meta.json`` rather than by re-reading the shards: the token
    counts there are the ones the report quotes, so the figure and the tables
    cannot drift apart.
    """
    per_source: dict[str, int] = defaultdict(int)
    for split in meta["splits"].values():
        for source, n in split["tokens_by_source"].items():
            per_source[source] += n
    rows = [
        {
            "source": source,
            "source_type": "downloaded" if source in DOWNLOADED_SOURCES else "manual",
            "chars": n,  # plot_source_inventory pivots on this column
        }
        for source, n in sorted(per_source.items(), key=lambda kv: -kv[1])
    ]
    # A 30-source axis is unreadable; keep the top 14 and pool the tail.
    head, tail = rows[:14], rows[14:]
    if tail:
        head.append(
            {
                "source": f"other ({len(tail)})",
                "source_type": "manual",
                "chars": sum(r["chars"] for r in tail),
            }
        )
    return pd.DataFrame(head)


def sweep_rows(lang: str, d_model: int = 512) -> list[dict]:
    """Tokenizer sweep results priced against the Phase-2 embedding budget."""
    path = REPORT_DIR / EVAL_JSON[lang]
    if not path.exists():
        return []
    rows = []
    for r in json.loads(path.read_text(encoding="utf-8")):
        if r.get("model_type") != "unigram":
            continue  # one series per plot; unigram is the selected family
        rows.append(
            {
                "vocab_size": r["vocab_size"],
                "fertility_tokens_per_word": r["fertility"],
                "embedding_params": embedding_params(r["vocab_size"], d_model),
            }
        )
    return sorted(rows, key=lambda r: r["vocab_size"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=["hindi", "nepali"], default=None)
    ap.add_argument("--skip-zipf", action="store_true",
                    help="skip the token-frequency figure, which must tokenize held-out text")
    ap.add_argument("--zipf-docs", type=int, default=4000,
                    help="held-out documents sampled for the Zipf figure")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for lang in ([a.lang] if a.lang else ["hindi", "nepali"]):
        label = lang.capitalize()
        p = paths(lang)
        meta = load_meta(lang)

        written.append(P.plot_source_inventory(
            inventory_frame(meta), FIG_DIR / f"{lang}_source_inventory.png", label))
        print(f"[fig] {lang}: source inventory", flush=True)

        written.append(P.plot_manual_fraction(
            S.manual_fraction_by_split(meta), FIG_DIR / f"{lang}_manual_fraction.png", label))
        print(f"[fig] {lang}: manual fraction per split", flush=True)

        rows = sweep_rows(lang)
        if rows:
            written.append(P.plot_vocab_sweep(
                rows, FIG_DIR / f"{lang}_vocab_sweep.png", label))
            print(f"[fig] {lang}: vocabulary sweep", flush=True)
        else:
            print(f"[fig] {lang}: no sweep JSON, skipping vocab sweep", flush=True)

        if not a.skip_zipf:
            code = "hi" if lang == "hindi" else "ne"
            model = p["tokenizer"] / f"{code}_unigram_10000.model"
            src = p["final"] if p["final"].exists() else p["dedup"]
            print(f"[fig] {lang}: tokenizing {a.zipf_docs:,} held-out docs for Zipf ...", flush=True)
            freq = S.token_frequency(model, src, p["splits"] / "val.txt", max_docs=a.zipf_docs)
            written.append(P.plot_zipf(freq, FIG_DIR / f"{lang}_zipf.png", label))
            # The brief also asks for token-frequency statistics as a table.
            out_csv = REPORT_DIR / f"{lang}_top50_tokens.csv"
            freq.head(50).to_csv(out_csv, index=False, encoding="utf-8")
            print(f"[fig] {lang}: Zipf + top-50 token table -> {out_csv.name}", flush=True)

    lid = ROOT / "models" / "langid_devanagari.report.json"
    if lid.exists():
        r = json.loads(lid.read_text(encoding="utf-8"))
        acc = r["classification_report"]["accuracy"]
        written.append(P.plot_confusion_matrix(
            r["confusion_matrix"], r["labels"], FIG_DIR / "langid_confusion.png",
            title=f"Six-class Devanagari language ID (held-out accuracy {acc:.3f})"))
        print("[fig] language-ID confusion matrix", flush=True)

    print(f"\n{len(written)} figures written to {FIG_DIR}")
    for w in written:
        print("   ", Path(w).name)


if __name__ == "__main__":
    main()
