"""Generate the report's corpus tables from measured artifacts.

Every number in sections 1-4 of ``report/Phase 1/phase1.md`` comes from here.
The tables were previously typed by hand and drifted from the data as soon as
the corpus was re-encoded — listing sources the corpus never contained, and
round document counts no hash-based splitter would produce.

Sources of truth, in order of preference:

* ``<lang>/data/tokens/meta.json``  — token and document counts per split and
  per source. This is the file the encode stage writes and the one Phase 2
  loads, so the report and the training data cannot disagree.
* ``<lang>/data/reports/*.json``    — per-stage survival counts for the funnel.

A stage with no report file on disk is printed as unavailable rather than
estimated: an estimated funnel row is exactly the class of number this script
exists to eliminate.

Usage:
    python scripts/build_report_tables.py                       # stdout
    python scripts/build_report_tables.py --out tables.md
    python scripts/build_report_tables.py --lang hindi
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Sources streamed from public corpora; everything else came from the crawl.
# The >=20% manual requirement is computed from this distinction.
DOWNLOADED_SOURCES = {"sangraha", "sangraha_unverified", "indiccorp_v2", "cc100"}


def load_meta(lang: str) -> dict:
    path = ROOT / lang / "data" / "tokens" / "meta.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — run the encode stage first")
    return json.loads(path.read_text(encoding="utf-8"))


def totals(meta: dict):
    """Aggregate per-source tokens plus corpus-level token/document/byte totals."""
    per_source: dict[str, int] = defaultdict(int)
    by_type: dict[str, int] = defaultdict(int)
    tokens = docs = raw_bytes = 0
    for split in meta["splits"].values():
        tokens += split["tokens"]
        docs += split["documents"]
        raw_bytes += split["raw_bytes"]
        for source, n in split["tokens_by_source"].items():
            per_source[source] += n
        for kind, n in split["tokens_by_source_type"].items():
            by_type[kind] += n
    return dict(per_source), tokens, docs, raw_bytes, dict(by_type)


def source_table(lang: str, meta: dict, num: str, top: int = 12) -> str:
    per_source, tokens, docs, _, by_type = totals(meta)
    rows = sorted(per_source.items(), key=lambda kv: -kv[1])
    rest = sum(n for _, n in rows[top:])
    out = [
        f"### 2.{num} {lang.capitalize()} Source Inventory",
        "",
        f"{len(per_source)} distinct sources. The {top} largest are shown; the remaining "
        f"{len(per_source) - top} contribute {rest / tokens:.2%} combined.",
        "",
        "| Source | Collection | Tokens | Share of corpus |",
        "|---|---|---|---|",
    ]
    for source, n in rows[:top]:
        kind = "Downloaded" if source in DOWNLOADED_SOURCES else "Manual"
        out.append(f"| `{source}` | {kind} | {n:,} | {n / tokens:.2%} |")
    out.append(
        f"| _{len(per_source) - top} further manual sources_ | Manual | {rest:,} | {rest / tokens:.2%} |"
    )
    out.append(f"| **Total** | | **{tokens:,}** | **100.00%** |")
    out += [
        "",
        f"Manual **{by_type.get('manual', 0):,}** tokens · "
        f"Downloaded **{by_type.get('downloaded', 0):,}** · "
        f"Documents **{docs:,}**",
        "",
    ]
    return "\n".join(out)


def split_table(lang: str, meta: dict) -> str:
    _, tokens, docs, raw_bytes, by_type = totals(meta)
    out = [
        f"#### {lang.capitalize()} (`{'hi' if lang == 'hindi' else 'ne'}`) Split Statistics",
        "",
        "| Split | Documents | Tokens | Share | Raw bytes | Manual tokens | Manual share |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("train", "val", "test"):
        s = meta["splits"][name]
        manual = s["tokens_by_source_type"].get("manual", 0)
        flag = " [SATISFIED]" if name == "train" and manual / s["tokens"] >= 0.20 else ""
        out.append(
            f"| **{name.capitalize()}** | {s['documents']:,} | {s['tokens']:,} | "
            f"{s['tokens'] / tokens:.1%} | {s['raw_bytes'] / 1e9:.2f} GB | {manual:,} | "
            f"**{manual / s['tokens']:.2%}**{flag} |"
        )
    manual_total = by_type.get("manual", 0)
    out.append(
        f"| **Total** | **{docs:,}** | **{tokens:,}** | **100%** | "
        f"**{raw_bytes / 1e9:.2f} GB** | **{manual_total:,}** | **{manual_total / tokens:.2%}** |"
    )
    train = meta["splits"]["train"]
    frac = train["tokens_by_source_type"].get("manual", 0) / train["tokens"]
    out += [
        "",
        f"Train manual fraction **{frac:.4f}** — requirement $\\ge 0.20$ "
        f"**{'SATISFIED' if frac >= 0.20 else 'NOT SATISFIED'}**. "
        f"Tokenizer vocabulary **{meta['vocab_size']:,}**.",
        "",
    ]
    return "\n".join(out)


def funnel_table() -> str:
    """Stage survival counts, from the per-stage reports the pipeline wrote."""
    lines = [
        "| Language | Collection | After clean & quality filters | After language ID | Quarantined by LID |",
        "|---|---|---|---|---|",
    ]
    for lang in ("hindi", "nepali"):
        base = ROOT / lang / "data" / "reports"

        def read(name: str):
            path = base / name
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None

        for kind in ("manual", "downloaded"):
            clean = read(f"clean_{kind}_stats.json") or read(f"clean_{kind}.json")
            lid = read(f"lid_{kind}_stats.json")
            fmt = lambda v: f"{v:,}" if isinstance(v, int) else "_not reported_"
            lines.append(
                f"| {lang.capitalize()} | {kind} | {fmt(clean.get('kept') if clean else None)} | "
                f"{fmt(lid.get('kept') if lid else None)} | "
                f"{fmt(lid.get('quarantined') if lid else None)} |"
            )

    dedup = None
    path = ROOT / "nepali" / "data" / "reports" / "dedup.json"
    if path.exists():
        dedup = json.loads(path.read_text(encoding="utf-8"))
    lines.append("")
    if dedup:
        lines.append(
            f"**Deduplication (Nepali).** {dedup['seen']:,} documents seen, {dedup['kept']:,} kept — "
            f"{dedup['exact_dupes']:,} exact duplicates and {dedup['near_dupes']:,} near-duplicates "
            f"removed at Jaccard $\\ge 0.80$. The Hindi `dedup.json` was not preserved, so the "
            f"equivalent Hindi row cannot be reported without re-running that stage; the post-dedup "
            f"Hindi totals in §2 and §4 are measured after the fact and are unaffected."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=["hindi", "nepali"], default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    langs = [a.lang] if a.lang else ["hindi", "nepali"]
    chunks = ["<!-- generated by scripts/build_report_tables.py — do not hand-edit -->", ""]
    for i, lang in enumerate(langs, start=1):
        meta = load_meta(lang)
        chunks += [source_table(lang, meta, str(i)), split_table(lang, meta)]
    chunks.append(funnel_table())
    text = "\n".join(chunks)

    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
