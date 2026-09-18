"""Plot the vocabulary sweep: the metric disagreement, and why it happens.

Three panels, because the result is a *contrast* rather than a single curve:

  1. Perplexity vs vocabulary       -- falls with smaller vocabulary
  2. Bits-per-byte vs vocabulary    -- rises with smaller vocabulary
  3. Tokens needed for the same text -- the mechanism explaining both

Panels 1 and 2 point in opposite directions on identical checkpoints, which is
the finding. Panel 3 is the explanation: a smaller vocabulary cuts the same text
into more pieces, so each prediction is easier (flattering perplexity) while the
total cost of encoding the bytes does not improve (bits-per-byte).

Usage:
    python scripts/plot_vocab_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "Phase 2" / "figures"
VOCABS = (5000, 8000, 10000)

# Where each run's evaluation landed. V=10,000 is the Phase-2 baseline itself,
# trained on byte-identical tokens, so it serves as that row rather than being
# retrained.
SOURCES = {
    ("hindi", 5000): "vocab_sweep/vocab_5000/evaluation_test.json",
    ("hindi", 8000): "vocab_sweep/vocab_8000/evaluation_test.json",
    ("hindi", 10000): "evaluation_test.json",
    ("nepali", 5000): "vocab_sweep/ne_vocab_5000/evaluation_test.json",
    ("nepali", 8000): "vocab_sweep/ne_vocab_8000/evaluation_test.json",
    ("nepali", 10000): "evaluation_test.json",
}

STYLE = {"hindi": dict(color="#C2410C", marker="o", label="Hindi (Model H)"),
         "nepali": dict(color="#1D4ED8", marker="s", label="Nepali (Model L)")}


def load() -> dict:
    """Read every sweep result into ``{(lang, vocab): metrics}``."""
    base = ROOT / "report" / "Phase 2"
    data = {}
    for (lang, vocab), rel in SOURCES.items():
        payload = json.loads((base / rel).read_text(encoding="utf-8"))
        entry = payload.get(lang, payload)
        data[(lang, vocab)] = entry["intrinsic"]
    return data


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    data = load()
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [
        ("perplexity", "Test perplexity",
         "1. Perplexity favours SMALL vocabularies", False),
        ("bits_per_byte", "Test bits per byte",
         "2. Bits-per-byte favours LARGE vocabularies", False),
        ("tokens_evaluated", "Tokens to encode the test split",
         "3. Why: smaller vocabulary = more tokens", True),
    ]

    for ax, (key, ylabel, title, millions) in zip(axes, panels):
        for lang, style in STYLE.items():
            ys = [data[(lang, v)][key] for v in VOCABS]
            if millions:
                ys = [y / 1e6 for y in ys]
            ax.plot(VOCABS, ys, linewidth=2, markersize=8, **style)
            for v, y in zip(VOCABS, ys):
                ax.annotate(f"{y:,.4f}" if key == "bits_per_byte" else f"{y:,.1f}",
                            (v, y), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=8, color=style["color"])
        ax.set_xlabel("Vocabulary size")
        ax.set_ylabel(ylabel + (" (millions)" if millions else ""))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(VOCABS)
        ax.set_xticklabels([f"{v//1000}k" for v in VOCABS])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        # Headroom for the annotations, which otherwise clip at the axis top.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.15)

    fig.suptitle("Vocabulary sweep at a fixed ~25.4M parameter budget: "
                 "the two intrinsic metrics disagree, in both languages",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = OUT / "vocab_sweep.png"
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")

    # ---- second figure: bits-per-byte alone, the metric that decides ---------
    # Panel 1 of the first figure invites the wrong conclusion if read alone, so
    # the decisive metric also gets a chart of its own.
    fig2, ax = plt.subplots(figsize=(6.5, 4.4))
    for lang, style in STYLE.items():
        ys = [data[(lang, v)]["bits_per_byte"] for v in VOCABS]
        ax.plot(VOCABS, ys, linewidth=2.2, markersize=9, **style)
        for v, y in zip(VOCABS, ys):
            ax.annotate(f"{y:.4f}", (v, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=style["color"])
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Test bits per byte  (lower is better)")
    ax.set_title("Bits-per-byte: monotone in both languages,\n"
                 "with no minimum inside the tested range", fontsize=11, fontweight="bold")
    ax.set_xticks(VOCABS)
    ax.set_xticklabels([f"{v:,}" for v in VOCABS])
    ax.grid(alpha=0.3)
    ax.legend()
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.18)
    fig2.tight_layout()
    path2 = OUT / "vocab_sweep_bpb.png"
    fig2.savefig(path2, dpi=150)
    print(f"  wrote {path2}")


if __name__ == "__main__":
    main()
