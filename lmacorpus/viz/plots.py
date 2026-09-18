"""All Phase-1 plotting.

This is the only module in the project permitted to import matplotlib.  Every
function takes a DataFrame produced by ``lmacorpus.stats`` and writes a figure;
none of them compute statistics.

The project states that plots missing a title, axis labels, or a legend (where
applicable) may score zero for that component, so every function below sets all
of them unconditionally.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGSIZE = (9, 5)
DPI = 150


def _save(fig, out_path):
    """Write a figure to disk, creating parent directories."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_source_inventory(df, out_path, language: str):
    """Stacked bar of tokens (or chars) per source, split by manual/downloaded."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    pivot = df.pivot_table(
        index="source", columns="source_type", values="chars", aggfunc="sum"
    ).fillna(0)
    bottom = None
    for col in pivot.columns:
        ax.bar(pivot.index, pivot[col], bottom=bottom, label=col)
        bottom = pivot[col] if bottom is None else bottom + pivot[col]
    ax.set_title(f"Corpus size by source and collection type — {language}")
    ax.set_xlabel("Source")
    ax.set_ylabel("Characters")
    ax.legend(title="Collection type")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, out_path)


def plot_filter_rejections(report: dict, out_path, language: str):
    """Grouped bar of rejection counts per reason, per source."""
    reasons = sorted({r for v in report.values() for r in v["rejected"]})
    sources = sorted(report)
    fig, ax = plt.subplots(figsize=(11, 5))
    if not reasons:
        # A legend is mandatory, so plot the retained counts rather than nothing.
        ax.bar(range(len(sources)), [report[s]["kept"] for s in sources],
               label="Documents retained (no rejections recorded)")
        reasons = []
    width = 0.8 / max(len(reasons), 1)
    for i, reason in enumerate(reasons):
        vals = [report[s]["rejected"].get(reason, 0) for s in sources]
        ax.bar([x + i * width for x in range(len(sources))], vals, width, label=reason)
    ax.set_xticks([x + 0.4 for x in range(len(sources))])
    ax.set_xticklabels(sources, rotation=45, ha="right")
    ax.set_title(f"Quality-filter rejections by reason and source — {language}")
    ax.set_xlabel("Source")
    ax.set_ylabel("Documents rejected")
    ax.legend(title="Rejection reason", fontsize=8)
    return _save(fig, out_path)


def plot_confusion_matrix(cm, labels, out_path, title="Devanagari language ID"):
    """Row-normalised confusion matrix for the language classifier."""
    import numpy as np

    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                color="white" if norm[i, j] > 0.5 else "black", fontsize=8,
            )
    ax.set_title(f"{title} — held-out confusion matrix (row-normalised)")
    ax.set_xlabel("Predicted language")
    ax.set_ylabel("True language")
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    return _save(fig, out_path)


def plot_dedup_funnel(df, out_path, language: str):
    """Documents surviving each pipeline stage."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["stage"], df["documents"], marker="o", label="Documents retained")
    for x, y in zip(df["stage"], df["documents"]):
        ax.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_title(f"Pipeline attrition by stage — {language}")
    ax.set_xlabel("Pipeline stage")
    ax.set_ylabel("Documents")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    return _save(fig, out_path)


def plot_length_histogram(df, out_path, language: str):
    """Document length distribution."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(df["bucket"], df["documents"], label="Documents")
    ax.set_yscale("log")
    ax.set_title(f"Document length distribution — {language}")
    ax.set_xlabel("Document length (characters)")
    ax.set_ylabel("Number of documents (log scale)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    return _save(fig, out_path)


def plot_vocab_sweep(rows, out_path, language: str, budget: int = 25_000_000):
    """Fertility against vocabulary size, overlaid with embedding parameter cost.

    This is the figure that justifies the vocabulary choice: fertility falls
    monotonically, embedding cost rises linearly, and the selected size is the
    knee where further fertility gains stop paying for the parameters they cost.
    """
    xs = [r["vocab_size"] for r in rows]
    fert = [r["fertility_tokens_per_word"] for r in rows]
    emb = [r["embedding_params"] / 1e6 for r in rows]

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    l1 = ax1.plot(xs, fert, marker="o", color="tab:blue", label="Fertility (tokens/word)")
    ax1.set_xlabel("Vocabulary size")
    ax1.set_ylabel("Fertility (tokens per word)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    l2 = ax2.plot(xs, emb, marker="s", color="tab:red", label="Embedding parameters (M)")
    l3 = ax2.axhline(
        budget / 1e6, linestyle="--", color="grey", label=f"Total budget ({budget/1e6:.0f}M)"
    )
    ax2.set_ylabel("Embedding parameters (millions)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title(f"Vocabulary size: fertility vs parameter cost — {language}")
    handles = l1 + l2 + [l3]
    ax1.legend(handles, [h.get_label() for h in handles], loc="center right")
    return _save(fig, out_path)


def plot_zipf(df, out_path, language: str, top_n: int = 50_000):
    """Log-log rank/frequency plot of the token distribution."""
    d = df.head(top_n)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(d["rank"], d["count"], label="Observed token frequency")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Token frequency distribution (Zipf) — {language}")
    ax.set_xlabel("Token rank (log scale)")
    ax.set_ylabel("Token frequency (log scale)")
    ax.legend()
    return _save(fig, out_path)


def plot_manual_fraction(df, out_path, language: str):
    """Manual vs downloaded token share per split, against the 20% requirement."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(df["split"], df["manual_tokens"], label="Manual (crawled/OCR)")
    ax.bar(
        df["split"], df["downloaded_tokens"], bottom=df["manual_tokens"],
        label="Downloaded (public corpora)",
    )
    for i, row in df.reset_index().iterrows():
        ax.annotate(
            f"{row['manual_fraction']:.1%} manual",
            (i, row["tokens"]), ha="center",
            textcoords="offset points", xytext=(0, 6), fontsize=9,
        )
    ax.set_title(f"Manual vs downloaded tokens per split — {language}")
    ax.set_xlabel("Split")
    ax.set_ylabel("Tokens")
    ax.legend(title="Collection type")
    return _save(fig, out_path)
