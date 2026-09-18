"""Figures for the Phase 3 ablation.

    ablation_summary.png    generation accuracy per run on the three unseen test
                            sets, and the language-modelling cost of each run
    validation_curves.png   in-distribution vs out-of-distribution validation loss,
                            without and with replay

Numbers come from ``<lang>/reasoning/ablation/ablation_results.json`` and, for
the pretrained model, ``<lang>/reasoning/sweep_results.json``. The final recipe
is read from the ``run`` field of ``<lang>/reasoning/checkpoints/best.pt``.

Usage:
    python scripts/plot_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "Phase 3" / "figures"

# First three slots of the reference categorical palette: validated all-pairs
# for colour-vision deficiency. Aqua is below 3:1 contrast on white, so every
# point also carries a direct value label.
SERIES = {"test_a": "#2a78d6", "test_b": "#eb6834", "test_c": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

RUNS = [("baseline", "plain finetuning"), ("ood_val", "new-name validation"),
        ("ood_val_2ep", "new-name validation, 2 passes"), ("name_pool", "name pool"),
        ("replay", "replay"), ("name_pool_replay", "name pool + replay")]


def final_run(lang: str) -> str | None:
    """The recipe of the installed final model, as recorded in its checkpoint."""
    import torch
    path = ROOT / lang / "reasoning" / "checkpoints" / "best.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False).get("run")


def load(lang: str):
    base = ROOT / lang / "reasoning" / "ablation"
    runs = json.loads((base / "ablation_results.json").read_text(encoding="utf-8"))["runs"]
    runs = {k: v for k, v in runs.items() if "error" not in v}
    sweep = json.loads((ROOT / lang / "reasoning" / "sweep_results.json").read_text(encoding="utf-8"))
    pre = next(m for m in sweep["models"] if m["label"] == "pretrained")["language"]["perplexity"]
    return runs, pre


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def summary() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), gridspec_kw={"width_ratios": [2.3, 1]})
    for row, lang in enumerate(("hindi", "nepali")):
        runs, pre_ppl = load(lang)
        final = final_run(lang)
        names = [(k, lbl) for k, lbl in RUNS if k in runs]
        ys = list(range(len(names)))[::-1]

        ax = axes[row][0]
        style(ax)
        for y, (k, lbl) in zip(ys, names):
            vals = {s: runs[k]["test_sets"][s]["exact_match"] * 100 for s in SERIES}
            ax.plot([min(vals.values()), max(vals.values())], [y, y], color=GRID, linewidth=2, zorder=1)
            for s, v in vals.items():
                ax.scatter(v, y, s=64, color=SERIES[s], edgecolor="white", linewidth=2, zorder=3)
            ax.text(max(vals.values()) + 1.2, y, f"{sum(vals.values()) / 3:.1f}",
                    va="center", fontsize=8.5, color=MUTED)
        ax.set_yticks(ys)
        ax.set_yticklabels([lbl + ("  (final)" if k == final else "") for k, lbl in names],
                           fontsize=9.5, color=INK)
        top = max(runs[k]["test_sets"][s]["exact_match"] * 100 for k, _ in names for s in SERIES)
        ax.set_xlim(0, max(50, top * 1.25))
        ax.set_title(f"{lang.title()}: generation accuracy on unseen tests (%)",
                     loc="left", fontsize=11, color=INK)
        if row == 1:
            ax.set_xlabel("first-word exact match, 600 questions per test set; number = mean of a, b, c",
                          fontsize=8.5, color=MUTED)

        ax = axes[row][1]
        style(ax)
        rise = [(runs[k]["language"]["perplexity"] / pre_ppl - 1) * 100 for k, _ in names]
        ax.barh(ys, rise, height=0.55, color="#2a78d6", edgecolor="white", linewidth=2)
        for y, r in zip(ys, rise):
            ax.text(r + 0.8, y, f"+{r:.0f}%", va="center", fontsize=8.5, color=MUTED)
        ax.set_yticks(ys)
        ax.set_yticklabels([])
        ax.set_xlim(0, max(45, max(rise) * 1.25))
        ax.set_title(f"PPL rise (pretrained {pre_ppl:.1f})", loc="left", fontsize=11, color=INK)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c,
                          markeredgecolor="white", label=s.replace("_", " ")) for s, c in SERIES.items()]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "ablation_summary.png", dpi=160)
    plt.close(fig)


def curves() -> None:
    colours = {"baseline": "#2a78d6", "replay": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, lang in zip(axes, ("hindi", "nepali")):
        runs, _ = load(lang)
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for run, c in colours.items():
            pts = runs[run]["curve"]
            steps = [p["step"] for p in pts]
            ax.plot(steps, [p["val_ood"] for p in pts], color=c, linewidth=2)
            ax.plot(steps, [p["val"] for p in pts], color=c, linewidth=2, linestyle=(0, (4, 3)))
            ax.text(steps[-1] + 12, pts[-1]["val_ood"], f"{'plain' if run == 'baseline' else run}, new names", va="center", fontsize=8.5, color=INK)
        # The two in-distribution curves coincide, so they share one label.
        ax.text(steps[-1] + 12, pts[-1]["val"], "both, normal", va="center",
                fontsize=8.5, color=MUTED)
        ax.set_xlim(0, steps[-1] * 1.32)
        ax.set_title(f"{lang.title()} ({runs['baseline']['n_samples'] // 1000}k examples)",
                     loc="left", fontsize=11, color=INK)
        ax.set_xlabel("finetuning step", fontsize=9, color=MUTED)
    axes[0].set_ylabel("answer-token loss", fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], color=INK, linewidth=2, label="validation with new names"),
               plt.Line2D([], [], color=INK, linewidth=2, linestyle=(0, (4, 3)),
                          label="normal validation")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "validation_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    summary()
    curves()
    print(f"wrote {OUT / 'ablation_summary.png'}\nwrote {OUT / 'validation_curves.png'}")


if __name__ == "__main__":
    main()
