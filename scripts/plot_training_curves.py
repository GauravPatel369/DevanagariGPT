"""Plot Phase-2 training curves from the JSONL logs.

Reads ``report/Phase 2/logs/<lang>_train_log.jsonl`` -- written a line at a time
during training, so the plots come from the run itself rather than from numbers
transcribed afterwards.

Every axis carries a title, labels and a legend: the brief states that plots
missing them may score zero for that component.

Usage:
    python scripts/plot_training_curves.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "report" / "Phase 2" / "logs"
FIG_DIR = ROOT / "report" / "Phase 2" / "figures"
COLOURS = {"hindi": "tab:blue", "nepali": "tab:orange"}


def load(lang: str):
    """Split the log into training rows and evaluation rows."""
    rows = [json.loads(l) for l in
            (LOG_DIR / f"{lang}_train_log.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return ([r for r in rows if "loss" in r],
            [r for r in rows if "val_loss" in r])


def plot_loss_curves(data, out_path):
    """Training and validation loss for both models on one axis."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for lang, (train, val) in data.items():
        c = COLOURS[lang]
        ax.plot([r["step"] for r in train], [r["loss"] for r in train],
                color=c, alpha=0.25, linewidth=0.8,
                label=f"{lang.capitalize()} — training (per step)")
        ax.plot([r["step"] for r in val], [r["val_loss"] for r in val],
                color=c, linewidth=2.2, marker="o", markersize=3,
                label=f"{lang.capitalize()} — validation")
    ax.set_title("Pretraining loss — Model H (Hindi) vs Model L (Nepali)")
    ax.set_xlabel("Training step (16,384 tokens per step)")
    ax.set_ylabel("Cross-entropy loss (nats per token)")
    ax.legend(title="Curve", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_perplexity(data, out_path):
    """Validation perplexity on a log axis.

    Log scale because perplexity falls from ~10,000 at initialisation to ~25;
    on a linear axis the entire useful range is squashed against zero.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for lang, (_, val) in data.items():
        ax.plot([r["step"] for r in val], [r["val_ppl"] for r in val],
                color=COLOURS[lang], linewidth=2, marker="o", markersize=3,
                label=f"{lang.capitalize()} (final {val[-1]['val_ppl']:.1f})")
    ax.set_yscale("log")
    ax.set_title("Validation perplexity during pretraining")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Perplexity (log scale)")
    ax.legend(title="Model")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_schedule_and_throughput(data, out_path):
    """Learning rate and tokens/sec, the two run-health signals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for lang, (train, _) in data.items():
        c = COLOURS[lang]
        ax1.plot([r["step"] for r in train], [r["lr"] for r in train],
                 color=c, linewidth=2, label=lang.capitalize())
        ax2.plot([r["step"] for r in train], [r.get("tokens_per_sec", 0) for r in train],
                 color=c, alpha=0.6, linewidth=1, label=lang.capitalize())

    ax1.set_title("Learning-rate schedule (warmup then cosine decay)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Learning rate")
    ax1.legend(title="Model")
    ax1.grid(alpha=0.3)

    ax2.set_title("Training throughput on Kaggle P100")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Tokens per second")
    ax2.legend(title="Model")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_grad_norm(data, out_path):
    """Gradient norm against the clipping threshold.

    Flat and below the clip line means the run never fought the clipper, which
    is the evidence that training was stable rather than merely finishing.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for lang, (train, _) in data.items():
        ax.plot([r["step"] for r in train], [r.get("grad_norm", 0) for r in train],
                color=COLOURS[lang], alpha=0.7, linewidth=0.9, label=lang.capitalize())
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label="Clipping threshold (1.0)")
    ax.set_title("Gradient norm during pretraining")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Global gradient L2 norm")
    ax.legend(title="Model")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {lang: load(lang) for lang in ("hindi", "nepali")}

    written = [
        plot_loss_curves(data, FIG_DIR / "training_loss.png"),
        plot_perplexity(data, FIG_DIR / "validation_perplexity.png"),
        plot_schedule_and_throughput(data, FIG_DIR / "lr_and_throughput.png"),
        plot_grad_norm(data, FIG_DIR / "gradient_norm.png"),
    ]
    for lang, (train, val) in data.items():
        print(f"  {lang}: {len(train)} train points, {len(val)} eval points, "
              f"final val {val[-1]['val_loss']:.4f} (ppl {val[-1]['val_ppl']:.2f})")
    print(f"\n  {len(written)} figures -> {FIG_DIR}")
    for w in written:
        print("   ", Path(w).name)


if __name__ == "__main__":
    main()
