"""Phase-2 attention analysis: heatmaps, entropy and mean attention distance.

Because the attention block is implemented from primitives rather than through a
fused kernel, the post-softmax weight matrix is available directly -- which is
what makes this analysis possible at all.  ``F.scaled_dot_product_attention``
never materialises it.

Three measurements, per head and layer:

* **Heatmaps** -- the raw ``(query, key)`` matrix for a short sentence.  Short
  matters: a 512x512 grid is unreadable, so the examples are ~25 tokens, where
  individual attention patterns are actually visible.
* **Entropy** -- ``-sum p log p`` over each query's distribution, in bits.  A
  head attending to one position scores ~0; a head spreading uniformly over t
  positions scores log2(t).  Low entropy means a sharp, selective head.
* **Mean attention distance** -- ``sum_k p(k) * (q - k)``, averaged over
  queries.  Small values mean a head looks just behind itself (local /
  positional); large values mean it reaches back across the sequence
  (content-based, plausibly syntactic or topical).

Both summary statistics are normalised for causality: query t can only attend to
t+1 positions, so early queries are mechanically low-entropy and short-distance.
Positions before ``min_query`` are excluded from the averages so the numbers
describe the head rather than the triangular mask.

Computation and plotting are kept in separate functions, per the project's code
guidelines.

Usage:
    python -m lmagpt.attention --lang both
    python -m lmagpt.attention --lang hindi --layers 0 6 --heads 0 1 2 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lmagpt.data import TokenDataset, resolve_data_dir, resolve_tokenizer  # noqa: E402
from lmagpt.model import GPT, GPTConfig  # noqa: E402

# Short, self-contained sentences so the heatmap axes stay legible.
EXAMPLES = {
    "hindi": [
        "भारतीय रिज़र्व बैंक ने डिजिटल मुद्रा प्रणाली की घोषणा की।",
        "पर्यावरणविदों ने प्रदूषण नियंत्रण के लिए कड़े कदम उठाने की मांग की।",
    ],
    "nepali": [
        "नेपालको संविधानले सबै नागरिकलाई समान अधिकार दिएको छ।",
        "विज्ञान र प्रविधिको क्षेत्रमा छिटो प्रगति भइरहेको छ।",
    ],
}


# ------------------------------------------------------------- computation


@torch.no_grad()
def collect_attention(model, token_ids, device):
    """Run one sequence and return every layer's attention weights.

    Returns:
        Array of shape ``(n_layer, n_head, T, T)``, post-softmax, on CPU.
    """
    model.eval()
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    _, _, attns = model(idx, need_weights=True)
    return torch.stack([a[0] for a in attns]).float().cpu().numpy()


def attention_entropy(weights: np.ndarray, min_query: int = 4) -> np.ndarray:
    """Mean entropy in bits, per layer and head.

    Query positions below ``min_query`` are skipped: position 0 can only attend
    to itself, so its entropy is 0 by construction and averaging it in would
    report the mask rather than the head's behaviour.
    """
    n_layer, n_head, T, _ = weights.shape
    out = np.zeros((n_layer, n_head))
    for l in range(n_layer):
        for h in range(n_head):
            ent = []
            for q in range(min_query, T):
                p = weights[l, h, q, : q + 1]
                p = p[p > 0]
                ent.append(float(-(p * np.log2(p)).sum()))
            out[l, h] = np.mean(ent) if ent else 0.0
    return out


def mean_attention_distance(weights: np.ndarray, min_query: int = 4) -> np.ndarray:
    """Mean distance ``q - k`` weighted by attention, per layer and head.

    A head that always attends to the token immediately before itself scores
    ~1; a head that spreads evenly over the whole prefix scores ~q/2.
    """
    n_layer, n_head, T, _ = weights.shape
    out = np.zeros((n_layer, n_head))
    for l in range(n_layer):
        for h in range(n_head):
            dists = []
            for q in range(min_query, T):
                p = weights[l, h, q, : q + 1]
                k = np.arange(q + 1)
                dists.append(float((p * (q - k)).sum()))
            out[l, h] = np.mean(dists) if dists else 0.0
    return out


def classify_heads(entropy: np.ndarray, distance: np.ndarray) -> list[dict]:
    """Label each head by where it sits in the (entropy, distance) plane.

    A crude taxonomy, but it makes the summary table readable and is defensible:
    sharpness and reach are the two things these statistics actually measure.
    """
    ent_med, dist_med = np.median(entropy), np.median(distance)
    rows = []
    for l in range(entropy.shape[0]):
        for h in range(entropy.shape[1]):
            e, d = entropy[l, h], distance[l, h]
            if e < ent_med and d < dist_med:
                kind = "local / positional"      # sharp and nearby
            elif e < ent_med:
                kind = "long-range selective"    # sharp but distant
            elif d < dist_med:
                kind = "diffuse local"
            else:
                kind = "diffuse global"
            rows.append({"layer": l, "head": h, "entropy_bits": round(float(e), 3),
                         "mean_distance": round(float(d), 2), "type": kind})
    return rows


# ---------------------------------------------------------------- plotting


# Fonts that actually carry Devanagari glyphs, most-preferred first. Nirmala UI
# ships with Windows, Noto with most Linux distributions, Kohinoor with macOS.
DEVANAGARI_FONTS = ["Nirmala UI", "Noto Sans Devanagari", "Mangal",
                    "Kohinoor Devanagari", "Arial Unicode MS", "FreeSerif"]


def use_devanagari_font() -> str | None:
    """Point matplotlib at a font with Devanagari coverage.

    Matplotlib's default (DejaVu Sans) has no Devanagari glyphs, so every
    character in a tick label renders as an empty box -- "tofu". The axis then
    looks like garbage rather than like text, which is worse than useless on a
    figure whose entire point is showing which *tokens* attend to which.

    Returns the font chosen, or None if the system has none -- in which case the
    caller should fall back to positional labels rather than emit tofu.
    """
    import matplotlib
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in DEVANAGARI_FONTS:
        if name in available:
            # Keep DejaVu behind it: the chosen font may lack glyphs that the
            # rest of the figure (axis numbers, minus signs) needs.
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None


def plot_heatmaps(weights, pieces, layers, heads, out_path, language, sentence):
    """Grid of (query x key) attention maps for the selected layers and heads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = use_devanagari_font()

    n_rows, n_cols = len(layers), len(heads)
    import textwrap

    # The full prompt, wrapped, sits above the grid; the extra height keeps it
    # clear of the first row's subplot titles.
    title = [f"Causal self-attention - {language}"] + textwrap.wrap(sentence, 95)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.1 * n_cols, 3.1 * n_rows + 0.3 * len(title)),
                             squeeze=False)
    T = weights.shape[-1]
    step = max(1, T // 12)
    ticks = list(range(0, T, step))
    # Index-prefixed labels ("3 राम") stay readable even where the script does
    # not render, and let a reader match the two axes without counting cells.
    # Without a Devanagari font, drop the token text: a row of empty boxes is
    # less informative than a plain position number.
    if font:
        labels = [f"{i} {pieces[i].replace(chr(0x2581), '_')}" for i in ticks]
    else:
        labels = [str(i) for i in ticks]

    im = None
    for r, layer in enumerate(layers):
        for c, head in enumerate(heads):
            ax = axes[r][c]
            im = ax.imshow(weights[layer, head], cmap="viridis", vmin=0.0,
                           aspect="auto", interpolation="nearest")
            ax.set_title(f"layer {layer}, head {head}", fontsize=9)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels, fontsize=5)
            if c == 0:
                ax.set_ylabel("Query position", fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("Key position", fontsize=8)

    fig.suptitle("\n".join(title), fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="Attention weight")
    # Rotated Devanagari tick labels are tall enough to run into the subplot
    # title of the row beneath, so the rows need explicit vertical separation.
    fig.subplots_adjust(hspace=0.75, top=0.90 - 0.03 * len(title))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight" here: it recomputes the layout and undoes the
    # hspace set above, putting the labels back under the row titles.
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def plot_head_summary(entropy, distance, out_path, language):
    """Per-layer/head entropy and mean distance, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    im1 = ax1.imshow(entropy, cmap="magma", aspect="auto")
    ax1.set_title(f"Attention entropy per head — {language}")
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Layer")
    ax1.set_xticks(range(entropy.shape[1]))
    ax1.set_yticks(range(entropy.shape[0]))
    fig.colorbar(im1, ax=ax1, label="Entropy (bits) — lower = sharper")

    im2 = ax2.imshow(distance, cmap="cividis", aspect="auto")
    ax2.set_title(f"Mean attention distance per head — {language}")
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Layer")
    ax2.set_xticks(range(distance.shape[1]))
    ax2.set_yticks(range(distance.shape[0]))
    fig.colorbar(im2, ax=ax2, label="Mean distance (tokens) — higher = longer range")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_distance_by_layer(per_lang, out_path):
    """How reach grows with depth, both models on one axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lang, dist in per_lang.items():
        ax.plot(range(dist.shape[0]), dist.mean(axis=1), marker="o", label=lang.capitalize())
    ax.set_title("Mean attention distance by layer — Model H vs Model L")
    ax.set_xlabel("Layer (0 = closest to the embeddings)")
    ax.set_ylabel("Mean attention distance (tokens)")
    ax.legend(title="Model")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


# -------------------------------------------------------------------- main


def analyse(lang: str, args, root: Path) -> dict:
    import sentencepiece as spm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint_dir:
        ckpt_path = Path(args.checkpoint_dir) / lang / "last.pt"
    else:
        ckpt_path = root / lang / "checkpoints" / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    sp = spm.SentencePieceProcessor(model_file=str(resolve_tokenizer(lang, root)))

    sentence = EXAMPLES[lang][0]
    ids = sp.encode(sentence, out_type=int)
    pieces = sp.encode(sentence, out_type=str)

    # Heatmaps and summary statistics need different inputs, and using one for
    # both is a measurement error rather than a shortcut. Mean attention
    # distance is bounded by sequence length -- query t can reach back at most t
    # tokens -- so a 13-token sentence caps the statistic at ~6 and reports the
    # sentence rather than the head. Statistics therefore run over a long
    # held-out passage, while the heatmap keeps the short sentence, where
    # individual cells are actually legible.
    weights_plot = collect_attention(model, ids, device)

    data_dir = resolve_data_dir(lang, root)
    ds = TokenDataset(data_dir / "val.bin", cfg.context)
    rng = np.random.default_rng(1337)
    passages = []
    for _ in range(args.stats_passages):
        off = int(rng.integers(0, len(ds) - args.stats_tokens - 1))
        passages.append(np.asarray(ds.tokens[off:off + args.stats_tokens]).astype(np.int64))

    ent_acc, dist_acc = [], []
    for p in passages:
        w = collect_attention(model, p.tolist(), device)
        ent_acc.append(attention_entropy(w))
        dist_acc.append(mean_attention_distance(w))
    entropy = np.mean(ent_acc, axis=0)
    distance = np.mean(dist_acc, axis=0)

    print(f"[attn] {lang}: step {ckpt['step']:,} | heatmap {len(ids)} tokens | "
          f"stats over {args.stats_passages} x {args.stats_tokens} tokens", flush=True)
    weights = weights_plot

    layers = args.layers or [0, cfg.n_layer // 2, cfg.n_layer - 1]
    heads = args.heads or list(range(min(4, cfg.n_head)))
    out_base = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    fig_dir = out_base / "figures"

    files = [
        plot_heatmaps(weights, pieces, layers, heads,
                      fig_dir / f"{lang}_attention_heatmaps.png", lang.capitalize(), sentence),
        plot_head_summary(entropy, distance,
                          fig_dir / f"{lang}_attention_summary.png", lang.capitalize()),
    ]
    print(f"[attn] {lang}: entropy {entropy.mean():.2f} bits mean, "
          f"distance {distance.mean():.2f} tokens mean", flush=True)

    return {
        "language": lang,
        "step": ckpt["step"],
        "sentence": sentence,
        "heatmap_tokens": len(ids),
        "stats_tokens": args.stats_tokens,
        "stats_passages": args.stats_passages,
        "layers_plotted": layers,
        "heads_plotted": heads,
        "entropy_bits": entropy.round(3).tolist(),
        "mean_distance": distance.round(2).tolist(),
        "entropy_mean": round(float(entropy.mean()), 3),
        "distance_mean": round(float(distance.mean()), 2),
        "entropy_by_layer": entropy.mean(axis=1).round(3).tolist(),
        "distance_by_layer": distance.mean(axis=1).round(2).tolist(),
        "heads": classify_heads(entropy, distance),
        "figures": files,
        "_distance_array": distance,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", default="both", choices=["hindi", "nepali", "both"])
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--heads", type=int, nargs="*", default=None)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="dir holding <lang>/last.pt; default <lang>/checkpoints/")
    ap.add_argument("--stats-tokens", type=int, default=256,
                    help="passage length for entropy/distance; longer = less mask-bound")
    ap.add_argument("--stats-passages", type=int, default=8,
                    help="held-out passages averaged for the statistics")
    ap.add_argument("--out", default="report/Phase 2")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parent.parent
    langs = ["hindi", "nepali"] if a.lang == "both" else [a.lang]

    # An absolute --out escapes the package root, which on Kaggle is a read-only
    # dataset mount. Resolved up front: the comparison figure below needs it too.
    out_dir = Path(a.out) if Path(a.out).is_absolute() else root / a.out

    results, dists = {}, {}
    for lang in langs:
        r = analyse(lang, a, root)
        dists[lang] = r.pop("_distance_array")
        results[lang] = r

    if len(dists) == 2:
        results["comparison_figure"] = plot_distance_by_layer(
            dists, out_dir / "figures" / "attention_distance_by_layer.png")

    out = out_dir / "attention_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[attn] wrote {out}")


if __name__ == "__main__":
    main()
