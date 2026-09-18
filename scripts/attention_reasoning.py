"""Pretrained vs finetuned attention on comparative-reasoning prompts.

Phase 3, Section 3.2.  Reuses the Phase-2 toolkit in ``lmagpt/attention.py``
(``collect_attention``, ``attention_entropy``, ``plot_heatmaps``) but runs it on
the reasoning prompts the brief asks about, not on news text.

Beyond heatmaps and entropy, it measures the one thing the evaluation implicated.
The finetuned model usually generates a name that is *not in the question* --
often one memorised from the training pool.  If that is a reading failure, it
should be visible in attention: at the answer position, the model should put
little weight on the names actually present.  So for every layer this reports
how the last position's attention splits between

    answer      tokens of the correct entity
    distractor  tokens of the other entities in the question
    other       everything else (verbs, numbers, the question, punctuation)

and whether finetuning moved weight onto the names.

The finetuned model is each language's best model: the sweep size that
generalised best (Hindi 20k, Nepali 30k), retrained by the ablation kernel's
``baseline`` run.  The pretrained reference is the checkpoint that run started
from.  For Hindi that is the two-epoch model: the kernel mounted
``lma-train-hindi``, whose latest output is the two-epoch continuation.

Usage:
    python scripts/attention_reasoning.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmagpt.attention import attention_entropy, collect_attention, plot_heatmaps  # noqa: E402
from lmagpt.eval_reasoning import load  # noqa: E402
import sentencepiece as spm  # noqa: E402

MODELS = {
    "hindi": {
        "pretrained": ROOT / "hindi" / "checkpoints" / "two_epoch" / "last.pt",
        "finetuned": ROOT / "hindi" / "reasoning" / "checkpoints" / "best.pt",
        "tokenizer": ROOT / "hindi" / "tokenizer" / "hi_unigram_10000.model",
    },
    "nepali": {
        "pretrained": ROOT / "nepali" / "checkpoints" / "last.pt",
        "finetuned": ROOT / "nepali" / "reasoning" / "checkpoints" / "best.pt",
        "tokenizer": ROOT / "nepali" / "tokenizer" / "ne_unigram_10000.model",
    },
}
N_PROMPTS = 300
RELATION = ("chain2_relation", "chain3_relation")
OUT = ROOT / "report" / "Phase 3"


def name_positions(ids: list[int], sp, name: str) -> list[int]:
    """Token positions covering every occurrence of ``name`` in the prompt.

    Matches the name's own token sequence as a contiguous run.  SentencePiece
    marks word-initial pieces with a leading space symbol, and entity names in
    these prompts are always word-initial, so encoding the bare name gives the
    same pieces the prompt contains.
    """
    pattern = sp.encode(name)
    if not pattern:
        return []
    hits = []
    for i in range(len(ids) - len(pattern) + 1):
        if ids[i:i + len(pattern)] == pattern:
            hits.extend(range(i, i + len(pattern)))
    return hits


def answer_attention_split(weights: np.ndarray, answer_pos: list[int],
                           distractor_pos: list[int]) -> np.ndarray:
    """Per layer: attention mass on answer / distractor / other, and selectivity.

    Averaged over heads.  The last position is the one that emits the answer,
    so its attention is what decides which entity gets named.

    Raw mass is biased: a prompt has more distractor tokens than answer tokens,
    so distractors collect more mass even under uniform attention. Selectivity
    removes that: attention per answer token divided by attention per distractor
    token. 1.0 means the model does not prefer the correct name; above 1.0 it
    does. The two per-token rates are returned here and divided only after
    averaging over prompts: a per-prompt ratio explodes on the few prompts where
    distractor attention is near zero.
    """
    n_layer = weights.shape[0]
    last = weights[:, :, -1, :].mean(axis=1)        # (n_layer, T)
    out = np.zeros((n_layer, 5))
    for l in range(n_layer):
        a = last[l, answer_pos].sum() if answer_pos else 0.0
        d = last[l, distractor_pos].sum() if distractor_pos else 0.0
        a_tok = a / len(answer_pos)
        d_tok = d / len(distractor_pos) if distractor_pos else np.nan
        out[l] = (a, d, max(0.0, 1.0 - a - d), a_tok, d_tok)
    return out


def analyse_model(lang: str, tag: str, rows: list[dict], sp, device) -> dict:
    """Entropy and the answer-attention split over reasoning prompts."""
    model, ckpt = load(lang, MODELS[lang][tag], device)
    splits, ents, fair = [], [], []
    for r in rows:
        ids = sp.encode(r["prompt"])
        if len(ids) > model.cfg.context:
            continue
        others = [c for c in r["candidates"] if c != r["answer"]]
        a_pos = name_positions(ids, sp, r["answer"])
        d_pos = sorted({p for c in others for p in name_positions(ids, sp, c)})
        if not a_pos:
            continue          # equality items have no entity to point at
        w = collect_attention(model, ids, device)
        splits.append(answer_attention_split(w, a_pos, d_pos))
        ents.append(attention_entropy(w, min_query=4).mean(axis=1))
        # Relation questions always name the answer first (see
        # scripts/shortcut_probe.py), so a preference for the answer there can
        # be a preference for the first-named entity. Kept apart.
        fair.append(r["category"] not in RELATION)
    split = np.nanmean(splits, axis=0)
    no_rel = np.nanmean([s for s, f in zip(splits, fair) if f], axis=0)
    ent = np.mean(ents, axis=0)
    print(f"  {lang} {tag:10s} step {ckpt['step']:>6,}  prompts used {len(splits)}")
    del model
    torch.cuda.empty_cache()
    return {"step": ckpt["step"], "n_prompts": len(splits),
            "answer_mass": split[:, 0].round(4).tolist(),
            "distractor_mass": split[:, 1].round(4).tolist(),
            "other_mass": split[:, 2].round(4).tolist(),
            "selectivity": (split[:, 3] / split[:, 4]).round(3).tolist(),
            "n_prompts_no_relation": int(sum(fair)),
            "selectivity_no_relation": (no_rel[:, 3] / no_rel[:, 4]).round(3).tolist(),
            "entropy_bits": ent.round(3).tolist()}


def heatmaps(lang: str, prompt_row: dict, sp, device) -> None:
    """Early and late layer heatmaps on one 2-hop prompt, both checkpoints."""
    ids = sp.encode(prompt_row["prompt"])
    pieces = sp.encode(prompt_row["prompt"], out_type=str)
    for tag in ("pretrained", "finetuned"):
        model, _ = load(lang, MODELS[lang][tag], device)
        w = collect_attention(model, ids, device)
        path = OUT / "figures" / f"{lang}_reasoning_heatmap_{tag}.png"
        plot_heatmaps(w, pieces, layers=[0, w.shape[0] - 1], heads=[0, 1, 2, 3],
                      out_path=path, language=f"{lang.title()} ({tag})",
                      sentence=prompt_row["text"])
        print(f"  wrote {path.name}")
        del model
        torch.cuda.empty_cache()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    results = {}

    for lang, paths in MODELS.items():
        sp = spm.SentencePieceProcessor(model_file=str(paths["tokenizer"]))
        rows = [json.loads(l) for l in
                (ROOT / lang / "reasoning" / "data" / "test_a.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()][:N_PROMPTS]
        print(f"\n=== {lang.upper()} ===")
        results[lang] = {tag: analyse_model(lang, tag, rows, sp, device)
                         for tag in ("pretrained", "finetuned")}

        # A 2-hop chain is where reading the entities matters most: the answer
        # is never stated, only derivable from the links between names. The
        # superlative form, because the relation form carries the answer-first
        # shortcut.
        example = next(r for r in rows if r["category"] == "chain2_superlative")
        heatmaps(lang, example, sp, device)
        results[lang]["heatmap_prompt"] = example["text"]

        p, f = results[lang]["pretrained"], results[lang]["finetuned"]
        print(f"  {'layer':>5} {'answer pre':>11} {'answer ft':>10} "
              f"{'distr pre':>10} {'distr ft':>9} {'select pre':>11} {'select ft':>10} "
              f"{'no-rel pre':>11} {'no-rel ft':>10} "
              f"{'entropy pre':>12} {'entropy ft':>11}")
        for L in range(len(p["answer_mass"])):
            print(f"  {L:>5} {p['answer_mass'][L]:>11.3f} {f['answer_mass'][L]:>10.3f} "
                  f"{p['distractor_mass'][L]:>10.3f} {f['distractor_mass'][L]:>9.3f} "
                  f"{p['selectivity'][L]:>11.2f} {f['selectivity'][L]:>10.2f} "
                  f"{p['selectivity_no_relation'][L]:>11.2f} {f['selectivity_no_relation'][L]:>10.2f} "
                  f"{p['entropy_bits'][L]:>12.2f} {f['entropy_bits'][L]:>11.2f}")

    out = OUT / "attention_reasoning.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
