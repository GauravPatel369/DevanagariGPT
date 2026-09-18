"""Score every technique run on the unseen-name validation set, to choose the final recipe.

The final recipe must be chosen without looking at the test sets. Each run's
selected checkpoint (``best.pt``) is scored on the same 500 ``val_ood``
questions the Kaggle kernel uses for checkpoint selection: new names, new
numbers, never used for training or testing. The pretrained model is scored too,
as the reference.

The recipe with the highest written-answer (first-word) accuracy wins, since
answering correctly is what finetuning is for; choice accuracy is recorded
alongside.

Usage:
    python scripts/score_val_ood.py --lang hindi
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmagpt.eval_reasoning import evaluate, load  # noqa: E402
import sentencepiece as spm  # noqa: E402

PRETRAINED = {"hindi": ROOT / "hindi" / "checkpoints" / "two_epoch" / "last.pt",
              "nepali": ROOT / "nepali" / "checkpoints" / "last.pt"}
TOKENIZER = {"hindi": "hi_unigram_10000.model", "nepali": "ne_unigram_10000.model"}
N_ROWS = 500


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--runs-dir", default=None, help="default <lang>/reasoning/ablation/techniques")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    reasoning = ROOT / a.lang / "reasoning"
    runs_dir = Path(a.runs_dir) if a.runs_dir else reasoning / "ablation" / "techniques"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / a.lang / "tokenizer" / TOKENIZER[a.lang]))
    rows = [json.loads(l) for l in (reasoning / "data" / "val_ood.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()][:N_ROWS]

    checkpoints = {"pretrained": PRETRAINED[a.lang]}
    checkpoints.update({p.parent.name: p for p in sorted(runs_dir.glob("*/best.pt"))})
    scores = {}
    for tag, path in checkpoints.items():
        model, ckpt = load(a.lang, path, device)
        r = evaluate(model, sp, rows, device, dtype, None, with_generation=True)
        scores[tag] = {"checkpoint": str(path.relative_to(ROOT)), "step": ckpt.get("step"),
                       "choice": r["accuracy"], "written": r["exact_match"],
                       "wrong_name_rate": r["generation_invalid_rate"]}
        print(f"  {tag:18s} choice {r['accuracy']:.1%}  written {r['exact_match']:.1%}  "
              f"wrong names {r['generation_invalid_rate']:.1%}", flush=True)
        del model
        torch.cuda.empty_cache()

    runs = {k: v for k, v in scores.items() if k != "pretrained"}
    best = max(runs, key=lambda k: (runs[k]["written"], runs[k]["choice"]))
    out = {"language": a.lang, "n_questions": len(rows), "rule": "highest written accuracy on val_ood",
           "chosen": best, "scores": scores}
    path = runs_dir / "val_ood_scores.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  chosen recipe: {best}\n  wrote {path}")


if __name__ == "__main__":
    main()
