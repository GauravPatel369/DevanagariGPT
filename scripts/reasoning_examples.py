"""Side-by-side answers from the pretrained, baseline and final models.

Phase 3 qualitative analysis.  The same test items are put to three checkpoints
of one language:

    pretrained  the model finetuning started from
    baseline    plain finetuning on the best sweep size (Hindi 20k, Nepali 30k)
    final       the best finetuned model after the ablation

For every item the file records the gold answer, each model's forced choice and
its greedy generation, so the report can quote successes and failures from the
final model and show which failures the ablation fixed.  test_b is used because
it holds out names, numbers and templates at once.

Usage:
    python scripts/reasoning_examples.py --lang hindi
    python scripts/reasoning_examples.py --lang nepali --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmagpt.eval_reasoning import generate_answer, load, score_candidates  # noqa: E402
import sentencepiece as spm  # noqa: E402

PRETRAINED = {"hindi": ROOT / "hindi" / "checkpoints" / "two_epoch" / "last.pt",
              "nepali": ROOT / "nepali" / "checkpoints" / "last.pt"}
TOKENIZER = {"hindi": "hi_unigram_10000.model", "nepali": "ne_unigram_10000.model"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--split", default="test_b")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    reasoning = ROOT / a.lang / "reasoning"
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / a.lang / "tokenizer" / TOKENIZER[a.lang]))
    rows = [json.loads(l) for l in (reasoning / "data" / f"{a.split}.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()][:a.n]

    models = {"pretrained": PRETRAINED[a.lang],
              "baseline": reasoning / "checkpoints" / "baseline" / "best.pt",
              "final": reasoning / "checkpoints" / "best.pt"}
    out = [{"text": r["text"], "category": r["category"], "hops": r["hops"],
            "gold": r["answer"], "candidates": r["candidates"]} for r in rows]
    for tag, path in models.items():
        model, ckpt = load(a.lang, path, device)
        for rec, r in zip(out, rows):
            rec[f"{tag}_choice"] = score_candidates(model, sp, r["prompt"], r["candidates"], device, dtype)
            rec[f"{tag}_generated"] = generate_answer(model, sp, r["prompt"], device, dtype).rstrip("।,.")
        gen_ok = sum(rec[f"{tag}_generated"] == rec["gold"] for rec in out)
        invalid = Counter(rec[f"{tag}_generated"] for rec in out
                          if rec[f"{tag}_generated"] not in rec["candidates"])
        print(f"  {tag:10s} {ckpt.get('run', '')} step {ckpt['step']}: generation {gen_ok}/{len(out)}, "
              f"most common outside names {invalid.most_common(4)}")
        del model
        torch.cuda.empty_cache()

    path = reasoning / f"examples_{a.split}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
