"""Written accuracy with and without Nepali case endings attached to the answer.

Nepali attaches endings to names: the number questions write "योगेशको उमेर"
(Yogesh's age), so the name never appears bare in those facts. A model that
copies the word from the question writes "योगेशको", which names the right
person but fails the strict first-word match. This script measures how much of
the written-accuracy gap that explains.

For each model the first generated word on test_b is scored two ways:

    strict   the word equals the answer
    ending   the word equals the answer, or the answer plus one of the endings
             को, ले, भन्दा, सँग, लाई, मा

Hindi writes these relations as separate words (का, से), so the two scores
should match there; it is run for comparison.

Usage:
    python scripts/score_with_endings.py --lang nepali
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmagpt.eval_reasoning import generate_answer, load  # noqa: E402
import sentencepiece as spm  # noqa: E402

PRETRAINED = {"hindi": ROOT / "hindi" / "checkpoints" / "two_epoch" / "last.pt",
              "nepali": ROOT / "nepali" / "checkpoints" / "last.pt"}
TOKENIZER = {"hindi": "hi_unigram_10000.model", "nepali": "ne_unigram_10000.model"}
ENDINGS = ("भन्दा", "सँग", "लाई", "को", "ले", "मा")


def strip_ending(word: str) -> str:
    for e in ENDINGS:
        if word.endswith(e) and len(word) > len(e):
            return word[: -len(e)]
    return word


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--limit", type=int, default=600)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    reasoning = ROOT / a.lang / "reasoning"
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / a.lang / "tokenizer" / TOKENIZER[a.lang]))
    rows = [json.loads(l) for l in (reasoning / "data" / "test_b.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()][:a.limit]

    models = {"pretrained": PRETRAINED[a.lang],
              "plain": reasoning / "checkpoints" / "baseline" / "best.pt",
              "final": reasoning / "checkpoints" / "best.pt"}
    out = {"language": a.lang, "split": "test_b", "n": len(rows), "endings": list(ENDINGS), "models": {}}
    for tag, path in models.items():
        model, ckpt = load(a.lang, path, device)
        strict = ending = named = 0
        for r in rows:
            word = generate_answer(model, sp, r["prompt"], device, dtype).rstrip("।,.")
            strict += word == r["answer"]
            ending += strip_ending(word) == r["answer"] or word == r["answer"]
            named += strip_ending(word) in r["candidates"] or word in r["candidates"]
        n = len(rows)
        out["models"][tag] = {"step": ckpt["step"], "strict": round(strict / n, 4),
                              "with_ending": round(ending / n, 4),
                              "names_a_candidate_with_ending": round(named / n, 4)}
        print(f"  {tag:10s} strict {strict / n:.1%}  allowing an ending {ending / n:.1%}  "
              f"answer is one of the question's names (ending allowed) {named / n:.1%}", flush=True)
        del model
        torch.cuda.empty_cache()

    path = reasoning / "written_with_endings.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
