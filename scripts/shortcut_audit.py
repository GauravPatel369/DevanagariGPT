"""Audit every question type for position shortcuts, in the data and in the models.

``scripts/shortcut_probe.py`` found one shortcut by hand: in relation questions
the answer is always the first name in the question. This script checks all nine
question types systematically.

Step 1, the data. For each type it measures how often a simple rule that needs
no reasoning picks the right answer:

    first_named     the first person (or object) mentioned in the facts
    last_named      the last one mentioned in the facts
    question_first  the first name inside the question itself
    most_first      "most" questions -> first named, "least" -> last named
    always_equal    always answer the equality word

A rule that is right far more often than chance is a shortcut the model could
learn instead of reasoning.

Step 2, the models. For each type, test_b is split by whether that type's best
rule gives the right answer. Each model's choice accuracy is measured on both
halves. A model that reasons scores about the same on both; a model that follows
the rule scores high where the rule is right and low where it is wrong.

Usage:
    python scripts/shortcut_audit.py --lang hindi
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))

from lmagpt.eval_reasoning import load, score_candidates  # noqa: E402
from shortcut_rules import RULES, rule_accuracy, rule_answers  # noqa: E402
import sentencepiece as spm  # noqa: E402

TOKENIZER = {"hindi": "hi_unigram_10000.model", "nepali": "ne_unigram_10000.model"}
PRETRAINED = {"hindi": ROOT / "hindi" / "checkpoints" / "two_epoch" / "last.pt",
              "nepali": ROOT / "nepali" / "checkpoints" / "last.pt"}


def templates(lang: str):
    spec = importlib.util.spec_from_file_location(f"{lang}_t", ROOT / lang / "reasoning" / "templates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    t = templates(a.lang)
    data = ROOT / a.lang / "reasoning" / "data"
    read = lambda s: [json.loads(l) for l in (data / f"{s}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    train, test_b = read("train"), read("test_b")

    train_rules = rule_accuracy(train, a.lang, t)
    best = {cat: max(RULES, key=lambda r: v[r]) for cat, v in train_rules.items()}
    print(f"\n{a.lang}: how often a no-reasoning rule is right in the training data")
    print(f"  {'type':24s} {'chance':>7s} " + " ".join(f"{r:>15s}" for r in RULES))
    for cat, v in train_rules.items():
        print(f"  {cat:24s} {v['chance']:>7.0%} " + " ".join(f"{v[r]:>15.0%}" for r in RULES))

    # Step 2: split test_b by whether each type's best rule is right.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / a.lang / "tokenizer" / TOKENIZER[a.lang]))
    models = {"pretrained": PRETRAINED[a.lang],
              "plain": ROOT / a.lang / "reasoning" / "checkpoints" / "baseline" / "best.pt",
              "final": ROOT / a.lang / "reasoning" / "checkpoints" / "best.pt"}
    groups = defaultdict(list)
    for r in test_b:
        ans = rule_answers(r, a.lang, t)[best[r["category"]]]
        groups[(r["category"], "rule_right" if ans == r["answer"] else "rule_wrong")].append(r)

    split_acc = defaultdict(dict)
    for tag, path in models.items():
        model, _ = load(a.lang, path, device)
        for (cat, half), rows in groups.items():
            ok = sum(score_candidates(model, sp, r["prompt"], r["candidates"], device, dtype) == r["answer"]
                     for r in rows)
            split_acc[cat].setdefault(tag, {})[half] = round(ok / len(rows), 3)
        del model
        torch.cuda.empty_cache()

    print(f"\n{a.lang} test_b, choice accuracy where the best rule is right / wrong")
    print(f"  {'type':24s} {'best rule':15s} {'right:wrong n':>14s} " + " ".join(f"{m:>18s}" for m in models))
    for cat in sorted(split_acc):
        n_r, n_w = len(groups[(cat, "rule_right")]), len(groups[(cat, "rule_wrong")])
        cells = []
        for m in models:
            v = split_acc[cat][m]
            cells.append(f"{v.get('rule_right', float('nan')):>7.0%} / {v.get('rule_wrong', float('nan')):>6.0%}")
        print(f"  {cat:24s} {best[cat]:15s} {n_r:>6d}:{n_w:<7d} " + " ".join(f"{c:>18s}" for c in cells))

    out = {"language": a.lang, "rules_train": train_rules, "rules_test_b": rule_accuracy(test_b, a.lang, t),
           "best_rule": best,
           "group_sizes": {f"{c}|{h}": len(v) for (c, h), v in groups.items()},
           "choice_accuracy_by_rule": split_acc,
           "models": {k: str(v.relative_to(ROOT)) for k, v in models.items()}}
    path = ROOT / a.lang / "reasoning" / "shortcut_audit.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
