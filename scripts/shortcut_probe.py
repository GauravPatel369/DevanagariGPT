"""Two order probes: does the answer change when only the surface order changes?

A model that reasons from the facts gives the same answer however the question
is laid out. A model leaning on position gives a different one. Two edits
change the layout while leaving the facts, and so the correct answer, the same:

    name swap      relation questions: swap the two names in the question
                   ("A और C में कौन लंबा है?" -> "C और A में कौन लंबा है?")
    fact reversal  every chain question: write the facts in reverse order

For each probe and model this reports accuracy on the original and the edited
questions, and consistency: how often the model gives the same answer to both.

This probe is how the shortcut in the first dataset was found: there, finetuned
models answered relation questions correctly 100% of the time and 0% after the
swap. On the regenerated data the two scores should match.

Usage:
    python scripts/shortcut_probe.py --lang hindi
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmagpt.eval_reasoning import generate_answer, load, score_candidates  # noqa: E402
import sentencepiece as spm  # noqa: E402

PRETRAINED = {"hindi": ROOT / "hindi" / "checkpoints" / "two_epoch" / "last.pt",
              "nepali": ROOT / "nepali" / "checkpoints" / "last.pt"}
TOKENIZER = {"hindi": "hi_unigram_10000.model", "nepali": "ne_unigram_10000.model"}
RELATION = ("chain2_relation", "chain3_relation")
CHAIN = RELATION + ("chain2_superlative", "chain3_superlative")


def templates(lang: str):
    spec = importlib.util.spec_from_file_location(f"{lang}_t", ROOT / lang / "reasoning" / "templates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def swap_names(row: dict, lang: str) -> dict | None:
    """The same item with the question's two names in the other order."""
    p, q = row["candidates"]
    if lang == "hindi":
        pairs = [(f"{p} और {q} ", f"{q} और {p} ")]
    else:
        # Nepali glues the genitive to the second name in one phrasing.
        pairs = [(f"{p} र {q}को ", f"{q} र {p}को "), (f"{p} र {q} ", f"{q} र {p} ")]
    for old, new in pairs:
        if row["text"].count(old) == 1:
            out = dict(row, text=row["text"].replace(old, new), prompt=row["prompt"].replace(old, new))
            out["candidates"] = [q, p]
            return out
    return None


def _split_facts(facts: str, join: str, and_word: str) -> list[str] | None:
    """Undo ``make_reasoning_data._join_facts`` for one join style."""
    style = join.removeprefix("short_")
    if style == "sentences":
        return facts.split("। ")
    if style == "semicolon":
        return facts.split("; ")
    last = f" {and_word} " if style == "and" else f", {and_word} "
    head, sep, tail = facts.rpartition(last)
    return head.split(", ") + [tail] if sep else None


def _join(parts: list[str], join: str, and_word: str) -> str:
    style = join.removeprefix("short_")
    if style == "sentences":
        return "। ".join(parts)
    if style == "semicolon":
        return "; ".join(parts)
    last = f" {and_word} " if style == "and" else f", {and_word} "
    return ", ".join(parts[:-1]) + last + parts[-1]


def reverse_facts(row: dict, t) -> dict | None:
    """The same chain question with its facts written in reverse order.

    The frame is recovered by matching each chain frame's fixed text around the
    facts; the number of facts is known from the hop count, which tells apart
    frames that share their text but join the facts differently.
    """
    n_facts = row["hops"]
    # Longest fixed prefix first: the plain "{facts}। {question}" frame has an
    # empty prefix and would otherwise swallow "हम जानते हैं कि ..." questions.
    frames = sorted(t.CHAIN_FRAMES, key=lambda f: -len(f["frame"].partition("{facts}")[0]))
    for frame in frames:
        before, _, rest = frame["frame"].partition("{facts}")
        middle, _, _ = rest.partition("{question}")
        text = row["text"]
        if not text.startswith(before) or middle not in text[len(before):]:
            continue
        # The last occurrence: facts written as sentences contain "। " themselves.
        cut = text.rindex(middle)
        if cut < len(before):
            continue
        facts, tail = text[len(before):cut], text[cut:]
        parts = _split_facts(facts, frame["join"], t.AND)
        if not parts or len(parts) != n_facts:
            continue
        new_text = before + _join(parts[::-1], frame["join"], t.AND) + tail
        if new_text == text:
            return None
        return dict(row, text=new_text, prompt=row["prompt"].replace(text, new_text))
    return None


def score(model, sp, rows, device, dtype):
    """Choice and first-word generation answers for each row."""
    out = []
    for r in rows:
        choice = score_candidates(model, sp, r["prompt"], r["candidates"], device, dtype)
        gen = generate_answer(model, sp, r["prompt"], device, dtype).rstrip("।,.")
        out.append((choice, gen))
    return out


def summarise(originals, edited, got_o, got_e) -> dict:
    n = len(originals)
    acc = lambda rows, got, k: sum(g[k] == r["answer"] for r, g in zip(rows, got)) / n
    return {
        "n": n,
        "choice_original": round(acc(originals, got_o, 0), 4),
        "choice_edited": round(acc(edited, got_e, 0), 4),
        "choice_consistent": round(sum(a[0] == b[0] for a, b in zip(got_o, got_e)) / n, 4),
        "generation_original": round(acc(originals, got_o, 1), 4),
        "generation_edited": round(acc(edited, got_e, 1), 4),
        "generation_consistent": round(sum(a[1] == b[1] for a, b in zip(got_o, got_e)) / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--per-split", type=int, default=300)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    t = templates(a.lang)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    reasoning = ROOT / a.lang / "reasoning"
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / a.lang / "tokenizer" / TOKENIZER[a.lang]))

    probes = {"name_swap": ([], []), "fact_reversal": ([], [])}
    for split in ("test_a", "test_b"):
        rows = [json.loads(l) for l in (reasoning / "data" / f"{split}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        rel = [r for r in rows if r["category"] in RELATION][:a.per_split]
        for r in rel:
            s = swap_names(r, a.lang)
            if s is None:
                raise SystemExit(f"could not swap names: {r['text']}")
            probes["name_swap"][0].append(r)
            probes["name_swap"][1].append(s)
        chain = [r for r in rows if r["category"] in CHAIN][:a.per_split]
        for r in chain:
            s = reverse_facts(r, t)
            if s is not None:
                probes["fact_reversal"][0].append(r)
                probes["fact_reversal"][1].append(s)
    for name, (o, _) in probes.items():
        print(f"  {name}: {len(o)} question pairs")
    example = probes["fact_reversal"]
    print(f"  example reversal:\n    {example[0][0]['text']}\n    {example[1][0]['text']}")

    models = {"pretrained": PRETRAINED[a.lang],
              "plain": reasoning / "checkpoints" / "baseline" / "best.pt",
              "final": reasoning / "checkpoints" / "best.pt"}
    results = {"models": {}}
    for tag, path in models.items():
        if not path.exists():
            print(f"  {tag}: {path} missing, skipped")
            continue
        model, ckpt = load(a.lang, path, device)
        rec = {"checkpoint": str(path.relative_to(ROOT)), "step": ckpt["step"]}
        for name, (orig, edited) in probes.items():
            rec[name] = summarise(orig, edited, score(model, sp, orig, device, dtype),
                                  score(model, sp, edited, device, dtype))
            v = rec[name]
            print(f"  {tag:10s} {name:13s} choice {v['choice_original']:.0%} -> {v['choice_edited']:.0%}"
                  f" (same answer {v['choice_consistent']:.0%})   generation {v['generation_original']:.0%}"
                  f" -> {v['generation_edited']:.0%} (same answer {v['generation_consistent']:.0%})")
        results["models"][tag] = rec
        del model
        torch.cuda.empty_cache()

    path = reasoning / "shortcut_probe.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
