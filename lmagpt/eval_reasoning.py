"""Score a model on the synthetic comparative-reasoning test sets.

Phase 3.  Reports pretrained against finetuned accuracy, sliced by category, hop
count and answer position, over three test sets that differ in what they hold
out.

**Two metrics, because they answer different questions.**

``accuracy`` is forced choice: the model's total log-probability of each
candidate name is computed given the prompt, and the argmax is taken.  This is
what makes the pretrained baseline meaningful -- it has never seen the answer
format, so a generation-only metric would score it 0% for formatting reasons and
the before/after comparison would say nothing about reasoning.

``exact_match`` is the first word of greedy free generation.  It is the harder
and more honest question: not "can the model rank the right answer above the
wrong one?" but "does it actually produce it?".  Forced choice can hide complete
failure -- a model that would emit a repeated token, or an entity that never
appeared in the question, still gets scored as if it answered.

Report both.  The gap between them is itself a result: a large one means the
model knows which answer is better but cannot produce it.

**Every number is read against the majority-position baseline.**  Roughly 53% of
items have their answer in the same position, so a model that always names that
position scores 53% while reasoning about nothing.  Accuracy below or near that
line is not evidence of anything.

Usage:
    python -m lmagpt.eval_reasoning --lang hindi
    python -m lmagpt.eval_reasoning --lang hindi --checkpoint <path> --tag finetuned
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lmagpt.model import GPT, GPTConfig  # noqa: E402


@torch.no_grad()
def score_candidates(model, sp, prompt: str, candidates: list[str], device,
                     autocast_dtype) -> str:
    """Return the candidate with the highest total log-probability.

    Each candidate is appended to the prompt and scored by summing the log
    probabilities of its own tokens.  Length normalisation is deliberately not
    applied: the candidates are single names of comparable length, and dividing
    by token count would bias toward whichever name the tokenizer happens to
    split into more pieces.
    """
    prompt_ids = sp.encode(prompt)
    best, best_score = candidates[0], float("-inf")

    for cand in candidates:
        cand_ids = sp.encode(" " + cand)
        ids = prompt_ids + cand_ids
        if len(ids) > model.cfg.context:
            ids = ids[-model.cfg.context:]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            logits, _, _ = model(x)
        logprobs = torch.log_softmax(logits[0].float(), dim=-1)

        # Sum log P(token | everything before it) over the candidate's tokens.
        total = 0.0
        start = len(ids) - len(cand_ids) - 1
        for i, tok in enumerate(cand_ids):
            total += logprobs[start + i, tok].item()
        if total > best_score:
            best, best_score = cand, total
    return best


@torch.no_grad()
def generate_answer(model, sp, prompt: str, device, autocast_dtype, max_new: int = 8) -> str:
    """Greedily continue the prompt and return the first whitespace-delimited word.

    The unconstrained counterpart to ``score_candidates``.  Candidate scoring
    asks "does the model rank the right answer above the wrong one?"; this asks
    "does the model actually produce it?", which is a strictly harder question
    and the one a user of the model would care about.

    The two can disagree sharply.  A model can rank correctly while generating
    something that is not an answer at all -- repeated tokens, or an entity that
    never appeared in the question -- and candidate scoring hides that entirely.
    Greedy decoding is used so the number is deterministic.
    """
    ids = sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                        enabled=autocast_dtype is not None):
        out = model.generate(x, max_new_tokens=max_new, greedy=True)
    text = sp.decode(out[0, len(ids):].cpu().numpy().tolist()).strip()
    return text.split()[0] if text.split() else ""


def evaluate(model, sp, rows: list[dict], device, autocast_dtype, limit: int | None,
             with_generation: bool = True):
    """Accuracy per slice, by two metrics, plus the majority-position baseline.

    ``accuracy`` is forced-choice over the candidate set; ``exact_match`` is the
    first word of free generation.  Both are reported because they answer
    different questions and their gap is itself informative: a large one means
    the model knows which answer is better but cannot produce it.
    """
    if limit:
        rows = rows[:limit]
    correct = 0
    gen_correct = 0
    gen_invalid = 0          # generated a word that is not one of the candidates
    by_cat, by_hops, by_pos = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    by_hops_gen = defaultdict(lambda: [0, 0])
    wrong_examples = []

    for i, r in enumerate(rows):
        pred = score_candidates(model, sp, r["prompt"], r["candidates"], device, autocast_dtype)
        ok = pred == r["answer"]
        correct += ok
        for d, k in ((by_cat, r["category"]), (by_hops, r["hops"]),
                     (by_pos, r["answer_position"])):
            d[k][0] += ok
            d[k][1] += 1

        gen = ""
        if with_generation:
            gen = generate_answer(model, sp, r["prompt"], device, autocast_dtype)
            # Strip the sentence-final danda so "मोहन।" still counts as "मोहन".
            gen_clean = gen.rstrip("।,.")
            gen_ok = gen_clean == r["answer"]
            gen_correct += gen_ok
            by_hops_gen[r["hops"]][0] += gen_ok
            by_hops_gen[r["hops"]][1] += 1
            if gen_clean not in r["candidates"]:
                gen_invalid += 1

        if not ok and len(wrong_examples) < 10:
            wrong_examples.append({"text": r["text"], "gold": r["answer"],
                                   "predicted": pred, "generated": gen,
                                   "category": r["category"]})
        if (i + 1) % 200 == 0:
            msg = f"    {i + 1}/{len(rows)}  choice {correct / (i + 1):.3f}"
            if with_generation:
                msg += f"  generation {gen_correct / (i + 1):.3f}"
            print(msg, flush=True)

    pos_counts = Counter(r["answer_position"] for r in rows)
    out = {
        "n": len(rows),
        "accuracy": round(correct / max(len(rows), 1), 4),
        "majority_position_baseline": round(max(pos_counts.values()) / len(rows), 4),
        "by_category": {k: round(v[0] / v[1], 4) for k, v in sorted(by_cat.items())},
        "by_hops": {str(k): round(v[0] / v[1], 4) for k, v in sorted(by_hops.items())},
        "by_answer_position": {str(k): round(v[0] / v[1], 4) for k, v in sorted(by_pos.items())},
        "errors": wrong_examples,
    }
    if with_generation:
        out["exact_match"] = round(gen_correct / max(len(rows), 1), 4)
        # How often free generation produces something that is not even one of
        # the candidates: the failure mode forced-choice scoring cannot see.
        out["generation_invalid_rate"] = round(gen_invalid / max(len(rows), 1), 4)
        out["exact_match_by_hops"] = {str(k): round(v[0] / v[1], 4)
                                      for k, v in sorted(by_hops_gen.items())}
    return out


def load(lang: str, ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.cfg = cfg
    return model, ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--checkpoint", default=None,
                    help="omit to score the pretrained model")
    ap.add_argument("--tag", default=None, help="label in the output json")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap items per test set")
    ap.add_argument("--no-generation", action="store_true",
                    help="skip the free-generation metric; roughly 4x faster")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parent.parent
    ckpt_path = Path(a.checkpoint) if a.checkpoint else root / a.lang / "checkpoints" / "last.pt"
    tag = a.tag or ("finetuned" if a.checkpoint else "pretrained")
    data_dir = Path(a.data_dir) if a.data_dir else root / a.lang / "reasoning" / "data"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif device.type == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    model, ckpt = load(a.lang, ckpt_path, device)
    print(f"[eval] {a.lang} / {tag}: {ckpt_path.name} @ step {ckpt['step']:,}, "
          f"{model.num_parameters():,} params")

    import sentencepiece as spm
    tok = Path(a.tokenizer) if a.tokenizer else (
        root / a.lang / "tokenizer" /
        f"{'hi' if a.lang == 'hindi' else 'ne'}_unigram_{model.cfg.vocab_size}.model")
    sp = spm.SentencePieceProcessor(model_file=str(tok))

    results = {"language": a.lang, "tag": tag, "checkpoint": str(ckpt_path),
               "step": ckpt["step"], "test_sets": {}}
    # Three test sets, each isolating a different kind of generalisation.
    for split, what in (("test_a", "new names + numbers, seen templates"),
                        ("test_b", "new names + numbers + templates"),
                        ("test_c", "new names + templates, seen numbers")):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n[eval] {split} ({what}) -- {len(rows) if not a.limit else a.limit} items")
        res = evaluate(model, sp, rows, device, autocast_dtype, a.limit,
                       with_generation=not a.no_generation)
        res["description"] = what
        results["test_sets"][split] = res
        print(f"  forced-choice {res['accuracy']:.1%}  (majority baseline "
              f"{res['majority_position_baseline']:.1%})")
        if 'exact_match' in res:
            print(f"  generation    {res['exact_match']:.1%}  "
                  f"(invalid output {res['generation_invalid_rate']:.1%})")
        for k, v in res["by_hops"].items():
            print(f"    {k}-hop: {v:.1%}")

    out_dir = Path(a.out) if a.out else root / a.lang / "reasoning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
