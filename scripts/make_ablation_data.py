"""Build the two extra data files the Phase-3 ablation needs.

    val_ood.jsonl         out-of-distribution validation, for checkpoint selection
    train_namepool.jsonl  the training examples with names drawn from a large pool

Neither touches the existing splits: train, val and the four test sets stay
byte-identical, so every ablation run is scored on the same test data as the
sweep.

**val_ood** uses names that appear in no training or test example
(``VAL_OOD_NAMES``) and numbers 91-140, which sit outside both the training
range (5-40) and the test range (41-90).  Templates are the seen ones: the only
unseen templates are the test templates, and selecting a checkpoint on those
would leak the test set into model selection.  So this file detects overfitting
to the name and number pools, which is the failure the sweep exposed, but not
overfitting to phrasing.

**train_namepool** keeps every example exactly as it is -- template, numbers,
answer, row order -- and changes only the people in it.  Each row's names are
mapped, consistently within the row, to names sampled from the 26 training names
plus ``EXTRA_NAMES``.  Rewriting the existing rows rather than regenerating them
matters: regeneration would draw different templates and numbers too, and the
ablation could no longer attribute a difference to the names alone.

Nepali glues four suffixes onto names (को, भन्दा, ले, सँग), so a name is matched
as a whole Devanagari word, optionally followed by one of them.  Every rewritten
row is checked: same number of name slots before and after, the answer still
one of the candidates and still present in the text.

Usage:
    python scripts/make_ablation_data.py --lang hindi
    python scripts/make_ablation_data.py --lang nepali
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_reasoning_data as gen  # noqa: E402
from shortcut_rules import violations  # noqa: E402

NUM_OOD = (91, 140)
N_VAL_OOD = 2000
N_NAMEPOOL = 30000        # covers both ablation sizes: Hindi 20k, Nepali 30k
SUFFIXES = ("भन्दा", "सँग", "को", "ले")   # longest first
WORD = re.compile(r"[ऀ-ॿ]+")


class OODPools(gen.Pools):
    """Seen templates, unseen names, unseen numbers."""

    def __init__(self, t):
        self.t = t
        self.names = list(t.VAL_OOD_NAMES)
        self.lo, self.hi = NUM_OOD
        self.seen_templates = True


def build_val_ood(t, seed: int) -> list[dict]:
    """``gen.build`` with the OOD pools swapped in."""
    original = gen.Pools
    gen.Pools = lambda t_, split: OODPools(t_)
    try:
        return gen.build(t, "val_ood", N_VAL_OOD, seed)
    finally:
        gen.Pools = original


def split_name(word: str, names: set[str]) -> tuple[str, str] | None:
    """``(name, suffix)`` if ``word`` is a name, bare or with a glued suffix."""
    if word in names:
        return word, ""
    for suf in SUFFIXES:
        if word.endswith(suf) and word[: -len(suf)] in names:
            return word[: -len(suf)], suf
    return None


def names_in(text: str, names: set[str]) -> list[str]:
    """Names in order of appearance, with repeats."""
    return [hit[0] for w in WORD.findall(text) if (hit := split_name(w, names))]


def rename_row(row: dict, names: set[str], pool: list[str],
               rng: random.Random) -> tuple[dict, dict]:
    """Swap every name in one row for a pool name, consistently within the row."""
    found = list(dict.fromkeys(names_in(row["text"], names)))
    mapping = dict(zip(found, rng.sample(pool, len(found))))

    def sub(text: str) -> str:
        def one(m):
            hit = split_name(m.group(0), names)
            return mapping[hit[0]] + hit[1] if hit and hit[0] in mapping else m.group(0)
        return WORD.sub(one, text)

    out = dict(row)
    out["text"] = sub(row["text"])
    out["prompt"] = sub(row["prompt"])
    out["answer"] = mapping.get(row["answer"], row["answer"])
    out["target"] = f" {out['answer']}"
    out["candidates"] = [mapping.get(c, c) for c in row["candidates"]]
    return out, mapping


def check_row(before: dict, after: dict, old: set[str], new: set[str]) -> None:
    """Fail loudly on any rewrite that changed more than the names."""
    n_before = len(names_in(before["text"], old))
    n_after = len(names_in(after["text"], new))
    assert n_before == n_after, (before["text"], after["text"])
    assert after["answer"] in after["candidates"], after
    # Equality items answer "बराबर", which the text never contains.
    if before["answer"] in before["text"]:
        assert after["answer"] in after["text"], after
    # The non-name skeleton must be untouched.
    assert skeleton(before["text"], old) == skeleton(after["text"], new), (before["text"], after["text"])


def skeleton(text: str, names: set[str]) -> str:
    """The text with every name replaced by a placeholder, suffixes kept."""
    def one(m):
        hit = split_name(m.group(0), names)
        return "<N>" + hit[1] if hit else m.group(0)
    return WORD.sub(one, text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    t = gen.load_templates(a.lang)
    data = ROOT / a.lang / "reasoning" / "data"
    train_names = t.NAMES[: -gen.N_TEST_NAMES]
    test_names = t.NAMES[-gen.N_TEST_NAMES:]

    # No template word may look like a name, or the rewrite would touch it.
    words = " ".join(json.dumps(v, ensure_ascii=False) for k, v in vars(t).items()
                     if k.isupper() and "NAMES" not in k)
    clash = {w for w in WORD.findall(words) if split_name(w, set(t.NAMES) | set(t.EXTRA_NAMES))}
    assert not clash, f"template words that match names: {clash}"

    # ---- val_ood
    rows = build_val_ood(t, a.seed)
    held = set(test_names) | set(t.VAL_OOD_NAMES)
    assert not set(t.EXTRA_NAMES) & held and not set(train_names) & held
    bad = violations(rows, a.lang, t)
    if bad:
        raise SystemExit("val_ood failed the shortcut check:\n  " + "\n  ".join(bad))
    path = data / "val_ood.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    base = max(Counter(r["answer_position"] for r in rows).values()) / len(rows)
    print(f"  val_ood          {len(rows):>6,} rows  names {len(t.VAL_OOD_NAMES)}  "
          f"numbers {NUM_OOD[0]}-{NUM_OOD[1]}  majority baseline {base:.1%}")

    # ---- train_namepool
    train = [json.loads(l) for l in (data / "train.jsonl").read_text(encoding="utf-8").splitlines()[:N_NAMEPOOL]]
    pool = list(train_names) + list(t.EXTRA_NAMES)
    old, new = set(train_names), set(pool)
    rng = random.Random(a.seed)
    out, used = [], Counter()
    for r in train:
        renamed, mapping = rename_row(r, old, pool, rng)
        check_row(r, renamed, old, new)
        used.update(mapping.values())
        out.append(renamed)
    # Renaming must not create a shortcut either (for example by making the
    # answer's name systematically longer or rarer).
    bad = violations(out, a.lang, t)
    if bad:
        raise SystemExit("train_namepool failed the shortcut check:\n  " + "\n  ".join(bad))
    path = data / "train_namepool.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    changed = sum(r["text"] != o["text"] for r, o in zip(train, out))
    print(f"  train_namepool   {len(out):>6,} rows  pool {len(pool)} names, {len(used)} used  "
          f"rows changed {changed:,}  (price rows name objects, not people)")
    print(f"  most used: {used.most_common(3)}  least used: {used.most_common()[-3:]}")
    print(f"  example: {train[0]['text']}\n        -> {out[0]['text']}")


if __name__ == "__main__":
    main()
