"""Generate the synthetic comparative-reasoning dataset for one language.

The Phase-3 brief requires a dataset built programmatically so the ground truth
is known, in the target language's own script and phrasing, with train/val/test
splits and documented leakage control.

**Nine categories**, covering the brief's three example families:

    direct comparison, numerical comparison, 3-entity ordering,
    mixed greater/smaller/equal          -> "comparison questions"
    2-hop and 3-hop transitive           -> "which is smallest/largest"
    2-hop and 3-hop A-to-C relation      -> "relation between A and C"
    ranking (middle/second)              -> ordering

The A-to-C categories matter: they ask for a specific pairwise fact rather than
an extreme, so a model that finds extremes by elimination cannot fake them.

**Leakage control on three axes at once.**  Test items differ from training in
names, templates *and* number range, so no surface feature carries over and only
the reasoning itself is shared:

    names      26 train / 6 test, disjoint
    templates  70% train / 30% test, disjoint
    numbers    5-40 train / 41-90 test, disjoint

Ranked by how much each actually tests: numbers is the strongest, because a
model that memorised value pairs cannot transfer to values it has never seen.
Names is the weakest, since names are interchangeable slots. Three test sets
separate these effects:

    test_a   new names + new numbers, SEEN templates   -> does it reason?
    test_b   new names + new numbers, NEW templates    -> does phrasing matter?
    test_c   new names + new templates, SEEN numbers   -> were numbers memorised?

**Shortcut suppression.**  A shortcut is a surface pattern that gives the answer
without reasoning. The first version of this dataset had two -- relation
questions always named the answer first, and chains mostly listed the largest
person first -- and both models learned them. Now:

    relation questions   random name order, asked as "more" or "less"
    chains               built from separate facts, each phrased at random
                         (half name the smaller person first), in shuffled order
    every split          checked by shortcut_rules.py; if any no-reasoning rule
                         beats chance by more than 5 points, nothing is written

Each record carries metadata (`hops`, `answer_position`, `category`,
`template_id`) so accuracy can be sliced afterwards.

Usage:
    python scripts/make_reasoning_data.py --lang hindi
    python scripts/make_reasoning_data.py --lang nepali --train 100000
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from shortcut_rules import THRESHOLD, rule_accuracy, violations  # noqa: E402

# Nine categories, weighted so the transitive families -- the heart of the
# brief -- carry the majority, with 3-hop A-to-C smallest because it is the
# hardest and least likely to yield signal at 25M parameters.
CATEGORY_WEIGHTS = {
    "direct_comparison": 0.12,
    "numeric_comparison": 0.12,
    "three_entity_ordering": 0.12,
    "mixed_relation": 0.12,
    "chain2_superlative": 0.12,
    "chain3_superlative": 0.12,
    "chain2_relation": 0.10,
    "chain3_relation": 0.08,
    "ranking": 0.10,
}

N_TEST_NAMES = 6          # held out from every training example
TEST_TEMPLATE_FRAC = 0.3  # last 30% of each template list is test-only
NUM_TRAIN = (5, 40)
NUM_TEST = (41, 90)


def load_templates(lang: str):
    """Import the language's own template module from its directory."""
    path = ROOT / lang / "reasoning" / "templates.py"
    if not path.exists():
        raise SystemExit(f"{path} missing")
    spec = importlib.util.spec_from_file_location(f"{lang}_templates", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Pools:
    """Names, templates and number ranges, split so train and test never meet.

    The split happens once, here, rather than at draw time. That makes leakage
    structurally impossible instead of something the sampling has to remember.
    """

    def __init__(self, t, split: str):
        self.t = t
        # test_d reuses the *training* names deliberately: it is the control for
        # the other three. If accuracy jumps there, the model needs familiar
        # entities; if it does not, the failure is not about entity novelty.
        train_names = split in ("train", "val", "test_d")
        self.names = t.NAMES[:-N_TEST_NAMES] if train_names else t.NAMES[-N_TEST_NAMES:]
        # test_c and test_d reuse the training number range to isolate whether
        # numbers specifically were memorised.
        self.lo, self.hi = NUM_TRAIN if split in ("train", "val", "test_c", "test_d") else NUM_TEST
        self.seen_templates = split in ("train", "val", "test_a", "test_d")

    def pick(self, templates: list[str]) -> tuple[str, int]:
        """Draw a template from this split's half of the list, with its id.

        The template and its id come from one draw. Drawing them separately
        would record an id that does not match the text, which silently breaks
        the per-template accuracy breakdown and the leakage check.

        The id is the template's own text, hashed, rather than its index. Index
        would collide across lists of different lengths -- index 5 of the 2-hop
        list is a different sentence from index 5 of the 3-hop list -- making a
        train/test id comparison meaningless.
        """
        cut = int(len(templates) * (1 - TEST_TEMPLATE_FRAC))
        pool = list(enumerate(templates))
        pool = pool[:cut] if self.seen_templates else pool[cut:]
        _, tmpl = random.choice(pool)
        # blake2b, not the builtin hash(): PYTHONHASHSEED randomises the latter
        # per process, so ids would differ between runs and break reproducibility.
        return tmpl, int(hashlib.blake2b(tmpl.encode(), digest_size=4).hexdigest(), 16)

    def pick_frame(self, frames: list[dict]) -> tuple[dict, int]:
        """``pick`` for chain frames, which are dicts rather than strings.

        Two frames can share their text and differ only in how the facts are
        joined, so the id hashes both.
        """
        cut = int(len(frames) * (1 - TEST_TEMPLATE_FRAC))
        pool = frames[:cut] if self.seen_templates else frames[cut:]
        frame = random.choice(pool)
        key = json.dumps(frame, ensure_ascii=False, sort_keys=True)
        return frame, int(hashlib.blake2b(key.encode(), digest_size=4).hexdigest(), 16)

    def entities(self, n: int) -> list[str]:
        return random.sample(self.names, n)

    def values(self, n: int, distinct: bool = True) -> list[int]:
        """Distinct values by default; equality cases ask for repeats."""
        span = list(range(self.lo, self.hi + 1))
        return random.sample(span, n) if distinct else [random.choice(span)] * n


def _attr(t, numeric_only: bool = False):
    """Pick an attribute; height and age take numeric templates naturally."""
    # Only age and height describe people. "price" would produce "Dinesh is
    # cheaper than Mohan" and "quantity" would give "Dinesh is more than Mohan",
    # neither of which is idiomatic. Price and quantity have their own
    # object-based generators instead.
    keys = ["age", "height"]
    k = random.choice(keys)
    return k, t.ATTRIBUTES[k]


def _q(t, key: str, a: dict, **extra) -> str:
    """Fill one question phrasing with the attribute's own adjectives.

    Defaults every attribute field for the same reason ``_fill`` does: a
    question that gains a placeholder should not raise in one caller only.
    """
    return random.choice(t.QUESTIONS[key]).format(
        more=a.get("more", ""), less=a.get("less", ""),
        more_q=a.get("more_q", ""), less_q=a.get("less_q", ""),
        noun=a.get("noun", ""), gen=a.get("gen", ""), unit=a.get("unit", ""),
        **extra)


def _fill(tmpl: str, attr: dict, **fields) -> str:
    """Format a template, always supplying every attribute-derived field.

    Every generator goes through here rather than calling ``.format`` directly.
    A template that gains a new placeholder would otherwise raise KeyError in
    whichever generator was not updated, which is exactly the bug this replaces.
    """
    return tmpl.format(
        noun=attr.get("noun", ""), gen=attr.get("gen", ""), unit=attr.get("unit", ""),
        more=attr.get("more", ""), less=attr.get("less", ""), **fields)


def _record(text: str, answer: str, candidates: list[str], category: str,
            hops: int, tid: int) -> dict:
    """One example plus the metadata needed to slice accuracy later.

    ``answer_position`` catches the commonest shortcut: if the model is much
    more accurate when the answer was mentioned last, it is using position
    rather than reasoning.
    """
    return {
        "text": text,
        "answer": answer,
        "candidates": candidates,
        "category": category,
        "hops": hops,
        "template_id": tid,
        "answer_position": candidates.index(answer) if answer in candidates else -1,
        "n_entities": len(candidates),
    }


# ------------------------------------------------------------- generators
# Each returns one record. They share a signature so the dispatch table below
# can call them uniformly.


def gen_direct(t, p: Pools) -> dict:
    """Two entities, numeric facts, asking which is greater or smaller."""
    a, b = p.entities(2)
    x, y = p.values(2)
    key, attr = _attr(t, numeric_only=True)
    tmpl, tid = p.pick(t.TWO_ENTITY)
    ask_max = random.random() < 0.5           # both directions, same facts
    q = _q(t, "max2" if ask_max else "min2", attr)
    text = _fill(tmpl, attr, a=a, b=b, x=x, y=y, question=q)
    answer = (a if x > y else b) if ask_max else (a if x < y else b)
    return _record(text, answer, [a, b], "direct_comparison", 1, tid)


def gen_numeric(t, p: Pools) -> dict:
    """Price or quantity, which use object nouns rather than people."""
    if random.random() < 0.5:
        o1, o2 = random.sample(t.OBJECTS, 2)
        x, y = p.values(2)
        attr = t.ATTRIBUTES["price"]
        tmpl, tid = p.pick(t.OBJECT_TWO)
        ask_max = random.random() < 0.5
        q = _q(t, "max2_obj" if ask_max else "min2_obj", attr)
        text = tmpl.format(a=o1, b=o2, x=x, y=y, question=q)
        answer = (o1 if x > y else o2) if ask_max else (o1 if x < y else o2)
        return _record(text, answer, [o1, o2], "numeric_comparison", 1, tid)

    a, b = p.entities(2)
    x, y = p.values(2)
    obj = random.choice(t.OBJECTS)
    attr = t.ATTRIBUTES["quantity"]
    tmpl, tid = p.pick(t.QUANTITY_TWO)
    ask_max = random.random() < 0.5
    # Quantity is about people holding objects, so the person form is correct.
    q = _q(t, "max2" if ask_max else "min2", attr)
    text = tmpl.format(a=a, b=b, x=x, y=y, obj=obj, question=q)
    answer = (a if x > y else b) if ask_max else (a if x < y else b)
    return _record(text, answer, [a, b], "numeric_comparison", 1, tid)


def gen_three(t, p: Pools) -> dict:
    """Three entities with numeric facts, asking for an extreme."""
    a, b, c = p.entities(3)
    x, y, z = p.values(3)
    key, attr = _attr(t, numeric_only=True)
    tmpl, tid = p.pick(t.THREE_ENTITY)
    ask_max = random.random() < 0.5
    q = _q(t, "max3" if ask_max else "min3", attr)
    text = _fill(tmpl, attr, a=a, b=b, c=c, x=x, y=y, z=z, question=q)
    pairs = [(a, x), (b, y), (c, z)]
    answer = max(pairs, key=lambda r: r[1])[0] if ask_max else min(pairs, key=lambda r: r[1])[0]
    return _record(text, answer, [a, b, c], "three_entity_ordering", 1, tid)


def gen_mixed(t, p: Pools) -> dict:
    """Includes equality, which a magnitude-guessing strategy cannot fake."""
    a, b = p.entities(2)
    key, attr = _attr(t, numeric_only=True)
    # One in three is a tie, matching the three candidates, so "always equal"
    # is no better than chance.
    if random.random() < 1 / 3:
        x, y = p.values(2, distinct=False)     # deliberate tie
        tmpl, tid = p.pick(t.TWO_ENTITY)
        q = _q(t, "equal", attr)
        text = _fill(tmpl, attr, a=a, b=b, x=x, y=y, question=q)
        return _record(text, t.EQUAL_ANSWER, [a, b, t.EQUAL_ANSWER], "mixed_relation", 1, tid)

    x, y = p.values(2)
    tmpl, tid = p.pick(t.TWO_ENTITY)
    ask_max = random.random() < 0.5
    q = _q(t, "max2" if ask_max else "min2", attr)
    text = _fill(tmpl, attr, a=a, b=b, x=x, y=y, question=q)
    answer = (a if x > y else b) if ask_max else (a if x < y else b)
    return _record(text, answer, [a, b, t.EQUAL_ANSWER], "mixed_relation", 1, tid)


def _join_facts(t, full: list[str], short: list[str], style: str) -> str:
    """Combine the link clauses the way the frame asks."""
    clauses = short if style.startswith("short_") else full
    style = style.removeprefix("short_")
    if style == "sentences":
        return "। ".join(clauses)
    if style == "semicolon":
        return "; ".join(clauses)
    last = f" {t.AND} " if style == "and" else f", {t.AND} "     # "known" keeps the comma
    return ", ".join(clauses[:-1]) + last + clauses[-1]


def _chain(t, p: Pools, n: int):
    """A chain of ``n`` people, largest first, written as shuffled facts.

    Returns ``(people, facts, frame, id, attr)`` where ``people[0]`` is the
    largest and ``people[-1]`` the smallest. The text never reveals that order:

      * each link "x is larger than y" is phrased at random, and two of the four
        phrasings name the smaller person first;
      * the links are shuffled.

    So the largest person can be named first, in the middle or last. The earlier
    whole-sentence templates mostly named the largest first, and that was a
    shortcut the models learned (see scripts/shortcut_audit.py).
    """
    people = p.entities(n)
    key, attr = _attr(t)
    frame, tid = p.pick_frame(t.CHAIN_FRAMES)
    links = [(people[i], people[i + 1]) for i in range(n - 1)]
    random.shuffle(links)
    forms = [random.randrange(len(t.CHAIN_LINKS)) for _ in links]
    full = [_fill(t.CHAIN_LINKS[f], attr, x=x, y=y) for (x, y), f in zip(links, forms)]
    short = [_fill(t.CHAIN_LINKS_SHORT[f], attr, x=x, y=y) for (x, y), f in zip(links, forms)]
    return people, _join_facts(t, full, short, frame["join"]), frame, tid, attr


def _superlative(t, p: Pools, n: int, category: str, hops: int) -> dict:
    """Who holds a given place in the chain: largest, smallest, or in between.

    Asking only for the largest or smallest leaves a structural cue even with
    shuffled facts: the two ends of a chain are each mentioned once and the
    middle people twice, so the answer is always one of the once-mentioned
    names. Asking for every place equally often (middle for three people,
    second largest or second smallest for four) makes each person the answer
    equally often.
    """
    people, facts, frame, tid, attr = _chain(t, p, n)
    if n == 3:
        place = random.choice(["max3", "min3", "middle"])
        answer = {"max3": people[0], "min3": people[2], "middle": people[1]}[place]
    else:
        place = random.choice(["max4", "min4", "second_max4", "second_min4"])
        answer = {"max4": people[0], "min4": people[3],
                  "second_max4": people[1], "second_min4": people[2]}[place]
    q = _q(t, place, attr)
    text = frame["frame"].format(facts=facts, question=q)
    # Candidate order is shuffled too, so the stored answer position carries no
    # information about which place was asked for.
    candidates = random.sample(people, len(people))
    return _record(text, answer, candidates, category, hops, tid)


def _relation(t, p: Pools, n: int, pair: tuple[str, str], people, facts, frame, tid, attr,
              category: str, hops: int) -> dict:
    """Which of two chain members is larger, or smaller.

    Both the order the two names are asked in and the direction of the question
    are random. The first version always asked "who is larger?" with the larger
    person named first, so the first name in the question was always the answer.
    """
    larger, smaller = pair
    ask_more = random.random() < 0.5
    first, second = (larger, smaller) if random.random() < 0.5 else (smaller, larger)
    q = _q(t, "relation" if ask_more else "relation_less", attr, p=first, q=second)
    text = frame["frame"].format(facts=facts, question=q)
    # Candidates keep the question's order: the name-swap probe relies on it.
    return _record(text, larger if ask_more else smaller, [first, second], category, hops, tid)


def gen_chain2_sup(t, p: Pools) -> dict:
    """A > B > C, asked for the extreme. Answer is never stated directly."""
    return _superlative(t, p, 3, "chain2_superlative", 2)


def gen_chain3_sup(t, p: Pools) -> dict:
    """A > B > C > D, asked for the extreme."""
    return _superlative(t, p, 4, "chain3_superlative", 3)


def gen_chain2_rel(t, p: Pools) -> dict:
    """A > B > C, asked for the A-to-C relation -- the brief's third bullet.

    Harder than the superlative form: the model must derive a specific pairwise
    fact rather than scan for an extreme, so elimination strategies fail.
    """
    people, facts, frame, tid, attr = _chain(t, p, 3)
    return _relation(t, p, 3, (people[0], people[2]), people, facts, frame, tid, attr,
                     "chain2_relation", 2)


def gen_chain3_rel(t, p: Pools) -> dict:
    """A > B > C > D, asked for the relation between two non-adjacent links."""
    people, facts, frame, tid, attr = _chain(t, p, 4)
    a, b, c, d = people
    pair = random.choice([(a, d), (a, c), (b, d)])     # vary the queried pair
    return _relation(t, p, 4, pair, people, facts, frame, tid, attr, "chain3_relation", 3)


def gen_ranking(t, p: Pools) -> dict:
    """Middle or second place: neither extreme, so scanning for one fails."""
    a, b, c = p.entities(3)
    x, y, z = p.values(3)
    key, attr = _attr(t, numeric_only=True)
    tmpl, tid = p.pick(t.THREE_ENTITY)
    q = _q(t, random.choice(["middle", "second"]), attr)
    text = _fill(tmpl, attr, a=a, b=b, c=c, x=x, y=y, z=z, question=q)
    middle = sorted([(a, x), (b, y), (c, z)], key=lambda r: r[1])[1][0]
    return _record(text, middle, [a, b, c], "ranking", 1, tid)


GENERATORS = {
    "direct_comparison": gen_direct,
    "numeric_comparison": gen_numeric,
    "three_entity_ordering": gen_three,
    "mixed_relation": gen_mixed,
    "chain2_superlative": gen_chain2_sup,
    "chain3_superlative": gen_chain3_sup,
    "chain2_relation": gen_chain2_rel,
    "chain3_relation": gen_chain3_rel,
    "ranking": gen_ranking,
}


def build(t, split: str, n: int, seed: int, exclude: set[str] | None = None) -> list[dict]:
    """Generate one split, balanced across categories, without duplicates.

    ``exclude`` holds texts that must not appear. It matters for test_d, which
    draws names, templates and numbers from the *training* pools: without it
    the split could reproduce training examples verbatim and would measure
    memorisation rather than recombination.
    """
    random.seed(seed)
    pools = Pools(t, split)
    out, seen = [], set(exclude or ())
    for cat, weight in CATEGORY_WEIGHTS.items():
        want = round(n * weight)
        gen, tries = GENERATORS[cat], 0
        made = 0
        # Reject duplicates rather than sampling with replacement: a repeated
        # example is one the model can memorise instead of solving.
        while made < want and tries < want * 50:
            tries += 1
            rec = gen(t, pools)
            if rec["text"] in seen:
                continue
            seen.add(rec["text"])
            rec["prompt"] = rec["text"] + f"\n{t.ANSWER_PREFIX}"
            rec["target"] = f" {rec['answer']}"
            out.append(rec)
            made += 1
        if made < want:
            print(f"    warning: {cat} produced {made}/{want} unique items")
    random.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--train", type=int, default=100_000)
    ap.add_argument("--val", type=int, default=5_000)
    ap.add_argument("--test", type=int, default=4_000, help="per test set; three are written")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    t = load_templates(a.lang)
    out_dir = ROOT / a.lang / "reasoning" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reasoning dataset for {a.lang}")
    print(f"  names   : {len(t.NAMES) - N_TEST_NAMES} train / {N_TEST_NAMES} test (disjoint)")
    print(f"  numbers : {NUM_TRAIN[0]}-{NUM_TRAIN[1]} train / {NUM_TEST[0]}-{NUM_TEST[1]} test")
    print(f"  templates: {int((1 - TEST_TEMPLATE_FRAC) * 100)}% train / "
          f"{int(TEST_TEMPLATE_FRAC * 100)}% test (disjoint)\n")

    splits = [("train", a.train), ("val", a.val),
              ("test_a", a.test), ("test_b", a.test), ("test_c", a.test),
              ("test_d", a.test)]
    built = {}
    train_texts: set[str] = set()
    for i, (split, n) in enumerate(splits):
        # test_d draws from the training pools, so it must be told what the
        # training split already contains or it would reproduce it verbatim.
        built[split] = build(t, split, n, a.seed + i,
                             exclude=train_texts if split == "test_d" else None)
        if split == "train":
            train_texts = {r["text"] for r in built[split]}

    # Shortcut gate: nothing is written if any split lets a no-reasoning rule
    # beat chance. The first dataset shipped two such shortcuts and both models
    # learned them, so this is checked on every split, not just train.
    failed = False
    audit = {}
    for split, rows in built.items():
        audit[split] = rule_accuracy(rows, a.lang, t)
        bad = violations(rows, a.lang, t)
        print(f"  shortcut check {split:7s}: " + ("PASS" if not bad else "FAIL"))
        for line in bad:
            print(f"      {line}")
        failed |= bool(bad)
    if failed:
        raise SystemExit("\nshortcut check failed; nothing written. Fix the generator and rerun.")

    summary = {}
    for split, rows in built.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # The majority-class rate is the score a model gets by always naming the
        # same position; accuracy must be read against it, not against zero.
        pos = Counter(r["answer_position"] for r in rows)
        base = max(pos.values()) / len(rows)
        summary[split] = dict(n=len(rows), majority_baseline=round(base, 4))
        print(f"  {split:7s} {len(rows):>7,} examples  majority-position baseline {base:.1%}")
    (out_dir / "shortcut_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    (out_dir / "meta.json").write_text(json.dumps({
        "language": a.lang,
        "categories": CATEGORY_WEIGHTS,
        "splits": summary,
        "leakage_control": {
            "names": f"{len(t.NAMES) - N_TEST_NAMES} train / {N_TEST_NAMES} test, disjoint",
            "templates": f"{int((1 - TEST_TEMPLATE_FRAC) * 100)}/"
                         f"{int(TEST_TEMPLATE_FRAC * 100)} split, disjoint",
            "numbers": f"train {NUM_TRAIN}, test {NUM_TEST}, disjoint",
            "test_a": "new names + numbers, seen templates",
            "test_b": "new names + numbers + templates",
            "test_c": "new names + templates, seen number range",
            "test_d": "seen names + templates + numbers, unseen combinations",
        },
        "shortcut_check": f"every split passed: no rule above chance + {int(THRESHOLD * 100)} points",
        "seed": a.seed,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {out_dir}")


if __name__ == "__main__":
    main()
