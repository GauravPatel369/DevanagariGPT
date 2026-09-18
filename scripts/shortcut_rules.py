"""No-reasoning answer rules, shared by the data generator and the shortcut audit.

A shortcut is a surface pattern that picks the right answer without reading the
facts. The first version of the dataset had two: relation questions always named
the answer first, and superlative chains mostly listed the largest person first.
Both models learned them.

Each rule here answers a question from surface position alone. On a dataset
without shortcuts every rule should score close to chance, so
``make_reasoning_data.py`` refuses to write a split in which any rule beats
chance by more than ``THRESHOLD`` for any question type.

No torch import, so the generator can use this without the model stack.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

WORD = re.compile(r"[ऀ-ॿ]+")
SUFFIXES = ("भन्दा", "सँग", "को", "ले")
# Where the question starts: after the last danda or semicolon, or after the
# "then" of an if-then sentence.
QUESTION_START = {"hindi": ["।", "; ", "तो "], "nepali": ["।", "; ", "भने, ", "भए, "]}
THRESHOLD = 0.05

RULES = [
    "first_named",      # first person or object mentioned in the facts
    "last_named",       # last one mentioned in the facts
    "question_first",   # first name inside the question
    "question_second",  # second name inside the question
    "most_first",       # "most" -> first named, "least" -> last named
    "most_last",        # "most" -> last named, "least" -> first named
    "always_equal",     # always the equality word
]


def entity(word: str, entities: set[str]) -> str | None:
    """The entity a word refers to, allowing a glued Nepali suffix."""
    if word in entities:
        return word
    for suf in SUFFIXES:
        if word.endswith(suf) and word[: -len(suf)] in entities:
            return word[: -len(suf)]
    return None


def rule_answers(row: dict, lang: str, t) -> dict[str, str | None]:
    """What each rule would answer for one question."""
    text = row["text"]
    start = max((text.rfind(m) + len(m)) if m in text else 0 for m in QUESTION_START[lang])
    facts, question = text[:start], text[start:]
    entities = {c for c in row["candidates"] if c != t.EQUAL_ANSWER}
    named = list(dict.fromkeys(e for w in WORD.findall(facts) if (e := entity(w, entities))))
    in_question = [e for w in WORD.findall(question) if (e := entity(w, entities))]
    more = {a["more"] for a in t.ATTRIBUTES.values()}
    less = {a["less"] for a in t.ATTRIBUTES.values()}
    asks_more = any(m in question for m in more) and not any(m in question for m in less)
    asks_less = any(m in question for m in less) and not any(m in question for m in more)
    first, last = (named[0], named[-1]) if named else (None, None)
    return {
        "first_named": first,
        "last_named": last,
        "question_first": in_question[0] if in_question else None,
        "question_second": in_question[1] if len(in_question) > 1 else None,
        "most_first": first if asks_more else last if asks_less else None,
        "most_last": last if asks_more else first if asks_less else None,
        "always_equal": t.EQUAL_ANSWER if t.EQUAL_ANSWER in row["candidates"] else None,
    }


def rule_accuracy(rows: list[dict], lang: str, t) -> dict[str, dict]:
    """Per question type: chance level and how often each rule is right."""
    counts = defaultdict(lambda: defaultdict(float))
    for r in rows:
        c = counts[r["category"]]
        c["n"] += 1
        c["chance"] += 1 / len(r["candidates"])
        for rule, ans in rule_answers(r, lang, t).items():
            c[rule] += ans == r["answer"]
    return {cat: {"n": int(c["n"]), "chance": round(c["chance"] / c["n"], 3),
                  **{rule: round(c[rule] / c["n"], 3) for rule in RULES}}
            for cat, c in sorted(counts.items())}


def violations(rows: list[dict], lang: str, t, threshold: float = THRESHOLD) -> list[str]:
    """Every (type, rule) pair that beats chance by more than the allowed margin.

    The margin is ``threshold`` or three standard errors of a chance-level rule,
    whichever is larger. On a 100,000-row training split 5 points is far outside
    noise, but a test split has only ~480 rows per type, where a rule at chance
    drifts by several points on its own; three standard errors keeps the check
    from failing on that noise while still catching real shortcuts.
    """
    out = []
    for cat, v in rule_accuracy(rows, lang, t).items():
        noise = 3 * math.sqrt(v["chance"] * (1 - v["chance"]) / v["n"])
        margin = max(threshold, noise)
        for rule in RULES:
            if v[rule] > v["chance"] + margin:
                out.append(f"{cat}: {rule} right {v[rule]:.0%} against chance {v['chance']:.0%}"
                           f" (allowed up to {v['chance'] + margin:.0%})")
    return out
