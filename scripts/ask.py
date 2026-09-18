"""Ask the finetuned reasoning model questions from a file.

Write one question per line in ``input.txt``, run this, read the answers in
``output.txt``.  Blank lines and lines starting with ``#`` are ignored, so a
file can be commented.

    python scripts/ask.py --lang hindi

Answers are chosen by comparing the model's probability of each candidate name
rather than by free generation, which is how the Phase-3 evaluation scores.  A
25M model rambles when asked to generate (Phase 2 Section 5.4), so generation
would measure formatting rather than reasoning.  Candidates are taken from the
Devanagari names in the question, or given explicitly:

    राम की उम्र 15 वर्ष है। श्याम की उम्र 23 वर्ष है। सबसे बड़ा कौन है?
    राम श्याम से लंबा है। श्याम मोहन से लंबा है। सबसे छोटा कौन है? | राम, श्याम, मोहन

The output shows the chosen answer and every candidate's score, so a confident
answer can be told from a near-tie.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmagpt.eval_reasoning import load  # noqa: E402


# Words that are grammar rather than entities. Without this the fallback below
# happily proposes "सेब" (apples) or "उम्र" (age) as candidate answers, and the
# model then scores nouns against names, which is meaningless.
STOPWORDS = {
    # Hindi
    "की", "का", "के", "है", "हैं", "है।", "हैं।", "में", "से", "और", "कौन",
    "उम्र", "कद", "मूल्य", "कीमत", "दाम", "वर्ष", "रुपये", "संख्या", "पास",
    "यदि", "तो", "जबकि", "दोनों", "तीनों", "सबसे", "इनमें", "एक", "दो", "तीन",
    "अधिक", "ज्यादा", "कम", "बड़ा", "छोटा", "लंबा", "भारी", "महंगा", "सस्ता",
    "व्यक्ति", "क्या", "संबंध", "स्थान", "दूसरे", "बीच", "किसकी", "किसके", "किसका",
    "बाइक", "क्षमता", "सिक्का",
    # Nepali
    "को", "छ", "छन्", "र", "मा", "भन्दा", "हो", "उमेर", "उचाइ", "मूल्य",
    "रुपैयाँ", "वटा", "सँग", "यदि", "भने", "दुवै", "तीनै", "सबैभन्दा",
    "यीमध्ये", "ठूलो", "सानो", "अग्लो", "होचो", "महँगो", "सस्तो", "बढी",
    "जना", "के", "सम्बन्ध", "कुन", "माझमा", "बीचमा", "दोस्रो",
}


def extract_candidates(question: str, names: list[str]) -> list[str]:
    """Entity names appearing in this question, in the order they appear.

    Known names from the template pool are preferred. Anything else falls back
    to Devanagari words that are not grammar and not adjacent to a digit, which
    catches names the pool has never seen -- but the fallback is a guess, so
    prefer giving candidates explicitly after a ``|`` when it matters.
    """
    found = [n for n in names if n in question]
    if len(found) >= 2:
        # Preserve order of appearance so a positional bias stays visible.
        return sorted(set(found), key=question.index)

    words = re.findall(r"[ऀ-ॿ]+", question)
    seen, cands = set(), []
    for w in words:
        if len(w) < 3 or w in STOPWORDS or w in seen:
            continue
        # A word adjacent to a number is a unit or a counted object, not the
        # entity being asked about: "45 सेब" (45 apples), "18 वर्ष" (18 years).
        if re.search(r"\d\s*" + re.escape(w) + r"|" + re.escape(w) + r"\s*\d", question):
            continue
        seen.add(w)
        cands.append(w)
    return cands[:4]


@torch.no_grad()
def answer(model, sp, question: str, candidates: list[str], device, autocast_dtype):
    """Return (best_candidate, {candidate: log-probability})."""
    prompt = question.rstrip() + "\nउत्तर:"
    prompt_ids = sp.encode(prompt)
    scores = {}
    for cand in candidates:
        cand_ids = sp.encode(" " + cand)
        ids = prompt_ids + cand_ids
        if len(ids) > model.cfg.context:
            ids = ids[-model.cfg.context:]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            logits, _, _ = model(x)
        lp = torch.log_softmax(logits[0].float(), dim=-1)
        start = len(ids) - len(cand_ids) - 1
        scores[cand] = sum(lp[start + i, t].item() for i, t in enumerate(cand_ids))
    best = max(scores, key=scores.get)
    return best, scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--input", default="input.txt")
    ap.add_argument("--output", default="output.txt")
    ap.add_argument("--checkpoint", default=None,
                    help="default: that language's finetuned best.pt")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    in_path, out_path = Path(a.input), Path(a.output)
    if not in_path.exists():
        # Seed a file rather than failing, so a first run is self-explanatory.
        sample = ("राम की उम्र 15 वर्ष है। श्याम की उम्र 23 वर्ष है। सबसे बड़ा कौन है?\n"
                  if a.lang == "hindi" else
                  "रामको उमेर 15 वर्ष छ। हरिको उमेर 23 वर्ष छ। सबैभन्दा ठूलो को हो?\n")
        in_path.write_text("# one question per line; '#' comments are ignored\n" + sample,
                           encoding="utf-8")
        print(f"created {in_path} with an example question -- edit it and run again")
        return

    ckpt = Path(a.checkpoint) if a.checkpoint else \
        ROOT / a.lang / "reasoning" / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found -- run the finetune first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif device.type == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    model, meta = load(a.lang, ckpt, device)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(
        ROOT / a.lang / "tokenizer" /
        f"{'hi' if a.lang == 'hindi' else 'ne'}_unigram_{model.cfg.vocab_size}.model"))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tmpl", ROOT / a.lang / "reasoning" / "templates.py")
    tmpl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tmpl)
    names = list(tmpl.NAMES) + list(tmpl.OBJECTS) + [tmpl.EQUAL_ANSWER]

    lines = [l.strip() for l in in_path.read_text(encoding="utf-8").splitlines()]
    questions = [l for l in lines if l and not l.startswith("#")]
    print(f"[ask] {a.lang}, {ckpt.name} @ step {meta['step']:,}, "
          f"{len(questions)} question(s)")

    out = [f"Model: {a.lang} finetuned ({ckpt.name}, step {meta['step']:,})",
           "=" * 70, ""]
    for i, line in enumerate(questions, 1):
        # "question | cand1, cand2" lets the candidate set be given explicitly.
        if "|" in line:
            question, rest = line.split("|", 1)
            cands = [c.strip() for c in rest.split(",") if c.strip()]
            question = question.strip()
        else:
            question = line
            cands = extract_candidates(question, names)

        # Always generate free text continuation
        prompt_ids = sp.encode(question + "\nउत्तर:")
        x_gen = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            gen_tokens = model.generate(x_gen, max_new_tokens=15, greedy=True)[0][len(prompt_ids):]
        free_text = sp.decode(gen_tokens.tolist()).strip().replace("\n", " ")

        if len(cands) < 2:
            out += [f"[{i}] {question}",
                    f"    GENERATED: \"{free_text}\"",
                    "    (could not find two candidates; add them after a | separator)", ""]
            print(f"  [{i}] GENERATED: {free_text}")
            continue

        best, scores = answer(model, sp, question, cands, device, autocast_dtype)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        margin = ranked[0][1] - ranked[1][1]
        out += [f"[{i}] {question}",
                f"    ANSWER   : {best}",
                f"    GENERATED: \"{free_text}\"",
                "    scores   : " + "  ".join(f"{c}={s:.2f}" for c, s in ranked),
                f"    margin   : {margin:.2f}"
                + ("  (close call)" if margin < 0.5 else ""),
                ""]
        print(f"  [{i}] {best}  (free: \"{free_text}\")")

    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"[ask] wrote {out_path}")


if __name__ == "__main__":
    main()
