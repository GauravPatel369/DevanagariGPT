"""Free-text autoregressive generation on questions from input.txt.

Shows what the fine-tuned model generates naturally without any multiple-choice candidate constraints.

Usage:
    python scripts/ask_free.py --lang hindi
    python scripts/ask_free.py --lang nepali
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmagpt.data import resolve_tokenizer  # noqa: E402
from lmagpt.eval_reasoning import load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="hindi", choices=["hindi", "nepali"])
    ap.add_argument("--input", default="input.txt")
    ap.add_argument("--output", default="output_free.txt")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    in_path = Path(a.input)
    out_path = Path(a.output)

    if not in_path.exists():
        print(f"Error: {in_path} not found.")
        sys.exit(1)

    ckpt_path = Path(a.checkpoint) if a.checkpoint else ROOT / a.lang / "reasoning" / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint {ckpt_path} not found.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {a.lang} model from: {ckpt_path.name} on {device}...")
    model, ckpt = load(a.lang, ckpt_path, device)

    import sentencepiece as spm
    tok_path = resolve_tokenizer(a.lang, ROOT)
    sp = spm.SentencePieceProcessor(model_file=str(tok_path))

    lines = in_path.read_text(encoding="utf-8").splitlines()
    questions = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip explicit candidates if present
        q = line.split("|")[0].strip()
        questions.append(q)

    print(f"Generating free-text answers for {len(questions)} question(s)...\n")

    output_blocks = [
        f"Model: {a.lang} finetuned ({ckpt_path.name}, step {ckpt.get('step', '?')}) - FREE GENERATION",
        "=" * 70,
        "",
    ]

    for i, q in enumerate(questions, 1):
        prompt = q + "\nउत्तर:"
        prompt_ids = sp.encode(prompt)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # 1. Greedy generation (argmax - the most confident tokens)
        gen_ids_greedy = model.generate(x, max_new_tokens=a.max_new_tokens, greedy=True)[0]
        new_ids_greedy = gen_ids_greedy[len(prompt_ids):].tolist()
        gen_text_greedy = sp.decode(new_ids_greedy).strip()

        # Extract the first word or stop at newline / punctuation
        first_word = gen_text_greedy.split()[0] if gen_text_greedy.split() else ""
        first_clause = gen_text_greedy.split("\n")[0].split("।")[0].strip()

        block = [
            f"[{i}] {q}",
            f"    Prompt Ending  : ...उत्तर:",
            f"    Free Generation: \"{gen_text_greedy}\"",
            f"    First Word     : {first_word}",
            f"    First Clause   : {first_clause}",
            "",
        ]
        output_blocks.extend(block)
        print(f"[{i}] Q: {q[:50]}...")
        print(f"    -> Generated: \"{gen_text_greedy}\" (First word: {first_word})\n")

    out_text = "\n".join(output_blocks)
    out_path.write_text(out_text, encoding="utf-8")
    print(f"Saved free generation output to: {out_path}")


if __name__ == "__main__":
    main()
