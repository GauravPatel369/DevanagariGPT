"""Stage 8a: tokenizer training (SentencePiece, from scratch).

Pretrained tokenizers are forbidden by the project, and a separate vocabulary is
required per language.  SentencePiece is trained here on a sample drawn from the
**training split only** -- sampling from val/test would leak held-out text into
the vocabulary and inflate Phase-2 numbers.

``byte_fallback=True`` means true unknown tokens essentially disappear.  The
project asks for unknown-token statistics, so we report the **byte-fallback
rate** instead (fraction of held-out characters that decompose into byte
pieces) and explain the substitution; a table of zeros would be uninformative.

``normalization_rule_name="identity"`` because stage 3 already normalized the
text.  Letting SentencePiece apply NMT normalization on top would silently
double-normalize and desynchronise the corpus from the tokenizer.
"""

from __future__ import annotations

import random
from pathlib import Path

from ..io_utils import read_dir
from ..splits import load_manifest

SPECIALS = ["<|endoftext|>"]


def write_training_sample(
    shard_dir,
    train_manifest,
    out_txt,
    max_lines: int = 8_000_000,
    seed: int = 1337,
) -> int:
    """Dump training-split sentences to a flat text file for SentencePiece.

    Sentences are split on the danda so SentencePiece sees natural units; long
    paragraphs are otherwise truncated by its internal sentence-length cap.

    Returns:
        Number of lines written.
    """
    train_ids = load_manifest(train_manifest)
    rng = random.Random(seed)
    n = 0
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fh:
        for doc in read_dir(shard_dir):
            if doc.doc_id not in train_ids:
                continue
            for line in doc.text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                for sent in _split_sentences(line):
                    if len(sent) < 10:
                        continue
                    fh.write(sent + "\n")
                    n += 1
                    if n >= max_lines:
                        return n
    return n


def _split_sentences(line: str) -> list[str]:
    """Split a line on Devanagari and ASCII sentence terminals, keeping them."""
    out, buf = [], []
    for ch in line:
        buf.append(ch)
        if ch in "।॥?!":
            out.append("".join(buf).strip())
            buf = []
    if buf:
        out.append("".join(buf).strip())
    return [s for s in out if s]


def train_spm(
    input_txt,
    model_prefix,
    vocab_size: int,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    input_sentence_size: int = 10_000_000,
    seed: int = 1337,
) -> str:
    """Train one SentencePiece model.

    Args:
        input_txt: Flat text file produced by :func:`write_training_sample`.
        model_prefix: Output path prefix; writes ``.model`` and ``.vocab``.
        vocab_size: Candidate vocabulary size (swept in stage 8b).
        model_type: "unigram" or "bpe"; both are trained for comparison.

    Returns:
        Path to the trained ``.model`` file.
    """
    import sentencepiece as spm

    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_txt),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        byte_fallback=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        user_defined_symbols=SPECIALS,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        num_threads=8,
        seed_sentencepiece_size=1_000_000,
        train_extremely_large_corpus=False,
    )
    return f"{model_prefix}.model"
