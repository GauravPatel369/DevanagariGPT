"""Stage 8c: encode the corpus to flat uint16 memmaps.

Output format is deliberately the one Phase-2 training wants: one contiguous
``.bin`` per split, ``np.uint16``, documents separated by the EOS id.  A
memmapped flat array makes the Phase-2 dataloader trivial and, crucially,
*resumable* -- a training step maps to a byte offset, so a killed Colab session
resumes at the exact same data position.

This is also the only point where the >=20% manual-token requirement can be
evaluated, because the requirement is stated in tokens, not documents or
characters.  ``meta.json`` therefore carries per-``source_type`` and per-source
token counts for every split.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np

from ..io_utils import dump_json, read_dir
from ..splits import load_manifest

DTYPE = np.uint16
DOC_BATCH = 1000
WRITE_BUFFER = 8 << 20  # 8 MiB


def encode_split(
    shard_dir,
    manifest,
    model_path,
    out_bin,
    add_eos: bool = True,
    num_threads: int | None = None,
    batch_size: int = DOC_BATCH,
) -> dict:
    """Tokenize one split into a flat uint16 binary and return token statistics.

    Single pass.  An earlier version tokenized the whole split twice — once to
    size a memmap exactly, once to fill it — which doubled the cost of the most
    expensive stage in Phase 1.  Appending to a buffered binary file produces a
    byte-identical result without needing the length up front, and documents are
    tokenized in batches so SentencePiece can use every core (it releases the
    GIL and threads internally on list input).
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    if sp.get_piece_size() > 65535:
        raise ValueError("vocab exceeds uint16 range; switch DTYPE to uint32")
    eos = sp.eos_id()
    ids_wanted = load_manifest(manifest)
    threads = num_threads or os.cpu_count() or 1

    total = 0
    by_type: Counter = Counter()
    by_source: Counter = Counter()
    docs = 0
    bytes_raw = 0

    Path(out_bin).parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb", buffering=WRITE_BUFFER) as fh:

        def flush(batch):
            nonlocal total, docs, bytes_raw
            if not batch:
                return
            encoded = sp.encode(
                [d.text for d in batch], out_type=int, num_threads=threads
            )
            for d, ids in zip(batch, encoded):
                if add_eos:
                    ids = ids + [eos]
                fh.write(np.asarray(ids, dtype=DTYPE).tobytes())
                n = len(ids)
                total += n
                by_type[d.source_type] += n
                by_source[d.source] += n
                docs += 1
                bytes_raw += d.flags.get("n_bytes", len(d.text.encode("utf-8")))

        batch: list = []
        for doc in read_dir(shard_dir):
            if doc.doc_id not in ids_wanted:
                continue
            batch.append(doc)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
        flush(batch)

    manual = by_type.get("manual", 0)
    return {
        "tokens": total,
        "documents": docs,
        "raw_bytes": bytes_raw,
        "bytes_per_token": round(bytes_raw / max(total, 1), 4),
        "tokens_by_source_type": dict(by_type),
        "tokens_by_source": dict(by_source),
        "manual_fraction": round(manual / max(total, 1), 4),
    }


def encode_all(shard_dir, splits_dir, model_path, out_dir, vocab_size: int,
               num_threads: int | None = None) -> dict:
    """Encode train/val/test and write ``meta.json``.

    ``meta.json`` is the single artifact Phase 2 needs to load the data, and the
    single artifact the Phase-1 report quotes for token counts and the manual
    fraction.
    """
    out = Path(out_dir)
    meta = {"vocab_size": vocab_size, "dtype": "uint16", "splits": {}}
    for split in ("train", "val", "test"):
        stats = encode_split(
            shard_dir, Path(splits_dir) / f"{split}.txt", model_path, out / f"{split}.bin",
            num_threads=num_threads,
        )
        meta["splits"][split] = stats
        print(
            f"[encode] {split}: {stats['tokens']:,} tokens "
            f"({stats['manual_fraction']:.1%} manual)",
            flush=True,
        )
    train = meta["splits"]["train"]
    meta["manual_requirement"] = {
        "train_manual_fraction": train["manual_fraction"],
        "required": 0.20,
        "satisfied": train["manual_fraction"] >= 0.20,
        "manual_tokens_needed_at_current_total": max(
            0, int(0.20 * train["tokens"]) - train["tokens_by_source_type"].get("manual", 0)
        ),
    }
    dump_json(out / "meta.json", meta)
    return meta
