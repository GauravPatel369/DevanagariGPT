"""Train & Evaluate 8k, 16k, and 32k Vocab Tokenizers on Validation Set (val.txt).

Evaluates on VALIDATION split (val.txt) with detailed Byte Fallback breakdown:
- ASCII Byte Fallback (0x00 - 0x7F): digits, punctuation, English letters, newlines (e.g. <0x0A>)
- Non-ASCII UTF-8 Byte Fallback (0x80 - 0xFF): rare unmapped Devanagari or foreign Unicode bytes

Usage:
    python scripts/train_and_eval_multivocab.py --lang hindi
    python scripts/train_and_eval_multivocab.py --lang nepali
"""

import sys
import os
import io
import time
import json
import argparse
from pathlib import Path
import zstandard as zstd
import sentencepiece as spm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from lmacorpus.cli import paths
from lmacorpus.io_utils import load_config, dump_json, read_dir
from lmacorpus.splits import load_manifest
from lmacorpus.tokenizer.train_spm import write_training_sample, train_spm

def evaluate_model_on_val_split(shard_dir: Path, val_manifest_path: Path, model_path: Path):
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    ids_wanted = load_manifest(val_manifest_path)
    unk_id = sp.unk_id()

    tot_docs = 0
    tot_words = 0
    tot_chars = 0
    tot_tokens = 0
    tot_unk = 0
    ascii_byte_fallback = 0
    non_ascii_byte_fallback = 0

    for doc in read_dir(shard_dir):
        if doc.doc_id not in ids_wanted:
            continue

        text = doc.text
        if not text:
            continue

        words = len(text.split())
        chars = len(text)

        ids = sp.encode(text, out_type=int)
        tok_strs = sp.encode(text, out_type=str)
        n_toks = len(ids)

        tot_docs += 1
        tot_words += words
        tot_chars += chars
        tot_tokens += n_toks

        if unk_id != -1:
            tot_unk += ids.count(unk_id)

        for t in tok_strs:
            if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                try:
                    b_val = int(t[3:5], 16)
                    if b_val <= 0x7F:
                        ascii_byte_fallback += 1
                    else:
                        non_ascii_byte_fallback += 1
                except ValueError:
                    pass

    tot_byte_fallback = ascii_byte_fallback + non_ascii_byte_fallback
    fertility = tot_tokens / max(1, tot_words)
    compression = tot_chars / max(1, tot_tokens)
    unk_rate = (tot_unk / max(1, tot_tokens)) * 100.0
    
    tot_byte_rate = (tot_byte_fallback / max(1, tot_tokens)) * 100.0
    ascii_byte_rate = (ascii_byte_fallback / max(1, tot_tokens)) * 100.0
    non_ascii_byte_rate = (non_ascii_byte_fallback / max(1, tot_tokens)) * 100.0

    return {
        "model_name": model_path.name,
        "vocab_size": sp.get_piece_size(),
        "split_evaluated": val_manifest_path.name,
        "docs": tot_docs,
        "words": tot_words,
        "chars": tot_chars,
        "tokens": tot_tokens,
        "fertility": round(fertility, 2),
        "compression": round(compression, 2),
        "unk_count": tot_unk,
        "unk_rate": round(unk_rate, 4),
        "ascii_byte_count": ascii_byte_fallback,
        "ascii_byte_rate": round(ascii_byte_rate, 4),
        "non_ascii_byte_count": non_ascii_byte_fallback,
        "non_ascii_byte_rate": round(non_ascii_byte_rate, 4),
        "total_byte_fallback_count": tot_byte_fallback,
        "total_byte_fallback_rate": round(tot_byte_rate, 4)
    }

def run_multivocab_eval(lang: str):
    p = paths(lang)
    cfg = load_config(p["configs"] / "tokenizer.yaml")
    src_dir = p["dedup"]
    val_manifest = p["splits"] / "val.txt"
    train_manifest = p["splits"] / "train.txt"

    if not val_manifest.exists():
        print(f"❌ Validation manifest not found: {val_manifest}. Run split first!")
        return

    print(f"\n========================================================================================")
    print(f"🔤 TOKENIZER VALIDATION & BYTE FALLBACK ANALYSIS FOR: {lang.upper()} (val.txt)")
    print(f"   Corpus Shards       : {src_dir}")
    print(f"   Validation Manifest : {val_manifest}")
    print(f"========================================================================================\n", flush=True)

    spm_input = p["tokenizer"] / "spm_input.txt"
    if not spm_input.exists():
        print(f"   Writing training text sample from train.txt manifest...", flush=True)
        write_training_sample(src_dir, train_manifest, spm_input, max_lines=cfg.get("max_sample_lines", 8_000_000))

    lang_prefix = cfg.get("lang_code", lang[:2])
    vocab_sizes = [5000, 8000, 10000]
    model_types = ["bpe", "unigram"]

    results = []

    print(f"{'Algorithm':<10} | {'Vocab':<7} | {'Val Tokens':<12} | {'Fertility':<10} | {'Compression':<12} | {'UNK Rate':<9} | {'ASCII Fallback (0x00-0x7F)':<26} | {'Non-ASCII Fallback (0x80-0xFF)':<28}")
    print(f"----------------------------------------------------------------------------------------------------------------------------------------------------------------")

    for mt in model_types:
        for V in vocab_sizes:
            model_prefix = p["tokenizer"] / f"{lang_prefix}_{mt}_{V}"
            model_file = Path(f"{model_prefix}.model")

            if not model_file.exists():
                print(f"   -> Training {mt.upper()} V={V:,} on train split...", flush=True)
                train_spm(spm_input, model_prefix, vocab_size=V, model_type=mt)

            eval_res = evaluate_model_on_val_split(src_dir, val_manifest, model_file)
            eval_res["model_type"] = mt

            ascii_str = f"{eval_res['ascii_byte_rate']:.4f}% ({eval_res['ascii_byte_count']:,})"
            non_ascii_str = f"{eval_res['non_ascii_byte_rate']:.4f}% ({eval_res['non_ascii_byte_count']:,})"

            marker = " 🏆 (SELECTED)" if mt == "bpe" and V == 16000 else ""
            print(f"{mt.upper():<10} | {V:<7,} | {eval_res['tokens']:<12,} | {eval_res['fertility']:<10.2f} | {eval_res['compression']:<12.2f} | {eval_res['unk_rate']:<8.4f}% | {ascii_str:<26} | {non_ascii_str:<28}{marker}")

            results.append(eval_res)

    print(f"================================================================================================================------------------------------------------------\n")

    # Save to report JSON
    out_json = p["reports"] / "multivocab_validation_evaluation_report.json"
    dump_json(out_json, results)
    print(f"📄 Saved detailed validation evaluation report to: {out_json}\n")

def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate 8k, 16k, and 32k Vocab Tokenizers on Validation Set (val.txt).")
    parser.add_argument("--lang", choices=["hindi", "nepali"], required=True, help="Target language")
    args = parser.parse_args()

    run_multivocab_eval(args.lang)

if __name__ == "__main__":
    main()
