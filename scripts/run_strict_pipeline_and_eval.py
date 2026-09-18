"""Master Pipeline Orchestrator & Strict Evaluation for Hindi & Nepali.

Performs:
1. Caps HF Downloaded dataset to ~400M tokens while keeping 100% Manual Webscraping tokens (~100M+).
2. Builds deterministic Group-Aware Train/Val/Test splits (train 98%, val 1%, test 1%).
3. Trains SentencePiece Tokenizer from scratch ONLY on the TRAIN split (zero data leakage).
4. Evaluates Tokenizer ONLY on HELD-OUT TEST text:
   - Fertility (Tokens/Word)
   - Compression Ratio (Chars/Token)
   - UNK Rate (%)
   - Byte Fallback Rate (%) & Count
5. Encodes corpus into uint16 memmaps (train.bin, val.bin, test.bin) & generates `meta.json`.

Usage:
    python scripts/run_strict_pipeline_and_eval.py --lang hindi
    python scripts/run_strict_pipeline_and_eval.py --lang nepali
"""

import sys
import os
import io
import time
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import zstandard as zstd
import sentencepiece as spm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import shutil
from lmacorpus.cli import paths
from lmacorpus.io_utils import load_config, dump_json, read_dir, shard_paths, ShardWriter, read_shard
from lmacorpus.splits import build_splits, load_manifest
from lmacorpus.tokenizer.train_spm import write_training_sample, train_spm
from lmacorpus.tokenizer.encode_corpus import encode_all

def cap_dataset(lang: str, max_hf_docs: int = 1000000):
    lang_dir = ROOT_DIR / lang
    src_dir = lang_dir / "data" / "dedup_combined" if (lang_dir / "data" / "dedup_combined").exists() else lang_dir / "data" / "dedup"
    target_dir = lang_dir / "data" / "dedup_combined"
    capped_dir = lang_dir / "data" / "dedup_capped"
    
    if capped_dir.exists():
        shutil.rmtree(capped_dir)

    manual_out = capped_dir / "manual"
    hf_out = capped_dir / "downloaded"

    print(f"\n========================================================================================")
    print(f"✂️ CAPPING HF DATASET FOR: {lang.upper()}")
    print(f"   Input Directory   : {src_dir}")
    print(f"   Output Directory  : {capped_dir}")
    print(f"   Max HF Documents  : {max_hf_docs:,} (~400M tokens)")
    print(f"========================================================================================\n", flush=True)

    all_shards = shard_paths(src_dir)
    print(f"   Found {len(all_shards)} total shards in input directory.")

    manual_writer = ShardWriter(manual_out)
    hf_writer = ShardWriter(hf_out)

    m_docs = m_toks = 0
    h_docs = h_toks = 0

    t0 = time.time()

    for i, s_path in enumerate(all_shards, 1):
        for doc in read_shard(s_path):
            stype = doc.source_type
            text = doc.text
            if not text:
                continue

            doc_toks = int(len(text) / 3.15)

            if stype == "manual":
                manual_writer.add(doc)
                m_docs += 1
                m_toks += doc_toks
            else:
                if h_docs < max_hf_docs:
                    hf_writer.add(doc)
                    h_docs += 1
                    h_toks += doc_toks

        if i % 10 == 0:
            print(f"   -> Processed {i}/{len(all_shards)} shards | Manual Docs: {m_docs:,} ({m_toks/1e6:.1f}M toks) | HF Docs: {h_docs:,} ({h_toks/1e6:.1f}M toks)", flush=True)

    manual_writer.flush()
    hf_writer.flush()

    # Update target_dir with capped dataset
    print(f"\n   Updating {target_dir} with capped dataset...", flush=True)
    if src_dir == target_dir:
        backup_dir = lang_dir / "data" / "dedup_combined_uncapped_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(src_dir), str(backup_dir))

    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.move(str(capped_dir), str(target_dir))

    print(f"\n========================================================================================")
    print(f"🎉 DATASET CAPPING COMPLETE!")
    print(f"   Manual Webscraping : {m_docs:,} docs | ~{m_toks/1e6:.2f}M clean tokens (✅ 100% kept)")
    print(f"   HF Downloaded Data : {h_docs:,} docs | ~{h_toks/1e6:.2f}M clean tokens (✅ Capped ~400M target)")
    print(f"   Total Combined     : {m_docs + h_docs:,} docs | ~{(m_toks + h_toks)/1e6:.2f}M total tokens")
    print(f"   Total Time Taken   : {time.time() - t0:.1f} seconds")
    print(f"   Updated Path       : {target_dir}")
    print(f"========================================================================================\n")

def evaluate_on_manifest(shard_dir: Path, manifest_path: Path, model_path: Path):
    """Evaluates tokenizer on a specific held-out manifest (e.g. test.txt or val.txt)."""
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    ids_wanted = load_manifest(manifest_path)
    unk_id = sp.unk_id()

    tot_docs = 0
    tot_words = 0
    tot_chars = 0
    tot_tokens = 0
    tot_unk = 0
    tot_byte_fallback = 0

    manual_tokens = 0
    downloaded_tokens = 0

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

        if doc.source_type == "manual":
            manual_tokens += n_toks
        else:
            downloaded_tokens += n_toks

        if unk_id != -1:
            tot_unk += ids.count(unk_id)

        b_cnt = sum(1 for t in tok_strs if t.startswith("<0x") and t.endswith(">"))
        tot_byte_fallback += b_cnt

    fertility = tot_tokens / max(1, tot_words)
    compression = tot_chars / max(1, tot_tokens)
    unk_rate = (tot_unk / max(1, tot_tokens)) * 100.0
    byte_fallback_rate = (tot_byte_fallback / max(1, tot_tokens)) * 100.0
    manual_frac = manual_tokens / max(1, tot_tokens)

    return {
        "manifest": manifest_path.name,
        "docs": tot_docs,
        "words": tot_words,
        "chars": tot_chars,
        "tokens": tot_tokens,
        "manual_tokens": manual_tokens,
        "downloaded_tokens": downloaded_tokens,
        "manual_fraction": round(manual_frac, 4),
        "manual_requirement_satisfied": manual_frac >= 0.20,
        "fertility": round(fertility, 2),
        "compression": round(compression, 2),
        "unk_count": tot_unk,
        "unk_rate": round(unk_rate, 4),
        "byte_fallback_count": tot_byte_fallback,
        "byte_fallback_rate": round(byte_fallback_rate, 4)
    }

def run_strict_pipeline(lang: str, max_hf_docs: int = 1000000, val_frac: float = 0.10, test_frac: float = 0.10):
    t0 = time.time()
    p = paths(lang)

    print(f"\n========================================================================================")
    print(f"🚀 STRICT END-TO-END PIPELINE & HELD-OUT EVALUATION FOR: {lang.upper()}")
    print(f"========================================================================================\n", flush=True)

    # 1. Cap HF Downloaded data to ~400M clean tokens while preserving 100% manual webscraping tokens
    print(f"--- STEP 1: CAPPING HF DATASET TO ~400M TOKENS ---", flush=True)
    cap_dataset(lang, max_hf_docs=max_hf_docs)
    src_dir = p["dedup"]

    # 2. Build Group-Aware Splits (train 80%, val 10%, test 10%)
    print(f"\n--- STEP 2: BUILDING TRAIN / VAL / TEST SPLITS (80% / 10% / 10%) ---", flush=True)
    split_stats = build_splits(src_dir, p["splits"], val_frac=val_frac, test_frac=test_frac)
    dump_json(p["reports"] / "splits.json", split_stats)
    print(f"   -> Train Docs: {split_stats['train']['docs']:,} | Val Docs: {split_stats['val']['docs']:,} | Test Docs: {split_stats['test']['docs']:,}", flush=True)

    # 3. Train SentencePiece Tokenizer ONLY on TRAIN Split
    print(f"\n--- STEP 3: TRAINING SENTENCEPIECE TOKENIZER ONLY ON TRAIN SPLIT ---", flush=True)
    cfg = load_config(p["configs"] / "tokenizer.yaml")
    spm_input = p["tokenizer"] / "spm_input.txt"

    print(f"   Writing training text sample from train.txt manifest...", flush=True)
    n_sample_lines = write_training_sample(src_dir, p["splits"] / "train.txt", spm_input, max_lines=cfg.get("max_sample_lines", 8_000_000))
    print(f"   -> Wrote {n_sample_lines:,} training lines from TRAIN split.", flush=True)

    model_prefix_bpe = p["tokenizer"] / f"{cfg['lang_code']}_bpe_10000"
    model_prefix_unigram = p["tokenizer"] / f"{cfg['lang_code']}_unigram_10000"

    print(f"   Training BPE (V=10,000) on TRAIN split...", flush=True)
    bpe_model_path = train_spm(spm_input, model_prefix_bpe, vocab_size=10000, model_type="bpe")

    print(f"   Training Unigram (V=10,000) on TRAIN split...", flush=True)
    unigram_model_path = train_spm(spm_input, model_prefix_unigram, vocab_size=10000, model_type="unigram")

    # Copy primary unigram winner model to default spm.model path
    shutil_target = p["tokenizer"] / f"{cfg['lang_code']}_spm.model"
    import shutil
    shutil.copy2(unigram_model_path, shutil_target)

    # 4. Evaluate Tokenizers ONLY on HELD-OUT TEST Manifest (test.txt)
    print(f"\n--- STEP 4: EVALUATING TOKENIZERS ON HELD-OUT TEST SPLIT (test.txt) ---", flush=True)
    test_manifest = p["splits"] / "test.txt"
    val_manifest = p["splits"] / "val.txt"

    eval_bpe_test = evaluate_on_manifest(src_dir, test_manifest, bpe_model_path)
    eval_unigram_test = evaluate_on_manifest(src_dir, test_manifest, unigram_model_path)

    print(f"\n{'Model Algorithm':<20} | {'Tokens':<12} | {'Tokens/Word':<12} | {'Chars/Token':<12} | {'UNK Rate':<10} | {'Byte Fallback Rate':<18}")
    print(f"------------------------------------------------------------------------------------------------------------------------")
    print(f"{'BPE (10,000)':<20} | {eval_bpe_test['tokens']:<12,} | {eval_bpe_test['fertility']:<12.2f} | {eval_bpe_test['compression']:<12.2f} | {eval_bpe_test['unk_rate']:<9.4f}% | {eval_bpe_test['byte_fallback_rate']:<17.4f}% ({eval_bpe_test['byte_fallback_count']:,} toks)")
    print(f"{'Unigram (10,000)':<20} | {eval_unigram_test['tokens']:<12,} | {eval_unigram_test['fertility']:<12.2f} | {eval_unigram_test['compression']:<12.2f} | {eval_unigram_test['unk_rate']:<9.4f}% | {eval_unigram_test['byte_fallback_rate']:<17.4f}% ({eval_unigram_test['byte_fallback_count']:,} toks)")
    print(f"================================================================================----------------------------------------\n")

    # 5. Encode Binary Memmaps & Write meta.json using Unigram 10,000 winner
    print(f"--- STEP 5: ENCODING TRAIN / VAL / TEST MEMMAPS & GENERATING META.JSON (V=10,000) ---", flush=True)
    meta = encode_all(src_dir, p["splits"], unigram_model_path, p["tokens"], 10000)
    satisfied = meta["manual_requirement"]["satisfied"]
    print(f"   -> Manual Token Requirement (>=20% of Train): {'✅ SATISFIED' if satisfied else '❌ NOT SATISFIED'}", flush=True)

    # 6. Save Final Comprehensive Report
    final_report = {
        "language": lang,
        "split_docs": {k: v["docs"] for k, v in split_stats.items()},
        "test_eval_bpe": eval_bpe_test,
        "test_eval_unigram": eval_unigram_test,
        "meta_summary": meta
    }
    out_json = p["reports"] / "final_strict_evaluation_report.json"
    dump_json(out_json, final_report)

    elapsed = time.time() - t0
    print(f"\n========================================================================================")
    print(f"🎉 STRICT PIPELINE & EVALUATION COMPLETE FOR: {lang.upper()}")
    print(f"   Total Time           : {elapsed / 60:.1f} minutes")
    print(f"   Clean Shards Source  : {src_dir}")
    print(f"   Tokenizer Model      : {bpe_model_path}")
    print(f"   Encoded Tokens Bin   : {p['tokens']}")
    print(f"   Final Report Saved   : {out_json}")
    print(f"========================================================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Master pipeline runner and strict held-out evaluation.")
    parser.add_argument("--lang", choices=["hindi", "nepali"], required=True, help="Target language")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Validation split fraction (default: 0.10 for 10%%)")
    parser.add_argument("--test-frac", type=float, default=0.10, help="Test split fraction (default: 0.10 for 10%%)")
    args = parser.parse_args()

    max_hf_docs = 1000000 if args.lang == "hindi" else 980000
    run_strict_pipeline(args.lang, max_hf_docs=max_hf_docs, val_frac=args.val_frac, test_frac=args.test_frac)

if __name__ == "__main__":
    main()
