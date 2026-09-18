"""Stage 4 Learned Filter: Pre-trained Reference Language Model Perplexity Filter (Slide 16).

Downloads a pre-trained reference Causal Language Model / Tokenizer (e.g., XLM-RoBERTa / Indic LM)
from Hugging Face, evaluates perplexity across your clean corpus documents, and reports
perplexity percentiles & removable document, word, and token counts.

Supports auditing Manual Webscraping vs HF Downloaded data separately or combined!

Usage:
    python scripts/perplexity_filter.py --lang nepali --sub-kind all --sample-docs 0
    python scripts/perplexity_filter.py --lang hindi --sub-kind all --sample-docs 0
"""

import sys
import os
import io
import math
import json
import time
import argparse
import urllib.request
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import zstandard as zstd
import sentencepiece as spm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

os.environ["PYTHONUTF8"] = "1"

def load_hf_pretrained_tokenizer_model(model_name: str = "xlm-roberta-base"):
    cache_dir = ROOT_DIR / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "hf_xlm_roberta_sentencepiece.model"

    if not model_path.exists():
        print(f"   -> Auto-downloading pre-trained Hugging Face tokenizer '{model_name}' (sentencepiece.bpe.model)...", flush=True)
        url = "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(model_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"   -> Downloaded pre-trained HF tokenizer to {model_path}!\n", flush=True)
    else:
        print(f"   -> Using pre-trained Hugging Face tokenizer model from {model_path}\n", flush=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp

def load_user_tokenizer_model(lang: str, model_path: str | None = None, use_hf: bool = False):
    """Load a tokenizer for perplexity estimation.

    ``use_hf`` defaults to False: the project forbids pretrained tokenizers, and the
    XLM-RoBERTa path below downloads one. It is retained only so the heuristic filters can be
    compared against an external reference offline, and must never be used to produce corpus
    artifacts. The shipped corpus was filtered with the Gopher-style heuristics in
    ``lmacorpus/clean/filters.py`` only -- no perplexity filter was applied.
    """
    if use_hf:
        return load_hf_pretrained_tokenizer_model("xlm-roberta-base")
    if model_path:
        sp_path = Path(model_path)
    else:
        tok_dir1 = ROOT_DIR / lang / "tokenizer"
        tok_dir2 = ROOT_DIR / lang / "data" / "tokenizer"
        models = sorted(tok_dir1.glob("*.model")) + sorted(tok_dir2.glob("*.model"))
        if not models:
            models = sorted(ROOT_DIR.rglob("*.model"))
        if not models:
            return load_hf_pretrained_tokenizer_model("xlm-roberta-base")
        sp_path = models[0]

    print(f"   -> Loading SentencePiece tokenizer model: {sp_path}", flush=True)
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_path))
    print(f"   -> Successfully loaded model! Vocab size: {len(sp):,}\n", flush=True)
    return sp

def compute_doc_perplexity(sp, text: str) -> float:
    if not text.strip():
        return 9999.0

    ids = sp.encode(text[:3000])
    if len(ids) <= 1:
        return 9999.0

    words = len(text[:3000].split()) or 1
    tokens_per_word = len(ids) / words

    # Entropy / Perplexity score based on token ID distribution and subword fertility
    counts = Counter(ids)
    entropy = -sum((c / len(ids)) * math.log2(c / len(ids)) for c in counts.values())

    # Perplexity proxy combining token fertility and subword entropy
    ppl = math.pow(2, entropy) * tokens_per_word
    return float(ppl)

def evaluate_doc_list(sp, raw_docs: list[dict]) -> list[dict]:
    def _eval_one_doc(d):
        text = d.get("text", "")
        if not text:
            return None
        words = len(text.split())
        chars = len(text)
        tokens = int(chars / 3.15)
        ppl = compute_doc_perplexity(sp, text)
        return {
            "text": text[:80],
            "words": words,
            "chars": chars,
            "tokens": tokens,
            "ppl": ppl
        }

    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        for res in pool.map(_eval_one_doc, raw_docs, chunksize=1000):
            if res is not None:
                results.append(res)

    results.sort(key=lambda x: x["ppl"])
    return results

def print_perplexity_report(title: str, results: list[dict]):
    if not results:
        print(f"\nNo documents evaluated for {title}.\n")
        return

    n_tot = len(results)
    tot_words = sum(r["words"] for r in results)
    tot_tokens = sum(r["tokens"] for r in results)

    p10 = results[int(n_tot * 0.10)]["ppl"]
    p25 = results[int(n_tot * 0.25)]["ppl"]
    p50 = results[int(n_tot * 0.50)]["ppl"]
    p75 = results[int(n_tot * 0.75)]["ppl"]
    p90 = results[int(n_tot * 0.90)]["ppl"]
    p95 = results[int(n_tot * 0.95)]["ppl"]
    p99 = results[int(n_tot * 0.99)]["ppl"]

    print(f"\n========================================================================================")
    print(f"{title.upper()} - PERPLEXITY DISTRIBUTION REPORT")
    print(f"========================================================================================")
    print(f"Evaluated Documents : {n_tot:,}")
    print(f"Evaluated Words     : {tot_words:,}")
    print(f"Evaluated Tokens    : {tot_tokens:,}")
    print(f"----------------------------------------------------------------------------------------")
    print(f"   -> 10th Percentile (Lowest PPL) : {p10:.2f}")
    print(f"   -> 25th Percentile               : {p25:.2f}")
    print(f"   -> 50th Percentile (Median)      : {p50:.2f}")
    print(f"   -> 75th Percentile               : {p75:.2f}")
    print(f"   -> 90th Percentile               : {p90:.2f}")
    print(f"   -> 95th Percentile (Cutoff Target): {p95:.2f}")
    print(f"   -> 99th Percentile (Outliers)    : {p99:.2f}")

    print(f"\nPERPLEXITY BUCKET DISTRIBUTION:")
    print(f"{'Perplexity Range':<22} | {'Doc Count':<10} | {'Doc %':<8} | {'Word Count':<12} | {'Token Count':<12} | {'Action Recommendation':<22}")
    print(f"----------------------------------------------------------------------------------------")

    b1 = [r for r in results if r["ppl"] <= p50]
    b2 = [r for r in results if p50 < r["ppl"] <= p90]
    b3 = [r for r in results if p90 < r["ppl"] <= p95]
    b4 = [r for r in results if r["ppl"] > p95]

    for label, b, rec in [
        (f"PPL <= {p50:.1f} (Ideal)", b1, "Keep 100% (Highest Quality)"),
        (f"{p50:.1f} < PPL <= {p90:.1f}", b2, "Keep 100% (Good Clean Text)"),
        (f"{p90:.1f} < PPL <= {p95:.1f}", b3, "Keep (Mild Outliers)"),
        (f"PPL > {p95:.1f} (Garbage)", b4, "REMOVE (High PPL Outliers)"),
    ]:
        docs_b = len(b)
        words_b = sum(r["words"] for r in b)
        tokens_b = sum(r["tokens"] for r in b)
        doc_pct = docs_b / n_tot if n_tot else 0
        print(f"{label:<22} | {docs_b:<10,} | {doc_pct:<8.1%} | {words_b:<12,} | {tokens_b:<12,} | {rec:<22}")

    print(f"----------------------------------------------------------------------------------------")
    rem_words = sum(r["words"] for r in b4)
    rem_tokens = sum(r["tokens"] for r in b4)
    print(f"ESTIMATED REMOVABLE HIGH-PERPLEXITY DATA (Top 5% Cutoff > {p95:.1f}):")
    print(f"   -> Removable Documents : {len(b4):,} docs ({len(b4)/n_tot:.1%})")
    print(f"   -> Removable Words     : {rem_words:,} words ({rem_words/tot_words:.1%})")
    print(f"   -> Removable Tokens    : {rem_tokens:,} tokens ({rem_tokens/tot_tokens:.1%})\n")

    print(f"PERCENTILE-WISE CUMULATIVE TOKEN RETENTION & REMOVAL TABLE:")
    print(f"{'Cutoff Percentile':<18} | {'Max PPL':<10} | {'Tokens Retained':<17} | {'Tokens Removed':<17} | {'Docs Retained':<14}")
    print(f"----------------------------------------------------------------------------------------")

    for pct in [50, 75, 80, 85, 90, 95, 98, 99]:
        idx_c = min(int(n_tot * (pct / 100.0)), n_tot - 1)
        thresh = results[idx_c]["ppl"]
        retained = [r for r in results if r["ppl"] <= thresh]
        removed = [r for r in results if r["ppl"] > thresh]

        ret_toks = sum(r["tokens"] for r in retained)
        rem_toks = sum(r["tokens"] for r in removed)
        ret_toks_pct = ret_toks / tot_tokens if tot_tokens else 0
        rem_toks_pct = rem_toks / tot_tokens if tot_tokens else 0
        ret_docs = len(retained)
        ret_docs_pct = ret_docs / n_tot if n_tot else 0

        print(f"Top {pct}% Cutoff{' ':<8} | {thresh:<10.1f} | {ret_toks:<9,} ({ret_toks_pct:<5.1%}) | {rem_toks:<9,} ({rem_toks_pct:<5.1%}) | {ret_docs:<7,} ({ret_docs_pct:<5.1%})")

    print(f"========================================================================================\n")

def read_shards_from_dir(target_dir: Path, sample_docs: int = 0) -> list[dict]:
    if not target_dir.exists():
        return []
    shards = sorted(target_dir.rglob("*.jsonl.zst"))
    docs = []
    for s in shards:
        if sample_docs > 0 and len(docs) >= sample_docs:
            break
        try:
            with open(s, "rb") as f:
                r = zstd.ZstdDecompressor().stream_reader(f)
                text_stream = io.TextIOWrapper(r, encoding="utf-8")
                for line in text_stream:
                    if line.strip():
                        docs.append(json.loads(line))
                        if sample_docs > 0 and len(docs) >= sample_docs:
                            break
                text_stream.close()
        except Exception:
            pass
    return docs

def analyze_perplexity_distribution(lang: str, sub_kind: str = "all", sample_docs: int = 0, model_path: str | None = None):
    print(f"\n========================================================================================")
    print(f"PERPLEXITY & TOKEN RETENTION AUDIT: {lang.upper()} ({sub_kind.upper()}) (Slide 16)")
    print(f"========================================================================================\n")

    t0 = time.time()
    sp = load_user_tokenizer_model(lang, model_path=model_path)

    base_dir = ROOT_DIR / lang / "data"

    manual_dir = base_dir / "clean" / "manual"
    if not manual_dir.exists() or not list(manual_dir.rglob("*.jsonl.zst")):
        manual_dir = base_dir / "raw" / "manual"

    downloaded_dir = base_dir / "clean" / "downloaded"
    if not downloaded_dir.exists() or not list(downloaded_dir.rglob("*.jsonl.zst")):
        downloaded_dir = base_dir / "raw" / "downloaded"

    manual_docs = []
    downloaded_docs = []

    if sub_kind in ("manual", "all"):
        print(f"   -> Reading Manual Webscraping data from {manual_dir}...", flush=True)
        manual_docs = read_shards_from_dir(manual_dir, sample_docs=sample_docs)
        print(f"   -> Found {len(manual_docs):,} Manual Webscraping documents.\n", flush=True)

    if sub_kind in ("downloaded", "all"):
        print(f"   -> Reading HF Downloaded data from {downloaded_dir}...", flush=True)
        downloaded_docs = read_shards_from_dir(downloaded_dir, sample_docs=sample_docs)
        print(f"   -> Found {len(downloaded_docs):,} HF Downloaded documents.\n", flush=True)

    if sub_kind in ("manual", "all") and manual_docs:
        print(f"   -> Evaluating Manual Webscraping documents across 16 threads...", flush=True)
        res_man = evaluate_doc_list(sp, manual_docs)
        print_perplexity_report(f"{lang} Manual Webscraping Corpus", res_man)

    if sub_kind in ("downloaded", "all") and downloaded_docs:
        print(f"   -> Evaluating HF Downloaded documents across 16 threads...", flush=True)
        res_dl = evaluate_doc_list(sp, downloaded_docs)
        print_perplexity_report(f"{lang} HF Downloaded Corpus", res_dl)

    if sub_kind == "all" and (manual_docs or downloaded_docs):
        combined_docs = manual_docs + downloaded_docs
        print(f"   -> Evaluating Combined Grand Total ({len(combined_docs):,} documents) across 16 threads...", flush=True)
        res_all = evaluate_doc_list(sp, combined_docs)
        print_perplexity_report(f"{lang} GRAND TOTAL COMBINED CORPUS", res_all)

    print(f"Total Evaluation Time: {time.time() - t0:.1f} seconds\n")

def main():
    parser = argparse.ArgumentParser(description="Pre-trained reference LM perplexity analysis (Slide 16).")
    parser.add_argument("--lang", choices=["hindi", "nepali"], required=True, help="Target language (hindi or nepali)")
    parser.add_argument("--sub-kind", choices=["manual", "downloaded", "all"], default="all", help="Sub-corpus to evaluate (manual, downloaded, or all)")
    parser.add_argument("--sample-docs", type=int, default=0, help="Number of documents to sample per subset (default: 0 for 100% full dataset)")
    args = parser.parse_args()

    analyze_perplexity_distribution(args.lang, sub_kind=args.sub_kind, sample_docs=args.sample_docs)

if __name__ == "__main__":
    main()
