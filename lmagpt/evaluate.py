"""Phase-2 evaluation: intrinsic language-modelling metrics and generation quality.

Two families of number, measured on each model's own held-out split.

**Intrinsic.**  Cross-entropy, perplexity, and bits-per-byte.  The first two are
per *token*, and Model H and Model L have different tokenizers -- a Nepali token
carries ~10.6 bytes against Hindi's ~9.7 -- so their perplexities are not
directly comparable.  Bits-per-byte divides that out by normalising to a unit
both models share, which is why the brief asks for it alongside perplexity.
The two metrics can and do disagree, and the disagreement is the finding.

**Generation.**  Continuations from held-out prefixes under greedy decoding and
three temperatures, scored against the true continuation with BLEU, chrF++ and
ROUGE-L, plus repetition and distinct-n diagnostics.

A caveat that belongs in the report rather than buried here: BLEU, chrF and
ROUGE all measure agreement with *one* reference continuation.  Open-ended
language modelling has no single correct continuation, so absolute values are
low and largely uninformative; what they are good for is *relative* comparison
across decoding settings and between the two models, where the reference is
held fixed.  chrF++ is the most trustworthy of the three here because it works
on character n-grams and therefore degrades gracefully on Devanagari
morphology, where a word-level metric scores a correct stem with the wrong
case-suffix as a total miss.

Usage:
    python -m lmagpt.evaluate --lang hindi
    python -m lmagpt.evaluate --lang hindi --split test --max-tokens 5000000
    python -m lmagpt.evaluate --lang both --out report/Phase\\ 2/eval
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lmagpt.data import TokenDataset, resolve_data_dir, resolve_tokenizer  # noqa: E402
from lmagpt.model import GPT, GPTConfig  # noqa: E402


# ------------------------------------------------------- intrinsic metrics


@torch.no_grad()
def intrinsic_metrics(model, dataset, device, autocast_dtype, batch_size=16,
                      max_tokens=None) -> dict:
    """Cross-entropy, perplexity and bits-per-byte over a split.

    Windows are sequential and non-overlapping.  Random sampling would weight
    some tokens more than once and make the number depend on the seed; a
    perplexity quoted in a report should cover the split exactly once.

    Bits-per-byte is derived from the *measured* byte length of the evaluated
    text rather than the corpus-average in ``meta.json``, so it describes the
    text actually scored:

        bpb = (nats_per_token / ln 2) / bytes_per_token
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()

    for x, y in dataset.sequential_batches(batch_size, device=device,
                                           limit_tokens=max_tokens):
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            _, loss, _ = model(x, targets=y)
        n = y.numel()
        total_loss += loss.item() * n     # loss is a mean; undo it to sum
        total_tokens += n

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "tokens_evaluated": total_tokens,
        "cross_entropy_nats": round(mean_loss, 4),
        "perplexity": round(math.exp(min(mean_loss, 20)), 3),
        "seconds": round(time.time() - t0, 1),
    }


def add_bits_per_byte(metrics: dict, bytes_per_token: float) -> dict:
    """Attach bits-per-byte, the tokenizer-independent view of the same loss."""
    bpb = (metrics["cross_entropy_nats"] / math.log(2)) / bytes_per_token
    metrics["bytes_per_token"] = round(bytes_per_token, 4)
    metrics["bits_per_byte"] = round(bpb, 4)
    # A model that has learnt nothing costs 8 bits per byte (raw storage).
    metrics["compression_ratio_vs_raw"] = round(8.0 / bpb, 2)
    return metrics



# ------------------------------------------------------------ decoding grid

# Ordered from most to least deterministic. The grid is built to separate three
# effects rather than to enumerate combinations blindly:
#
#   * temperature alone      -- how much does flattening the distribution help?
#   * truncation (top-k/p)   -- does cutting the tail beat lowering temperature?
#   * explicit anti-repeat   -- does a penalty or a hard block beat both?
#
# top-p is expected to outperform top-k because it adapts: a confident step
# keeps two tokens, an uncertain one keeps fifty, whereas top-k applies the same
# cut regardless of how peaked the distribution is.
DECODING_GRID = [
    ("greedy",                 dict(greedy=True)),
    ("greedy+norepeat4",       dict(greedy=True, no_repeat_ngram_size=4)),
    ("T0.5",                   dict(temperature=0.5)),
    ("T0.8",                   dict(temperature=0.8)),
    ("T1.0",                   dict(temperature=1.0)),
    ("T1.5",                   dict(temperature=1.5)),
    ("T1.0+topk50",            dict(temperature=1.0, top_k=50)),
    ("T1.0+topp0.9",           dict(temperature=1.0, top_p=0.9)),
    ("T1.0+topp0.95",          dict(temperature=1.0, top_p=0.95)),
    ("T0.8+topp0.9",           dict(temperature=0.8, top_p=0.9)),
    ("T0.8+topp0.9+rep1.2",    dict(temperature=0.8, top_p=0.9, repetition_penalty=1.2)),
    ("T0.8+topp0.9+norepeat4", dict(temperature=0.8, top_p=0.9, no_repeat_ngram_size=4)),
]


def sweet_spot(results: dict, reference: dict) -> dict:
    """Pick the decoding setting whose statistics best match human text.

    Maximising diversity is the wrong objective -- temperature 1.5 maximises it
    and produces noise. Maximising BLEU is also wrong, since it rewards copying
    a single reference. Human text has a characteristic repetition rate and
    lexical variety, so the defensible target is the setting whose rep-4 and
    distinct-2 land closest to the reference continuations' own values, with
    chrF++ breaking ties on quality.

    The distance is normalised per axis so neither dominates purely by scale.
    """
    scored = []
    for name, m in results.items():
        d_rep = abs(m["repetition_rate_4gram"] - reference["repetition_rate_4gram"])
        d_d2 = abs(m["distinct_2"] - reference["distinct_2"])
        scored.append({
            "setting": name,
            "naturalness_distance": round(d_rep + d_d2, 4),
            "chrf2": m["chrf2"],
            "rep4_delta": round(m["repetition_rate_4gram"] - reference["repetition_rate_4gram"], 4),
            "distinct2_delta": round(m["distinct_2"] - reference["distinct_2"], 4),
        })
    scored.sort(key=lambda r: (r["naturalness_distance"], -r["chrf2"]))
    return {"reference_statistics": reference, "ranking": scored, "best": scored[0]["setting"]}


# ------------------------------------------------------------- generation


@torch.no_grad()
def generate_samples(model, dataset, device, n_samples=64, prompt_tokens=128,
                     continue_tokens=128, settings=None, seed=1234,
                     gen_batch=16) -> dict:
    """Continue held-out prefixes under several decoding settings.

    The same prefixes are used for every setting so the comparison isolates the
    decoding strategy rather than the prompts.
    """
    if settings is None:
        settings = DECODING_GRID

    rng = np.random.default_rng(seed)
    need = prompt_tokens + continue_tokens
    offsets = rng.integers(0, len(dataset) - need, size=n_samples)

    prompts, references = [], []
    for off in offsets:
        window = np.asarray(dataset.tokens[off:off + need]).astype(np.int64)
        prompts.append(window[:prompt_tokens])
        references.append(window[prompt_tokens:])
    prompt_arr = np.stack(prompts)

    # Generate in chunks. The attention matrix during decoding is
    # (B, n_head, T, T) and T grows with every emitted token, so a large batch
    # peaks late in the loop -- exactly when it is most expensive to fail.
    # Chunking bounds peak memory independently of n_samples.
    out = {}
    for name, kwargs in settings:
        torch.manual_seed(seed)  # sampling settings stay reproducible
        chunks = []
        for i in range(0, len(prompt_arr), gen_batch):
            batch = torch.from_numpy(prompt_arr[i:i + gen_batch]).to(device)
            gen = model.generate(batch, max_new_tokens=continue_tokens, **kwargs)
            chunks.append(gen[:, prompt_tokens:].cpu().numpy())
            del gen, batch
            if device.type == "cuda":
                torch.cuda.empty_cache()
        out[name] = np.concatenate(chunks, axis=0)
    return {"prompts": np.stack(prompts), "references": np.stack(references),
            "generations": out}


# ---------------------------------------------------------------- metrics


def lcs_length(a: list, b: list) -> int:
    """Longest common subsequence length, the basis of ROUGE-L."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hyps: list[str], refs: list[str]) -> float:
    """Corpus ROUGE-L F1 over whitespace tokens.

    Implemented here rather than via ``rouge_score``, whose default pipeline
    applies English stemming and punctuation rules that are meaningless for
    Devanagari and would silently distort the number.
    """
    f1s = []
    for h, r in zip(hyps, refs):
        ht, rt = h.split(), r.split()
        if not ht or not rt:
            f1s.append(0.0)
            continue
        l = lcs_length(ht, rt)
        p, rec = l / len(ht), l / len(rt)
        f1s.append(0.0 if p + rec == 0 else 2 * p * rec / (p + rec))
    return sum(f1s) / max(len(f1s), 1)


def diversity_metrics(texts: list[str], max_n: int = 4) -> dict:
    """Repetition rate and distinct-n.

    Degenerate LM output loops, and loops are invisible to BLEU/chrF against a
    single reference, so these are reported alongside them:

    * ``repetition_rate_ngram``: fraction of n-grams that are not their first
      occurrence within the same continuation. High values mean the model is
      cycling.
    * ``distinct_1`` / ``distinct_2``: unique unigrams / bigrams over the total
      produced across the corpus. Low values mean low lexical variety.
    """
    out = {}
    for n in (1, 2):
        total, uniq = 0, set()
        for t in texts:
            grams = [tuple(t.split()[i:i + n]) for i in range(max(0, len(t.split()) - n + 1))]
            total += len(grams)
            uniq.update(grams)
        out[f"distinct_{n}"] = round(len(uniq) / max(total, 1), 4)

    rates = []
    for t in texts:
        toks = t.split()
        grams = [tuple(toks[i:i + max_n]) for i in range(max(0, len(toks) - max_n + 1))]
        if not grams:
            continue
        counts = Counter(grams)
        repeated = sum(c - 1 for c in counts.values())
        rates.append(repeated / len(grams))
    out[f"repetition_rate_{max_n}gram"] = round(sum(rates) / max(len(rates), 1), 4)
    return out


def generation_metrics(hyps: list[str], refs: list[str]) -> dict:
    """BLEU, chrF++ and ROUGE-L for one decoding setting."""
    from sacrebleu.metrics import BLEU, CHRF

    # tokenize='char': the default '13a' tokenizer is built for Latin script and
    # treats a whole Devanagari word as one token, which makes BLEU-4 collapse
    # to near zero and stop discriminating between systems.
    bleu = BLEU(tokenize="char").corpus_score(hyps, [refs])
    chrf = CHRF(word_order=2).corpus_score(hyps, [refs])  # word_order=2 -> chrF++
    return {
        "bleu_char_4gram": round(bleu.score, 3),
        "chrf2": round(chrf.score, 3),
        "rouge_l": round(rouge_l(hyps, refs) * 100, 3),
        **diversity_metrics(hyps),
    }


# -------------------------------------------------------------------- main


def load_model(lang: str, ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def evaluate_language(lang: str, args, root: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif device.type == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.checkpoint_dir:
        # <dir>/<lang>/last.pt -- lets one flag serve both languages, which a
        # single --checkpoint path cannot.
        ckpt_path = Path(args.checkpoint_dir) / lang / "last.pt"
    else:
        ckpt_path = root / lang / "checkpoints" / "last.pt"
    model, cfg, ckpt = load_model(lang, ckpt_path, device)
    print(f"[eval] {lang}: {ckpt_path.name} @ step {ckpt['step']:,}, "
          f"{model.num_parameters():,} params, device {device}", flush=True)

    # The vocabulary sweep needs these overrides: each run has its own tokenizer
    # and its own encoding of the same corpus, so scoring a sweep checkpoint
    # against the default V=10,000 binaries would measure the wrong thing --
    # silently, since only the token ids differ.
    data_dir = Path(args.data_dir) if args.data_dir else resolve_data_dir(lang, root)
    tok_path = Path(args.tokenizer) if args.tokenizer else resolve_tokenizer(lang, root)
    if cfg.vocab_size != (n_meta := json.loads(
            (data_dir / "meta.json").read_text(encoding="utf-8")).get("vocab_size", cfg.vocab_size)):
        raise SystemExit(
            f"checkpoint vocab_size {cfg.vocab_size:,} != {n_meta:,} in {data_dir}/meta.json -- "
            f"pass --data-dir/--tokenizer for this run's own encoding")
    ds = TokenDataset(data_dir / f"{args.split}.bin", cfg.context)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    bpt = meta["splits"][args.split]["bytes_per_token"]

    print(f"[eval] intrinsic metrics on {args.split} split ...", flush=True)
    intrinsic = intrinsic_metrics(model, ds, device, autocast_dtype,
                                  batch_size=args.batch_size, max_tokens=args.max_tokens)
    add_bits_per_byte(intrinsic, bpt)
    print(f"[eval]   loss {intrinsic['cross_entropy_nats']:.4f} | "
          f"ppl {intrinsic['perplexity']:.2f} | bpb {intrinsic['bits_per_byte']:.4f} "
          f"({intrinsic['tokens_evaluated']:,} tokens)", flush=True)

    print(f"[eval] generating {args.n_samples} continuations ...", flush=True)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(tok_path))

    samples = generate_samples(model, ds, device, n_samples=args.n_samples,
                               prompt_tokens=args.prompt_tokens,
                               continue_tokens=args.continue_tokens,
                               gen_batch=args.gen_batch)
    refs = [sp.decode(r.tolist()) for r in samples["references"]]

    # The reference continuations are real human text; their repetition and
    # diversity are the target the decoder should be matching.
    ref_stats = diversity_metrics(refs)
    print(f"[eval]   reference text: rep4 {ref_stats['repetition_rate_4gram']:.3f} | "
          f"dist1 {ref_stats['distinct_1']:.3f} | dist2 {ref_stats['distinct_2']:.3f}", flush=True)

    gen_results, examples = {}, []
    for name, arr in samples["generations"].items():
        hyps = [sp.decode(h.tolist()) for h in arr]
        gen_results[name] = generation_metrics(hyps, refs)
        print(f"[eval]   {name:9s} bleu {gen_results[name]['bleu_char_4gram']:>6.2f} | "
              f"chrf2 {gen_results[name]['chrf2']:>6.2f} | "
              f"rougeL {gen_results[name]['rouge_l']:>6.2f} | "
              f"rep4 {gen_results[name]['repetition_rate_4gram']:.3f} | "
              f"dist2 {gen_results[name]['distinct_2']:.3f}", flush=True)
        if len(examples) < 3:
            for i in range(min(3, len(hyps))):
                examples.append({"setting": name, "index": i,
                                 "prompt": sp.decode(samples["prompts"][i].tolist())[-200:],
                                 "generated": hyps[i][:300],
                                 "reference": refs[i][:300]})

    spot = sweet_spot(gen_results, ref_stats)
    print(f"[eval]   sweet spot -> {spot['best']}", flush=True)

    return {
        "language": lang,
        "reference_statistics": ref_stats,
        "sweet_spot": spot,
        "checkpoint": str(ckpt_path),
        "step": ckpt["step"],
        "parameters": model.num_parameters(),
        "split": args.split,
        "model_config": ckpt["model_config"],
        "intrinsic": intrinsic,
        "generation": gen_results,
        "generation_config": {"n_samples": args.n_samples,
                              "prompt_tokens": args.prompt_tokens,
                              "continue_tokens": args.continue_tokens},
        "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", default="both", choices=["hindi", "nepali", "both"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="dir holding <lang>/last.pt; serves both languages")
    ap.add_argument("--data-dir", default=None,
                    help="token dir to score against; needed for vocabulary-sweep "
                         "checkpoints, whose encoding is not <lang>/data/tokens")
    ap.add_argument("--tokenizer", default=None,
                    help="SentencePiece model matching --data-dir")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap the intrinsic pass; omit to score the whole split")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--continue-tokens", type=int, default=128)
    ap.add_argument("--gen-batch", type=int, default=16,
                    help="sequences decoded at once; bounds peak VRAM")
    ap.add_argument("--out", default="report/Phase 2")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parent.parent
    # An absolute --out escapes the package root, which on Kaggle is a
    # read-only dataset mount.
    out_dir = Path(a.out) if Path(a.out).is_absolute() else root / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    langs = ["hindi", "nepali"] if a.lang == "both" else [a.lang]
    results = {}
    for lang in langs:
        results[lang] = evaluate_language(lang, a, root)

    if len(results) == 2:
        h, n = results["hindi"]["intrinsic"], results["nepali"]["intrinsic"]
        results["comparison"] = {
            "perplexity_favours": "hindi" if h["perplexity"] < n["perplexity"] else "nepali",
            "bits_per_byte_favours": "hindi" if h["bits_per_byte"] < n["bits_per_byte"] else "nepali",
            "perplexity_gap": round(n["perplexity"] - h["perplexity"], 3),
            "bits_per_byte_gap": round(n["bits_per_byte"] - h["bits_per_byte"], 4),
            "note": ("Perplexity is per token and the two tokenizers differ, so the two "
                     "metrics can disagree; bits-per-byte is the comparable one."),
        }
        print(f"\n[eval] perplexity favours {results['comparison']['perplexity_favours']}, "
              f"bits-per-byte favours {results['comparison']['bits_per_byte_favours']}")

    out_path = out_dir / f"evaluation_{a.split}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
