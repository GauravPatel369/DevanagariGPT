"""Evaluate sweep checkpoints (5k, 10k, etc.) and produce the complete 5k-100k report."""

import json
from pathlib import Path
import torch

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lmagpt.data import resolve_tokenizer
from lmagpt.eval_reasoning import evaluate, load
from lmagpt.model import GPTConfig, GPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None

lang = "hindi"
data_dir = ROOT / lang / "reasoning" / "data"
sweep_dir = ROOT / lang / "reasoning" / "sweep"
out_json = ROOT / lang / "reasoning" / "sample_sweep_results.json"

import sentencepiece as spm
tok_path = resolve_tokenizer(lang, ROOT)
sp = spm.SentencePieceProcessor(model_file=str(tok_path))

test_sets = {}
for split in ["test_a", "test_b", "test_c"]:
    p = data_dir / f"{split}.jsonl"
    test_sets[split] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

results = {}
if out_json.exists():
    try:
        results = json.loads(out_json.read_text(encoding="utf-8"))
    except Exception:
        results = {}

for size_str in ["5k", "10k", "20k", "40k", "80k", "100k"]:
    if size_str in results and "test_a_acc" in results[size_str]:
        print(f"[{size_str}] Already evaluated: Test A = {results[size_str]['test_a_acc']:.2%}")
        continue

    ckpt_file = sweep_dir / f"model_{size_str}.pt"
    if not ckpt_file.exists():
        print(f"[{size_str}] {ckpt_file} not found.")
        continue

    print(f"\n[{size_str}] Loading {ckpt_file.name}...")
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    eval_res = {}
    for split, rows in test_sets.items():
        res = evaluate(model, sp, rows, device, autocast_dtype, limit=None)
        eval_res[split] = res
        print(f"  -> {split.upper()} Acc: {res['accuracy']:.2%} (Base: {res['majority_position_baseline']:.2%})")

    n_samples = int(size_str.replace("k", "")) * 1000
    results[size_str] = {
        "n_samples": n_samples,
        "best_step": ckpt.get("best_step", 0),
        "best_val_loss": round(ckpt.get("best_val", 0.0), 4),
        "test_a_acc": eval_res["test_a"]["accuracy"],
        "test_b_acc": eval_res["test_b"]["accuracy"],
        "test_c_acc": eval_res["test_c"]["accuracy"],
        "test_a_hops": eval_res["test_a"]["by_hops"],
    }

out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n" + "="*85)
print(f" COMPLETE SCALING CURVE: 5K TO 100K ({lang.upper()})")
print("="*85)
print(f"{'Sample Size':<12} | {'Val Loss':<10} | {'Test A (Seen)':<15} | {'Test B (Unseen)':<16} | {'Test C':<10} | {'2-Hop Acc':<10}")
print("-"*85)
for k in ["5k", "10k", "20k", "40k", "80k", "100k"]:
    if k in results:
        v = results[k]
        two_hop = v.get("test_a_hops", {}).get("2", 0.0)
        print(f"{k:<12} | {v.get('best_val_loss', 0.0):<10.4f} | {v['test_a_acc']:<15.2%} | {v['test_b_acc']:<16.2%} | {v['test_c_acc']:<10.2%} | {two_hop:<10.2%}")
print("="*85)
