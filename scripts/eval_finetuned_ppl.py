"""High-Performance Intrinsic Perplexity (PPL) Evaluator for Fine-Tuned Model.

Optimizations:
1. Vectorized flat-binary chunking (eliminates slow Python list-comprehension file seeks).
2. torch.inference_mode() for zero-overhead tensor dispatch.
3. GPU-side asynchronous loss accumulation (eliminates CPU-GPU synchronization bottleneck).
4. Memory-optimized batch size (prevents Windows WDDM VRAM paging).

Usage:
    python scripts/eval_finetuned_ppl.py --lang hindi --split test
    python scripts/eval_finetuned_ppl.py --lang nepali --split test
    python scripts/eval_finetuned_ppl.py --lang both --split test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lmagpt.data import resolve_data_dir  # noqa: E402
from lmagpt.model import GPT, GPTConfig  # noqa: E402


def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


@torch.inference_mode()
def evaluate_fast(model, tokens_memmap, cfg, device, autocast_dtype, batch_size=32):
    """Ultra-fast sequential evaluation streaming contiguous tokens."""
    model.eval()
    T = cfg.context
    stride = T
    step_tokens = batch_size * stride

    # We need step_tokens + 1 for shifted targets
    total_tokens = len(tokens_memmap)
    n_steps = (total_tokens - 1) // step_tokens

    print(f"Total test tokens : {total_tokens:,} tokens (~{total_tokens / 1e6:.1f}M)")
    print(f"Batch size        : {batch_size} sequences of {T} context ({step_tokens:,} tokens/step)")
    print(f"Total steps       : {n_steps:,} batches\n", flush=True)

    gpu_loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    tokens_evaluated = 0

    t0 = time.time()
    last_print = t0

    # Stream contiguous chunks directly from memmap without Python list comprehension
    for step in range(n_steps):
        start_idx = step * step_tokens
        end_idx = start_idx + step_tokens + 1

        chunk = tokens_memmap[start_idx:end_idx]

        # Reshape directly: (batch_size, T)
        x_np = chunk[:-1].reshape(batch_size, T).astype(np.int64)
        y_np = chunk[1:].reshape(batch_size, T).astype(np.int64)

        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        y = torch.from_numpy(y_np).to(device, non_blocking=True)

        try:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                _, loss, _ = model(x, targets=y)
        except torch.cuda.OutOfMemoryError:
            print(f"\n[CUDA OOM] Batch size {batch_size} exceeded your 6 GB VRAM!")
            print(f"Please reduce to --batch-size 64 or --batch-size 32.")
            sys.exit(1)

        n = y.numel()
        gpu_loss_sum += loss.double() * n
        tokens_evaluated += n

        now = time.time()
        if now - last_print >= 5.0:  # live print every 5 seconds
            elapsed = now - t0
            speed = tokens_evaluated / elapsed
            pct = (tokens_evaluated / total_tokens) * 100
            current_loss = (gpu_loss_sum.item() / tokens_evaluated)
            current_ppl = math.exp(min(current_loss, 20))
            eta_sec = (total_tokens - tokens_evaluated) / max(speed, 1)

            print(f"  [{pct:5.1f}%] {tokens_evaluated:,}/{total_tokens:,} tokens | "
                  f"running PPL: {current_ppl:6.2f} | {speed:,.0f} tok/s | ETA: {eta_sec/60:.1f}m", flush=True)
            last_print = now

    elapsed = time.time() - t0
    final_loss = gpu_loss_sum.item() / max(tokens_evaluated, 1)
    final_ppl = math.exp(min(final_loss, 20))

    return {
        "tokens_evaluated": tokens_evaluated,
        "cross_entropy_nats": round(final_loss, 4),
        "perplexity": round(final_ppl, 3),
        "seconds": round(elapsed, 1),
        "throughput_tokens_sec": round(tokens_evaluated / max(elapsed, 1), 1),
    }


def evaluate_language(lang: str, split: str, batch_size: int, device, autocast_dtype, checkpoint_override: str | None):
    print(f"\n======================================================================")
    print(f" Evaluating Fine-Tuned Model: {lang.upper()}")
    print(f" Split: {split.upper()} (Complete Phase 1/2 Test Dataset)")
    print(f" Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"======================================================================")

    data_dir = resolve_data_dir(lang, ROOT)
    bin_path = data_dir / f"{split}.bin"
    if not bin_path.exists():
        print(f"Error: {bin_path} not found.")
        return

    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bytes_per_token = meta["splits"][split]["bytes_per_token"]

    ckpt_path = Path(checkpoint_override) if checkpoint_override else ROOT / lang / "reasoning" / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return

    print(f"Loading checkpoint: {ckpt_path.name}")
    model, cfg, ckpt = load_model(ckpt_path, device)
    print(f"Model parameters: {model.num_parameters():,} | Training step: {ckpt.get('step', '?')}\n")

    # Fast memory map
    tokens_memmap = np.memmap(bin_path, dtype=np.uint16, mode="r")

    res = evaluate_fast(model, tokens_memmap, cfg, device, autocast_dtype, batch_size=batch_size)

    # Calculate bits-per-byte
    bpb = (res["cross_entropy_nats"] / math.log(2)) / bytes_per_token
    res["bytes_per_token"] = round(bytes_per_token, 4)
    res["bits_per_byte"] = round(bpb, 4)
    res["compression_ratio_vs_raw"] = round(8.0 / bpb, 2)

    print(f"\n======================================================================")
    print(f" FINAL RESULTS: {lang.upper()} (Fine-Tuned Model on Whole Test Set)")
    print(f"======================================================================")
    print(f"  Total Tokens Scored   : {res['tokens_evaluated']:,}")
    print(f"  Cross-Entropy Loss    : {res['cross_entropy_nats']:.4f} nats/token")
    print(f"  Test Perplexity (PPL) : {res['perplexity']:.2f}")
    print(f"  Test Bits-per-Byte    : {res['bits_per_byte']:.4f} BPB")
    print(f"  Throughput            : {res['throughput_tokens_sec']:,.0f} tokens/sec")
    print(f"  Evaluation Time       : {res['seconds']/60:.2f} minutes ({res['seconds']}s)")
    print(f"======================================================================\n")

    out_file = ROOT / lang / "reasoning" / f"eval_test_ppl_finetuned.json"
    out_file.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="hindi", choices=["hindi", "nepali", "both"],
                        help="Language to evaluate (default: hindi)")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Split to evaluate on (default: test)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32 - optimal for 6GB RTX 4050)")
    parser.add_argument("--checkpoint", default=None,
                        help="Custom path to fine-tuned checkpoint")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif device.type == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    langs = ["hindi", "nepali"] if args.lang == "both" else [args.lang]
    for lang in langs:
        evaluate_language(lang, args.split, args.batch_size, device, autocast_dtype, args.checkpoint)


if __name__ == "__main__":
    main()
