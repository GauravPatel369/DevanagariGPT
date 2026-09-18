"""Sample Efficiency Sweep: Train and Evaluate models across custom sample sizes (e.g. 5k, 10k, 20k).

Merges results with any existing sweep data to provide a comprehensive scaling curve.

Usage:
    python scripts/sample_sweep.py --lang hindi --sizes 5000 10000 20000
    python scripts/sample_sweep.py --lang hindi --sizes 5000 10000 20000 --test-limit 1000
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

from lmagpt.data import resolve_tokenizer  # noqa: E402
from lmagpt.eval_reasoning import evaluate, load  # noqa: E402
from lmagpt.finetune import ReasoningDataset, validation_loss  # noqa: E402
from lmagpt.train import build_optimizer  # noqa: E402


def run_sweep(lang: str, sample_sizes: list[int], batch_size: int, lr: float, device, autocast_dtype, test_limit: int | None = None):
    print("=" * 75)
    print(f" SAMPLE EFFICIENCY SWEEP: {lang.upper()}")
    print(f" Target Sample Sizes: {sample_sizes}")
    if test_limit:
        print(f" Fast Evaluation    : Capped at {test_limit:,} items per test set")
    print(f" Device             : {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print("=" * 75)

    data_dir = ROOT / lang / "reasoning" / "data"
    tok_path = resolve_tokenizer(lang, ROOT)

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(tok_path))

    # Load full dataset once into memory
    print(f"\n[data] Loading full train and val datasets into memory...")
    full_train_ds = ReasoningDataset(data_dir / "train.jsonl", sp, max_len=128)
    val_ds = ReasoningDataset(data_dir / "val.jsonl", sp, max_len=128)
    print(f"[data] Total train examples available: {len(full_train_ds):,}")
    print(f"[data] Validation examples: {len(val_ds):,}")

    # Load test sets for evaluation
    test_sets = {}
    for split, desc in [("test_a", "seen templates, new names+nums"),
                        ("test_b", "unseen templates + names+nums"),
                        ("test_c", "unseen templates, seen nums")]:
        p = data_dir / f"{split}.jsonl"
        if p.exists():
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            test_sets[split] = (rows, desc)

    sweep_dir = ROOT / lang / "reasoning" / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    out_json = ROOT / lang / "reasoning" / "sample_sweep_results.json"
    sweep_results = {}
    if out_json.exists():
        try:
            sweep_results = json.loads(out_json.read_text(encoding="utf-8"))
            print(f"[data] Loaded {len(sweep_results)} existing results from {out_json.name}")
        except Exception:
            sweep_results = {}

    pretrained_ckpt_path = ROOT / lang / "checkpoints" / "last.pt"

    for n_samples in sample_sizes:
        label = f"{n_samples // 1000}k" if n_samples >= 1000 else f"{n_samples}"
        print("\n" + "-" * 75)
        print(f" RUNNING SAMPLE SIZE: {n_samples:,} examples ({label})")
        print("-" * 75)

        # Slice the training dataset to exactly n_samples
        train_items = full_train_ds.items[:n_samples]

        # Fresh model loaded from pretrained checkpoint
        model, ckpt = load(lang, pretrained_ckpt_path, device)
        cfg = model.cfg

        tcfg = dict(ckpt.get("train_config", {}))
        tcfg.update(learning_rate=lr, weight_decay=0.1, beta1=0.9, beta2=0.95)
        optimizer = build_optimizer(model, tcfg)

        steps_per_epoch = len(train_items) // batch_size
        # A fixed step budget is what makes the sizes comparable. Training one
        # epoch each would give 5k only 156 updates against 100k's 3,125, so a
        # low score could mean "too little data" or simply "too little training"
        # and the two could not be separated. With the budget fixed, compute is
        # equal and the only variable is how much distinct data the model saw.
        total_steps = fixed_steps or steps_per_epoch
        epochs = total_steps / max(steps_per_epoch, 1)
        warmup_steps = min(100, max(10, total_steps // 10))

        print(f"[train] Steps: {total_steps:,} ({epochs:.1f} epochs over {len(train_items):,} "
              f"examples) | Warmup: {warmup_steps} | Batch: {batch_size}")

        model.train()
        best_val = float("inf")
        best_state = None
        best_step = 0
        step = 0
        t0 = time.time()

        # Batch iteration
        order = np.arange(len(train_items))
        np.random.default_rng(1337).shuffle(order)

        pos = 0
        for _ in range(total_steps):
            # Reshuffle and wrap when the data runs out, so a small set simply
            # cycles more times within the same step budget.
            if pos + batch_size > len(order):
                np.random.default_rng(1337 + step).shuffle(order)
                pos = 0

            # Cosine LR over the whole budget, not over one epoch.
            if step < warmup_steps:
                cur_lr = lr * (step + 1) / warmup_steps
            else:
                prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cur_lr = lr / 10 + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * prog))
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            sl = order[pos:pos + batch_size]
            pos += batch_size
            x_batch = torch.from_numpy(np.stack([train_items[j][0] for j in sl])).to(device)
            y_batch = torch.from_numpy(np.stack([train_items[j][1] for j in sl])).to(device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                _, loss, _ = model(x_batch, targets=y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            # Periodic validation (every 100 steps for small sets, or at end)
            eval_interval = min(250, max(50, steps_per_epoch // 4))
            if step % eval_interval == 0 or step == steps_per_epoch:
                vl = validation_loss(model, val_ds, device, autocast_dtype, batch_size)
                if vl < best_val:
                    best_val = vl
                    best_step = step
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        train_time = time.time() - t0
        print(f"[train] Completed in {train_time:.1f}s | Best val_loss: {best_val:.4f} @ step {best_step}")

        # Load best weights for evaluation
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

        # Save checkpoint
        ckpt_save_path = sweep_dir / f"model_{label}.pt"
        torch.save({"model": model.state_dict(), "model_config": cfg.__dict__,
                    "n_samples": n_samples, "best_val": best_val, "best_step": best_step},
                   ckpt_save_path)

        # Run Evaluation on Test A, Test B, Test C
        print(f"[eval] Scoring on holdout test sets...")
        eval_res = {}
        for split, (rows, desc) in test_sets.items():
            res = evaluate(model, sp, rows, device, autocast_dtype, limit=test_limit)
            eval_res[split] = {
                "accuracy": res["accuracy"],
                "majority_baseline": res["majority_position_baseline"],
                "delta_over_baseline": round(res["accuracy"] - res["majority_position_baseline"], 4),
                "by_hops": res["by_hops"],
            }
            print(f"  -> {split.upper()} Acc: {res['accuracy']:.2%} (Base: {res['majority_position_baseline']:.2%})")

        sweep_results[label] = {
            "n_samples": n_samples,
            "training_seconds": round(train_time, 1),
            "best_step": best_step,
            "best_val_loss": round(best_val, 4),
            "test_a_acc": eval_res.get("test_a", {}).get("accuracy", 0.0),
            "test_b_acc": eval_res.get("test_b", {}).get("accuracy", 0.0),
            "test_c_acc": eval_res.get("test_c", {}).get("accuracy", 0.0),
            "test_a_hops": eval_res.get("test_a", {}).get("by_hops", {}),
        }

    # Save merged summary json
    out_json.write_text(json.dumps(sweep_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Sort results by sample size for display
    def parse_size(k):
        return int(k.replace("k", "")) * 1000 if "k" in k else int(k)

    sorted_keys = sorted(sweep_results.keys(), key=parse_size)

    # Print Final Markdown Comparison Table
    print("\n" + "=" * 85)
    print(f" COMPLETE SCALING CURVE: SAMPLE SIZE SWEEP ({lang.upper()})")
    print("=" * 85)
    print(f"{'Sample Size':<12} | {'Val Loss':<10} | {'Test A (Seen)':<15} | {'Test B (Unseen)':<16} | {'Test C':<10} | {'2-Hop Acc':<10}")
    print("-" * 85)
    for k in sorted_keys:
        v = sweep_results[k]
        two_hop = v.get("test_a_hops", {}).get("2", 0.0)
        print(f"{k:<12} | {v['best_val_loss']:<10.4f} | {v['test_a_acc']:<15.2%} | {v['test_b_acc']:<16.2%} | {v['test_c_acc']:<10.2%} | {two_hop:<10.2%}")
    print("=" * 85)
    print(f"Saved complete scaling table to: {out_json}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", default="hindi", choices=["hindi", "nepali"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[5000, 10000, 20000],
                        help="Sample sizes to sweep (default: 5000 10000 20000)")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Optional cap on test items for faster evaluation (e.g. 1000)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=6e-5,
                        help="Learning rate (default: 6e-5)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif device.type == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    run_sweep(args.lang, args.sizes, args.batch_size, args.lr, device, autocast_dtype, args.test_limit)


if __name__ == "__main__":
    main()
