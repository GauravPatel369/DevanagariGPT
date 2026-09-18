"""Pretraining loop with resume-capable checkpointing.

Resume is not a convenience here.  Kaggle kills a GPU session at 12 hours with
no warning and no interaction, and free Colab is worse, so a run that cannot
pick up where it stopped cannot finish at all.  Every checkpoint therefore
carries model weights, optimizer state, scheduler position, the step counter,
the RNG streams and the config -- everything needed to make the resumed run a
continuation rather than a restart.

Two details make resume actually faithful rather than approximately so:

* **The data generator is re-seeded from the step number** (``seed + step``), so
  a resumed run draws different windows than it already trained on instead of
  replaying the same ones.
* **``--max-hours`` stops on wall clock**, saves, and exits 0. Set it below the
  platform's hard kill (11.0 for Kaggle's 12h) so the checkpoint is written
  while the process is still alive.

Usage:
    python -m lmagpt.train --lang hindi                     # start or resume
    python -m lmagpt.train --lang hindi --max-hours 11      # Kaggle session
    python -m lmagpt.train --lang hindi --max-steps 200 --eval-every 50   # smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lmagpt.data import TokenDataset, resolve_data_dir  # noqa: E402
from lmagpt.model import GPT, GPTConfig  # noqa: E402


# ------------------------------------------------------------------ schedule


def lr_at(step: int, cfg: dict) -> float:
    """Linear warmup then cosine decay to ``min_lr``.

    Warmup exists because Adam's second-moment estimate is unreliable in the
    first few hundred steps: stepping at full learning rate before it settles
    is the classic way to destabilise a fresh Transformer. Cosine decay then
    spends most of the budget at a high rate and anneals smoothly at the end.
    """
    warmup, total = cfg["warmup_steps"], cfg["max_steps"]
    lr, min_lr = cfg["learning_rate"], cfg["min_lr"]
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: GPT, cfg: dict) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only.

    Decaying LayerNorm gains and biases pulls them toward zero, which fights the
    normalisation itself; the convention is to decay parameters of rank >= 2 and
    leave the rest alone. Embeddings are 2-D and are decayed with the other
    matrices -- with tied weights they are also the output projection.
    """
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]), eps=1e-8,
    )


# --------------------------------------------------------------- checkpoints


def save_checkpoint(path: Path, model, optimizer, step, cfg, model_cfg, best_val, rng_state):
    """Write a resumable checkpoint atomically.

    Written to a temporary file and renamed: a session killed mid-write would
    otherwise leave a truncated checkpoint that fails to load, losing the run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "train_config": cfg,
        "model_config": model_cfg.to_dict(),
        "best_val": best_val,
        "numpy_rng": rng_state,
        "torch_rng": torch.get_rng_state(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer, device):
    """Restore a run. Returns ``(step, best_val)``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("torch_rng") is not None:
        torch.set_rng_state(ckpt["torch_rng"].cpu() if hasattr(ckpt["torch_rng"], "cpu") else ckpt["torch_rng"])
    return ckpt.get("step", 0), ckpt.get("best_val", float("inf"))


# --------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(model, dataset, cfg, device, autocast_dtype, n_batches: int = 40) -> float:
    """Mean loss over fixed validation windows."""
    model.eval()
    losses = []
    for x, y in dataset.deterministic_batches(cfg["micro_batch"], n_batches, device=device):
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


# -------------------------------------------------------------------- train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--config", default=None, help="default: <lang>/configs/model.yaml")
    ap.add_argument("--out", default=None, help="checkpoint dir; default <lang>/checkpoints")
    ap.add_argument("--data-dir", default=None, help="override token directory")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop, checkpoint and exit after this wall-clock budget")
    ap.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--resume", default="auto", help="'auto', 'none', or a checkpoint path")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parent.parent
    cfg_path = Path(a.config) if a.config else root / a.lang / "configs" / "model.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mcfg, tcfg = raw["model"], raw["train"]
    if a.max_steps:
        tcfg["max_steps"] = a.max_steps
    if a.eval_every:
        tcfg["eval_every"] = a.eval_every

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 needs Ampere+ (compute capability >= 8.0). It has fp32's exponent
    # range, so unlike fp16 it needs no GradScaler and will not silently produce
    # inf during the backward pass. Older cards fall back to fp16.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype, amp_name = torch.bfloat16, "bf16"
    elif device.type == "cuda":
        autocast_dtype, amp_name = torch.float16, "fp16"
    else:
        autocast_dtype, amp_name = None, "fp32"
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_name == "fp16"))

    torch.manual_seed(tcfg["seed"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = GPTConfig(**mcfg)
    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(model, tcfg)

    out_dir = Path(a.out) if a.out else root / a.lang / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt, best_ckpt = out_dir / "last.pt", out_dir / "best.pt"

    step, best_val = 0, float("inf")
    if a.resume == "auto" and last_ckpt.exists():
        step, best_val = load_checkpoint(last_ckpt, model, optimizer, device)
        print(f"[resume] continuing from step {step:,} (best val {best_val:.4f})", flush=True)
    elif a.resume not in ("auto", "none"):
        step, best_val = load_checkpoint(Path(a.resume), model, optimizer, device)
        print(f"[resume] loaded {a.resume} at step {step:,}", flush=True)

    if a.compile:
        model = torch.compile(model)

    data_dir = Path(a.data_dir) if a.data_dir else resolve_data_dir(a.lang, root)
    train_ds = TokenDataset(data_dir / "train.bin", model_cfg.context)
    val_ds = TokenDataset(data_dir / "val.bin", model_cfg.context)

    micro, accum = tcfg["micro_batch"], tcfg["grad_accum"]
    tokens_per_step = micro * accum * model_cfg.context

    print(f"[setup] lang={a.lang} device={device} amp={amp_name}")
    print(f"[setup] params={model.num_parameters() if not a.compile else GPT(model_cfg).num_parameters():,}")
    print(f"[setup] data={data_dir}  train={train_ds.n_tokens:,} tok  val={val_ds.n_tokens:,} tok")
    print(f"[setup] micro_batch={micro} x accum={accum} x T={model_cfg.context} "
          f"= {tokens_per_step:,} tokens/step")
    print(f"[setup] max_steps={tcfg['max_steps']:,} "
          f"({tcfg['max_steps']*tokens_per_step/1e6:.0f}M tokens, "
          f"{tcfg['max_steps']*tokens_per_step/train_ds.n_tokens:.2f} epochs)", flush=True)

    log_path = out_dir / "train_log.jsonl"
    started, t_last = time.time(), time.time()
    deadline = started + a.max_hours * 3600 if a.max_hours else None
    model.train()

    while step < tcfg["max_steps"]:
        lr = lr_at(step, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Re-seed per step so a resumed run does not replay windows it has seen.
        gen = np.random.default_rng(tcfg["seed"] + step)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(accum):
            x, y = train_ds.batch(micro, gen, device=device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                _, loss, _ = model(x, targets=y)
                loss = loss / accum          # mean over the whole effective batch
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            total_loss += loss.item()

        # Clip on the full accumulated gradient, after unscaling.
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        step += 1

        if step % tcfg["log_every"] == 0:
            now = time.time()
            tps = tokens_per_step * tcfg["log_every"] / (now - t_last)
            t_last = now
            rec = {"step": step, "loss": round(total_loss, 4), "lr": round(lr, 6),
                   "grad_norm": round(float(grad_norm), 3), "tokens_per_sec": round(tps),
                   "elapsed_s": round(now - started)}
            print(f"[train] step {step:>7,}/{tcfg['max_steps']:,} loss {total_loss:.4f} "
                  f"lr {lr:.2e} |g| {float(grad_norm):.2f} {tps:,.0f} tok/s", flush=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

        if step % tcfg["eval_every"] == 0 or step == tcfg["max_steps"]:
            val = evaluate(model, val_ds, tcfg, device, autocast_dtype)
            ppl = math.exp(min(val, 20))
            print(f"[eval ] step {step:>7,} val_loss {val:.4f} val_ppl {ppl:.2f}", flush=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"step": step, "val_loss": round(val, 4),
                                     "val_ppl": round(ppl, 3)}) + "\n")
            rng = None
            save_checkpoint(last_ckpt, model, optimizer, step, tcfg, model_cfg, best_val, rng)
            if val < best_val:
                best_val = val
                save_checkpoint(best_ckpt, model, optimizer, step, tcfg, model_cfg, best_val, rng)
                print(f"[eval ] new best -> {best_ckpt.name}", flush=True)

        if deadline and time.time() > deadline:
            save_checkpoint(last_ckpt, model, optimizer, step, tcfg, model_cfg, best_val, None)
            hrs = (time.time() - started) / 3600
            print(f"\n[budget] wall-clock limit reached after {hrs:.2f}h at step {step:,}.")
            print(f"[budget] checkpoint saved to {last_ckpt}. Re-run the same command to continue.",
                  flush=True)
            return

    save_checkpoint(last_ckpt, model, optimizer, step, tcfg, model_cfg, best_val, None)
    print(f"\n[done] {step:,} steps in {(time.time()-started)/3600:.2f}h, best val {best_val:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
