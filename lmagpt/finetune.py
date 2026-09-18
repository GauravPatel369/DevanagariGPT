"""Supervised finetuning on the synthetic comparative-reasoning data.

Phase 3.  Starts from a language's own pretrained checkpoint, keeps its
tokenizer and vocabulary fixed, and never touches the other language.

Two things differ from pretraining, and both matter:

**The prompt is masked out of the loss.**  Each example is
``facts + question + "उत्तर:" + answer``.  The facts and the question are given
to the model at inference time, so learning to reproduce them teaches nothing
and spends capacity on trivially predictable text.  Only the answer tokens
carry gradient, which is what ``ignore_index=-100`` in ``GPT.forward`` is for.

**The learning rate is an order of magnitude lower.**  6e-5 against pretraining's
6e-4.  A fresh-scale rate on 3.5M tokens would overwrite the language model that
450M tokens of pretraining produced -- catastrophic forgetting -- and the result
would answer reasoning questions in broken Hindi.

Checkpoints keep the pretraining format, so a finetuned model resumes and
evaluates through exactly the same code paths.

Usage:
    python -m lmagpt.finetune --lang hindi
    python -m lmagpt.finetune --lang hindi --epochs 3 --lr 6e-5
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lmagpt.model import GPT, GPTConfig  # noqa: E402
from lmagpt.train import build_optimizer, save_checkpoint  # noqa: E402


# --------------------------------------------------------------- data


class ReasoningDataset:
    """Tokenised prompt/answer pairs with the prompt masked out of the loss.

    Each item is padded to a fixed length so a batch stacks without a collate
    function.  Padding is masked exactly like the prompt, so it contributes no
    gradient either.
    """

    def __init__(self, path: Path, sp, max_len: int = 128):
        self.rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.sp = sp
        self.max_len = max_len
        self.pad = 0                      # id 0 is <unk>; masked, so never predicted
        self._skipped = 0
        self.items = [enc for enc in (self._encode(r) for r in self.rows) if enc is not None]
        if self._skipped:
            print(f"[data] skipped {self._skipped} examples longer than {max_len} tokens")

    def _encode(self, row: dict):
        """Return (input_ids, targets) with prompt positions set to -100.

        The usual next-token shift applies: position t predicts t+1.  A target
        of -100 means "no loss here", which covers the whole prompt and any
        padding, leaving only the answer.
        """
        prompt = self.sp.encode(row["prompt"])
        answer = self.sp.encode(row["target"])
        ids = prompt + answer
        if len(ids) + 1 > self.max_len:
            self._skipped += 1
            return None

        x = ids[:-1]
        y = ids[1:]
        # Positions before the answer predict prompt tokens: mask them.
        n_prompt = len(prompt) - 1
        y = [-100] * n_prompt + y[n_prompt:]

        pad = self.max_len - 1 - len(x)
        x = x + [self.pad] * pad
        y = y + [-100] * pad
        return np.array(x, dtype=np.int64), np.array(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.items)

    def batches(self, batch_size: int, device, shuffle: bool = True, seed: int = 0):
        """Yield ``(x, y)`` batches for one epoch."""
        order = np.arange(len(self.items))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            sl = order[i:i + batch_size]
            x = torch.from_numpy(np.stack([self.items[j][0] for j in sl])).to(device)
            y = torch.from_numpy(np.stack([self.items[j][1] for j in sl])).to(device)
            yield x, y


# ----------------------------------------------------------- evaluation


@torch.no_grad()
def validation_loss(model, ds: ReasoningDataset, device, autocast_dtype, batch_size: int) -> float:
    """Mean loss over the answer tokens of the validation split."""
    model.eval()
    total, n = 0.0, 0
    for x, y in ds.batches(batch_size, device, shuffle=False):
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            _, loss, _ = model(x, targets=y)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


# ----------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    ap.add_argument("--checkpoint", default=None, help="pretrained weights to start from")
    ap.add_argument("--data-dir", default=None, help="dir holding train/val jsonl")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default=None)
    # 1 epoch, not 3. A smoke run showed training loss falling from 1.23 to 0.4
    # within 500 steps while validation rose from 1.37 to 1.93: the task is
    # learned quickly and the model then overfits. `best.pt` is selected on
    # validation so the saved weights stay good either way, but spending three
    # epochs would mostly buy worse checkpoints.
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-5,
                    help="10x below pretraining; higher risks catastrophic forgetting")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--max-hours", type=float, default=None)
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parent.parent

    ckpt_path = Path(a.checkpoint) if a.checkpoint else root / a.lang / "checkpoints" / "last.pt"
    data_dir = Path(a.data_dir) if a.data_dir else root / a.lang / "reasoning" / "data"
    out_dir = Path(a.out) if a.out else root / a.lang / "reasoning" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype, amp = torch.bfloat16, "bf16"
    elif device.type == "cuda":
        autocast_dtype, amp = torch.float16, "fp16"
    else:
        autocast_dtype, amp = None, "fp32"

    # Start from this language's own pretrained weights, never the other's.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[ft] {a.lang}: loaded {ckpt_path.name} @ step {ckpt['step']:,}, "
          f"{model.num_parameters():,} params, device {device}, amp {amp}")

    import sentencepiece as spm
    tok = Path(a.tokenizer) if a.tokenizer else (
        root / a.lang / "tokenizer" /
        f"{'hi' if a.lang == 'hindi' else 'ne'}_unigram_{cfg.vocab_size}.model")
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    if sp.get_piece_size() != cfg.vocab_size:
        raise SystemExit(f"tokenizer has {sp.get_piece_size()} pieces, "
                         f"checkpoint expects {cfg.vocab_size}")

    train_ds = ReasoningDataset(data_dir / "train.jsonl", sp, a.max_len)
    val_ds = ReasoningDataset(data_dir / "val.jsonl", sp, a.max_len)
    print(f"[ft] train {len(train_ds):,} examples | val {len(val_ds):,}")

    tcfg = dict(ckpt.get("train_config", {}))
    tcfg.update(learning_rate=a.lr, weight_decay=0.1, beta1=0.9, beta2=0.95)
    optimizer = build_optimizer(model, tcfg)

    steps_per_epoch = len(train_ds) // a.batch_size
    total_steps = steps_per_epoch * a.epochs
    print(f"[ft] {steps_per_epoch:,} steps/epoch x {a.epochs} epochs = {total_steps:,} steps")
    print(f"[ft] lr {a.lr:g} with {a.warmup}-step warmup, then cosine to {a.lr / 10:g}")

    log_path = out_dir / "finetune_log.jsonl"
    log = log_path.open("w", encoding="utf-8")
    best_val, step, started = float("inf"), 0, time.time()
    model.train()

    for epoch in range(a.epochs):
        for x, y in train_ds.batches(a.batch_size, device, seed=epoch):
            # Warmup then cosine decay, as a pure function of the step so a
            # resumed run needs no scheduler state.
            if step < a.warmup:
                lr = a.lr * (step + 1) / a.warmup
            else:
                prog = (step - a.warmup) / max(1, total_steps - a.warmup)
                lr = a.lr / 10 + 0.5 * (a.lr - a.lr / 10) * (1 + math.cos(math.pi * prog))
            for g in optimizer.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                _, loss, _ = model(x, targets=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 50 == 0:
                rec = dict(step=step, epoch=epoch, loss=round(loss.item(), 4),
                           lr=lr, grad_norm=round(float(gnorm), 3))
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[ft] step {step:>6,}/{total_steps:,} ep {epoch} "
                      f"loss {loss.item():.4f} lr {lr:.2e}", flush=True)

            if step % a.eval_every == 0 or step == total_steps:
                vl = validation_loss(model, val_ds, device, autocast_dtype, a.batch_size)
                rec = dict(step=step, epoch=epoch, val_loss=round(vl, 4))
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[eval] step {step:>6,} val_loss {vl:.4f}", flush=True)
                if vl < best_val:
                    best_val = vl
                    save_checkpoint(out_dir / "best.pt", model, optimizer, step,
                                    tcfg, cfg, best_val, np.random.get_state())
                    print(f"[eval] new best -> best.pt", flush=True)

            if a.max_hours and (time.time() - started) / 3600 >= a.max_hours:
                print(f"[budget] wall-clock limit reached at step {step:,}", flush=True)
                break
        else:
            continue
        break

    save_checkpoint(out_dir / "last.pt", model, optimizer, step, tcfg, cfg,
                    best_val, np.random.get_state())
    log.close()
    print(f"\n[ft] done: {step:,} steps in {(time.time() - started) / 60:.1f} min, "
          f"best val {best_val:.4f}")
    print(f"[ft] checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
