"""Kaggle kernel -- Phase 3 data-scaling sweep for one language.

LANG is rewritten by scripts/kaggle_run.py before each push, so one kernel
serves both languages.

For every training-set size, and for the pretrained baseline, this measures:

    reasoning accuracy   forced choice over the candidate set
    reasoning accuracy   first word of free greedy generation
    invalid-output rate  generations that name nobody from the question
    language PPL / BPB   on the Phase-1/2 held-out split, the forgetting check

all of it broken out per test set (a/b/c/d) and per hop count.

Each size trains one full epoch over its own slice, so a bigger slice also gets
more steps.  That is the realistic setting -- more data normally means more
training -- but it means the sweep measures data and compute together, so the
step count is recorded alongside every result.

The pretrained baseline is expected to score near zero on generation: it has
never seen the answer format and continues in news register instead.  That is
the point of reporting both metrics rather than one.
"""
# stdlib only until the bootstrap has settled which torch build to use.
import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

LANG = "nepali"          # rewritten per push
SIZES = [5000, 10000, 20000, 30000, 100000]
TEST_LIMIT = 600        # per test set; 4 sets per model
BATCH_SIZE = 32
LR = 6e-5
MAX_HOURS = 8.0         # hang guard; expected ~1.5 h


def find(pattern):
    """First path matching a glob under /kaggle/input, or None."""
    hits = glob.glob("/kaggle/input/**/%s" % pattern, recursive=True)
    return hits[0] if hits else None


print("[diag] /kaggle/input tree:", flush=True)
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if _root.count(os.sep) - 2 > 2:
        _dirs[:] = []
        continue
    print("   %s dirs=%s" % (_root, sorted(_dirs)[:6]), flush=True)

CODE = os.path.dirname(find("lmagpt") or "")
# A dataset holding one language only is flattened by Kaggle on upload (the
# second account's Nepali data), so fall back to the top level when the
# language folder is absent.
TRAIN_JSONL = find("%s/train.jsonl" % LANG) or find("train.jsonl")
TOKENIZER = find("%s_unigram_10000.model" % ("hi" if LANG == "hindi" else "ne"))
TEST_BIN = find("%s/test.bin" % LANG) or find("test.bin")
if not CODE or not TRAIN_JSONL or not TOKENIZER:
    raise SystemExit("missing input: code=%s data=%s tok=%s" % (CODE, TRAIN_JSONL, TOKENIZER))
DATA = os.path.dirname(TRAIN_JSONL)
print("[diag] code=%s\n[diag] data=%s\n[diag] tok=%s\n[diag] test.bin=%s"
      % (CODE, DATA, TOKENIZER, TEST_BIN), flush=True)
sys.path.insert(0, CODE)

# Must run BEFORE torch is imported: Kaggle allocates a P100 (sm_60) and the
# preinstalled torch has no Pascal kernels.
from lmagpt.kaggle_bootstrap import ensure_compatible_torch, report_device  # noqa: E402

ensure_compatible_torch()
report_device()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import sentencepiece as spm  # noqa: E402

from lmagpt.eval_reasoning import evaluate, load  # noqa: E402
from lmagpt.finetune import ReasoningDataset, validation_loss  # noqa: E402
from lmagpt.train import build_optimizer  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda" and torch.cuda.is_bf16_supported():
    autocast_dtype = torch.bfloat16
elif device.type == "cuda":
    autocast_dtype = torch.float16
else:
    autocast_dtype = None

# The pretrained checkpoint arrives as the Phase-2 training kernel's output.
cands = [p for p in glob.glob("/kaggle/input/**/last.pt", recursive=True)
         if "reasoning" not in p and "sweep" not in p]
if not cands:
    raise SystemExit("no pretrained last.pt; is kernel_sources set?")
PRETRAINED = max(cands, key=os.path.getmtime)
print("[kaggle] pretrained -> %s" % PRETRAINED, flush=True)

sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
full_train = ReasoningDataset(Path(DATA) / "train.jsonl", sp, 128)
val_ds = ReasoningDataset(Path(DATA) / "val.jsonl", sp, 128)
print("[data] train %d | val %d" % (len(full_train), len(val_ds)), flush=True)

# The size is chosen on these unseen-name validation questions, not on a test
# set: choosing it on test_a would let the test data steer a training decision.
VAL_OOD_ROWS = []
_vo = os.path.join(DATA, "val_ood.jsonl")
if os.path.exists(_vo):
    with open(_vo, encoding="utf-8") as f:
        VAL_OOD_ROWS = [json.loads(l) for l in f if l.strip()][:500]
print("[data] val_ood rows for size selection: %d" % len(VAL_OOD_ROWS), flush=True)

TEST_SETS = {}
for split in ("test_a", "test_b", "test_c", "test_d"):
    p = os.path.join(DATA, "%s.jsonl" % split)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            TEST_SETS[split] = [json.loads(l) for l in f if l.strip()]
print("[data] test sets: %s" % sorted(TEST_SETS), flush=True)


def language_metrics(model, cfg):
    """Perplexity and bits-per-byte on the Phase-1/2 held-out split.

    This is the catastrophic-forgetting check: how much general language
    ability did finetuning on reasoning text cost?  Scored with the Phase-2
    code path so the numbers are directly comparable to that report.
    """
    if not TEST_BIN:
        return None
    from lmagpt.data import TokenDataset
    from lmagpt.evaluate import intrinsic_metrics, add_bits_per_byte

    meta_path = os.path.join(os.path.dirname(TEST_BIN), "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        bpt = json.load(f)["splits"]["test"]["bytes_per_token"]
    ds = TokenDataset(Path(TEST_BIN), cfg.context)
    m = intrinsic_metrics(model, ds, device, autocast_dtype, batch_size=16)
    add_bits_per_byte(m, bpt)
    return m


def train_one(n_samples):
    """Finetune a fresh copy of the pretrained model on one slice, 1 epoch."""
    items = full_train.items[:n_samples]
    model, ckpt = load(LANG, PRETRAINED, device)
    cfg = model.cfg
    tcfg = dict(ckpt.get("train_config", {}))
    tcfg.update(learning_rate=LR, weight_decay=0.1, beta1=0.9, beta2=0.95)
    optimizer = build_optimizer(model, tcfg)

    steps = len(items) // BATCH_SIZE
    warmup = min(100, max(10, steps // 10))
    order = np.arange(len(items))
    np.random.default_rng(1337).shuffle(order)

    best_val, best_state, best_step, step = float("inf"), None, 0, 0
    model.train()
    t0 = time.time()
    for i in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE):
        lr = (LR * (step + 1) / warmup) if step < warmup else (
            LR / 10 + 0.5 * (LR - LR / 10) *
            (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        for g in optimizer.param_groups:
            g["lr"] = lr

        sl = order[i:i + BATCH_SIZE]
        x = torch.from_numpy(np.stack([items[j][0] for j in sl])).to(device)
        y = torch.from_numpy(np.stack([items[j][1] for j in sl])).to(device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            _, loss, _ = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        # Validation selects the checkpoint: training loss keeps falling well
        # after validation turns, so the last step is not the best model.
        if step % max(50, steps // 6) == 0 or step == steps:
            vl = validation_loss(model, val_ds, device, autocast_dtype, BATCH_SIZE)
            if vl < best_val:
                best_val, best_step = vl, step
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, cfg, dict(steps=steps, best_step=best_step,
                            best_val=round(best_val, 4),
                            train_seconds=round(time.time() - t0, 1))


def measure(model, cfg, label, meta):
    """Every metric for one model, per test set, plus the selection metric."""
    rec = {"label": label, **meta, "test_sets": {}}
    if VAL_OOD_ROWS:
        v = evaluate(model, sp, VAL_OOD_ROWS, device, autocast_dtype, None, with_generation=True)
        v.pop("errors", None)
        rec["val_ood"] = v
        print("    val_ood (selection): choice %.1f%%  generation %.1f%%"
              % (v["accuracy"] * 100, v["exact_match"] * 100), flush=True)
    for split, rows in sorted(TEST_SETS.items()):
        r = evaluate(model, sp, rows, device, autocast_dtype, TEST_LIMIT,
                     with_generation=True)
        r.pop("errors", None)          # keep the results file small
        rec["test_sets"][split] = r
        print("    %s: choice %.1f%%  generation %.1f%%  invalid %.1f%%"
              % (split, r["accuracy"] * 100, r["exact_match"] * 100,
                 r["generation_invalid_rate"] * 100), flush=True)
    lm = language_metrics(model, cfg)
    if lm:
        rec["language"] = lm
        print("    language: ppl %.2f  bpb %.4f" % (lm["perplexity"], lm["bits_per_byte"]),
              flush=True)
    return rec


results = {"language": LANG, "test_limit": TEST_LIMIT, "models": []}
started = time.time()

print("\n" + "=" * 70 + "\n  PRETRAINED BASELINE\n" + "=" * 70, flush=True)
model, ckpt = load(LANG, PRETRAINED, device)
results["models"].append(measure(model, model.cfg, "pretrained",
                                 dict(n_samples=0, steps=0)))
del model
torch.cuda.empty_cache()

for n in SIZES:
    if (time.time() - started) / 3600 > MAX_HOURS:
        print("[budget] wall-clock limit reached, stopping", flush=True)
        break
    label = "%dk" % (n // 1000)
    print("\n" + "=" * 70 + "\n  %s (%d examples)\n" % (label, n) + "=" * 70, flush=True)
    model, cfg, meta = train_one(n)
    meta["n_samples"] = n
    print("  trained %d steps in %.0fs, best val %.4f @ %d"
          % (meta["steps"], meta["train_seconds"], meta["best_val"], meta["best_step"]),
          flush=True)
    results["models"].append(measure(model, cfg, label, meta))
    del model
    torch.cuda.empty_cache()

    out = "/kaggle/working/sweep_results_%s.json" % LANG
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("  [saved] %s" % out, flush=True)

print("\n[done] %.1f min total" % ((time.time() - started) / 60), flush=True)
