"""Kaggle GPU benchmark: measure tokens/sec before committing quota.

Reports throughput and peak VRAM at several batch shapes so the training config
and the ETA are measured rather than guessed. Costs ~10 minutes of the
30 h/week budget and settles two questions the docs cannot: what the allocated
GPU actually does with this model, and which micro-batch fits its memory.
"""
# stdlib only until the bootstrap has settled which torch build to use.
import glob
import json
import os
import sys
import time


def find_code_dir():
    """Directory containing the lmagpt package, under /kaggle/input."""
    for root, dirs, _files in os.walk("/kaggle/input"):
        if root.count(os.sep) - 2 > 6:
            dirs[:] = []
            continue
        if "lmagpt" in dirs:
            return root
    return None


def find_data_dir(lang):
    """Directory holding <lang>/train.bin.

    Searching for the *language name* is not enough: the code dataset also ships
    a ``hindi/configs/`` folder, so the marker has to be a file that only the
    token dataset has.
    """
    for root, dirs, _files in os.walk("/kaggle/input"):
        if root.count(os.sep) - 2 > 6:
            dirs[:] = []
            continue
        if os.path.exists(os.path.join(root, lang, "train.bin")):
            return root
    return None


print("[diag] /kaggle/input tree:", flush=True)
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if _root.count(os.sep) - 2 > 2:
        _dirs[:] = []
        continue
    print("   %s dirs=%s files=%s" % (_root, sorted(_dirs)[:6], sorted(_files)[:4]), flush=True)

CODE = find_code_dir()
DATA_ROOT = find_data_dir("hindi")
if CODE is None:
    raise SystemExit("lmagpt/ not found under /kaggle/input (see tree above)")
if DATA_ROOT is None:
    raise SystemExit("hindi/train.bin not found under /kaggle/input (see tree above)")
print("[diag] code -> %s" % CODE, flush=True)
print("[diag] data -> %s" % DATA_ROOT, flush=True)
sys.path.insert(0, CODE)

# Must run BEFORE torch is imported: Kaggle allocates a P100 (sm_60) and the
# preinstalled torch has no Pascal kernels.
from lmagpt.kaggle_bootstrap import ensure_compatible_torch, report_device  # noqa: E402

ensure_compatible_torch()
report_device()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lmagpt.data import TokenDataset  # noqa: E402
from lmagpt.model import GPT, GPTConfig  # noqa: E402

DATA = os.path.join(DATA_ROOT, "hindi")
cfg = GPTConfig()
model = GPT(cfg).cuda()
print("[bench] params %d" % model.num_parameters(), flush=True)

ds = TokenDataset(os.path.join(DATA, "train.bin"), cfg.context)
opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
print("[bench] autocast dtype %s" % dtype, flush=True)
gen = np.random.default_rng(0)

results = {
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "amp": str(dtype),
    "params": model.num_parameters(),
    "shapes": {},
}

for micro, accum in [(16, 2), (32, 1), (48, 1)]:
    key = "micro%d_accum%d" % (micro, accum)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):  # warm up kernels before timing
            x, y = ds.batch(micro, gen, device="cuda")
            with torch.autocast("cuda", dtype=dtype):
                _, loss, _ = model(x, targets=y)
            (scaler.scale(loss) if scaler.is_enabled() else loss).backward()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        N = 30
        t0 = time.time()
        for _ in range(N):
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                x, y = ds.batch(micro, gen, device="cuda")
                with torch.autocast("cuda", dtype=dtype):
                    _, loss, _ = model(x, targets=y)
                    loss = loss / accum
                (scaler.scale(loss) if scaler.is_enabled() else loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
        torch.cuda.synchronize()

        dt = time.time() - t0
        tps = N * micro * accum * cfg.context / dt
        peak = torch.cuda.max_memory_allocated() / 1e9
        hrs = 450_667_252 / tps / 3600
        results["shapes"][key] = {
            "tokens_per_sec": round(tps),
            "peak_vram_gb": round(peak, 2),
            "hours_per_epoch": round(hrs, 2),
        }
        print("[bench] micro=%d accum=%d: %8.0f tok/s | peak %.2f GB | %.2f h/epoch"
              % (micro, accum, tps, peak, hrs), flush=True)
    except torch.cuda.OutOfMemoryError:
        print("[bench] micro=%d accum=%d: OOM" % (micro, accum), flush=True)
        results["shapes"][key] = {"error": "OOM"}
        torch.cuda.empty_cache()

with open("/kaggle/working/benchmark.json", "w") as fh:
    json.dump(results, fh, indent=2)
print("\n[bench] wrote /kaggle/working/benchmark.json", flush=True)
