"""Kaggle kernel -- Phase 3 reasoning finetune, one language per push.

LANG is rewritten by scripts/kaggle_run.py before each push, so one kernel
serves both languages.

Finetuning is cheap next to pretraining: 3.5M tokens against 450M, roughly
6,250 steps for one epoch, about 10 minutes on a P100.  MAX_HOURS is therefore
a hang guard rather than a schedule.

Three inputs are mounted:

    lma-phase2-code       lmagpt package and configs
    lma-reasoning-data    the synthetic train/val/test jsonl, both languages
    lma-train-<lang>      the pretrained checkpoint, via kernel_sources

The pretrained checkpoint arrives as another kernel's output, exactly as
resuming did in Phase 2, so no separate upload of a 300 MB file is needed.
"""
# stdlib only until the bootstrap has settled which torch build to use.
import glob
import os
import shutil
import subprocess
import sys

LANG = "nepali"        # rewritten per push
EPOCHS = 1            # validation rises after ~1 epoch; see lmagpt/finetune.py
MAX_HOURS = 2.0       # ~10 min expected; this only catches a hang


def find_dir(marker):
    """Locate a directory containing `marker` under /kaggle/input.

    Globs rather than walking with a depth cut-off. Kaggle nests mounts as
    /kaggle/input/datasets/<user>/<slug>/..., and an over-tight depth limit
    makes this return None with no error, which then surfaces as a confusing
    "not found" much later.
    """
    hits = glob.glob("/kaggle/input/**/%s" % marker, recursive=True)
    return os.path.dirname(hits[0]) if hits else None


print("[diag] /kaggle/input tree:", flush=True)
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if _root.count(os.sep) - 2 > 2:
        _dirs[:] = []
        continue
    print("   %s dirs=%s files=%s" % (_root, sorted(_dirs)[:6], sorted(_files)[:4]), flush=True)

CODE = find_dir("lmagpt")
if CODE is None:
    raise SystemExit("lmagpt/ not found under /kaggle/input")

# The reasoning data ships as <lang>/train.jsonl inside one dataset covering
# both languages. Glob for the file itself rather than walking for a marker
# directory: Kaggle nests mounts as
# /kaggle/input/datasets/<user>/<slug>/<lang>/, which is deeper than find_dir's
# cut-off, and a marker search silently returned nothing.
matches = glob.glob("/kaggle/input/**/%s/train.jsonl" % LANG, recursive=True)
if not matches:
    raise SystemExit("%s/train.jsonl not found under /kaggle/input -- is "
                     "lma-reasoning-data attached?" % LANG)
DATA = os.path.dirname(matches[0])

print("[diag] code -> %s" % CODE, flush=True)
print("[diag] data -> %s" % DATA, flush=True)
sys.path.insert(0, CODE)

# Must run BEFORE torch is imported: Kaggle allocates a P100 (sm_60) and the
# preinstalled torch has no Pascal kernels.
from lmagpt.kaggle_bootstrap import ensure_compatible_torch, report_device  # noqa: E402

ensure_compatible_torch()
report_device()

WORK = "/kaggle/working"
OUT = os.path.join(WORK, "reasoning_checkpoints")
os.makedirs(OUT, exist_ok=True)

# ---- the pretrained checkpoint, mounted from the Phase-2 training kernel ----
# Prefer last.pt from a checkpoints/ directory; the finetune output of a
# previous run of *this* kernel would live under reasoning_checkpoints/ and
# must not be picked up as the starting point.
cands = [p for p in glob.glob("/kaggle/input/**/last.pt", recursive=True)
         if "reasoning" not in p]
if not cands:
    raise SystemExit("no pretrained last.pt found: is kernel_sources set to "
                     "the lma-train-%s kernel?" % LANG)
PRETRAINED = max(cands, key=os.path.getmtime)
print("[kaggle] pretrained checkpoint -> %s" % PRETRAINED, flush=True)

TOKENIZER = os.path.join(
    CODE, LANG, "tokenizer",
    "%s_unigram_10000.model" % ("hi" if LANG == "hindi" else "ne"))
if not os.path.exists(TOKENIZER):
    # The code dataset carries configs but not tokenizers; fall back to the
    # data dataset, which ships them alongside the jsonl.
    found = glob.glob("/kaggle/input/**/%s_unigram_10000.model"
                      % ("hi" if LANG == "hindi" else "ne"), recursive=True)
    if not found:
        raise SystemExit("tokenizer not found under /kaggle/input")
    TOKENIZER = found[0]
print("[kaggle] tokenizer -> %s" % TOKENIZER, flush=True)

cmd = [
    sys.executable, "-m", "lmagpt.finetune",
    "--lang", LANG,
    "--checkpoint", PRETRAINED,
    "--data-dir", DATA,
    "--tokenizer", TOKENIZER,
    "--out", OUT,
    "--epochs", str(EPOCHS),
    "--max-hours", str(MAX_HOURS),
]
print("[kaggle] %s | %s" % (LANG, " ".join(cmd)), flush=True)
rc = subprocess.call(cmd, env=dict(os.environ, PYTHONPATH=CODE, PYTHONUNBUFFERED="1"),
                     cwd=CODE)

best = os.path.join(OUT, "best.pt")
if os.path.exists(best):
    import torch

    ck = torch.load(best, map_location="cpu", weights_only=False)
    print("\n[kaggle] %s finetuned: step %d, best val %.4f"
          % (LANG, ck["step"], ck["best_val"]), flush=True)
else:
    print("\n[kaggle] WARNING: no checkpoint written", flush=True)

sys.exit(rc)
