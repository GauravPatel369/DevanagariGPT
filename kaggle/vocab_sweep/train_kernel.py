"""Kaggle kernel -- one run of the vocabulary sweep (Hindi).

VOCAB is rewritten by scripts/kaggle_run.py before each push, so one kernel
serves all four vocabulary sizes.

The experiment holds the PARAMETER COUNT constant rather than the architecture:
d_ff absorbs the change in embedding size, so every run lands within +/-0.8% of
25.35M and vocabulary trades directly against FFN width.

    V= 5,000  d_ff 2176 (4.2x)   25,546,112 params   496.0M train tokens
    V= 8,000  d_ff 1920 (3.8x)   25,245,312 params   463.1M
    V=10,000  d_ff 1792 (3.5x)   25,350,912 params   450.7M   <- Phase-1 choice
    V=16,000  d_ff 1344 (2.6x)   25,208,512 params   430.8M

Compare the runs on BITS PER BYTE, never perplexity: each tokenizer emits a
different number of tokens for the same text, so a small vocabulary makes more
predictions that are each individually easier and its perplexity flatters it.
"""
# stdlib only until the bootstrap has settled which torch build to use.
import os
import subprocess
import sys

VOCAB = 5000          # rewritten per push
LANG = "nepali"        # rewritten per push

# Per-run wall-clock cap. Three separate limits are in play, and they are easy
# to confuse:
#
#   30 h  Kaggle's WEEKLY GPU quota, shared across every kernel in the account.
#   12 h  Kaggle's hard stop on any single GPU session.
#    8 h  this cap.
#
# THIS IS A HANG GUARD, NOT THE SCHEDULE. What ends a run is `max_steps` in the
# config, set to exactly one epoch. The cap consumes no quota unless it actually
# fires -- quota is billed on real runtime -- so it is set generously.
#
# Sizing it, from a measured P100 median of 22,200 tok/s (Phase-2 logs):
#
#     V= 5,000   496.0M tokens   6.21 h
#     V= 8,000   463.1M          5.79 h
#     V=10,000   450.7M          5.64 h
#     V=16,000   430.8M          5.39 h        total ~23 h of the 30 h quota
#
# Throughput barely moves across the four despite d_ff ranging 2176->1344:
# embeddings are tied, so the output projection's d_model*V cost grows exactly
# as the FFN's cost shrinks, and per-token compute stays ~constant.
#
# 8 h leaves the worst case (V=5,000) 29% headroom. An earlier 7 h cap gave it
# only 13%, close enough that a slow allocation would have truncated the epoch.
#
# If the cap ever does fire, the run exits with a resumable checkpoint: re-push
# that vocabulary with --resume and it continues from `last.pt`.
MAX_HOURS = 8.0


def find_dir(marker):
    """Locate a directory containing `marker` under /kaggle/input."""
    for root, dirs, _files in os.walk("/kaggle/input"):
        if root.count(os.sep) - 2 > 6:
            dirs[:] = []
            continue
        if marker in dirs or marker in _files:
            return root
    return None


print("[diag] /kaggle/input tree:", flush=True)
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if _root.count(os.sep) - 2 > 2:
        _dirs[:] = []
        continue
    print("   %s dirs=%s files=%s" % (_root, sorted(_dirs)[:6], sorted(_files)[:4]), flush=True)

CODE = find_dir("lmagpt")
DATA_ROOT = find_dir("v%d" % VOCAB)
if CODE is None:
    raise SystemExit("lmagpt/ not found under /kaggle/input (see tree above)")
if DATA_ROOT is None:
    raise SystemExit("v%d/ not found under /kaggle/input (see tree above)" % VOCAB)

DATA = os.path.join(DATA_ROOT, "v%d" % VOCAB)
print("[diag] code -> %s" % CODE, flush=True)
print("[diag] data -> %s" % DATA, flush=True)
sys.path.insert(0, CODE)

# Must run BEFORE torch is imported: Kaggle allocates a P100 (sm_60) and the
# preinstalled torch has no Pascal kernels.
from lmagpt.kaggle_bootstrap import ensure_compatible_torch, report_device  # noqa: E402

ensure_compatible_torch()
report_device()

CKPT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# ---- recover this vocabulary's previous checkpoint, if any ------------------
# Only reachable when kernel-metadata mounts this kernel's own prior output, so
# a run stopped by MAX_HOURS continues instead of restarting from step 0. The
# data dataset contains no .pt files, so this cannot pick up the wrong run.
import glob                                                        # noqa: E402
import shutil                                                      # noqa: E402

_prev = glob.glob("/kaggle/input/**/last.pt", recursive=True)
if _prev:
    _src = max(_prev, key=os.path.getmtime)
    shutil.copy2(_src, os.path.join(CKPT_DIR, "last.pt"))
    import torch

    _step = torch.load(os.path.join(CKPT_DIR, "last.pt"),
                       map_location="cpu", weights_only=False)["step"]
    print("[kaggle] resuming V=%d from %s at step %d" % (VOCAB, _src, _step), flush=True)
else:
    print("[kaggle] V=%d: no previous checkpoint -> training from scratch" % VOCAB, flush=True)

cfg = os.path.join(CODE, LANG, "configs", "model_v%d.yaml" % VOCAB)
if not os.path.exists(cfg):
    raise SystemExit("config %s missing -- sync the code dataset" % cfg)

cmd = [
    sys.executable, "-m", "lmagpt.train",
    "--lang", LANG,
    "--config", cfg,
    "--data-dir", DATA,
    "--out", CKPT_DIR,
    "--max-hours", str(MAX_HOURS),
]
print("[kaggle] V=%d | %s" % (VOCAB, " ".join(cmd)), flush=True)
rc = subprocess.call(cmd, env=dict(os.environ, PYTHONPATH=CODE, PYTHONUNBUFFERED="1"), cwd=CODE)

last = os.path.join(CKPT_DIR, "last.pt")
if os.path.exists(last):
    import torch

    ck = torch.load(last, map_location="cpu", weights_only=False)
    print("\n[kaggle] V=%d finished at step %d, best val %.4f"
          % (VOCAB, ck["step"], ck["best_val"]), flush=True)
    print("[kaggle] compare these runs on bits-per-byte, not perplexity", flush=True)
else:
    print("\n[kaggle] WARNING: no checkpoint written", flush=True)

sys.exit(rc)
