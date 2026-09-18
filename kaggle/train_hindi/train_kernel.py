"""Kaggle training kernel -- Model H (Hindi).

Kaggle gives every run a fresh, empty /kaggle/working, so a checkpoint written
by run N is invisible to run N+1 unless it is wired in as an INPUT. That is what
``kernel_sources`` in kernel-metadata.json does: it mounts this kernel's own
previous output, and the block below copies the checkpoint found there into
/kaggle/working so training continues instead of restarting.

    run 1 -> from scratch          -> output: checkpoints/last.pt @ step N
    run 2 -> mounts run 1's output -> resumes at N -> output @ step 2N
    ...   -> repeat until the log prints [done] rather than [budget]

MAX_HOURS stops and checkpoints BEFORE Kaggle's 12 h hard kill, so a session
always ends with a loadable checkpoint rather than a truncated one.
"""
# stdlib only until the bootstrap has settled which torch build to use.
import glob
import os
import shutil
import subprocess
import sys

LANG = "hindi"
# Wall-clock stop, NOT a reservation: Kaggle bills the time actually used, so a
# run that reaches max_steps at 5.7 h costs 5.7 h of the weekly quota and this
# value is never reached. It exists for two cases:
#   * a session that would otherwise hit Kaggle's 12 h hard kill mid-write,
#   * a run that hangs -- this caps how much quota that can burn.
# Set it a little above the expected duration (~5.8 h for a full epoch) rather
# than at the platform maximum, so a stall wastes ~1 h instead of ~5.
MAX_HOURS = 7.0
# Overwritten by scripts/kaggle_run.py --max-steps. None = full run from config.
MAX_STEPS = None


def find_code_dir():
    """Directory containing the lmagpt package, under /kaggle/input.

    Kaggle's mount layout varies between /kaggle/input/<slug>/ and
    /kaggle/input/datasets/<user>/<slug>/, so walk for the marker rather than
    hardcoding a path that silently yields an empty mount.
    """
    for root, dirs, _files in os.walk("/kaggle/input"):
        if root.count(os.sep) - 2 > 6:
            dirs[:] = []
            continue
        if "lmagpt" in dirs:
            return root
    return None


def find_data_dir(lang):
    """Directory holding <lang>/train.bin.

    The language name alone is not a safe marker: the code dataset also ships a
    <lang>/configs/ folder, so the search must key on a file only the token
    dataset has.
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
DATA_ROOT = find_data_dir(LANG)
if CODE is None:
    raise SystemExit("lmagpt/ not found under /kaggle/input (see tree above)")
if DATA_ROOT is None:
    raise SystemExit("%s/train.bin not found under /kaggle/input (see tree above)" % LANG)
print("[diag] code -> %s" % CODE, flush=True)
print("[diag] data -> %s" % DATA_ROOT, flush=True)
sys.path.insert(0, CODE)

# Must run BEFORE torch is imported anywhere in this process: Kaggle's P100 is
# sm_60 and the preinstalled torch has no Pascal kernels.
from lmagpt.kaggle_bootstrap import ensure_compatible_torch, report_device  # noqa: E402

ensure_compatible_torch()
report_device()

WORK = "/kaggle/working"
CKPT_DIR = os.path.join(WORK, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ---- recover the previous run's checkpoint, if this is a continuation -------
candidates = glob.glob("/kaggle/input/**/last.pt", recursive=True)
if candidates:
    src = max(candidates, key=os.path.getmtime)
    shutil.copy2(src, os.path.join(CKPT_DIR, "last.pt"))
    import torch

    prev_step = torch.load(os.path.join(CKPT_DIR, "last.pt"),
                           map_location="cpu", weights_only=False)["step"]
    print("[kaggle] resuming from %s at step %d" % (src, prev_step), flush=True)
else:
    print("[kaggle] no previous checkpoint -> training from scratch", flush=True)

cmd = [
    sys.executable, "-m", "lmagpt.train",
    "--lang", LANG,
    "--config", os.path.join(CODE, LANG, "configs", "model.yaml"),
    "--data-dir", os.path.join(DATA_ROOT, LANG),
    "--out", CKPT_DIR,
    "--max-hours", str(MAX_HOURS),
]
if MAX_STEPS:
    # Smoke run: exercise the whole path (schedule, eval, checkpoint) in minutes.
    cmd += ["--max-steps", str(MAX_STEPS), "--eval-every", str(max(10, MAX_STEPS // 2))]
print("[kaggle] " + " ".join(cmd), flush=True)
rc = subprocess.call(cmd, env=dict(os.environ, PYTHONPATH=CODE, PYTHONUNBUFFERED="1"), cwd=CODE)

last = os.path.join(CKPT_DIR, "last.pt")
if os.path.exists(last):
    import torch

    ck = torch.load(last, map_location="cpu", weights_only=False)
    mb = os.path.getsize(last) / 1e6
    print("\n[kaggle] checkpoint at step %d (%.0f MB), best val %.4f"
          % (ck["step"], mb, ck["best_val"]), flush=True)
    print("[kaggle] push this kernel again to continue from here", flush=True)
else:
    print("\n[kaggle] WARNING: no checkpoint written", flush=True)

sys.exit(rc)
