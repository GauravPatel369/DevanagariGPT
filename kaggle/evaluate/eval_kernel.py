"""Kaggle evaluation kernel -- full Phase-2 evaluation for both models.

Produces, into /kaggle/working:

    results/evaluation_test.json     perplexity, bits-per-byte, BLEU / chrF++ /
                                     ROUGE-L, repetition and distinct-n, per
                                     decoding setting
    results/attention_analysis.json  per-head entropy and mean attention distance
    results/figures/*.png            attention heatmaps and head summaries

The trained checkpoints are NOT uploaded: ``kernel_sources`` in
kernel-metadata.json mounts the two training kernels' own outputs, so
``checkpoints/last.pt`` from each run appears under /kaggle/input.  Same
mechanism the training chain uses to resume, reused here to avoid pushing
600 MB of weights back and forth.
"""
# stdlib only until the bootstrap has settled which torch build to use.
import glob
import os
import shutil
import subprocess
import sys

LANGS = ("hindi", "nepali")
SPLIT = "test"

WORK = "/kaggle/working"
STAGE = os.path.join(WORK, "stage")        # <lang>/last.pt, matches --checkpoint-dir
OUTDIR = os.path.join(WORK, "results")     # absolute: the code mount is read-only
os.makedirs(STAGE, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)


def find_code_dir():
    """Directory containing the lmagpt package."""
    for root, dirs, _files in os.walk("/kaggle/input"):
        if root.count(os.sep) - 2 > 6:
            dirs[:] = []
            continue
        if "lmagpt" in dirs:
            return root
    return None


def find_checkpoints():
    """Map language -> checkpoint path from the mounted training-kernel outputs.

    Both kernels write ``checkpoints/last.pt``, so the filename alone is
    ambiguous; the language is recovered from the mount path, which carries the
    kernel slug (lma-train-hindi / lma-train-nepali).
    """
    found = {}
    for path in glob.glob("/kaggle/input/**/last.pt", recursive=True):
        low = path.replace("\\", "/").lower()
        for lang in LANGS:
            if lang in low:
                if lang not in found or os.path.getmtime(path) > os.path.getmtime(found[lang]):
                    found[lang] = path
    return found


print("[diag] /kaggle/input tree:", flush=True)
for _root, _dirs, _files in os.walk("/kaggle/input"):
    if _root.count(os.sep) - 2 > 3:
        _dirs[:] = []
        continue
    print("   %s dirs=%s files=%s" % (_root, sorted(_dirs)[:6], sorted(_files)[:4]), flush=True)

CODE = find_code_dir()
if CODE is None:
    raise SystemExit("lmagpt/ not found under /kaggle/input (see tree above)")
print("[diag] code -> %s" % CODE, flush=True)
sys.path.insert(0, CODE)

ckpts = find_checkpoints()
print("[diag] checkpoints: %s" % ckpts, flush=True)
missing = [l for l in LANGS if l not in ckpts]
if missing:
    raise SystemExit(
        "no checkpoint for %s. Are the training kernels in kernel_sources, and "
        "have they completed at least one version?" % ", ".join(missing)
    )

# Must run BEFORE torch is imported: Kaggle allocates a P100 (sm_60) and the
# preinstalled torch has no Pascal kernels.
from lmagpt.kaggle_bootstrap import ensure_compatible_torch, report_device  # noqa: E402

ensure_compatible_torch()

# sacrebleu supplies BLEU and chrF++; not on the Kaggle image.
print("[eval] installing sacrebleu ...", flush=True)
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "sacrebleu"])

report_device()

for lang, src in ckpts.items():
    dst = os.path.join(STAGE, lang)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(src, os.path.join(dst, "last.pt"))
    print("[eval] staged %s -> %s/last.pt" % (lang, dst), flush=True)

env = dict(os.environ, PYTHONPATH=CODE, PYTHONUNBUFFERED="1")

print("\n" + "=" * 70, flush=True)
print("INTRINSIC + GENERATION METRICS", flush=True)
print("=" * 70, flush=True)
rc1 = subprocess.call([
    sys.executable, "-m", "lmagpt.evaluate",
    "--lang", "both",
    "--split", SPLIT,
    "--checkpoint-dir", STAGE,
    "--n-samples", "128",
    "--gen-batch", "32",     # 16 GB here against 6 GB on the laptop
    "--batch-size", "32",
    "--out", OUTDIR,
], env=env, cwd=CODE)

print("\n" + "=" * 70, flush=True)
print("ATTENTION ANALYSIS", flush=True)
print("=" * 70, flush=True)
rc2 = subprocess.call([
    sys.executable, "-m", "lmagpt.attention",
    "--lang", "both",
    "--checkpoint-dir", STAGE,
    "--stats-tokens", "256",
    "--stats-passages", "16",
    "--out", OUTDIR,
], env=env, cwd=CODE)

print("\n[eval] /kaggle/working now holds: %s" % sorted(os.listdir(WORK)), flush=True)
if os.path.isdir(OUTDIR):
    print("[eval] results/: %s" % sorted(os.listdir(OUTDIR)), flush=True)

# The staged checkpoint copies would otherwise be re-uploaded as kernel output.
shutil.rmtree(STAGE, ignore_errors=True)

sys.exit(rc1 or rc2)
