"""Make Kaggle's GPU usable before any CUDA code runs.

Kaggle allocates a Tesla P100 (Pascal, sm_60) but preinstalls PyTorch 2.10+cu128,
whose binaries are compiled only for sm_70 and above.  The mismatch is not a
warning that can be ignored: the first CUDA op dies with

    CUDA error: no kernel image is available for execution on the device
    (cudaErrorNoKernelImageForDevice)

PyTorch 2.5.1+cu121 is the last line that still ships Pascal kernels, so this
module installs it when -- and only when -- the allocated device is older than
sm_70.  On a T4/L4/A100 the check is a no-op and the preinstalled torch is used.

The capability probe deliberately shells out to ``nvidia-smi`` rather than
importing torch: once torch is imported the wrong build is already resident in
the process, and swapping it out would require re-exec.  Stdlib only, called
before any ``import torch``.
"""

from __future__ import annotations

import subprocess
import sys

# Last PyTorch release whose CUDA binaries include sm_60 (Pascal) kernels.
PASCAL_TORCH = "torch==2.5.1"
PASCAL_INDEX = "https://download.pytorch.org/whl/cu121"
MIN_SUPPORTED_CAP = (7, 0)


def detect_capability() -> tuple[int, int] | None:
    """Compute capability of GPU 0, via nvidia-smi.  None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=60,
        ).strip().splitlines()[0]
        major, minor = out.strip().split(".")
        return int(major), int(minor)
    except Exception as exc:  # no GPU, no driver, unexpected format
        print(f"[bootstrap] could not read compute capability: {exc}", flush=True)
        return None


def ensure_compatible_torch(verbose: bool = True) -> bool:
    """Install a Pascal-capable torch if the device needs one.

    Returns:
        True if an install was performed (the caller has not yet imported torch,
        so no re-exec is needed), False if the preinstalled build is fine.
    """
    cap = detect_capability()
    if cap is None:
        return False
    if verbose:
        print(f"[bootstrap] GPU compute capability sm_{cap[0]}{cap[1]}", flush=True)

    if cap >= MIN_SUPPORTED_CAP:
        if verbose:
            print("[bootstrap] preinstalled torch supports this device", flush=True)
        return False

    if "torch" in sys.modules:
        raise RuntimeError(
            "ensure_compatible_torch() must be called before 'import torch'; "
            "the incompatible build is already loaded in this process"
        )

    print(f"[bootstrap] sm_{cap[0]}{cap[1]} predates sm_70 -- Kaggle's torch has no "
          f"kernels for it. Installing {PASCAL_TORCH} (~3-5 min).", flush=True)
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           PASCAL_TORCH, "--index-url", PASCAL_INDEX]
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(
            f"pip install failed (exit {rc}). Is 'enable_internet' true in "
            f"kernel-metadata.json? Without it the kernel has no network."
        )
    print("[bootstrap] install complete", flush=True)
    return True


def report_device() -> None:
    """Print the device summary, after torch is safely importable."""
    import torch
    if not torch.cuda.is_available():
        print("[bootstrap] WARNING: no CUDA device visible", flush=True)
        return
    free, total = torch.cuda.mem_get_info()
    print(f"[bootstrap] torch {torch.__version__} | {torch.cuda.get_device_name(0)} "
          f"| sm_{''.join(map(str, torch.cuda.get_device_capability(0)))} "
          f"| bf16={torch.cuda.is_bf16_supported()} "
          f"| vram {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)

    # Fail loudly here rather than 200 steps into training.
    try:
        torch.zeros(8, device="cuda").sum().item()
        print("[bootstrap] CUDA smoke test passed", flush=True)
    except Exception as exc:
        raise SystemExit(f"[bootstrap] CUDA is not usable on this device: {exc}")
