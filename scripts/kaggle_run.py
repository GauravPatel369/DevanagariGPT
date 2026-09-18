"""Drive Kaggle training from the command line.

Kaggle gives every run a fresh, empty ``/kaggle/working``, so resuming across
sessions means wiring run N's *output* in as run N+1's *input*.  This script
automates that chain, plus the push / poll / fetch cycle around it.

Typical session:

    python scripts/kaggle_run.py status                  # what is running
    python scripts/kaggle_run.py train --lang hindi      # start (or continue)
    python scripts/kaggle_run.py watch --lang hindi      # poll until it ends
    python scripts/kaggle_run.py fetch --lang hindi      # download checkpoint + log
    python scripts/kaggle_run.py train --lang hindi      # continue where it stopped

Repeat ``train`` until the log reports ``[done]`` rather than ``[budget]``.

Kaggle has no streaming-log API: ``watch`` reports state transitions only.  The
live console is in the browser at the URL each command prints.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "gauravpatel123"
KERNEL = {"hindi": f"{USER}/lma-train-hindi",
          "nepali": f"{USER}/lma-train-nepali",
          "benchmark": f"{USER}/lma-benchmark"}
CODE_DATASET = f"{USER}/lma-phase2-code"
VOCABS = [5000, 8000, 10000, 16000]


def target(a) -> tuple[str, str]:
    """Resolve ``--lang``/``--vocab`` to ``(kernel slug, local name)``.

    ``watch`` and ``fetch`` serve both the two language models and the four
    vocabulary-sweep runs, which are addressed differently.
    """
    if getattr(a, "vocab", None):
        return f"{USER}/lma-vocab-{a.vocab}", f"vocab_{a.vocab}"
    if getattr(a, "lang", None):
        return KERNEL[a.lang], a.lang
    raise SystemExit("pass either --lang or --vocab")


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def kernel_status(slug: str) -> tuple[str, str]:
    """Return ``(status, message)`` for one kernel."""
    try:
        r = api().kernels_status(slug)
        if isinstance(r, dict):
            st = str(r.get("status", "unknown"))
            msg = str(r.get("failureMessage") or "")
        else:
            st = str(getattr(r, "status", "unknown"))
            msg = str(getattr(r, "failureMessage", "") or "")
        # The API returns e.g. "KernelWorkerStatus.COMPLETE"; keep the leaf.
        return st.rsplit(".", 1)[-1].lower(), msg
    except Exception as e:
        return "unavailable", f"{type(e).__name__}: {e}"


def account(config_dir: str | None) -> tuple[dict, str]:
    """Environment and username for the account a command should act as.

    Phase 3 runs Hindi on the main account and Nepali on a second account, so
    each language has its own GPU quota. ``--config-dir`` points at the second
    account's kaggle.json; the key itself is never read here, only the username.
    """
    import os
    env = dict(os.environ)
    if not config_dir:
        return env, USER
    env["KAGGLE_CONFIG_DIR"] = config_dir
    user = json.loads((Path(config_dir) / "kaggle.json").read_text(encoding="utf-8"))["username"]
    return env, user


def phase3_inputs(lang: str, user: str) -> tuple[list[str], list[str]]:
    """Datasets and kernel sources a Phase-3 kernel mounts on this account.

    On the main account the pretrained checkpoint arrives as the Phase-2
    training kernel's output. Kernel outputs cannot be mounted across accounts,
    so on the second account the checkpoint and the token files are uploaded as
    datasets by ``sync-pretrained``.
    """
    code = "hi" if lang == "hindi" else "ne"
    if user == USER:
        return ([f"{USER}/lma-phase2-code", f"{USER}/lma-reasoning-data",
                 f"{USER}/lma-phase1-tokens"], [KERNEL[lang]])
    return ([f"{user}/lma-phase2-code", f"{user}/lma-reasoning-data",
             f"{user}/lma-phase1-tokens-{code}", f"{user}/lma-pretrained-{lang}"], [])


# ------------------------------------------------------------------ commands


def cmd_status(a):
    print(f"{'kernel':<34} {'status':<14} note")
    print("-" * 74)
    for name, slug in KERNEL.items():
        st, msg = kernel_status(slug)
        print(f"{slug:<34} {st:<14} {msg[:28]}")
    print(f"\nlive logs: https://www.kaggle.com/code/{KERNEL['hindi']}  (and /lma-train-nepali)")


def cmd_sync_code(a):
    """Push the local lmagpt package + configs as a new dataset version.

    Run this after ANY change to lmagpt/ or the configs, otherwise Kaggle keeps
    executing the previous version and the change silently has no effect.
    """
    import shutil, tempfile
    stage = Path(tempfile.mkdtemp(prefix="lma-code-"))
    (stage / "lmagpt").mkdir()
    for p in (ROOT / "lmagpt").glob("*.py"):
        shutil.copy2(p, stage / "lmagpt" / p.name)
    for lang in ("hindi", "nepali"):
        (stage / lang / "configs").mkdir(parents=True)
        # Every config, not just model.yaml: the ablation (model_norope.yaml) and
        # the vocabulary sweep (model_v*.yaml) are selected by the kernel at run
        # time, and a missing one fails the run only after the GPU is allocated.
        for cfg in sorted((ROOT / lang / "configs").glob("*.yaml")):
            shutil.copy2(cfg, stage / lang / "configs")
        print(f"[sync] {lang}: {len(list((stage / lang / 'configs').glob('*.yaml')))} configs")
    env, user = account(a.config_dir)
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": "LMA Phase 2 Code", "id": f"{user}/lma-phase2-code",
         "licenses": [{"name": "unknown"}]}, indent=2), encoding="utf-8")
    print(f"[sync] pushing code from {stage} to {user}/lma-phase2-code")
    subprocess.run([sys.executable, "-m", "kaggle", "datasets", "version",
                    "-p", str(stage), "-m", a.message, "--dir-mode", "zip"], check=False, env=env)
    shutil.rmtree(stage, ignore_errors=True)


def cmd_train(a):
    """Push the training kernel, chaining it to its own previous output."""
    kdir = ROOT / "kaggle" / f"train_{a.lang}"
    meta_path = kdir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    slug = KERNEL[a.lang]
    if not a.fresh:
        # Mount the previous run's output so training continues instead of
        # restarting. Harmless on the first run: Kaggle ignores a self-reference
        # that has no completed version yet.
        meta["kernel_sources"] = [slug]
    else:
        meta["kernel_sources"] = []
        print("[train] --fresh: starting from scratch, previous checkpoint ignored")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Inject the step cap into the kernel source so one kernel serves both the
    # smoke run and the real one.
    ksrc = kdir / "train_kernel.py"
    src = ksrc.read_text(encoding="utf-8")
    import re as _re
    src = _re.sub(r"^MAX_STEPS = .*$",
                  f"MAX_STEPS = {a.max_steps if a.max_steps else None}",
                  src, count=1, flags=_re.M)
    ksrc.write_text(src, encoding="utf-8")
    if a.max_steps:
        print(f"[train] SMOKE RUN: {a.max_steps} steps only")

    print(f"[train] pushing {slug}")
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)], check=False)
    print(f"[train] live log: https://www.kaggle.com/code/{slug}")
    print(f"[train] poll with: python scripts/kaggle_run.py watch --lang {a.lang}")


def cmd_vocab(a):
    """Push one vocabulary-sweep run.

    A single kernel serves all four sizes: VOCAB is rewritten in the source and
    the kernel id is suffixed, so each vocabulary gets its own kernel and its own
    retrievable output.

    Each run is capped at 8 h (see the kernel). One epoch is ~5-6 h, so the cap
    normally never fires -- but if it does, ``--resume`` re-pushes the same
    vocabulary with its own previous output mounted, continuing from that
    checkpoint rather than restarting.
    """
    import os
    import re as _re
    kdir = ROOT / "kaggle" / "vocab_sweep"
    src = kdir / "train_kernel.py"
    body = src.read_text(encoding="utf-8")
    body = _re.sub(r"^VOCAB = \d+", f"VOCAB = {a.vocab}", body, count=1, flags=_re.M)
    body = _re.sub(r'^LANG = ".*?"', f'LANG = "{a.lang}"', body, count=1, flags=_re.M)
    src.write_text(body, encoding="utf-8")

    # --config-dir points KAGGLE_CONFIG_DIR at a different credentials file, so a
    # run can be pushed from a second account. Every id and dataset reference
    # must then use THAT account's username: datasets are per-account, and a
    # cross-account reference fails at mount time, after the GPU is allocated.
    env = dict(os.environ)
    user = USER
    if a.config_dir:
        env["KAGGLE_CONFIG_DIR"] = a.config_dir
        user = json.loads((Path(a.config_dir) / "kaggle.json")
                          .read_text(encoding="utf-8"))["username"]
        print(f"[vocab] account: {user} (credentials from {a.config_dir})")

    # "-train" distinguishes the KERNEL from the token DATASET of the same
    # vocabulary: on the second account both would be `lma-ne-vocab-8000`, and
    # Kaggle rejects the push with a 409 Conflict rather than disambiguating.
    prefix = "lma-vocab" if a.lang == "hindi" else "lma-ne-vocab"
    slug = f"{user}/{prefix}-{a.vocab}-train"
    meta_path = kdir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["id"] = slug
    # The title must slugify to the id, or Kaggle silently creates the kernel at
    # a DIFFERENT url than the one in meta["id"] -- which then breaks watch/fetch.
    # Deriving it from the slug keeps the two in step by construction.
    meta["title"] = slug.split("/", 1)[1]
    if a.datasets:
        meta["dataset_sources"] = a.datasets
    # Self-reference only on --resume: on a first push the kernel has no
    # completed version, and naming a nonexistent source fails the push.
    meta["kernel_sources"] = [slug] if a.resume else []
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[vocab] pushing {a.lang} V={a.vocab:,} -> {slug}"
          + ("  (resuming from its own last.pt)" if a.resume else ""))
    print(f"[vocab] datasets: {meta['dataset_sources']}")
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)],
                   check=False, env=env)
    print(f"[vocab] live log: https://www.kaggle.com/code/{slug}")


def cmd_finetune(a):
    """Push one Phase-3 reasoning finetune.

    The pretrained checkpoint arrives through ``kernel_sources`` as the Phase-2
    training kernel's output, so no 300 MB upload is needed. The reasoning data
    comes from the ``lma-reasoning-data`` dataset covering both languages.
    """
    import re as _re
    kdir = ROOT / "kaggle" / "finetune"
    src = kdir / "finetune_kernel.py"
    body = _re.sub(r'^LANG = ".*?"', f'LANG = "{a.lang}"',
                   src.read_text(encoding="utf-8"), count=1, flags=_re.M)
    src.write_text(body, encoding="utf-8")

    slug = f"{USER}/lma-ft-{a.lang}"
    meta_path = kdir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["id"] = slug
    meta["title"] = slug.split("/", 1)[1]      # must slugify to the id
    meta["kernel_sources"] = [KERNEL[a.lang]]  # the pretrained checkpoint
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ft] pushing {a.lang} -> {slug}")
    print(f"[ft] pretrained from: {KERNEL[a.lang]}")
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)],
                   check=False)
    print(f"[ft] live log: https://www.kaggle.com/code/{slug}")


def cmd_sweep(a):
    """Push the Phase-3 data-scaling sweep for one language.

    One kernel covers the whole sweep: the pretrained baseline plus every
    training-set size, each measured on all four test sets with both accuracy
    metrics and with language perplexity. Roughly 1.5 h per language, so the
    two run concurrently in Kaggle's two GPU slots.
    """
    import re as _re
    kdir = ROOT / "kaggle" / "sweep"
    src = kdir / "sweep_kernel.py"
    body = _re.sub(r'^LANG = ".*?"', f'LANG = "{a.lang}"',
                   src.read_text(encoding="utf-8"), count=1, flags=_re.M)
    if a.sizes:
        body = _re.sub(r"^SIZES = \[.*?\]", f"SIZES = {a.sizes}", body, count=1, flags=_re.M)
    if a.test_limit:
        body = _re.sub(r"^TEST_LIMIT = \d+", f"TEST_LIMIT = {a.test_limit}",
                       body, count=1, flags=_re.M)
    src.write_text(body, encoding="utf-8")

    env, user = account(a.config_dir)
    slug = f"{user}/lma-sweep-{a.lang}"
    meta_path = kdir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["id"] = slug
    meta["title"] = slug.split("/", 1)[1]      # must slugify to the id
    meta["dataset_sources"], meta["kernel_sources"] = phase3_inputs(a.lang, user)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[sweep] pushing {a.lang} -> {slug}")
    print(f"[sweep] sizes {a.sizes or 'default'} | test limit {a.test_limit or 'default'}")
    print(f"[sweep] inputs {meta['dataset_sources']} {meta['kernel_sources']}")
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)],
                   check=False, env=env)
    print(f"[sweep] live log: https://www.kaggle.com/code/{slug}")


ABLATION_RUNS = ["baseline", "ood_val", "ood_val_2ep", "name_pool", "replay",
                 "name_pool_replay", "epochs3", "epochs4", "ood_val_4ep", "name_pool_replay_4ep"]


def cmd_ablation(a):
    """Push the Phase-3 best-model retrain and technique ablation for one language.

    ``--runs baseline`` retrains the official best model alone; the technique
    runs follow once it is in hand. All runs in one push share a session, so the
    list decides how long the kernel runs (~20 min per run).
    """
    import re as _re
    kdir = ROOT / "kaggle" / "ablation"
    src = kdir / "ablation_kernel.py"
    body = _re.sub(r'^LANG = ".*?"', f'LANG = "{a.lang}"',
                   src.read_text(encoding="utf-8"), count=1, flags=_re.M)
    for spec in a.runs:
        if spec.partition("@")[0] not in ABLATION_RUNS:
            raise SystemExit(f"unknown run {spec!r}; choose from {ABLATION_RUNS} (optionally name@seed)")
    body = _re.sub(r"^RUNS = \[.*?\]", f"RUNS = {a.runs!r}".replace("'", '"'),
                   body, count=1, flags=_re.M)
    body = _re.sub(r"^SIZE = \d+", f"SIZE = {a.size}", body, count=1, flags=_re.M)
    body = _re.sub(r"^LR = [0-9.e-]+", f"LR = {a.lr}", body, count=1, flags=_re.M)
    src.write_text(body, encoding="utf-8")

    env, user = account(a.config_dir)
    # A separate kernel per batch of runs: Kaggle only serves the output of a
    # kernel's latest version, so pushing new runs to a running kernel would make
    # the earlier runs' checkpoints unreachable.
    slug = f"{user}/lma-ablation-{a.lang}" + (f"-{a.name}" if a.name else "")
    meta_path = kdir / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["id"] = slug
    meta["title"] = slug.split("/", 1)[1]      # must slugify to the id
    meta["dataset_sources"], meta["kernel_sources"] = phase3_inputs(a.lang, user)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ablation] pushing {a.lang} size {a.size:,} lr {a.lr} runs {a.runs} -> {slug}")
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)],
                   check=False, env=env)
    print(f"[ablation] live log: https://www.kaggle.com/code/{slug}")


def _upload_dataset(stage: Path, dataset_id: str, title: str, env: dict, message: str) -> None:
    """Create ``dataset_id`` from ``stage``, or add a version if it exists."""
    (stage / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": dataset_id, "licenses": [{"name": "unknown"}]}, indent=2),
        encoding="utf-8")
    probe = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "files", dataset_id],
                           capture_output=True, text=True, env=env)
    # Look for the error text, not "404": a listed file size can contain "404".
    out = (probe.stdout + probe.stderr).lower()
    exists = probe.returncode == 0 and "not found" not in out and "size" in out
    verb = ["version", "-m", message] if exists else ["create"]
    print(f"[sync] {'new version of' if exists else 'creating'} {dataset_id}")
    subprocess.run([sys.executable, "-m", "kaggle", "datasets", verb[0], "-p", str(stage),
                    "--dir-mode", "zip", *verb[1:]], check=False, env=env)


def cmd_sync_reasoning(a):
    """Upload the generated reasoning data for the chosen languages to one account."""
    import shutil, tempfile
    stage = Path(tempfile.mkdtemp(prefix="lma-reason-"))
    for lang in a.langs:
        src = ROOT / lang / "reasoning" / "data"
        if not (src / "train.jsonl").exists():
            raise SystemExit(f"{src}/train.jsonl missing -- run make_reasoning_data.py first")
        dst = stage / lang
        dst.mkdir(parents=True)
        for f in src.glob("*.jsonl"):
            shutil.copy2(f, dst)
        for f in ("meta.json", "shortcut_audit.json"):
            if (src / f).exists():
                shutil.copy2(src / f, dst)
        # Tokenizers ship alongside so the kernel need not depend on the code
        # dataset carrying them.
        code = "hi" if lang == "hindi" else "ne"
        shutil.copy2(ROOT / lang / "tokenizer" / f"{code}_unigram_10000.model", stage / lang)
        n = sum(1 for _ in (src / "train.jsonl").open(encoding="utf-8"))
        print(f"[sync] {lang}: {n:,} train examples")
    env, user = account(a.config_dir)
    _upload_dataset(stage, f"{user}/lma-reasoning-data", "LMA Reasoning Data", env,
                    "regenerated reasoning data")
    shutil.rmtree(stage, ignore_errors=True)


def cmd_sync_pretrained(a):
    """Upload one language's pretrained checkpoint and token files to an account.

    Only needed on an account that cannot mount the Phase-2 training kernel's
    output (the second account used for Nepali). Files are hard-linked into the
    staging folder rather than copied, since the token file alone is ~900 MB.
    """
    import os, shutil, tempfile
    env, user = account(a.config_dir)
    code = "hi" if a.lang == "hindi" else "ne"
    ckpt = ROOT / a.lang / "checkpoints" / ("two_epoch/last.pt" if a.lang == "hindi" else "last.pt")
    tokens = ROOT / a.lang / "data" / "tokens"

    def link(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    stage = Path(tempfile.mkdtemp(prefix="lma-tok-"))
    for f in ("train.bin", "test.bin", "meta.json"):
        link(tokens / f, stage / a.lang / f)
    _upload_dataset(stage, f"{user}/lma-phase1-tokens-{code}", f"LMA Phase1 Tokens {code.upper()}",
                    env, "token files")
    shutil.rmtree(stage, ignore_errors=True)

    stage = Path(tempfile.mkdtemp(prefix="lma-ckpt-"))
    link(ckpt, stage / "pretrained" / "last.pt")
    _upload_dataset(stage, f"{user}/lma-pretrained-{a.lang}", f"LMA Pretrained {a.lang.title()}",
                    env, "pretrained checkpoint")
    shutil.rmtree(stage, ignore_errors=True)


def cmd_watch(a):
    """Poll until the kernel leaves the running state."""
    slug, name = target(a)
    print(f"[watch] {slug}  (Ctrl-C to stop watching; the run continues on Kaggle)")
    last = None
    t0 = time.time()
    try:
        while True:
            st, msg = kernel_status(slug)
            if st != last:
                print(f"[watch] {time.strftime('%H:%M:%S')}  {st}  {msg[:60]}")
                last = st
            if st.lower() in ("complete", "error", "cancelacknowledged", "cancelrequested"):
                mins = (time.time() - t0) / 60
                print(f"[watch] finished after {mins:.1f} min of watching -> {st}")
                if st.lower() == "complete":
                    flag = f"--vocab {a.vocab}" if getattr(a, "vocab", None) else f"--lang {a.lang}"
                    print(f"[watch] fetch it: python scripts/kaggle_run.py fetch {flag}")
                return 0 if st.lower() == "complete" else 1
            time.sleep(a.interval)
    except KeyboardInterrupt:
        # Ctrl-C stops the polling, never the run: the job lives on Kaggle.
        print("\n[watch] stopped watching. The run CONTINUES on Kaggle.")
        print(f"[watch] live log: https://www.kaggle.com/code/{slug}")
        print(f"[watch] do NOT re-run 'train' while it is running -- that restarts it from zero.")
        return 0


def cmd_fetch(a):
    """Download the kernel output: checkpoints, training log, console log."""
    slug, name = target(a)
    # Sweep runs land under report/, beside the analysis that consumes them;
    # language runs land beside their language's other artifacts.
    dest = (ROOT / "report" / "Phase 2" / "vocab_sweep" / name if getattr(a, "vocab", None)
            else ROOT / name / "kaggle_output")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {slug} -> {dest}")
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", slug,
                    "-p", str(dest)], check=False)

    ckpt = dest / "checkpoints" / "last.pt"
    if ckpt.exists():
        import torch
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        print(f"[fetch] checkpoint at step {c['step']:,}, best val {c['best_val']:.4f}")
        if not getattr(a, "vocab", None):
            # Sweep checkpoints stay where they landed: they are compared with
            # each other, never used as *the* model for a language.
            live = ROOT / name / "checkpoints"
            live.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(ckpt, live / "last.pt")
            print(f"[fetch] copied to {live/'last.pt'} (local eval can use it directly)")
    else:
        print(f"[fetch] no checkpoint in output; check the console log in {dest}")


def cmd_progress(a):
    """Summarise the downloaded training log."""
    log = ROOT / a.lang / "kaggle_output" / "checkpoints" / "train_log.jsonl"
    if not log.exists():
        log = ROOT / a.lang / "checkpoints" / "train_log.jsonl"
    if not log.exists():
        print(f"no training log yet — run: python scripts/kaggle_run.py fetch --lang {a.lang}")
        return
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    train = [r for r in rows if "loss" in r]
    val = [r for r in rows if "val_loss" in r]
    if train:
        last = train[-1]
        print(f"  step        : {last['step']:,}")
        print(f"  train loss  : {last['loss']:.4f}")
        print(f"  tokens/sec  : {last.get('tokens_per_sec', 0):,}")
    if val:
        best = min(val, key=lambda r: r["val_loss"])
        print(f"  best val    : {best['val_loss']:.4f} (ppl {best['val_ppl']:.1f}) at step {best['step']:,}")
        print(f"  latest val  : {val[-1]['val_loss']:.4f} (ppl {val[-1]['val_ppl']:.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="status of every kernel").set_defaults(fn=cmd_status)

    s = sub.add_parser("sync-code", help="push local lmagpt/ + configs to Kaggle")
    s.add_argument("-m", "--message", default="update code")
    s.add_argument("--config-dir", default=None, help="second account's KAGGLE_CONFIG_DIR")
    s.set_defaults(fn=cmd_sync_code)

    s = sub.add_parser("train", help="start or continue training")
    s.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    s.add_argument("--fresh", action="store_true", help="ignore previous checkpoint")
    s.add_argument("--max-steps", type=int, default=None,
                   help="cap steps (smoke run); omit for the full schedule")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("vocab", help="launch one vocabulary-sweep run")
    s.add_argument("--vocab", type=int, required=True, choices=VOCABS)
    s.add_argument("--lang", default="hindi", choices=["hindi", "nepali"])
    s.add_argument("--config-dir", default=None,
                   help="KAGGLE_CONFIG_DIR holding a different account's kaggle.json")
    s.add_argument("--datasets", nargs="+", default=None,
                   help="override dataset_sources (required when pushing from an "
                        "account whose datasets differ from the default)")
    s.add_argument("--resume", action="store_true",
                   help="continue this vocabulary from its own last checkpoint "
                        "(only needed if a run hit the 8 h cap)")
    s.set_defaults(fn=cmd_vocab)

    s = sub.add_parser("sync-reasoning", help="upload the Phase-3 reasoning data")
    s.add_argument("--langs", nargs="+", default=["hindi", "nepali"], choices=["hindi", "nepali"])
    s.add_argument("--config-dir", default=None, help="second account's KAGGLE_CONFIG_DIR")
    s.set_defaults(fn=cmd_sync_reasoning)

    s = sub.add_parser("sync-pretrained", help="upload a pretrained checkpoint + tokens to an account")
    s.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    s.add_argument("--config-dir", default=None, help="second account's KAGGLE_CONFIG_DIR")
    s.set_defaults(fn=cmd_sync_pretrained)

    s = sub.add_parser("finetune", help="run a Phase-3 reasoning finetune")
    s.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    s.set_defaults(fn=cmd_finetune)

    s = sub.add_parser("sweep", help="run the Phase-3 data-scaling sweep")
    s.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    s.add_argument("--sizes", type=int, nargs="+", default=None,
                   help="training-set sizes; default 5k 10k 20k 30k 100k")
    s.add_argument("--test-limit", type=int, default=None,
                   help="items per test set; default 600")
    s.add_argument("--config-dir", default=None, help="second account's KAGGLE_CONFIG_DIR")
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("ablation", help="run the Phase-3 best-model retrain and ablation")
    s.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    s.add_argument("--runs", nargs="+", default=["baseline"],
                   help=f"any of {ABLATION_RUNS}; append @seed to repeat with another seed")
    s.add_argument("--size", type=int, required=True, help="training slice, chosen from the sweep")
    s.add_argument("--name", default=None, help="kernel suffix, so separate batches keep their outputs")
    s.add_argument("--lr", type=float, default=6e-5, help="peak learning rate (default 6e-5)")
    s.add_argument("--config-dir", default=None, help="second account's KAGGLE_CONFIG_DIR")
    s.set_defaults(fn=cmd_ablation)

    s = sub.add_parser("watch", help="poll until the run ends")
    s.add_argument("--lang", choices=["hindi", "nepali", "benchmark"])
    s.add_argument("--vocab", type=int, choices=VOCABS, help="watch a sweep run instead")
    s.add_argument("--interval", type=int, default=60)
    s.set_defaults(fn=cmd_watch)

    s = sub.add_parser("fetch", help="download checkpoints and logs")
    s.add_argument("--lang", choices=["hindi", "nepali", "benchmark"])
    s.add_argument("--vocab", type=int, choices=VOCABS, help="fetch a sweep run instead")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("progress", help="summarise the training log")
    s.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    s.set_defaults(fn=cmd_progress)

    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
