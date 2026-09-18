"""Interactive text generation with the trained models.

Type a Hindi or Nepali prompt and the model continues it. Decoding settings can
be changed mid-session, so the effect of temperature, top-p, top-k and the
anti-repetition controls can be felt directly rather than read off a table.

    python scripts/generate.py                      # interactive, Hindi
    python scripts/generate.py --lang nepali
    python scripts/generate.py --model ablation     # the no-positional model
    python scripts/generate.py --text "..." --compare   # one prompt, every decoder

Inside the session:

    <any text>        continue it
    :lang nepali      switch model (hindi | nepali | ablation)
    :set T=0.8        temperature
    :set top_p=0.9    nucleus threshold        (:set top_p=off to disable)
    :set top_k=50     fixed-size truncation
    :set rep=1.2      repetition penalty
    :set norepeat=4   block repeated 4-grams
    :set n=100        tokens to generate
    :greedy           switch to greedy decoding
    :sample           switch back to sampling
    :show             current settings
    :compare          run the current prompt through every preset
    :quit
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lmagpt.data import resolve_tokenizer  # noqa: E402
from lmagpt.model import GPT, GPTConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# The sweet spot measured in report/Phase 2/evaluation_test.json: the setting
# whose repetition rate and distinct-2 land closest to real human text.
DEFAULTS = {"temperature": 1.0, "top_p": 0.95, "top_k": None,
            "repetition_penalty": None, "no_repeat_ngram_size": None,
            "greedy": False, "n": 80}

PRESETS = [
    ("greedy", dict(greedy=True)),
    ("greedy+norepeat4", dict(greedy=True, no_repeat_ngram_size=4)),
    ("T0.5", dict(temperature=0.5)),
    ("T0.8", dict(temperature=0.8)),
    ("T1.0 (sweet spot base)", dict(temperature=1.0)),
    ("T1.0+topp0.95 (best)", dict(temperature=1.0, top_p=0.95)),
    ("T1.5", dict(temperature=1.5)),
]

MODELS = {
    "hindi": ("hindi/checkpoints/last.pt", "hindi"),
    "nepali": ("nepali/checkpoints/last.pt", "nepali"),
    "ablation": ("hindi_norope/checkpoints/last.pt", "hindi"),
}


def log_turn(path: Path, record: dict) -> None:
    """Append one interaction to the session log.

    JSONL rather than plain text: every turn keeps its decoding settings and
    timing alongside the output, so a sample quoted later in the report can
    always be traced back to the exact configuration that produced it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load(name: str, device):
    """Load one checkpoint plus the tokenizer for its language."""
    import sentencepiece as spm

    rel, lang = MODELS[name]
    path = ROOT / rel
    if not path.exists():
        raise SystemExit(f"{path} not found. Train it, or fetch it from Kaggle.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=str(resolve_tokenizer(lang, ROOT)))
    return model, sp, ckpt


def describe(name: str, model, ckpt) -> str:
    rope = ckpt["model_config"].get("use_rope", True)
    return (f"{name} | {model.num_parameters():,} params | step {ckpt['step']:,} | "
            f"val {ckpt['best_val']:.4f} | positional={'RoPE' if rope else 'NONE'}")


@torch.no_grad()
def run(model, sp, prompt: str, device, settings: dict):
    """Generate a continuation and report tokens/sec."""
    try:
        ids = sp.encode(prompt, out_type=int)
    except Exception as exc:
        print(f"  could not tokenize that input ({type(exc).__name__}). "
              f"If you pasted Devanagari into a Windows terminal, try "
              f"`chcp 65001` first.")
        return "", 0.0, 0
    if not ids:
        return "", 0.0, 0
    ctx = model.cfg.context
    x = torch.tensor([ids[-(ctx - settings["n"]):]], dtype=torch.long, device=device)

    kw = {k: v for k, v in settings.items() if k != "n" and v is not None}
    t0 = time.time()
    out = model.generate(x, max_new_tokens=settings["n"], **kw)
    dt = time.time() - t0
    new = out[0, x.size(1):].tolist()
    text = sp.decode(new)
    # sp.decode() strips the leading word-boundary marker, so a continuation
    # that begins a new word would otherwise be glued onto the prompt.
    if new and sp.id_to_piece(new[0]).startswith("▁") and text and not text[0].isspace():
        text = " " + text
    return text, dt, len(new)


def settings_line(s: dict) -> str:
    if s["greedy"]:
        base = "greedy"
    else:
        base = f"T={s['temperature']}"
        if s["top_p"]:
            base += f" top_p={s['top_p']}"
        if s["top_k"]:
            base += f" top_k={s['top_k']}"
    if s["repetition_penalty"]:
        base += f" rep={s['repetition_penalty']}"
    if s["no_repeat_ngram_size"]:
        base += f" norepeat={s['no_repeat_ngram_size']}"
    return f"{base}  ({s['n']} tokens)"


def compare(model, sp, prompt: str, device, n: int):
    """Run one prompt through every preset, so the differences are side by side."""
    print(f"\n  prompt: {prompt}\n" + "  " + "-" * 76)
    for label, kw in PRESETS:
        s = dict(DEFAULTS, **{"top_p": None, "n": n})
        s.update(kw)
        text, dt, ntok = run(model, sp, prompt, device, s)
        print(f"\n  [{label}]  {ntok/max(dt,1e-9):,.0f} tok/s")
        print(f"  {text[:400]}")
    print()


def interactive(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.model or args.lang
    model, sp, ckpt = load(name, device)
    settings = dict(DEFAULTS)
    if args.n:
        settings["n"] = args.n
    log_path = Path(args.log) if args.log else ROOT / "report" / "Phase 2" / "generation_session.jsonl"

    print("=" * 78)
    print("  " + describe(name, model, ckpt))
    print(f"  device: {device} | decoding: {settings_line(settings)}")
    print(f"  logging to: {log_path}")
    print("  type a prompt, or :help for commands, :quit to exit")
    print("=" * 78)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            continue

        if line in (":quit", ":q", ":exit"):
            print("bye")
            return
        if line in (":help", ":h"):
            print(__doc__[__doc__.index("Inside the session:"):])
            continue
        if line == ":show":
            print("  " + describe(name, model, ckpt))
            print("  decoding: " + settings_line(settings))
            continue
        if line == ":greedy":
            settings["greedy"] = True
            print("  -> greedy")
            continue
        if line == ":sample":
            settings["greedy"] = False
            print("  -> sampling: " + settings_line(settings))
            continue
        if line.startswith(":lang "):
            want = line.split(maxsplit=1)[1].strip()
            if want not in MODELS:
                print(f"  unknown model {want!r}; choose from {', '.join(MODELS)}")
                continue
            name = want
            model, sp, ckpt = load(name, device)
            print("  -> " + describe(name, model, ckpt))
            continue
        if line.startswith(":set "):
            try:
                key, val = line[5:].split("=", 1)
            except ValueError:
                print("  usage: :set T=0.8")
                continue
            key, val = key.strip().lower(), val.strip()
            mapping = {"t": "temperature", "temp": "temperature", "temperature": "temperature",
                       "top_p": "top_p", "topp": "top_p", "top_k": "top_k", "topk": "top_k",
                       "rep": "repetition_penalty", "norepeat": "no_repeat_ngram_size",
                       "n": "n"}
            if key not in mapping:
                print(f"  unknown setting {key!r}")
                continue
            field = mapping[key]
            if val.lower() in ("off", "none", "0"):
                settings[field] = None if field != "n" else DEFAULTS["n"]
            elif field in ("top_k", "no_repeat_ngram_size", "n"):
                settings[field] = int(val)
            else:
                settings[field] = float(val)
            if field != "n":
                settings["greedy"] = False
            print("  decoding: " + settings_line(settings))
            continue
        if line == ":compare":
            print("  give a prompt first, then :compare <prompt>")
            continue
        if line.startswith(":compare "):
            compare(model, sp, line[9:].strip(), device, settings["n"])
            continue

        text, dt, ntok = run(model, sp, line, device, settings)
        print(f"\n{line}{text}")
        print(f"\n  [{ntok} tokens in {dt:.2f}s = {ntok/max(dt,1e-9):,.0f} tok/s | "
              f"{settings_line(settings)}]")
        log_turn(log_path, {
            "time": _dt.datetime.now().isoformat(timespec="seconds"),
            "model": name,
            "prompt": line,
            "generated": text,
            "settings": settings_line(settings),
            "tokens": ntok,
            "seconds": round(dt, 2),
            "tokens_per_sec": round(ntok / max(dt, 1e-9)),
        })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--lang", default="hindi", choices=["hindi", "nepali"])
    ap.add_argument("--model", default=None, choices=list(MODELS),
                    help="overrides --lang; 'ablation' is the no-positional model")
    ap.add_argument("--text", default=None, help="one-shot prompt, no REPL")
    ap.add_argument("--compare", action="store_true", help="run --text through every preset")
    ap.add_argument("-n", type=int, default=None, help="tokens to generate")
    ap.add_argument("--log", default=None,
                    help="session log; default report/Phase 2/generation_session.jsonl")
    a = ap.parse_args()

    # Windows reads the console/pipe in the local codepage by default, which
    # mangles Devanagari into bytes SentencePiece cannot encode. Force UTF-8 on
    # both directions.
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass
    log_path = Path(a.log) if a.log else ROOT / "report" / "Phase 2" / "generation_session.jsonl"

    if a.text:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name = a.model or a.lang
        model, sp, ckpt = load(name, device)
        print("  " + describe(name, model, ckpt))
        if a.compare:
            # Every preset is logged, so the comparison is reproducible from the
            # file rather than only visible in the terminal.
            for label, kw in PRESETS:
                s = dict(DEFAULTS, **{"top_p": None, "n": a.n or DEFAULTS["n"]})
                s.update(kw)
                text, dt, ntok = run(model, sp, a.text, device, s)
                print(f"\n  [{label}]  {ntok/max(dt,1e-9):,.0f} tok/s")
                print(f"  {text[:400]}")
                log_turn(log_path, {
                    "time": _dt.datetime.now().isoformat(timespec="seconds"),
                    "model": name, "preset": label, "prompt": a.text,
                    "generated": text, "settings": settings_line(s),
                    "tokens": ntok, "seconds": round(dt, 2),
                    "tokens_per_sec": round(ntok / max(dt, 1e-9)),
                })
            print()
        else:
            s = dict(DEFAULTS)
            if a.n:
                s["n"] = a.n
            text, dt, ntok = run(model, sp, a.text, device, s)
            print(f"\n{a.text}{text}")
            print(f"\n  [{ntok} tokens in {dt:.2f}s | {settings_line(s)}]")
            log_turn(log_path, {
                "time": _dt.datetime.now().isoformat(timespec="seconds"),
                "model": name, "prompt": a.text, "generated": text,
                "settings": settings_line(s), "tokens": ntok,
                "seconds": round(dt, 2),
                "tokens_per_sec": round(ntok / max(dt, 1e-9)),
            })
        print(f"  logged to {log_path}")
        return
    interactive(a)


if __name__ == "__main__":
    main()
