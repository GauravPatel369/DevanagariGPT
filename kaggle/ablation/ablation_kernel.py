"""Kaggle kernel -- Phase 3 technique ablation and final models, one language.

LANG, RUNS and SIZE are rewritten by scripts/kaggle_run.py before each push.

Every run starts from the same pretrained checkpoint and trains on the same
slice (SIZE, the size the sweep chose on validation data) with the same learning
rate and batch order, and changes exactly one thing:

    baseline          plain finetuning, checkpoint chosen on normal validation loss
    ood_val           checkpoint chosen on answer accuracy for unseen names/numbers
    ood_val_2ep       the same with a two-epoch budget: is one epoch enough?
    name_pool         training names drawn from a pool of ~175 instead of 26
    replay            each step also trains on Phase-1 corpus windows (LM loss)
    name_pool_replay  both of the last two

A run written as "replay@1338" repeats that configuration with another seed
(data order, dropout, replay windows), which is how run-to-run noise is measured.

Each run is measured the same way: forced-choice accuracy, first-word generation
accuracy and invalid rate on 600 items of each test set, plus perplexity and
bits-per-byte on the Phase-1/2 test split.
"""
# stdlib only until the bootstrap has settled which torch build to use.
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

LANG = "nepali"                 # rewritten per push
RUNS = ["name_pool_replay_4ep"]            # rewritten per push
SIZE = 10000                   # rewritten per push: the size the sweep chose
SEED = 1337                    # default seed; a run can override it as "name@seed"
TEST_LIMIT = 600
BATCH_SIZE = 32
LR = 0.0001                      # rewritten per push when --lr is given
EVAL_EVERY = 50
REPLAY_BATCH = 4               # 4 x 512 tokens of corpus text per step
REPLAY_CONTEXT = 512

# Local smoke test only: LMA_INPUT/LMA_WORK point at a mock of the Kaggle mounts
# and LMA_SMOKE caps the run to a few steps. Unset on Kaggle.
INPUT = os.environ.get("LMA_INPUT", "/kaggle/input")
SMOKE = int(os.environ.get("LMA_SMOKE", "0"))
if SMOKE:
    SIZE, TEST_LIMIT, EVAL_EVERY = SMOKE, 4, 10

# Runs that are, or may become, a final model. The rest are comparisons only
# and keep just the selected weights (a resumable checkpoint is ~300 MB).
KEEP_RESUMABLE = {"baseline", "replay", "name_pool", "name_pool_replay", "ood_val", "ood_val_2ep",
                  "ood_val_4ep", "name_pool_replay_4ep"}

CONFIGS = {
    "baseline":    dict(train="train.jsonl", select="val", epochs=1, replay=False),
    # Checkpoint chosen by answer accuracy on unseen names and numbers. The
    # first dataset used loss here, which stopped Nepali before it had learned
    # the answer format; accuracy measures what the evaluation cares about.
    "ood_val":     dict(train="train.jsonl", select="val_ood_acc", epochs=1, replay=False),
    "ood_val_2ep": dict(train="train.jsonl", select="val_ood_acc", epochs=2, replay=False),
    "name_pool":   dict(train="train_namepool.jsonl", select="val", epochs=1, replay=False),
    "replay":      dict(train="train.jsonl", select="val", epochs=1, replay=True),
    "name_pool_replay": dict(train="train_namepool.jsonl", select="val", epochs=1, replay=True),
    # The two best recipes, given four passes over the data: does longer
    # training finally teach the comparison?
    "ood_val_4ep": dict(train="train.jsonl", select="val_ood_acc", epochs=4, replay=False),
    "name_pool_replay_4ep": dict(train="train_namepool.jsonl", select="val", epochs=4, replay=True),
    # More epochs of plain finetuning. Accuracy on the unseen-name validation set
    # is recorded at the end of every epoch, to show whether each extra pass
    # helps or only memorises the training names.
    "epochs3": dict(train="train.jsonl", select="val", epochs=3, replay=False),
    "epochs4": dict(train="train.jsonl", select="val", epochs=4, replay=False),
}


def find(pattern):
    """First path matching a glob under /kaggle/input, or None."""
    hits = glob.glob("%s/**/%s" % (INPUT, pattern), recursive=True)
    return hits[0] if hits else None


CODE = os.path.dirname(find("lmagpt") or "")
# A dataset holding one language only is flattened by Kaggle on upload (the
# second account's Nepali data), so fall back to the top level when the
# language folder is absent.
TRAIN_JSONL = find("%s/train.jsonl" % LANG) or find("train.jsonl")
TOKENIZER = find("%s_unigram_10000.model" % ("hi" if LANG == "hindi" else "ne"))
TEST_BIN = find("%s/test.bin" % LANG) or find("test.bin")
TRAIN_BIN = find("%s/train.bin" % LANG) or find("train.bin")
if not CODE or not TRAIN_JSONL or not TOKENIZER:
    raise SystemExit("missing input: code=%s data=%s tok=%s" % (CODE, TRAIN_JSONL, TOKENIZER))
DATA = os.path.dirname(TRAIN_JSONL)
print("[diag] code=%s\n[diag] data=%s (%s)\n[diag] tok=%s\n[diag] test.bin=%s\n[diag] train.bin=%s"
      % (CODE, DATA, sorted(os.listdir(DATA)), TOKENIZER, TEST_BIN, TRAIN_BIN), flush=True)


def parse_run(spec):
    """``"replay@1338"`` -> ``("replay", 1338)``; a bare name uses SEED."""
    name, _, seed = spec.partition("@")
    return name, int(seed) if seed else SEED


for spec in RUNS:
    r = parse_run(spec)[0]
    need = CONFIGS[r]["train"]
    if not os.path.exists(os.path.join(DATA, need)):
        raise SystemExit("run %s needs %s, missing from the data dataset" % (r, need))
    if CONFIGS[r]["replay"] and not TRAIN_BIN:
        raise SystemExit("run %s needs %s/train.bin from lma-phase1-tokens" % (r, LANG))
sys.path.insert(0, CODE)

# Must run BEFORE torch is imported: Kaggle allocates a P100 (sm_60) and the
# preinstalled torch has no Pascal kernels.
from lmagpt.kaggle_bootstrap import ensure_compatible_torch, report_device  # noqa: E402

ensure_compatible_torch()
report_device()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import sentencepiece as spm  # noqa: E402

from lmagpt.data import TokenDataset  # noqa: E402
from lmagpt.eval_reasoning import evaluate, load  # noqa: E402
from lmagpt.evaluate import add_bits_per_byte, intrinsic_metrics  # noqa: E402
from lmagpt.finetune import ReasoningDataset, validation_loss  # noqa: E402
from lmagpt.train import build_optimizer, save_checkpoint  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda" and torch.cuda.is_bf16_supported():
    autocast_dtype = torch.bfloat16
elif device.type == "cuda":
    autocast_dtype = torch.float16
else:
    autocast_dtype = None

cands = [p for p in glob.glob("%s/**/last.pt" % INPUT, recursive=True)
         if "reasoning" not in p and "sweep" not in p and "ablation" not in p]
if not cands:
    raise SystemExit("no pretrained last.pt; is kernel_sources set?")
PRETRAINED = max(cands, key=os.path.getmtime)
print("[kaggle] pretrained -> %s" % PRETRAINED, flush=True)

WORK = Path(os.environ.get("LMA_WORK", "/kaggle/working"))
sp = spm.SentencePieceProcessor(model_file=TOKENIZER)


def reasoning_subset(name, n):
    """The first ``n`` rows of one jsonl, tokenised with the prompt masked.

    Rows are cut before encoding so every run sees the same examples in the
    same order. No row exceeds the length cap, which the skip count confirms.
    """
    src = os.path.join(DATA, name)
    with open(src, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    tmp = WORK / "tmp" / name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if SMOKE and not n:
        n = 200
    tmp.write_text("".join(lines[:n] if n else lines), encoding="utf-8")
    return ReasoningDataset(tmp, sp, 128)


VAL = {"val": reasoning_subset("val.jsonl", None)}
if os.path.exists(os.path.join(DATA, "val_ood.jsonl")):
    VAL["val_ood"] = reasoning_subset("val_ood.jsonl", None)
print("[data] validation sets: %s" % {k: len(v) for k, v in VAL.items()}, flush=True)
# 500 unseen-name questions for accuracy-based checkpoint selection: enough to
# rank checkpoints, small enough to score every 50 steps.
VAL_OOD_ROWS = []
if os.path.exists(os.path.join(DATA, "val_ood.jsonl")):
    with open(os.path.join(DATA, "val_ood.jsonl"), encoding="utf-8") as f:
        VAL_OOD_ROWS = [json.loads(l) for l in f if l.strip()][:500 if not SMOKE else 20]
for spec in RUNS:
    r = parse_run(spec)[0]
    need = CONFIGS[r]["select"]
    if (need == "val_ood_acc" and not VAL_OOD_ROWS) or (need != "val_ood_acc" and need not in VAL):
        raise SystemExit("run %s selects on %s, which is missing" % (r, need))

TEST_SETS = {}
for split in ("test_a", "test_b", "test_c", "test_d"):
    with open(os.path.join(DATA, "%s.jsonl" % split), encoding="utf-8") as f:
        TEST_SETS[split] = [json.loads(l) for l in f if l.strip()]

REPLAY = TokenDataset(Path(TRAIN_BIN), REPLAY_CONTEXT) if TRAIN_BIN and any(
    CONFIGS[parse_run(r)[0]]["replay"] for r in RUNS) else None


def language_metrics(model, cfg):
    """Perplexity and bits-per-byte on the Phase-1/2 test split (forgetting)."""
    if not TEST_BIN:
        return None
    with open(os.path.join(os.path.dirname(TEST_BIN), "meta.json"), encoding="utf-8") as f:
        bpt = json.load(f)["splits"]["test"]["bytes_per_token"]
    m = intrinsic_metrics(model, TokenDataset(Path(TEST_BIN), cfg.context), device,
                          autocast_dtype, batch_size=16,
                          max_tokens=20000 if SMOKE else None)
    add_bits_per_byte(m, bpt)
    return m


def ood_accuracy(model):
    """First-word answer accuracy on the unseen-name validation questions."""
    model.eval()
    r = evaluate(model, sp, VAL_OOD_ROWS, device, autocast_dtype, None, with_generation=True)
    model.train()
    return r["exact_match"]


def train_run(spec):
    """One finetune under one configuration; returns the selected model."""
    name, seed = parse_run(spec)
    conf = CONFIGS[name]
    torch.manual_seed(seed)
    train = reasoning_subset(conf["train"], SIZE)
    items = train.items
    model, ckpt = load(LANG, PRETRAINED, device)
    cfg = model.cfg
    tcfg = dict(ckpt.get("train_config", {}))
    tcfg.update(learning_rate=LR, weight_decay=0.1, beta1=0.9, beta2=0.95)
    optimizer = build_optimizer(model, tcfg)

    per_epoch = len(items) // BATCH_SIZE
    steps = per_epoch * conf["epochs"]
    warmup = min(100, max(10, steps // 10))
    replay_rng = np.random.default_rng(seed)
    print("  %d examples, %d steps (%d epoch), select on %s, replay %s, seed %d"
          % (len(items), steps, conf["epochs"], conf["select"], conf["replay"], seed), flush=True)

    curve, best_val, best_state, best_step, step = [], float("inf"), None, 0, 0
    epoch_acc, best_opt = [], None
    model.train()
    t0 = time.time()
    for epoch in range(conf["epochs"]):
        # With the default seed, epoch 0 uses the sweep's shuffle, so the
        # baseline reproduces the sweep's model for the same size.
        order = np.arange(len(items))
        np.random.default_rng(seed + epoch).shuffle(order)
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
                lm_loss = None
                if conf["replay"]:
                    rx, ry = REPLAY.batch(REPLAY_BATCH, replay_rng, device)
                    _, lm_loss, _ = model(rx, targets=ry)
                    loss = loss + lm_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if conf["epochs"] > 1 and VAL_OOD_ROWS and i + 2 * BATCH_SIZE > len(order):
                # Last batch of this epoch: one accuracy check per epoch.
                acc = round(ood_accuracy(model), 4)
                epoch_acc.append({"epoch": epoch + 1, "step": step, "val_ood_acc": acc})
                print("    end of epoch %d: val_ood accuracy %.1f%%" % (epoch + 1, acc * 100), flush=True)
            if step % EVAL_EVERY == 0 or step == steps:
                point = {"step": step, "train_loss": round(loss.item(), 4)}
                if lm_loss is not None:
                    point["replay_lm_loss"] = round(lm_loss.item(), 4)
                for vname, vds in VAL.items():
                    point[vname] = round(validation_loss(model, vds, device, autocast_dtype,
                                                         BATCH_SIZE), 4)
                if conf["select"] == "val_ood_acc":
                    point["val_ood_acc"] = round(ood_accuracy(model), 4)
                curve.append(point)
                # Accuracy is better when higher, loss when lower; compare as a cost.
                cost = -point["val_ood_acc"] if conf["select"] == "val_ood_acc" else point[conf["select"]]
                if cost < best_val:
                    best_val, best_step = cost, step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if name in KEEP_RESUMABLE:
                        # The selected step can be early (new-name validation
                        # stops as soon as accuracy peaks), so the optimizer state
                        # at that step is kept too: best.pt itself is resumable.
                        import copy
                        best_opt = copy.deepcopy(optimizer.state_dict())
                print("    %s" % point, flush=True)

    out_dir = WORK / (name if seed == SEED else "%s_s%d" % (name, seed))
    # The end-of-training weights are kept as well, so a run whose selected
    # step is early can also be scored after all its epochs.
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if name in KEEP_RESUMABLE:
        # Final models also keep a resumable checkpoint: weights, optimizer and
        # step at the end of training, so finetuning can continue from it.
        save_checkpoint(out_dir / "last.pt", model, optimizer, step, tcfg, cfg, best_val,
                        np.random.default_rng(1337).bit_generator.state)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model": best_state, "model_config": cfg.to_dict(), "step": best_step,
               "best_val": best_val, "selected_on": conf["select"], "run": name, "seed": seed,
               "n_samples": len(items), "train_config": tcfg}
    if best_opt is not None:
        payload.update(optimizer=best_opt, numpy_rng=np.random.default_rng(seed).bit_generator.state,
                       torch_rng=torch.get_rng_state())
    torch.save(payload, out_dir / "best.pt")
    return model, cfg, dict(run=name, seed=seed, n_samples=len(items), steps=steps, epochs=conf["epochs"],
                            selected_on=conf["select"], best_step=best_step,
                            best_val=round(best_val, 4), train_seconds=round(time.time() - t0, 1),
                            curve=curve, epoch_val_ood_acc=epoch_acc), final_state


def measure(model, cfg, meta):
    """Every metric for one model, per test set."""
    rec = dict(meta, test_sets={})
    for split, rows in sorted(TEST_SETS.items()):
        r = evaluate(model, sp, rows, device, autocast_dtype, TEST_LIMIT, with_generation=True)
        r.pop("errors", None)
        rec["test_sets"][split] = r
        print("    %s: choice %.1f%%  generation %.1f%%  invalid %.1f%%"
              % (split, r["accuracy"] * 100, r["exact_match"] * 100,
                 r["generation_invalid_rate"] * 100), flush=True)
    lm = language_metrics(model, cfg)
    if lm:
        rec["language"] = lm
        print("    language: ppl %.2f  bpb %.4f" % (lm["perplexity"], lm["bits_per_byte"]), flush=True)
    return rec


out_path = WORK / ("ablation_results_%s.json" % LANG)
results = {"language": LANG, "size": SIZE, "test_limit": TEST_LIMIT, "lr": LR,
           "pretrained": PRETRAINED, "runs": {}}
started = time.time()
for name in RUNS:
    print("\n" + "=" * 70 + "\n  %s  %s\n" % (LANG.upper(), name) + "=" * 70, flush=True)
    try:
        model, cfg, meta, final_state = train_run(name)
        print("  selected step %d of %d (%s %.4f)" % (meta["best_step"], meta["steps"],
                                                    meta["selected_on"], meta["best_val"]), flush=True)
        results["runs"][name] = measure(model, cfg, meta)
        if meta["best_step"] != meta["steps"]:
            # Also score the model as it stands after every epoch has run.
            print("  scoring the end-of-training model (step %d)" % meta["steps"], flush=True)
            model.load_state_dict({k: v.to(device) for k, v in final_state.items()})
            model.eval()
            end_meta = dict(meta, best_step=meta["steps"], note="end of training, not selected")
            results["runs"][name + "_end"] = measure(model, cfg, end_meta)
        del model, final_state
    except Exception as e:          # one broken run must not lose the others
        import traceback
        traceback.print_exc()
        results["runs"][name] = {"error": repr(e)}
    torch.cuda.empty_cache()
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  [saved] %s" % out_path, flush=True)

import shutil  # noqa: E402

shutil.rmtree(WORK / "tmp", ignore_errors=True)   # copies of the inputs, not outputs
print("\n[done] %.1f min total" % ((time.time() - started) / 60), flush=True)
