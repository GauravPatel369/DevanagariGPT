"""Command-line entry point for the Phase-1 corpus pipeline.

    python -m lmacorpus.cli <stage> --lang hindi [options]

Stages (run in this order):

    discover   1  expand sitemaps into the SQLite crawl frontier
    crawl      1  fetch, archive and extract manual documents
    download   2  stream Hugging Face corpora into shards
    clean     3+4 normalize, de-boilerplate, quality filter
    lid-train  5  train the six-class Devanagari classifier
    lid        5  apply it; quarantine out-of-language documents
    dedup      6  exact + near-duplicate dedup within one language
    cross      6  cross-corpus collision detection and removal (both languages)
    split      7  group-aware train/val/test manifests
    sweep     8b  vocabulary sweep and selection
    tokenizer 8a  train the final SentencePiece model
    encode    8c  write uint16 memmaps + meta.json
    stats      8  tables and figures for the report
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .io_utils import dump_json, load_config, read_dir


def paths(lang: str) -> dict:
    """Standard on-disk layout for one language directory."""
    root = Path(lang)
    d = root / "data"
    return {
        "root": root,
        "configs": root / "configs",
        "raw_manual": d / "raw" / "manual",
        "raw_downloaded": d / "raw" / "downloaded",
        "raw_html": d / "raw_html",
        "clean": d / "clean",
        "lid": d / "lid",
        "quarantine": d / "quarantine",
        "dedup": d / "dedup_combined" if (d / "dedup_combined").exists() else d / "dedup",
        "final": d / "final",
        "splits": d / "splits",
        "reports": d / "reports",
        "frontier": d / "frontier.sqlite",
        "tokenizer": root / "tokenizer",
        "tokens": d / "tokens",
    }


def tokenizer_model(p: dict, cfg: dict) -> Path:
    """Resolve the tokenizer selected in ``<lang>/configs/tokenizer.yaml``.

    The path is derived from ``model_type`` and ``vocab_size`` -- the same naming
    scheme :func:`cmd_tokenizer` writes -- so the model that gets loaded is always
    the one the config selects.

    An earlier version loaded a fixed ``<lang_code>_spm.model`` and fell back to
    the alphabetically first ``*.model`` when it was missing.  Both are silently
    wrong once a sweep has left several vocabularies in the directory: a stale
    ``hi_spm.model`` from an abandoned V=16000 run kept being loaded while
    ``cfg["vocab_size"]`` (10000) was written into ``meta.json`` beside it,
    producing token ids far outside the declared vocabulary.  The size check
    below makes that class of mismatch impossible: it costs one model load and
    catches a corruption that is otherwise invisible until Phase-2 training
    indexes off the end of its embedding table.
    """
    import sentencepiece as spm

    vocab_size = cfg["vocab_size"]
    model_type = cfg.get("model_type", "unigram")
    model = p["tokenizer"] / f"{cfg['lang_code']}_{model_type}_{vocab_size}.model"
    if not model.exists():
        available = sorted(x.name for x in p["tokenizer"].glob("*.model"))
        raise SystemExit(
            f"tokenizer {model} not found. Train it first:\n"
            f"    python -m lmacorpus.cli tokenizer --lang {p['root'].name}\n"
            f"Models present: {', '.join(available) if available else '(none)'}"
        )
    actual = spm.SentencePieceProcessor(model_file=str(model)).get_piece_size()
    if actual != vocab_size:
        raise SystemExit(
            f"{model.name} has vocab_size {actual}, but "
            f"{p['configs'] / 'tokenizer.yaml'} declares {vocab_size}. Encoding "
            f"with this model would write token ids the metadata does not admit."
        )
    return model


# --------------------------------------------------------------------- stages


def cmd_discover(a):
    """Walk each site's sitemap tree and load the URLs into the frontier.

    Sitemap walking is pure network work and is not cached, so re-running this
    over an already-discovered domain re-downloads its entire sitemap tree just
    for ``INSERT OR IGNORE`` to throw every row away.  ``--new-only`` and
    ``--domains`` exist to skip that; domains are also walked concurrently,
    since they are separate hosts and nothing is gained by serialising them.
    """
    from .acquire.crawler import Crawler
    from .acquire.frontier import Frontier

    p = paths(a.lang)
    cfg = load_config(p["configs"] / "sources.yaml")
    fr = Frontier(p["frontier"])
    known = fr.url_counts()
    configured = {s["domain"] for s in cfg["manual_sites"]}

    wanted = None
    if a.domains:
        wanted = {d.strip() for d in a.domains.split(",") if d.strip()}
        missing = wanted - configured
        if missing:
            raise SystemExit(
                f"--domains not present in {a.lang}/configs/sources.yaml: "
                f"{', '.join(sorted(missing))}"
            )

    todo, skipped = [], []
    for site in cfg["manual_sites"]:
        n = known.get(site["domain"], 0)
        if wanted is not None and site["domain"] not in wanted:
            skipped.append((site["domain"], n, "not in --domains"))
        elif a.new_only and n:
            skipped.append((site["domain"], n, "--new-only"))
        else:
            todo.append(site)

    for dom, n, why in skipped:
        print(f"[discover] skip {dom:34s} ({n:,} urls already known; {why})")
    if not todo:
        print("[discover] nothing to do")
        return
    print(f"[discover] walking {len(todo)} domains, {a.concurrency} at a time")

    async def go():
        sem = asyncio.Semaphore(a.concurrency)
        async with Crawler(fr, p["raw_manual"], cfg["lang_code"]) as c:

            async def one(site):
                async with sem:
                    urls = await c.discover(site["root"], a.max_urls)
                # add_many is synchronous, so no interleaving on the connection
                fr.add_many(urls, site["domain"])
                before = known.get(site["domain"], 0)
                after = fr.url_counts().get(site["domain"], 0)
                print(f"[discover] {site['domain']:34s} found={len(urls):>7,} "
                      f"new={after - before:>7,}", flush=True)

            await asyncio.gather(*(one(s) for s in todo))

    asyncio.run(go())
    dump_json(p["reports"] / "frontier_stats.json", fr.stats())


def cmd_crawl(a):
    from .acquire.crawler import Crawler
    from .acquire.frontier import Frontier

    p = paths(a.lang)
    cfg = load_config(p["configs"] / "sources.yaml")
    fr = Frontier(p["frontier"])
    source_of = {s["domain"]: s["name"] for s in cfg["manual_sites"]}

    rps = getattr(a, "rps", None) or cfg.get("rps_per_domain", 1.5)
    concurrency = getattr(a, "concurrency", None) or 5

    # Category priority -> per-domain tier index.  Every domain is crawled
    # simultaneously; the rank only decides how deeply each one is claimed per
    # round, so the under-represented registers fill faster without idling hosts.
    priority = list(cfg.get("category_priority", []))
    tier_of = {c: i for i, c in enumerate(priority)}
    domain_rank = {
        s["domain"]: tier_of[s["category"]]
        for s in cfg["manual_sites"]
        if s.get("category") in tier_of
    }
    if priority:
        print(f"[crawl] priority weighting (x{a.priority_boost} deepest first): "
              f"{' > '.join(priority)}")

    async def go():
        async with Crawler(
            fr,
            p["raw_manual"],
            cfg["lang_code"],
            raw_html_dir=p["raw_html"],
            rps_per_domain=rps,
            concurrency_per_domain=concurrency,
            max_connections=a.max_connections,
        ) as c:
            await c.run(
                source_of,
                batch=a.batch,
                max_docs=a.max_docs,
                domain_rank=domain_rank,
                priority_boost=a.priority_boost,
            )

    asyncio.run(go())
    dump_json(p["reports"] / "frontier_stats.json", fr.stats())


def cmd_download(a):
    from .acquire.hf_stream import stream_source

    p = paths(a.lang)
    cfg = load_config(p["configs"] / "sources.yaml")
    out = []
    for src in cfg["hf_sources"]:
        if a.only and src["name"] != a.only:
            continue
        out.append(
            stream_source(
                repo=src["repo"],
                config=src.get("config"),
                split=src.get("split", "train"),
                lang=cfg["lang_code"],
                source=src["name"],
                out_dir=p["raw_downloaded"] / src["name"],
                text_key=src.get("text_key", "text"),
                url_key=src.get("url_key"),
                max_docs=a.max_docs or src.get("max_docs"),
            )
        )
    dump_json(p["reports"] / "download_stats.json", out)


def cmd_clean(a):
    from .pipeline import run_clean

    p = paths(a.lang)
    cfg = load_config(p["configs"] / "clean.yaml")
    for kind in ("manual", "downloaded"):
        src = p["raw_manual"] if kind == "manual" else p["raw_downloaded"]
        if not src.exists():
            continue
        run_clean(src, p["clean"] / kind, p["reports"] / f"clean_{kind}.json", cfg,
                  workers=a.workers)


def cmd_lid_train(a):
    from .clean.langid import build_training_set, train

    X, y = build_training_set(n_per_class=a.n_per_class)
    result = train(X, y, a.out)
    dump_json(Path(a.out).with_suffix(".report.json"), result)
    print(json.dumps(result["classification_report"]["accuracy"], indent=2))


def cmd_lid(a):
    from .pipeline import run_langid

    p = paths(a.lang)
    cfg = load_config(p["configs"] / "clean.yaml")
    run_langid(
        p["clean"],
        p["lid"],
        p["quarantine"],
        a.model,
        cfg["lang_code"],
        p["reports"] / "langid.json",
        min_conf=cfg.get("lid_min_conf", 0.90),
        batch_size=a.batch_size,
    )


def cmd_dedup(a):
    from .clean.dedup import dedup_near

    p = paths(a.lang)
    stats = dedup_near(p["lid"], p["dedup"], threshold=a.threshold)
    dump_json(p["reports"] / "dedup.json", stats)
    print(json.dumps(stats, indent=2))


def cmd_cross(a):
    """Cross-corpus collisions.  Drops offending documents from BOTH languages."""
    from .clean.dedup import cross_corpus_collisions, drop_ids

    pa, pb = paths(a.lang_a), paths(a.lang_b)
    res = cross_corpus_collisions(pa["dedup"], pb["dedup"], threshold=a.threshold)
    bad_a, bad_b = set(res["collisions_a"]), set(res["collisions_b"])
    res["kept_a"] = drop_ids(pa["dedup"], pa["final"], bad_a)
    res["kept_b"] = drop_ids(pb["dedup"], pb["final"], bad_b)
    for p in (pa, pb):
        dump_json(p["reports"] / "cross_corpus.json", res)
    print(
        f"[cross] collisions: {res['n_collisions_a']} in {a.lang_a}, "
        f"{res['n_collisions_b']} in {a.lang_b}"
    )


def cmd_split(a):
    from .splits import build_splits

    p = paths(a.lang)
    src = p["final"] if p["final"].exists() else p["dedup"]
    stats = build_splits(src, p["splits"], a.val_frac, a.test_frac)
    dump_json(p["reports"] / "splits.json", stats)
    print(json.dumps({k: v["docs"] for k, v in stats.items()}, indent=2))


def cmd_sweep(a):
    from .tokenizer.train_spm import train_spm, write_training_sample
    from .tokenizer.vocab_sweep import evaluate_model, held_out_text, sweep_table

    p = paths(a.lang)
    cfg = load_config(p["configs"] / "tokenizer.yaml")
    src = p["final"] if p["final"].exists() else p["dedup"]
    sample = p["tokenizer"] / "spm_input.txt"
    if not sample.exists():
        n = write_training_sample(src, p["splits"] / "train.txt", sample,
                                  max_lines=cfg.get("max_sample_lines", 8_000_000))
        print(f"[sweep] wrote {n:,} training lines")

    texts = held_out_text(src, p["splits"] / "val.txt", max_docs=cfg.get("eval_docs", 5000))
    results = []
    for V in cfg["sweep_vocab_sizes"]:
        for mt in cfg.get("sweep_model_types", ["unigram"]):
            prefix = p["tokenizer"] / "sweep" / f"{cfg['lang_code']}_{mt}_{V}"
            model = train_spm(sample, prefix, V, model_type=mt)
            r = evaluate_model(model, texts)
            r["model_type"] = mt
            results.append(r)
            print(f"[sweep] {mt} V={V}: fertility={r['fertility_tokens_per_word']}")
    table = sweep_table(results, d_model=cfg.get("d_model", 512),
                        budget=cfg.get("param_budget", 25_000_000))
    dump_json(p["reports"] / "vocab_sweep.json", table)


def cmd_tokenizer(a):
    from .tokenizer.train_spm import train_spm, write_training_sample

    p = paths(a.lang)
    cfg = load_config(p["configs"] / "tokenizer.yaml")
    src = p["final"] if p["final"].exists() else p["dedup"]

    vocab_size = a.vocab_size or cfg["vocab_size"]
    model_type = a.model_type or cfg.get("model_type", "unigram")

    prefix = p["tokenizer"] / f"{cfg['lang_code']}_{model_type}_{vocab_size}"
    model_path = Path(f"{prefix}.model")
    if model_path.exists():
        print(f"[tokenizer] Model already exists at '{model_path}', skipping training.")
        return

    sample = p["tokenizer"] / "spm_input.txt"
    if not sample.exists():
        write_training_sample(src, p["splits"] / "train.txt", sample,
                              max_lines=cfg.get("max_sample_lines", 8_000_000))
    model = train_spm(
        sample, prefix, vocab_size, model_type=model_type
    )
    print(f"[tokenizer] Trained {model_type} (V={vocab_size:,}) model at {model}")


def cmd_encode(a):
    from .tokenizer.encode_corpus import encode_all

    p = paths(a.lang)
    cfg = load_config(p["configs"] / "tokenizer.yaml")
    src = p["final"] if p["final"].exists() else p["dedup"]
    model = tokenizer_model(p, cfg)
    print(f"[encode] tokenizer {model.name} (V={cfg['vocab_size']:,})", flush=True)
    meta = encode_all(src, p["splits"], model, p["tokens"], cfg["vocab_size"],
                      num_threads=a.workers)
    ok = meta["manual_requirement"]["satisfied"]
    print(f"[encode] manual requirement satisfied: {ok}")


def cmd_stats(a):
    from . import stats as S
    from .viz import plots as P

    p = paths(a.lang)
    cfg = load_config(p["configs"] / "tokenizer.yaml")
    lang_name = a.lang.capitalize()
    fig = Path("report/figures")
    src = p["final"] if p["final"].exists() else p["dedup"]

    inv = S.source_inventory(src)
    inv.to_csv(p["reports"] / "source_inventory.csv", index=False)
    P.plot_source_inventory(inv, fig / f"{a.lang}_source_inventory.png", lang_name)

    hist = S.length_histogram(src)
    P.plot_length_histogram(hist, fig / f"{a.lang}_length_hist.png", lang_name)

    meta = json.loads((p["tokens"] / "meta.json").read_text())
    mf = S.manual_fraction_by_split(meta)
    mf.to_csv(p["reports"] / "manual_fraction.csv", index=False)
    P.plot_manual_fraction(mf, fig / f"{a.lang}_manual_fraction.png", lang_name)

    sweep = json.loads((p["reports"] / "vocab_sweep.json").read_text())
    P.plot_vocab_sweep(sweep, fig / f"{a.lang}_vocab_sweep.png", lang_name,
                       budget=cfg.get("param_budget", 25_000_000))

    model = tokenizer_model(p, cfg)
    tf = S.token_frequency(model, src, p["splits"] / "val.txt")
    tf.head(50).to_csv(p["reports"] / "top50_tokens.csv", index=False)
    P.plot_zipf(tf, fig / f"{a.lang}_zipf.png", lang_name)

    ex = S.tokenization_examples(model, cfg.get("example_sentences", []))
    ex.to_csv(p["reports"] / "tokenization_examples.csv", index=False)
    print("[stats] tables in", p["reports"], "figures in", fig)


# ------------------------------------------------------------------------ CLI


def main():
    ap = argparse.ArgumentParser(prog="lmacorpus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, lang=True):
        s = sub.add_parser(name)
        if lang:
            s.add_argument("--lang", required=True, help="hindi | nepali")
        s.set_defaults(fn=fn)
        return s

    s = add("discover", cmd_discover)
    s.add_argument("--max-urls", type=int, default=200_000)
    s.add_argument("--new-only", action="store_true",
                   help="skip domains that already have URLs in the frontier "
                        "(avoids re-downloading sitemaps you already walked)")
    s.add_argument("--domains", default=None,
                   help="comma-separated domains to walk; must appear in "
                        "sources.yaml. Overrides nothing else -- combine with "
                        "--new-only freely")
    s.add_argument("--concurrency", type=int, default=6,
                   help="domains walked simultaneously")

    s = add("crawl", cmd_crawl)
    s.add_argument("--max-docs", type=int, default=None)
    s.add_argument("--rps", type=float, default=None, help="requests per second per domain (e.g. 4.0 for 3x faster crawl)")
    s.add_argument("--concurrency", type=int, default=None, help="concurrent connections per domain (e.g. 5 or 8)")
    s.add_argument("--batch", type=int, default=200,
                   help="base URLs claimed per domain per round")
    s.add_argument("--priority-boost", type=int, default=4,
                   help="round-share multiplier for the highest-priority "
                        "categories (1 disables weighting)")
    s.add_argument("--max-connections", type=int, default=None,
                   help="httpx pool size (default: domains x concurrency, "
                        "capped at 1024)")


    s = add("download", cmd_download)
    s.add_argument("--max-docs", type=int, default=None)
    s.add_argument("--only", default=None, help="run a single named source")

    s = add("clean", cmd_clean)
    s.add_argument("--workers", type=int, default=None,
                   help="process pool size (default: all cores)")

    s = add("lid-train", cmd_lid_train, lang=False)
    s.add_argument("--out", default="models/langid_devanagari.pkl")
    s.add_argument("--n-per-class", type=int, default=20_000)

    s = add("lid", cmd_lid)
    s.add_argument("--model", default="models/langid_devanagari.pkl")
    s.add_argument("--batch-size", type=int, default=512,
                   help="documents classified per predict_proba call")

    s = add("dedup", cmd_dedup)
    s.add_argument("--threshold", type=float, default=0.8)

    s = add("cross", cmd_cross, lang=False)
    s.add_argument("--lang-a", default="hindi")
    s.add_argument("--lang-b", default="nepali")
    s.add_argument("--threshold", type=float, default=0.8)

    s = add("split", cmd_split)
    s.add_argument("--val-frac", type=float, default=0.01)
    s.add_argument("--test-frac", type=float, default=0.01)

    add("sweep", cmd_sweep)
    s = add("tokenizer", cmd_tokenizer)
    s.add_argument("--vocab-size", type=int, default=None, help="vocabulary size (e.g. 8000, 16000, 24000, 32000)")
    s.add_argument("--model-type", choices=["unigram", "bpe"], default=None, help="SentencePiece algorithm (unigram or bpe)")
    s = add("encode", cmd_encode)
    s.add_argument("--workers", type=int, default=None,
                   help="SentencePiece encode threads (default: all cores)")
    add("stats", cmd_stats)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
